"""Official TrimPolicy with the unchanged C7 physical/penalized CI kernel.

No half-sampler pseudo-pairs, invented CI coefficients, or new quantum draws.
The high-filling root is mapped back WITH fermionic phases before supplying
a physical SCIState to the official policy. This is a common-setup component
study, not the protein embedding/SBD-G/ExtSQD workflow reproduction.
"""
from time import perf_counter, process_time
import numpy as np
import ffsim
from qiskit.primitives import BitArray
from qiskit_addon_sqd.fermion import SCIState, SCIResult, diagonalize_fermionic_hamiltonian
from qiskit_addon_sqd.trim import TrimPolicy
from .lowest_orbital import Reader


def electron_state(reader, strings, n, nelec):
    strings = np.asarray(strings, np.int64)
    vector = reader.previous
    if vector is None:
        raise ValueError('No qualified CI eigenvector')
    if reader.holes:
        lookup = {int(x): i for i, x in enumerate(reader.s)}
        order = [lookup[((1 << n)-1)^int(x)] for x in strings]
        # Overall fixed-nelec phase is irrelevant; determinant-dependent
        # phases are essential for a physical electron-space SCIState.
        signs = np.array([(-1)**sum(q for q in range(n) if int(x) >> q & 1) for x in strings])
        vector = vector[np.ix_(order, order)]*signs[:, None]*signs[None, :]
    state = SCIState(np.array(vector, dtype=float, order='C', copy=True), strings.copy(), strings.copy(), n, tuple(nelec))
    if abs(np.sum(abs(state.amplitudes)**2)-1) > 1e-9:
        raise ValueError('Electron CI vector lost normalization')
    return state


def bit_array(ids, counts, n, nelec):
    ids, counts = list(ids), list(counts)
    if any(not isinstance(x, (int, np.integer)) or isinstance(x, (bool, np.bool_)) for x in ids+counts):
        raise ValueError('Do not truncate noninteger or reinterpret unknown counts/IDs')
    ids, counts = list(map(int, ids)), list(map(int, counts))
    mask = (1 << n)-1
    if len(ids) != len(counts) or len(ids) != len(set(ids)) or any(c <= 0 for c in counts):
        raise ValueError('Actual unique positive source counts required')
    for x in ids:
        if x < 0 or x >= 1 << (2*n) or ((x & mask).bit_count(), (x >> n).bit_count()) != tuple(nelec):
            raise ValueError('Source outside registered blocked particle sector')
    return BitArray.from_counts(dict(zip(ids, counts)), num_bits=2*n)


class Solver:
    def __init__(self, h, g, core, nelec, settings, observer=None):
        self.h, self.g, self.core = np.asarray(h), np.asarray(g), float(core)
        self.n, self.nelec = len(h), tuple(nelec)
        self.settings, self.observer = settings, observer
        self.cpu, self.wall = process_time(), perf_counter()
        self.calls, self.rounds, self.screen_results = [], 0, []

    def __call__(self, subspaces, h, g, n, nelec):
        if n != self.n or tuple(nelec) != self.nelec or not np.array_equal(h, self.h) or not np.array_equal(g, self.g):
            raise ValueError('Official policy/kernel physical identity changed')
        results = []
        for batch, (sa, sb) in enumerate(subspaces):
            remaining = min(self.settings['total_wall_cap_seconds']-(perf_counter()-self.wall),
                            self.settings['total_CPU_cap_seconds']-(process_time()-self.cpu))
            if remaining <= 0:
                raise TimeoutError('Whole Trim readout work cap; preserve all completed screens')
            sa, sb = np.sort(np.asarray(sa, np.int64)), np.sort(np.asarray(sb, np.int64))
            if not np.array_equal(sa, sb) or len(sa) > self.settings['r_cap'] or len(sa) < 1:
                raise ValueError('Common spin-exchange-symmetric R cap violated')
            ham = ffsim.MolecularHamiltonian(h, g, constant=self.core)
            spec = {**self.settings, 'spin_target_S_squared': 0,
                'per_CI_wall_cap_seconds': min(self.settings['per_CI_wall_cap_seconds'], remaining)}
            reader = Reader(n, nelec, sa, spec)
            cpu, wall = process_time(), perf_counter()
            row, rdm = reader.diagonalize(ham)
            state = electron_state(reader, sa, n, nelec)
            occupations = ((sa[:, None] >> np.arange(n)) & 1).astype(float)
            probabilities = abs(state.amplitudes)**2
            occa, occb = probabilities.sum(axis=1)@occupations, probabilities.sum(axis=0)@occupations
            row.update(round=self.rounds, batch=batch, CI_strings=sa.tolist(),
                actual_CI_dimension=len(sa)**2, actual_r=len(sa),
                CPU_seconds=process_time()-cpu, wall_seconds=perf_counter()-wall,
                eigenvector_weight=sum(map(float, probabilities.ravel())),
                electron_marginal_weights_alpha=probabilities.sum(axis=1).tolist(),
                electron_marginal_weights_beta=probabilities.sum(axis=0).tolist(),
                new_model_shots=0, reference_information_used=False)
            self.calls.append(row)
            result = SCIResult(energy=row['energy_Ha'], sci_state=state,
                orbital_occupancies=(occa, occb), rdm1=rdm.one_rdm, rdm2=rdm.two_rdm)
            results.append(result)
            if self.rounds == 0:
                self.screen_results.append(result)
            if self.observer:
                self.observer(self.calls)
        self.rounds += 1
        return results


def run(h, g, core, nelec, ids, counts, hf_string, settings, observer=None):
    n = len(h)
    if nelec[0] != nelec[1] or settings['r_cap'] > 128 or settings['max_iterations'] != 1:
        raise ValueError('Registered one-pass balanced/R<=128 component study only')
    samples = bit_array(ids, counts, n, nelec)
    solver = Solver(h, g, core, nelec, settings, observer)
    result = diagonalize_fermionic_hamiltonian(h, g, samples,
        samples_per_batch=settings['samples_per_batch'], norb=n, nelec=tuple(nelec),
        num_batches=settings['num_batches'], max_iterations=1,
        policy=TrimPolicy(trim_ratio=settings['trim_ratio'],
            max_strings_per_trim=settings['max_strings_per_trim']),
        sci_solver=solver, symmetrize_spin=True, max_dim=settings['r_cap'],
        include_configurations=[int(hf_string)], seed=settings['seed'])
    if solver.rounds != 2 or len(solver.calls) != settings['num_batches']+1:
        raise RuntimeError('Native Trim did not execute the expected screen/merge rounds')
    merged = solver.calls[-1]
    if abs(result.energy-merged['energy_Ha']) > 1e-10:
        raise RuntimeError('Native Trim output not bound to merged root')
    best = min(solver.calls[:-1], key=lambda r: (r['energy_Ha'], r['batch']))
    return {'status': 'COMPLETED', 'calls': solver.calls,
        'native_Trim_energy_Ha': merged['energy_Ha'],
        'native_Trim_penalized_energy_Ha': merged['penalized_energy_Ha'],
        'native_Trim_S2': merged['spin_square'],
        'native_Trim_actual_CI_dimension': merged['actual_CI_dimension'],
        'matched_disjoint_screen_control_energy_Ha': best['energy_Ha'],
        'matched_disjoint_screen_control_S2': best['spin_square'],
        'matched_disjoint_screen_control_batch': best['batch'],
        'screen_CPU_seconds_attributed_to_both_arms': sum(x['CPU_seconds'] for x in solver.calls[:-1]),
        'merge_additional_CPU_seconds': merged['CPU_seconds'],
        'whole_readout_CPU_seconds': process_time()-solver.cpu,
        'whole_readout_wall_seconds': perf_counter()-solver.wall,
        'merged_not_guaranteed_better_than_screen_or_frequency': True,
        'best_screen_is_matched_control_not_native_StandardPolicy_reproduction': True,
        'new_model_shots': 0, 'exact_singlet_certificate': False,
        'FCI_reference_used_for_selection': False}, result.sci_state
