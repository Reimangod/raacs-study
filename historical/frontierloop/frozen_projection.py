"""Exact fixed-occupation restriction of an ordered Pauli network.

Source labels use mode 0 first (MSB in source integer IDs). Qiskit wire j
represents active_modes[j]. This is source-sector/state-preparation equivalence,
NOT full-unitary or arbitrary noisy-channel equivalence. No angle pruning.
"""
import math
from qiskit import QuantumCircuit
from qiskit.synthesis import synth_pauli_network_rustiq
from qiskit.quantum_info import Clifford, Pauli


def fermionic_relabel(sequence, n, hf, permutation):
    """Relabel fermionic modes BEFORE JW, using adjacent FSWAP Clifford action.

    A plain qubit permutation of JW strings is not a fermionic relabeling.
    permutation[old_mode] is the new physical wire, matching the SDK metadata.
    """
    if sorted(permutation) != list(range(n)):
        raise ValueError('Invalid native fermionic permutation')
    wires = list(range(n))
    q = QuantumCircuit(n)
    for dest in range(n):
        old = permutation.index(dest)
        pos = wires.index(old)
        while pos > dest:
            q.swap(pos-1, pos)
            q.cz(pos-1, pos)
            wires[pos-1], wires[pos] = wires[pos], wires[pos-1]
            pos -= 1
    clifford = Clifford(q)
    transformed = []
    for group in sequence:
        result = []
        for label, angle in group:
            pauli = Pauli(label[::-1]).evolve(clifford, frame='s')
            if int(pauli.phase) not in (0, 2):
                raise ValueError('Hermitian Pauli gained imaginary coefficient')
            sign = 1 if int(pauli.phase) == 0 else -1
            unsigned = Pauli((pauli.z, pauli.x, 0)).to_label()[::-1]
            result.append([unsigned, sign*float(angle)])
        transformed.append(result)
    new_hf = sum(((hf >> (n-1-old)) & 1) << (n-1-permutation[old]) for old in range(n))
    return transformed, new_hf


def restrict(sequence, n, hf):
    if n < 0 or hf < 0 or hf >= 1 << n:
        raise ValueError('Invalid source register/HF')
    for group in sequence:
        for label, angle in group:
            if len(label) != n or set(label) - set('IXYZ') or not math.isfinite(angle):
                raise ValueError('Invalid Pauli/angle')
    active = sorted({q for group in sequence for label, _ in group
                     for q, p in enumerate(label) if p in 'XY'})
    frozen = [q for q in range(n) if q not in active]
    occupations = {q: (hf >> (n - 1 - q)) & 1 for q in frozen}
    groups = []
    witnesses = []
    global_phase = 0.
    for group in sequence:
        reduced = []
        for label, angle in group:
            sign = (-1) ** sum(occupations[q] for q in frozen if label[q] == 'Z')
            word = ''.join(label[q] for q in active)
            theta = sign * float(angle)
            dropped = not any(p != 'I' for p in word)
            if dropped:
                global_phase -= theta
            else:
                reduced.append([word, theta])
            witnesses.append({'source_label': label, 'source_angle': float(angle),
                'restricted_label': word, 'sign': sign, 'restricted_angle': theta,
                'global_phase_only': dropped})
        groups.append(reduced)
    return {'n_source': n, 'source_hf': hf, 'active_modes': active,
        'frozen_occupations': {str(k): v for k, v in occupations.items()},
        'groups': groups, 'global_phase': global_phase, 'witnesses': witnesses,
        'scope': 'registered fixed occupations; ideal source measurement distribution',
        'full_unitary_equivalence': False, 'arbitrary_noise_channel_equivalence': False,
        'angle_pruning': False}


def verify(certificate, sequence, n, hf):
    # Independent bit-mask restriction audit; it does not invoke restrict().
    active = certificate['active_modes']
    expected = {i for group in sequence for label, _ in group
                for i in range(n) if label[i] == 'X' or label[i] == 'Y'}
    if active != sorted(expected) or certificate['n_source'] != n or certificate['source_hf'] != hf:
        raise ValueError('Incorrect whole-network active set/source')
    fixed = {str(q): (hf >> (n - 1 - q)) & 1 for q in range(n) if q not in expected}
    if fixed != certificate['frozen_occupations']:
        raise ValueError('Incorrect frozen occupations')
    groups = []
    phase = 0.
    position = 0
    for group in sequence:
        result = []
        for label, angle in group:
            parity = sum((hf >> (n - 1 - q)) & 1 for q in range(n)
                         if q not in expected and label[q] == 'Z') % 2
            sign = 1 if parity == 0 else -1
            word = ''.join(label[q] for q in active)
            dropped = set(word) <= {'I'}
            witness = {'source_label': label, 'source_angle': float(angle),
                'restricted_label': word, 'sign': sign, 'restricted_angle': sign*float(angle),
                'global_phase_only': dropped}
            if certificate['witnesses'][position] != witness:
                raise ValueError('Restriction witness mismatch')
            position += 1
            if dropped:
                phase -= sign*float(angle)
            else:
                result.append([word, sign*float(angle)])
        groups.append(result)
    if position != len(certificate['witnesses']) or groups != certificate['groups'] or phase != certificate['global_phase']:
        raise ValueError('Restriction group/order/phase mismatch')
    return True


def restore(active_id, certificate):
    """Qiskit little-endian active ID -> original source MSB integer ID."""
    active = certificate['active_modes']
    if not 0 <= active_id < 1 << len(active):
        raise ValueError('Active outcome outside register')
    n = certificate['n_source']
    value = sum(int(bit) << (n - 1 - int(q)) for q, bit in certificate['frozen_occupations'].items())
    return value | sum(((active_id >> j) & 1) << (n - 1 - q) for j, q in enumerate(active))


def synthesize(sequence, modes, source_hf, source_n, global_phase=0.):
    """Same order-preserving Rustiq backend for full and restricted labels."""
    n = len(modes)
    qc = QuantumCircuit(n)
    for j, q in enumerate(modes):
        if (source_hf >> (source_n - 1 - q)) & 1:
            qc.x(j)
    network = []
    for group in sequence:
        for label, angle in group:
            support = [j for j, p in enumerate(label) if p != 'I']
            if support:
                network.append((''.join(label[j] for j in support), support, 2*float(angle)))
            else:
                qc.global_phase -= float(angle)
    if network:
        qc.compose(synth_pauli_network_rustiq(n, network, optimize_count=True,
            preserve_order=True, upto_clifford=False, upto_phase=False), inplace=True)
    qc.global_phase += global_phase
    return qc
