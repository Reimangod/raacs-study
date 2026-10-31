"""N44 actual source files/resources/t1-U provenance audit; no new outcome evaluation."""
from io import BytesIO
from pathlib import Path
from time import process_time, perf_counter
import numpy as np
import ffsim
from qiskit import qpy
from paperstudy.lucj import amplitudes_from_saved, naive
from . import fixed_rotation_source as s
from .io import load, save, sha256
from .paths import ROOT


def run():
    cpu, wall = process_time(), perf_counter(); p = s.verified(); t = load(s.RUN/'terminal.json')
    if t['status'] != 'ALL4_FIXED_U_LINEAR_J_SOURCES_CONSTRUCTED' or len(t['rows']) != 4:
        raise ValueError('All4 actual source outputs required')
    rows = []
    for r in t['rows']:
        for f in r['files'].values():
            if sha256(Path(f['path']).read_bytes()) != f['SHA256']: raise ValueError('Source file drift')
        for filename, view in [('preparation.qpy', 'preparation'), ('sampling.qpy', 'Z_sampling_equivalent')]:
            circuit = qpy.load(BytesIO(Path(r['files'][filename]['path']).read_bytes()))[0]
            resources = r['resources'][view]
            if (circuit.count_ops().get('cx', 0), circuit.depth(filter_function=lambda x: x.operation.name == 'cx'), circuit.depth()) != (resources['cnot_count'], resources['cnot_depth'], resources['total_unitary_depth']):
                raise ValueError('Actual QPY resource mismatch')
        t1, t2, kind = amplitudes_from_saved(r['amplitude_file'], r['norb'], r['nelec'][0])
        base = naive(t1, t2, r['norb'], 1)
        with np.load(r['files']['operator.npz']['path'], allow_pickle=False) as z:
            final_error = float(np.max(abs(base.final_orbital_rotation-z['final_orbital_rotation'])))
            U_error = float(np.max(abs(base.orbital_rotations-z['orbital_rotations'])))
        if max(final_error, U_error) > 1e-12: raise ValueError('Registered t1 or explicit U changed')
        rows.append(dict(case=r['case'], amplitude_kind=kind, loss_before=r['naive_masked_loss'],
             loss_after=r['linear_fit']['loss_half_squared'],
             relative_loss_improvement=(r['naive_masked_loss']-r['linear_fit']['loss_half_squared'])/max(r['naive_masked_loss'], 1e-300),
             normal_equation_residual=r['linear_fit']['normal_equation_residual'],
             linear_map_rank=r['linear_fit']['linear_matrix_rank'],
             initialization_CPU_seconds=r['initialization_CPU_seconds'], compile_CPU_seconds=r['compile_CPU_seconds'],
             actual20q_state_error=r['actual20q_state_error'], resources=r['resources']['Z_sampling_equivalent'],
             coefficient_slots=r['coefficient_slots'], independent_parameter_rank=None,
             t1_final_rotation_error=final_error, explicit_naive_U_error=U_error,
             amplitude_creation_cost_record=r['source_amplitude_creation_provenance'],
             upstream_CCSD_record=r['upstream_CCSD_record'], old_fit_cost_record=r['old_initialization_cost']))
    result = dict(status='N44_ALL4_SOURCE_IDENTITY_RESOURCE_AND_INITIALIZATION_AUDITED', rows=rows,
         source_CPU_seconds=t['CPU_seconds'], source_wall_seconds=t['wall_seconds'],
         audit_CPU_seconds=process_time()-cpu, audit_wall_seconds=perf_counter()-wall,
         source_terminal_SHA256=sha256((s.RUN/'terminal.json').read_bytes()),
         source_protocol_SHA256=sha256((s.RUN/'protocol.json').read_bytes()),
         new_samples_CI_FCI_optimizer_GPU=0, physical_energy_or_Recovery_superiority_proven=False,
         mathematical_loss_not_actual_energy=True, goal_complete=False)
    save(ROOT/'analysis/N44-fixed-U-source-summary.json', result)
    for r in rows: print(r['case'], 'loss improvement', r['relative_loss_improvement'], 'CPU', r['initialization_CPU_seconds'])
    print('N44 all16 source artifacts8 QPYs4 explicit-U/t1 lineages audited')


if __name__ == '__main__': run()
