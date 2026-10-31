"""N41 post-run source/hash, HF-anchor and actual QPY resource audit; no new sampling/CI."""
from io import BytesIO
from pathlib import Path
from time import process_time, perf_counter
import numpy as np
from qiskit import qpy
from . import cheap_mp2_qualification_v2 as source
from .io import load, save, sha256
from .paths import ROOT, OLD


def run():
    cpu, wall = process_time(), perf_counter(); p = source.verified(); t = load(source.RUN/'terminal.json')
    if t['status'] != 'ALL4_SOURCES_QUALIFIED_AND_CONSTRUCTED' or len(t['rows']) != 4:
        raise ValueError('All four source outcomes required')
    rows = []
    for r in t['rows']:
        item = next(i for i in p['catalog'] if i['source']['case'] == r['case']); s, d = item['source'], item['physical']
        for f in r['files'].values():
            if sha256(Path(f['path']).read_bytes()) != f['SHA256']:
                raise ValueError('Source artifact changed')
        for view, filename in [('preparation', 'preparation.qpy'), ('Z_sampling_equivalent', 'sampling.qpy')]:
            qc = qpy.load(BytesIO(Path(r['files'][filename]['path']).read_bytes()))[0]
            actual = dict(cnot_count=int(qc.count_ops().get('cx', 0)),
                          cnot_depth=int(qc.depth(filter_function=lambda x: x.operation.name == 'cx')),
                          total_unitary_depth=int(qc.depth()))
            if any(actual[k] != r['resources'][view][k] for k in actual):
                raise ValueError('Independent ordinary-gate resource recount failed')
        with np.load(d['local_integrals'], allow_pickle=False) as z:
            h, g = z['h1'], z['eri']
        k = d['nelec'][0]
        hf = float(d['core_energy']+2*np.trace(h[:k, :k])+2*np.einsum('iijj', g[:k, :k, :k, :k])-np.einsum('ijji', g[:k, :k, :k, :k]))
        metadata = load(OLD/'data/physical'/r['case']/'problem.json')
        hf_error = abs(hf-metadata['hf_energy_Ha'])
        if hf_error > 1e-8:
            raise ValueError('Wrong active-space HF/core/physical orbital frame')
        old = load(s['construction']); cost = r['old_initialization_cost']
        rows.append(dict(case=r['case'], HF_anchor_error_Ha=hf_error,
                         MP2_plus_factory_CPU_seconds=r['MP2_CPU_seconds']+r['factory_CPU_seconds'],
                         new_compile_CPU_seconds=r['compile_CPU_seconds'],
                         old_compressed_fit_CPU_seconds=cost['compressed_initialization_CPU_seconds'],
                         old_amplitude_CPU_seconds=cost['amplitude_creation_CPU_seconds'],
                         old_all_initializer_stages_converged=cost['all_initializer_stages_converged'],
                         new_sampling_resources=r['resources']['Z_sampling_equivalent'],
                         old_sampling_resources=old['resources']['Z_sampling_equivalent'],
                         new_coefficient_slots=r['coefficient_slots'], old_coefficient_slots=old['coefficient_slots'],
                         independent_parameter_rank=None,
                         actual20q_state_error=r['actual20q_state_error_up_to_phase'],
                         large_molecular_state_verification='NOT_RUN_NO_GPU_ALLOCATION',
                         HF_gradient_norm=r['MP2_qualification']['HF_RHF_orbital_gradient_norm']))
    result = dict(version='N41_ALL4_SOURCE_AUDIT_V1', rows=rows,
                  terminal_SHA256=sha256((source.RUN/'terminal.json').read_bytes()),
                  protocol_SHA256=sha256((source.RUN/'protocol.json').read_bytes()),
                  audit_CPU_seconds=process_time()-cpu, audit_wall_seconds=perf_counter()-wall,
                  molecular_CI_FCI_or_sampling_outcomes=0,
                  decision='SOURCE_PREPARATION_FEASIBLE_CHEAP_NOT_ACCURACY_OR_BEST_PAPER_SUPERIORITY',
                  scope='KNOWN_MP2_NAIVE_FULLSPIN_L1_NOT_OPTIMIZED_COMPRESSED_FIT_EQUIVALENCE;ALL4_KEPT')
    save(ROOT/'analysis/N41-cheap-mp2-summary.json', result)
    for r in rows:
        print(r['case'], 'MP2+factory CPU', round(r['MP2_plus_factory_CPU_seconds'], 6),
              'new/old CX-depth-total', [(r[x]['cnot_count'], r[x]['cnot_depth'], r[x]['total_unitary_depth'])
                                       for x in ('new_sampling_resources', 'old_sampling_resources')])
    print('N41 all20 source artifact hashes,8 QPY resources,4 HF anchors verified')


if __name__ == '__main__':
    run()
