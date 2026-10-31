"""Post-outcome Naph source diagnostic; no new samples, CI, selection or rescue."""
from pathlib import Path
from time import process_time, perf_counter
import numpy as np
import ffsim
from .paths import ROOT, OLD
from .io import load, save, sha256
from . import cheap_source_sampling as s


def run():
    cpu, wall = process_time(), perf_counter(); p = s.verified()
    if (ROOT/'analysis/N42-Naph-source-diagnostic.json').exists():
        raise RuntimeError('Post-outcome diagnostic already recorded; no rerun')
    current = next(r for r in p['sources'] if r['case'] == 'NAPH20-BLOCK-BOYS')
    cost = load(ROOT/'analysis/SOURCE-cost-components-v1.json')
    naive = next(r for r in cost['rows'] if r['case'] == current['case'] and r['method'] == 'C5_B_LUCJ_NAIVE_L1')
    oldresult = load(naive['source_result']); oldconstruct = oldresult['detail']['construction']
    if oldresult['problem_id'] != current['problem_id'] or sha256(Path(naive['source_result']).read_bytes()) != naive['source_result_SHA256']:
        raise ValueError('Existing naive source physical identity differs')
    oldnaive = OLD/'prepared/NAPH20-BLOCK-BOYS/B_LUCJ_NAIVE_L1-L1/operator.npz'
    if sha256(oldnaive.read_bytes()) != oldconstruct['operator_sha256']:
        raise ValueError('Actual existing naive operator drift')
    item = next(x for x in load(s.init.RUN/'protocol.json')['catalog'] if x['source']['case'] == current['case'])
    definitions = [('NEW_MP2_NAIVE_L1', Path(current['files']['operator.npz']['path'])),
                   ('EXISTING_CCSD_NAIVE_L1', oldnaive),
                   ('EXISTING_CCSD_COMPRESSED_L1', Path(item['source']['operator']))]
    rows = []
    for name, path in definitions:
        with np.load(path, allow_pickle=False) as z:
            op = ffsim.UCJOpSpinBalanced(z['diag_coulomb_mats'], z['orbital_rotations'], z['final_orbital_rotation'])
        hf = ffsim.hartree_fock_state(10, (5, 5)); state = ffsim.apply_unitary(hf, op, 10, (5, 5))
        prob = abs(state)**2; overlap = float(abs(np.vdot(hf, state))**2)
        if abs(prob.sum()-1) > 1e-8:
            raise ValueError('Actual source norm failed')
        rows.append(dict(name=name, problem_id=current['problem_id'], operator=str(path), operator_SHA256=sha256(path.read_bytes()),
                         exact_HF_probability=overlap, expected_nonHF_draws_per1000=1000*(1-overlap),
                         probability_all3000_HF=overlap**3000,
                         diag_Coulomb_Frobenius_norm=float(np.linalg.norm(op.diag_coulomb_mats)),
                         fixed_20q_sector_dimension=len(state)))
    output = dict(status='POST_OUTCOME_PRIVILEGED_DISTRIBUTION_DIAGNOSTIC_ONLY', case=current['case'], rows=rows,
                  original_N42_outcome_unchanged=True, used_for_source_selection=False,
                  no_causal_assignment_of_MP2_vs_factorization=True,
                  existing_naive100k_not_same_sampling_or_spin_objective_as_N42=True,
                  new_samples_CI_FCI_GPU_optimizer=0, CPU_seconds=process_time()-cpu, wall_seconds=perf_counter()-wall,
                  source_protocol_SHA256=sha256((s.LOCAL/'protocol.json').read_bytes()))
    save(ROOT/'analysis/N42-Naph-source-diagnostic.json', output)
    for row in rows: print(row['name'], 'P(HF)', row['exact_HF_probability'], 'expected nonHF/1000', row['expected_nonHF_draws_per1000'])


if __name__ == '__main__': run()
