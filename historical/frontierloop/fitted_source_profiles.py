"""N45 input profiles: frozen N44 fit plus newly materialized same-amplitude naive.
No paid-fit U/J input and no new CCSD/CISD/energy/sampling here.
"""
from pathlib import Path
from time import process_time, perf_counter
import copy
import numpy as np
import ffsim
from paperstudy.lucj import amplitudes_from_saved, naive, interaction_pairs
from .fixed_rotation_source import RUN as N44, verified
from .cheap_mp2_qualification_v2 import live_views, serialize
from .direct_sample_pilot import error_up_to_phase
from .prefix_performance_sampling import qualified_module
from .paths import ROOT, OLD
from .io import load, save, write, arrays, sha256

METHOD = 'FIXED_U_LINEAR_J_FIT_L1'
NAIVE_METHOD = 'SAME_AMPLITUDE_NAIVE_L1'
LOCAL = ROOT/'runs/N45-FIXED-FIT-AND-NAIVE-FINITE-v1'


def profiles(rows):
    verified(); cpu, wall = process_time(), perf_counter(); out = []
    for source in rows:
        new = copy.deepcopy(source); new['method_id'] = METHOD; out.append(new)
        r = copy.deepcopy(source); n, ne = r['norb'], tuple(r['nelec']); tick = process_time(); w = perf_counter()
        t1, t2, kind = amplitudes_from_saved(r['amplitude_file'], n, ne[0]); op = naive(t1, t2, n, 1)
        initcpu, initwall = process_time()-tick, perf_counter()-w
        olddir = OLD/'prepared'/r['case']/'B_LUCJ_NAIVE_L1-L1'
        old = load(olddir/'construction.json'); oldfile = olddir/'operator.npz'
        if old['problem_id'] != r['problem_id'] or old['operator_sha256'] != sha256(oldfile.read_bytes()):
            raise ValueError('Existing naive reference/gauge identity differs')
        with np.load(oldfile, allow_pickle=False) as z:
            errors = [float(np.max(abs(z[name]-getattr(op, name)))) for name in ('diag_coulomb_mats', 'orbital_rotations', 'final_orbital_rotation')]
        if max(errors) > 1e-10: raise ValueError('Fresh same-amplitude naive differs from registered source')
        tick = process_time(); w = perf_counter(); logical, prep, sampling, resources = live_views(op, n, ne)
        compilecpu, compilewall = process_time()-tick, perf_counter()-w
        folder = LOCAL/'naive-sources'/r['case']
        write(folder/'operator.npz', arrays(diag_coulomb_mats=op.diag_coulomb_mats, orbital_rotations=op.orbital_rotations, final_orbital_rotation=op.final_orbital_rotation))
        for name, qc in [('logical.qpy', logical), ('preparation.qpy', prep), ('sampling.qpy', sampling)]: write(folder/name, serialize(qc))
        error = None; checkcpu = 0.
        if n == 10:
            tick = process_time(); module = qualified_module(); cfg, values = module.plan(logical, n, ne)
            expected = ffsim.apply_unitary(ffsim.hartree_fock_state(n, ne), op, n, ne)
            error = error_up_to_phase(expected, module.numpy_state(cfg, values))
            if error > 1e-8: raise ValueError('Actual20q naive primitive mismatch')
            checkcpu = process_time()-tick
        r.update(method_id=NAIVE_METHOD, initializer='SAME_AMP_EXPLICIT_NAIVE_L1', linear_fit=None,
                 initialization_CPU_seconds=initcpu, compile_CPU_seconds=compilecpu,
                 initialization_wall_seconds=initwall, compile_wall_seconds=compilewall, actual20q_state_error=error,
                 actual20q_state_check_CPU_seconds=checkcpu,
                 resources=resources, coefficient_slots=len(op.to_parameters(interaction_pairs=interaction_pairs(n))),
                 existing_naive_operator_SHA256=sha256(oldfile.read_bytes()), existing_naive_array_errors=errors,
                 initialization_cost_context='NEW_NAIVE_COSTS_HERE;old_initialization_cost_IS_COMPARATOR_CDF_NOT_NAIVE_COST',
                 files={name: dict(path=str(folder/name), SHA256=sha256((folder/name).read_bytes()))
                        for name in ('operator.npz', 'logical.qpy', 'preparation.qpy', 'sampling.qpy')})
        out.append(r)
        if process_time()-cpu > 30 or perf_counter()-wall > 60: raise TimeoutError('N45 profile construction cap')
    if len(out) != 8 or len({(r['case'], r['method_id']) for r in out}) != 8: raise ValueError('All8 source profiles required')
    save(LOCAL/'profile-construction.json', dict(rows=out, CPU_seconds=process_time()-cpu, wall_seconds=perf_counter()-wall,
         new_molecular_energy_CI_CCSD_samples_GPU=0, comparison_is_same_physical_amplitudes=True))
    return out
