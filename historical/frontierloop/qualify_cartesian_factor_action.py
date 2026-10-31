"""N52 one bounded actual-integral H/S action parity, no CI eigensolve."""
from pathlib import Path
from time import process_time,perf_counter
import numpy as np
from .paths import ROOT,OLD
from .io import load,save,sha256,space
from .cartesian_factor_action import factor_action
from .native_CI_kernel_feasibility import action_inputs
from .energy_cache import hole_integrals
from .joint_ci_sparse import matrices
from .fe_native_readout_diagnostic import LIBRARY

LOCAL=ROOT/'runs/N52-CARTESIAN-FACTOR-ACTION-MATH-v1'


def run():
    if (LOCAL/'terminal.json').exists():raise RuntimeError('Completed pilot; no duplicate')
    space();cpu,wall=process_time(),perf_counter()
    n51=load(ROOT/'runs/N51-FULL-BATCH-CI-v1/protocol.json')
    task=next(t for t in n51['tasks'] if t['method']=='C5_C_LUCJ_COMPRESSED_L1')
    # Outcome-free first 12 sorted hole strings from the fixed source basis.
    s=np.sort(((1<<20)-1)^np.asarray(task['strings'],np.int64))[:12]
    file=OLD/'data/physical/FE2S2-40-P2/integrals.npz'
    with np.load(file,allow_pickle=False) as z:h,g=z['h1'],z['eri']
    hh,gg,shift=hole_integrals(h,g,0.)
    hop,spin,info=factor_action(hh,gg,(5,5),s,s,LIBRARY)
    native,_=action_inputs(hh,gg,20,(5,5),s)
    H,S,_=matrices(hh,gg,(5,5),[(int(a),int(b)) for a in s for b in s],LIBRARY)
    x=np.random.default_rng(5211).normal(size=len(s)**2)
    errors=dict(H_vs_independent_sparse=float(np.max(abs(hop(x)-H@x))),
                S2_vs_independent_sparse=float(np.max(abs(spin(x)-S@x))),
                H_vs_PySCF=float(np.max(abs(hop(x)-native(x)))))
    if max(errors.values())>1e-8:raise ValueError('Structured factor action differs')
    if process_time()-cpu>30 or perf_counter()-wall>60:raise RuntimeError('Bounded math pilot exceeded')
    result=dict(status='N52_REAL_FE_SMALL_SPACE_EXACT_CARTESIAN_FACTOR_ACTION_PARITY_PASSED',
         R=len(s),D=len(s)**2,errors=errors,factorization=info,
         physical_integrals_SHA256=sha256(file.read_bytes()),
         code_SHA256=sha256(Path(__file__).read_bytes()),
         action_SHA256=sha256((ROOT/'frontierloop/cartesian_factor_action.py').read_bytes()),
         CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
         new_CI_eigensolves=0,ERI_matrix_factorizations=1,new_model_samples=0,new_GPU_jobs=0,
         production_CUDA_implementation=False,large_CI_energy_improvement_claimed=False,goal_complete=False)
    save(LOCAL/'terminal.json',result)
    print(result)


if __name__=='__main__':run()
