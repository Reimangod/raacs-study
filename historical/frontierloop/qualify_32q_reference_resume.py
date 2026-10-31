"""Check only the previously unstarted second32q case; preserve v1 prefix."""
from io import BytesIO
from pathlib import Path
from time import process_time,perf_counter
import numpy as np
import ffsim
from qiskit import qpy
from .paths import ROOT,OLD
from .io import load,save,sha256,space
from .compression_trial import guard,validate_row,DIRECTORY


def main():
    guard();prior=ROOT/'qualification/D2-full32q-reference-v1'
    out=ROOT/'qualification/D2-full32q-reference-resume-v2';out.mkdir(exist_ok=True)
    first=prior/'N2-32-R1p2-P2.json';record=load(first)
    def verify(r,case):
        source=OLD/'prepared'/case/'C_LUCJ_COMPRESSED_L1-L1/operator.npz';row=validate_row(case,'EXACT')
        if r['original_operator_SHA256']!=sha256(source.read_bytes()) or r['source_QPY_SHA256']!=row['exact_execution_QPY_SHA256']:
            raise RuntimeError('Prior32q reference input changed')
        if max(r['state_max_error'],r['probability_L1_error'])>1e-8:raise RuntimeError('Prior parity did not pass')
    verify(record,'N2-32-R1p2-P2')
    if (out/'result.json').exists():print('ALREADY_QUALIFIED_NO_REEXECUTION');return
    if (out/'started.json').exists():raise RuntimeError('Partial successor preserved; do not repeat')
    case='N2-32-R2p4-P2'
    if (prior/(case+'.json')).exists():raise RuntimeError('Second case already observed, reconcile rather than recompute')
    space();save(out/'started.json',{'first_qualification_SHA256':sha256(first.read_bytes()),
        'first_not_recomputed':True,'only_new_case':case,'code_SHA256':sha256(Path(__file__).read_bytes()),'new_model_shots':0})
    m=load(OLD/'data/physical'/case/'problem.json');n=m['norb'];ne=m['nalpha'],m['nbeta']
    source=OLD/'prepared'/case/'C_LUCJ_COMPRESSED_L1-L1/operator.npz'
    with np.load(source,allow_pickle=False) as a:op=ffsim.UCJOpSpinBalanced(a['diag_coulomb_mats'],a['orbital_rotations'],a['final_orbital_rotation'])
    row=validate_row(case,'EXACT');payload=(DIRECTORY/case/'EXACT/source.qpy').read_bytes()
    if sha256(payload)!=row['exact_execution_QPY_SHA256']:raise RuntimeError('Stored primitive input drift')
    circuit=qpy.load(BytesIO(payload))[0];cpu,tick=process_time(),perf_counter()
    expected=ffsim.apply_unitary(ffsim.hartree_fock_state(n,ne),op,n,ne,copy=False)
    original_cpu,original_wall=process_time()-cpu,perf_counter()-tick;cpu,tick=process_time(),perf_counter()
    actual=ffsim.qiskit.final_state_vector(circuit,norb=n,nelec=ne).vec
    primitive_cpu,primitive_wall=process_time()-cpu,perf_counter()-tick
    actual*=np.exp(-1j*np.angle(np.vdot(expected,actual)))
    state_error=float(np.max(abs(actual-expected)));probability_error=float(np.sum(abs(abs(actual)**2-abs(expected)**2)))
    if max(state_error,probability_error)>1e-8:raise RuntimeError('Second full32q source/primitive mismatch')
    check={'case':case,'amplitudes_compared':int(actual.size),'state_max_error':state_error,'probability_L1_error':probability_error,
        'source_operator_CPU_seconds':original_cpu,'source_operator_wall_seconds':original_wall,
        'saved_primitive_CPU_seconds':primitive_cpu,'saved_primitive_wall_seconds':primitive_wall,
        'original_operator_SHA256':sha256(source.read_bytes()),'source_QPY_SHA256':row['exact_execution_QPY_SHA256'],
        'new_model_shots':0,'molecular_energy_evaluations':0,'statevector_saved':False}
    save(out/(case+'.json'),check)
    save(out/'result.json',{'status':'BOTH_FULL32Q_SOURCE_PRIMITIVE_PARITIES_PASSED_ADDITIVE_SUCCESSOR',
        'first_prior_result':str(first),'first_prior_SHA256':sha256(first.read_bytes()),'first_not_recomputed':True,
        'checks':[record,check],'new_model_shots':0,'old_partial_artifacts_unchanged':True,'goal_complete':False})
    print(check,flush=True)


if __name__=='__main__':main()
