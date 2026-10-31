"""Full32q source-operator vs saved primitive circuit, no shots or energies.

Independent execution paths: original UCJ orbital-matrix application versus
QPY roundtrip elementary quantum gates. Large vectors exist only in RAM.
"""
from io import BytesIO
from time import perf_counter,process_time
import numpy as np
import ffsim
from qiskit import qpy
from .paths import ROOT,OLD
from .io import load,save,sha256,space
from .compression_trial import guard,validate_row,DIRECTORY


def main():
    guard();d=ROOT/'qualification/D2-full32q-reference-v1';d.mkdir(parents=True,exist_ok=True)
    if (d/'result.json').exists():return load(d/'result.json')
    if (d/'started.json').exists():raise RuntimeError('Partial full32q check preserved')
    save(d/'started.json',{'code_SHA256':sha256(__import__('pathlib').Path(__file__).read_bytes()),'new_model_shots':0})
    rows=[]
    for case in ['N2-32-R1p2-P2','N2-32-R2p4-P2']:
        space();m=load(OLD/'data/physical'/case/'problem.json');n=m['norb'];ne=(m['nalpha'],m['nbeta'])
        source=OLD/'prepared'/case/'C_LUCJ_COMPRESSED_L1-L1/operator.npz'
        with np.load(source,allow_pickle=False) as a:
            op=ffsim.UCJOpSpinBalanced(a['diag_coulomb_mats'],a['orbital_rotations'],a['final_orbital_rotation'])
        row=validate_row(case,'EXACT');cell=DIRECTORY/case/'EXACT'
        assert sha256((cell/'source.qpy').read_bytes())==row['exact_execution_QPY_SHA256']
        q=qpy.load(BytesIO((cell/'source.qpy').read_bytes()))[0]
        cpu,tick=process_time(),perf_counter()
        expected=ffsim.apply_unitary(ffsim.hartree_fock_state(n,ne),op,n,ne,copy=False)
        operator_cpu=process_time()-cpu;operator_wall=perf_counter()-tick
        cpu,tick=process_time(),perf_counter()
        actual=ffsim.qiskit.final_state_vector(q,norb=n,nelec=ne).vec
        primitive_cpu=process_time()-cpu;primitive_wall=perf_counter()-tick
        phase=np.angle(np.vdot(expected,actual));actual*=np.exp(-1j*phase)
        state_error=float(np.max(abs(actual-expected)))
        probability_error=float(np.sum(abs(abs(actual)**2-abs(expected)**2)))
        if max(state_error,probability_error)>1e-8:raise RuntimeError('Full32q exact compiler changed source state')
        check={'case':case,'amplitudes_compared':int(actual.size),'state_max_error':state_error,
            'probability_L1_error':probability_error,'source_operator_CPU_seconds':operator_cpu,
            'source_operator_wall_seconds':operator_wall,'saved_primitive_CPU_seconds':primitive_cpu,
            'saved_primitive_wall_seconds':primitive_wall,'new_model_shots':0,'molecular_energy_evaluations':0,
            'original_operator_SHA256':sha256(source.read_bytes()),'source_QPY_SHA256':row['exact_execution_QPY_SHA256'],
            'statevector_saved':False}
        save(d/(case+'.json'),check);rows.append(check);del actual,expected,op
        print(check,flush=True)
    result={'status':'TWO_FULL32Q_SOURCE_PRIMITIVE_PARITIES_PASSED','checks':rows,
        'new_model_shots':0,'molecular_energy_evaluations':0,'goal_complete':False}
    save(d/'result.json',result);return result


if __name__=='__main__':main()
