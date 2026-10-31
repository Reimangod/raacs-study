"""Independent full-Fock Qiskit verification of actual saved Naph 20q outputs."""
from io import BytesIO
from pathlib import Path
from time import process_time,perf_counter
import ffsim
import numpy as np
from pyscf.fci import cistring
from qiskit import qpy
from qiskit.quantum_info import Statevector
from .paths import ROOT
from .io import load,save,sha256,space
from .slater_prefix_preflight import LOCAL
from paperstudy.lucj import make_circuit


def decode_tensor(vector,mapping):
    n=len(mapping)
    if n>20:raise ValueError('Full-Fock qualification cap20')
    return np.transpose(np.asarray(vector).reshape([2]*n),[n-1-mapping[j] for j in reversed(range(n))]).reshape(-1)


def run():
    space();cpu,tick=process_time(),perf_counter();spec=load(LOCAL/'protocol.json');a=load(ROOT/'analysis/N15-prefix-rounding-summary.json')
    s=next(s for s in spec['sources'] if s['case']=='NAPH20-BLOCK-BOYS' and s['method'].endswith('L1'))
    n=s['norb'];ne=tuple(s['nelec'])
    with np.load(s['operator'],allow_pickle=False) as z:op=ffsim.UCJOpSpinBalanced(z['diag_coulomb_mats'],z['orbital_rotations'],z['final_orbital_rotation'])
    reference=ffsim.qiskit.final_state_vector(make_circuit(op,n,ne),norb=n,nelec=ne).vec
    sa=cistring.make_strings(range(n),ne[0]);sb=cistring.make_strings(range(n),ne[1]);addresses=(sa[:,None]|(sb[None,:]<<n)).reshape(-1)
    selected=[r for r in a['rows'] if r['case']==s['case'] and r['status']=='RESOURCE_PREFLIGHT_ONLY'];cache={};records=[]
    for r in selected:
        mapping=r['selected_decoder_logical_to_physical'];states={}
        for role,key in [('preparation','selected_preparation'),('sampling','selected_sampling')]:
            record=r[key];path=LOCAL/record['path']
            if sha256(path.read_bytes())!=record['SHA256']:raise RuntimeError('Actual QPY changed')
            cache_key=(record['path'],tuple(mapping))
            if cache_key not in cache:
                q=qpy.load(BytesIO(path.read_bytes()))[0]
                if q.num_qubits!=20:raise RuntimeError('Bounded full20 verifier only')
                full=decode_tensor(Statevector.from_instruction(q).data,mapping);sector=full[addresses]
                norm=float(np.vdot(full,full).real);sector_norm=float(np.vdot(sector,sector).real)
                if abs(norm-1)>1e-8 or abs(norm-sector_norm)>1e-8:raise RuntimeError('Compiled circuit norm/sector mismatch')
                cache[cache_key]=sector.copy()
            states[role]=cache[cache_key]
        infidelity=float(max(0.,1-abs(np.vdot(reference,states['preparation']))**2))
        error=abs(infidelity-r['selected_preparation_infidelity'])
        probability_error=float(np.sum(np.abs(abs(states['preparation'])**2-abs(states['sampling'])**2)))
        if error>1e-8 or probability_error>1e-8:raise RuntimeError('Actual compiled decoder/fidelity/probability certificate failed')
        records.append(dict(policy=r['policy'],tier=r['tier'],adopted=r['adopted'],actual_compiled_preparation_infidelity=infidelity,
            determinant_formula_error=error,prep_vs_sampling_probability_L1=probability_error,
            selected_preparation_SHA256=r['selected_preparation']['SHA256'],selected_sampling_SHA256=r['selected_sampling']['SHA256']))
    out=dict(status='ALL6_SELECTED_NAPH20_POINTS_INDEPENDENT_FULL_FOCK_COMPILED_PARITY_PASSED',records=records,
        unique_full_Fock_simulations=len(cache),CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-tick,
        new_CI_solves=0,new_model_shots=0,new_QPU_shots=0,GPU_allocations=0,statevectors_saved=False,
        scope='ACTUAL_20Q_COMPILED_CASE_ONLY_NOT_ALL_32Q40Q_CIRCUITS',implementation_SHA256=sha256(Path(__file__).read_bytes()),
        N15_summary_SHA256=sha256((ROOT/'analysis/N15-prefix-rounding-summary.json').read_bytes()),goal_complete=False)
    save(ROOT/'analysis/N15-compiled20-state-parity.json',out);print(out['status'],'CPU',out['CPU_seconds'],flush=True)


if __name__=='__main__':run()
