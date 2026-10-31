"""N56 outcome-free native SDK filtering inventory; no circuit draws or CI."""
import hashlib,json,os,resource,sys
from pathlib import Path
from time import process_time,perf_counter
from collections import Counter


def diagonal(actions):
    return Counter(int(q) for create,q in actions if create)==Counter(int(q) for create,q in actions if not create)


def run(root):
    cpu,wall=process_time(),perf_counter();p=json.loads((root/'protocol.json').read_text())
    if (root/'terminal.json').exists():raise RuntimeError('No replay')
    resource.setrlimit(resource.RLIMIT_CPU,(30,31))
    import numpy as np
    from scipy.linalg import expm
    from paperstudy.sqdrift_native_worker import Exporter
    from qiskit_fermions.operators import FermionOperator
    from qiskit_fermions.operators.terms.filtering import filter_diagonal_terms
    def save(path,value):
        tmp=path.with_name(path.name+'.tmp')
        with tmp.open('w') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    toy=FermionOperator.from_terms([((),1.),(((True,0),(False,0)),2.),
        (((True,0),(False,1)),.3),(((True,1),(False,0)),.3)]).normal_ordered().simplify(atol=1e-12)
    filter_diagonal_terms(toy);assert len(list(toy.iter_terms()))==2
    assert all(not diagonal(a) for a,c in toy.iter_terms())
    x=np.array([[0.,1.],[1.,0.]]);z=np.diag([1.,-1.]);v=np.array([1.,0.])
    pfull=abs(expm(-1j*(x+z))@v)**2;pfiltered=abs(expm(-1j*x)@v)**2
    qualification=dict(SDK_filter_toy_passed=True,diagonal_filter_not_exact_Z_sampling=True,
        counterexample_probability_L1=float(np.sum(abs(pfull-pfiltered))))
    assert qualification['counterexample_probability_L1']>.1
    rows=[];save(root/'qualification.json',qualification)
    for item in p['cases']:
        if perf_counter()-wall>60:raise TimeoutError('Finite N56 preparation wall cap')
        source=Path(item['source']);blob=source.read_bytes()
        assert hashlib.sha256(blob).hexdigest()==item['SHA256']
        request=json.loads(blob);start=process_time();exporter=Exporter(request)
        groups=exporter.operator.split_out_groups();weights=list(exporter.weights)
        assert len(groups)==len(weights)
        flags=[[diagonal(a) for a,c in group.iter_terms()] for group in groups]
        pure=[i for i,f in enumerate(flags) if f and all(f)];mixed=[i for i,f in enumerate(flags) if any(f) and not all(f)]
        before={tuple((bool(a),int(q)) for a,q in actions):complex(c) for actions,c in exporter.operator.iter_terms()}
        expected={a:c for a,c in before.items() if not diagonal(a)}
        lam=float(sum(weights));filter_diagonal_terms(exporter.operator)
        after={tuple((bool(a),int(q)) for a,q in actions):complex(c) for actions,c in exporter.operator.iter_terms()}
        assert set(after)==set(expected) and all(abs(after[a]-expected[a])<1e-12 for a in after)
        assert all(not diagonal(a) for a in after)
        filtered_weights=list(exporter.operator.group_weights());remaining=float(sum(filtered_weights))
        row=dict(case=item['case'],problem_id=request['problem_id'],n_qubits=request['n_qubits'],
            original_terms=len(before),filtered_terms=len(after),removed_diagonal_terms=len(before)-len(after),
            original_groups=len(groups),filtered_groups=len(filtered_weights),pure_diagonal_groups=len(pure),mixed_groups=len(mixed),
            original_group_weight_sum=lam,filtered_group_weight_sum=remaining,
            pure_diagonal_draw_probability=float(sum(weights[i] for i in pure)/lam),
            removed_group_weight_fraction=1-remaining/lam,CPU_seconds=process_time()-start,
            source_SHA256=item['SHA256'],same_original_physical_H_for_future_CI_required=True,
            distribution_preserved=False,new_model_samples=0,new_CI_roots=0)
        rows.append(row);save(root/'progress.json',dict(rows=rows));print('N56_CASE',item['case'],row['pure_diagonal_draw_probability'],flush=True)
    result=dict(status='N56_ALL4_FILTER_PREFLIGHT_COMPLETED',rows=rows,qualification=qualification,
        CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,new_model_samples=0,new_QPU_shots=0,
        new_CI_roots=0,circuit_resource_gain_or_energy_gain_proven=False,goal_complete=False)
    save(root/'terminal.json',result);print(result['status'],flush=True)


if __name__=='__main__':run(Path(sys.argv[1]))
