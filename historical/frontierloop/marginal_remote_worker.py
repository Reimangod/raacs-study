"""Standalone F2 experiment using pinned pure kernels on a Slurm CPU node.

No local source paths, FCI, reference energies, GPU or environment mutation.
All positive-mixture outcomes and resource costs, including poor ones, survive.
"""
import hashlib
from importlib.metadata import version
from io import BytesIO
import json
import os
from pathlib import Path
import resource
import shutil
import sqlite3
import sys
from time import process_time,perf_counter
import numpy as np
import ffsim
from qiskit import QuantumCircuit,qpy,transpile
from qiskit.quantum_info import Statevector
from frontierloop.conditional_marginal import components,conditional_state,circuit
from frontierloop.primitive_plan import strings


def sha(blob):return hashlib.sha256(blob).hexdigest()
def digest(value):return sha(json.dumps(value,sort_keys=True,separators=(',',':')).encode())
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.pending')
    temporary.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n');temporary.replace(path)
def npz(**a):
    f=BytesIO();np.savez_compressed(f,**a);return f.getvalue()
def write(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(str(path)+'.pending');temp.write_bytes(payload);temp.replace(path)


def guard(root,spec):
    if shutil.disk_usage(root).free<spec['remote_free_floor_bytes']:
        raise OSError('Remote free floor; no automatic deletion')
    if sum(p.stat().st_size for p in root.rglob('*') if p.is_file())>spec['new_namespace_bytes_cap']:
        raise OSError('New remote namespace cap')


def fixture(seed):
    rng=np.random.default_rng(seed);n=4
    aa=rng.normal(0,.2,(n,n));aa=(aa+aa.T)/2
    ab=rng.normal(0,.2,(n,n));ab=(ab+ab.T)/2
    return ffsim.UCJOpSpinBalanced(np.array([[aa,ab]]),np.array([ffsim.random.random_unitary(n,seed=seed)]),
        ffsim.random.random_unitary(n,seed=seed+1))


def qualify():
    checks=[]
    for seed,ne in [(4801,(2,2)),(4802,(1,2)),(4803,(3,3)),(4804,(2,1))]:
        op=fixture(seed);v=ffsim.apply_unitary(ffsim.hartree_fock_state(4,ne),op,4,ne)
        v=v.reshape(len(strings(4,ne[0])),len(strings(4,ne[1])));target=v@v.conj().T
        actual=np.zeros_like(target);terms=components(op,ne)
        for t in terms:
            x=conditional_state(op,ne,t);actual+=t['probability']*np.outer(x,x.conj())
        error=float(np.max(abs(actual-target)))
        if error>1e-11:raise RuntimeError('Remote reduced-state identity failed')
        checks.append({'seed':seed,'reduced_state_max_error':error})
    op=fixture(4805);ne=(2,2);term=components(op,ne)[5]
    expected=conditional_state(op,ne,term)
    actual=Statevector.from_instruction(circuit(op,ne,term).decompose(reps=5)).data[strings(4,2)]
    error=float(abs(1-abs(np.vdot(actual,expected))))
    if error>1e-10:raise RuntimeError('Remote half-circuit full-Fock check failed')
    checks.append({'seed':4805,'half_circuit_independent_state_infidelity':error})
    prep,sampling,_=resource_circuit(circuit(op,ne,term))
    a=Statevector.from_instruction(prep).probabilities()
    b=Statevector.from_instruction(sampling).probabilities()
    resource_error=float(np.max(abs(a-b)))
    if resource_error>1e-11:raise RuntimeError('Physical Z-sampling resource view changed distribution')
    checks.append({'half_circuit_sampling_resource_probability_max_error':resource_error})
    rejected=False
    try:components(fixture(4806),(2,2),max_partner_columns=1)
    except ValueError:rejected=True
    if not rejected:raise RuntimeError('Pattern cap did not fail closed')
    checks.append({'over_cap_rejected':True})
    return checks


BASIS=['rz','ry','rx','h','s','sdg','x','cx']
def resource_circuit(q):
    preparation=transpile(ffsim.qiskit.PRE_INIT.run(q),basis_gates=BASIS,
        optimization_level=1,seed_transpiler=8137)
    diagonal={'rz','z','s','sdg','t','tdg','p','cp','cz','rzz','id'};blocked=set();keep=[]
    for item in reversed(preparation.data):
        wires={preparation.find_bit(w).index for w in item.qubits}
        if item.operation.name in diagonal and not wires&blocked:continue
        keep.append(item);blocked|=wires
    sampling=preparation.copy_empty_like()
    for item in reversed(keep):sampling.append(item.operation,[sampling.qubits[preparation.find_bit(w).index] for w in item.qubits])
    sampling=transpile(sampling,basis_gates=BASIS,optimization_level=1,seed_transpiler=8137)
    def counters(c):return {'n_qubits':c.num_qubits,'cnot_count':int(c.count_ops().get('cx',0)),
        'cnot_depth':int(c.depth(filter_function=lambda i:i.operation.name=='cx')),
        'total_depth':int(c.depth()),'numeric_gate_coefficient_slots_not_rank':sum(len(i.operation.params) for i in c)}
    return preparation,sampling,{'preparation':counters(preparation),'sampling':counters(sampling)}


def select_half_pool(ids,counts,n,na,r):
    if int(counts.sum())<=0 or any(int(x).bit_count()!=na for x in ids):raise RuntimeError('Wrong one-spin sector')
    hf=(1<<na)-1;frequency={int(x):int(c) for x,c in zip(ids,counts)}
    ranked=sorted((x for x in frequency if x!=hf),key=lambda x:(-frequency[x],x))
    return np.array(sorted([hf,*ranked[:r-1]]),dtype=np.int64)


def draw_case(root,case,row,spec):
    d=root/'outputs'/case;d.mkdir(parents=True,exist_ok=True)
    if (d/'draw-result.json').exists():return load(d/'draw-result.json')
    if (d/'draw-started.json').exists():raise RuntimeError('Partial F2 draw preserved; no automatic retry')
    guard(root,spec);m=row['physical'];n=m['norb'];ne=m['nalpha'],m['nbeta']
    if ne[0]!=ne[1]:raise ValueError('F2 pilot only uses registered balanced problems')
    with np.load(root/row['operator_file'],allow_pickle=False) as a:
        op=ffsim.UCJOpSpinBalanced(a['diag_coulomb_mats'],a['orbital_rotations'],a['final_orbital_rotation'])
    if not np.allclose(op.diag_coulomb_mats[0,1],op.diag_coulomb_mats[0,1].T,atol=1e-12):
        raise ValueError('Spin-exchange symmetry needed for shared alpha/beta pool')
    cpu,tick=process_time(),perf_counter();terms=components(op,ne,spec['partner_columns_cap'])
    seed=int(digest([spec['seed_family'],m['problem_id']])[:8],16)
    allocation=np.random.default_rng(seed).multinomial(spec['model_shots_per_case'],[x['probability'] for x in terms])
    baseline=QuantumCircuit(2*n);baseline.append(ffsim.qiskit.PrepareHartreeFockJW(n,ne),baseline.qubits)
    baseline.append(ffsim.qiskit.UCJOpSpinBalancedJW(op),baseline.qubits)
    _,_,full_resources=resource_circuit(baseline);members=[]
    for index,(term,shots) in enumerate(zip(terms,allocation)):
        guard(root,spec);q=circuit(op,ne,term);prep,sampling,resources_=resource_circuit(q)
        hashes={}
        for name,c in [('preparation',prep),('sampling',sampling)]:
            stream=BytesIO();qpy.dump(c,stream);payload=stream.getvalue()
            write(d/'circuits'/f'{index:03d}-{name}.qpy',payload);hashes[name]=sha(payload)
        members.append({'member':index,'partner_columns':term['partner_columns'],'partner_bits':term['bits'],
            'probability':term['probability'],'conditional_phases':term['conditional_phases'].tolist(),
            'model_shots':int(shots),'resources':resources_,'QPY_SHA256':hashes})
    compile_cpu=process_time()-cpu;compile_wall=perf_counter()-tick
    save(d/'members.json',{'source':row,'seed':seed,'members':members,'baseline_resources':full_resources,
        'pattern_and_all_compilation_CPU_seconds':compile_cpu,'wall_seconds':compile_wall})
    db=sqlite3.connect(d/'draws.sqlite');db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE draws(member INTEGER PRIMARY KEY,request TEXT,counts BLOB,CPU_seconds REAL,wall_seconds REAL)');db.commit()
    save(d/'draw-started.json',{'run_id':row['run_id'],'model_shots':spec['model_shots_per_case'],'seed':seed})
    frequency={};cpu,tick=process_time(),perf_counter()
    for member,term in zip(members,terms):
        shots=member['model_shots']
        if not shots:continue
        guard(root,spec)
        if process_time()-cpu>=spec['per_case_sampling_boundary_CPU_seconds']:
            raise TimeoutError('F2 sampler CPU boundary cap; preserve partial member draws')
        c,t=process_time(),perf_counter();v=conditional_state(op,ne,term)
        if abs(float(np.vdot(v,v).real)-1)>1e-10:raise RuntimeError('Conditional state norm failed')
        draw_seed=int(digest([seed,member['member'],'independent-member-draw'])[:8],16)
        values=ffsim.sample_state_vector(v,norb=n,nelec=ne[0],shots=shots,
            bitstring_type=ffsim.BitstringType.INT,seed=draw_seed)
        ids,counts=np.unique(values,return_counts=True);payload=npz(half_strings=ids.astype(np.uint64),counts=counts.astype(np.uint64))
        db.execute('INSERT INTO draws VALUES(?,?,?,?,?)',(member['member'],json.dumps({**member,'seed':draw_seed},sort_keys=True),
            payload,process_time()-c,perf_counter()-t));db.commit()
        for x,count in zip(ids,counts):frequency[int(x)]=frequency.get(int(x),0)+int(count)
    db.close();ids=np.array(sorted(frequency),np.uint64);counts=np.array([frequency[int(x)] for x in ids],np.uint64)
    payload=npz(half_strings=ids,counts=counts);write(d/'half-counts.npz',payload)
    if int(counts.sum())!=spec['model_shots_per_case']:raise RuntimeError('F2 allocation accounting mismatch')
    weighted={k:sum(m['model_shots']*m['resources']['sampling'][k] for m in members)/int(counts.sum())
        for k in ('cnot_count','cnot_depth','total_depth','numeric_gate_coefficient_slots_not_rank')}
    result={'status':'HALF_REGISTER_POSITIVE_MIXTURE_DRAW_COMPLETED','run_id':row['run_id'],
        'physical_problem_id':m['problem_id'],'sampler_qubits':n,'source_full_qubits':2*n,
        'physical_nelec':list(ne),'counts_are_half_strings_not_full_determinants':True,
        'known_model_shots':int(counts.sum()),'half_configurations':int(counts.sum()),
        'full_register_comparator_half_configurations_at_same_shots':2*int(counts.sum()),
        'counts_SHA256':sha(payload),'members_SHA256':sha((d/'members.json').read_bytes()),
        'model_CNOT_applications':int(round(weighted['cnot_count']*int(counts.sum()))),
        'sampling_resources_shot_weighted_mean':weighted,'sampling_member_count':len(members),
        'baseline_resources_same_runtime':full_resources,'source_model_coordinates_unchanged':True,
        'nonredundant_parameter_rank':None,'source_initialization_record':row['source_initialization_record'],
        'pattern_and_all_compilation_CPU_seconds':compile_cpu,
        'sampling_CPU_seconds':process_time()-cpu,'sampling_wall_seconds':perf_counter()-tick,
        'actual_QPU_shots':0,'GPU_used':False,'statevector_saved':False}
    save(d/'draw-result.json',result);print('F2_DRAW',case,result['known_model_shots'],weighted,flush=True)
    return result


def load(path):return json.loads(Path(path).read_text())
def readouts(root,case,row,spec):
    from frontierloop.orbital import optimize as original
    from frontierloop.spin_orbital import optimize as penalized
    from frontierloop.fast_orbital import optimize as singlet
    d=root/'outputs'/case;draw=load(d/'draw-result.json');m=row['physical'];n=m['norb'];ne=m['nalpha'],m['nbeta']
    if sha((d/'half-counts.npz').read_bytes())!=draw['counts_SHA256']:raise RuntimeError('F2 counts binding drift')
    with np.load(d/'half-counts.npz',allow_pickle=False) as a:ids,counts=a['half_strings'],a['counts']
    selected=select_half_pool(ids,counts,n,ne[0],spec['R_cap'])
    with np.load(root/row['integrals_file'],allow_pickle=False) as a:h,g=a['h1'],a['eri']
    for mode in spec['readouts']:
        guard(root,spec);out=d/mode;out.mkdir(exist_ok=True)
        if (out/'result.json').exists():continue
        if (out/'started.json').exists():raise RuntimeError('Partial F2 OO preserved')
        save(out/'started.json',{'counts_SHA256':draw['counts_SHA256'],'CI_strings':selected.tolist()})
        cpu,tick=process_time(),perf_counter()
        try:
            ospec=spec['readout_specs'][mode]
            def checkpoint(history,rotations):save(out/'checkpoint.json',{'history':history,'new_model_shots':0})
            if mode=='C4-SINGLET-OO':r=singlet(h,g,m['core_energy'],ne,selected,ospec,mode,checkpoint)
            else:r=(original if mode=='C2-OO' else penalized)(h,g,m['core_energy'],ne,selected,ospec,checkpoint)
            rotations,hh,gg=(r.pop(k) for k in ('rotations','final_h1','final_eri'))
            payload=npz(rotations=np.array(rotations),h1=hh,eri=gg,CI_strings=selected)
            write(out/'optimized-representation.npz',payload)
            r.update(status='COMPLETED',case=case,method='F2_POSITIVE_MARGINAL',readout=mode,
                physical_problem_id=m['problem_id'],source_run_id=row['run_id'],counts_SHA256=draw['counts_SHA256'],
                CI_strings=selected.tolist(),actual_CI_dimension=len(selected)**2,r_cap=spec['R_cap'],
                representation_SHA256=sha(payload),whole_body_CPU_seconds=process_time()-cpu,
                whole_body_wall_seconds=perf_counter()-tick,new_model_shots=0,actual_QPU_shots=0,
                CI_state_not_sampler_prepared_state=True,goal_complete=False)
            save(out/'result.json',r);print('F2_READOUT',case,mode,r['final_energy_Ha'],flush=True)
        except Exception as exc:
            save(out/'failure.json',{'status':'FAILED_ENGINEERING_PRESERVED','error':repr(exc),
                'CPU_seconds':process_time()-cpu,'wall_seconds':perf_counter()-tick,'new_model_shots':0})
            print('F2_READOUT_FAILED',case,mode,repr(exc),flush=True)


def main(root):
    root=Path(root).resolve();manifest=load(root/'manifest.json');spec=manifest['protocol']
    if (root/'terminal.json').exists():print('ALREADY_TERMINAL_NO_REEXECUTION');return
    if (root/'started.json').exists():raise RuntimeError('Existing remote prefix; preserve without automatic retry')
    guard(root,spec)
    for name,hash_ in manifest['input_SHA256'].items():
        if sha((root/name).read_bytes())!=hash_:raise RuntimeError('Uploaded input/runtime code changed: '+name)
    runtime={p:version(p) for p in manifest['expected_runtime']}
    if runtime!=manifest['expected_runtime']:raise RuntimeError('Qualified package versions differ')
    save(root/'started.json',{'runtime':runtime,'job_id':os.environ.get('SLURM_JOB_ID'),'GPU_requested':False})
    cpu,tick=process_time(),perf_counter();save(root/'qualification.json',{'checks':qualify(),'model_shots':0})
    results=[]
    for case in spec['cases']:
        row=manifest['cases'][case]
        try:
            draw=draw_case(root,case,row,spec);readouts(root,case,row,spec)
            results.append({'case':case,'draw_status':draw['status'],'known_model_shots':draw['known_model_shots']})
        except Exception as exc:
            save(root/'outputs'/case/'failure.json',{'status':'FAILED_ENGINEERING_PRESERVED','error':repr(exc),
                'no_unknown_draws_refunded':True});results.append({'case':case,'status':'FAILED_ENGINEERING_PRESERVED'})
    save(root/'terminal.json',{'status':'F2_REMOTE_PREFIX_TERMINAL_NOT_SUPERIORITY','cases':results,
        'CPU_seconds':process_time()-cpu,'wall_seconds':perf_counter()-tick,
        'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'actual_QPU_shots':0,'GPU_used':False,'goal_complete':False})


if __name__=='__main__':main(sys.argv[1])
