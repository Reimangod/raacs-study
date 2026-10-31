"""N57 qualified finite sampling. No old output or reference enters the policy."""
from collections import Counter
from io import BytesIO
from pathlib import Path
from time import process_time,perf_counter
import gzip,json,os,resource,select,subprocess,sys,traceback
import numpy as np
from qiskit import qpy
from qiskit.quantum_info import Statevector
from .paths import ROOT,OLD,PROJECT
from .io import load,save,write,arrays,sha256,digest,space
from .filtered_SqDRIFT_plan import LOCAL,verified,length
from .strong_bank_prepare import prepare as prepare_compiler,verify_batch
from paperstudy.sqdrift_execution import member_state
from paperstudy.lucj import circuit_views
from paperstudy.ci import interleaved_to_blocked
from paperstudy.budget import BudgetLedger


def cpu_total():
    r=resource.getrusage(resource.RUSAGE_CHILDREN)
    return process_time()+r.ru_utime+r.ru_stime


class Session:
    def __init__(self,base,directory):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        if (self.directory/'owner.json').exists():raise RuntimeError('Existing exporter owner; reconcile first')
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
            PYTHONPATH=os.pathsep.join([str(PROJECT/'.runtime-overlays/sqdrift-qiskit25-v1'),str(OLD)]))
        self.log=(self.directory/'stderr.txt').open('wb')
        command=['rtk','proxy',sys.executable,'-u',str(ROOT/'frontierloop/filtered_SqDRIFT_exporter.py'),
            '--base',str(base),'--directory',str(self.directory)]
        self.p=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,env=env,cwd=PROJECT)
        save(self.directory/'owner.json',dict(pid=self.p.pid,command=command))
        self.ready=self.read();self.live_CPU=self.ready['setup_CPU_seconds']
        save(self.directory/'setup.json',self.ready)
    def read(self):
        if not select.select([self.p.stdout],[],[],60)[0]:raise TimeoutError('Exporter observation timed out; no restart')
        line=self.p.stdout.readline()
        if not line:raise RuntimeError('Exporter exited; preserved stderr at '+str(self.directory))
        return json.loads(line)
    def export(self,arm,spec,target):
        self.p.stdin.write(json.dumps(dict(arm=arm,spec=spec,relative_directory=str(target.relative_to(self.directory))))+'\n');self.p.stdin.flush()
        r=self.read();self.live_CPU=r['worker_CPU_seconds'];return r
    def close(self):
        if self.p.poll() is None:
            self.p.stdin.write('{"stop":true}\n');self.p.stdin.flush();self.p.wait(timeout=15)
        self.log.close()


def qualify():
    directory=LOCAL/'qualification'
    if directory.exists():raise RuntimeError('Existing qualifier; inspect rather than repeat')
    request=dict(n_qubits=4,hf_id=12,problem_id='SYNTHETIC_N57',fermion_terms=[
        dict(term=t,coefficient=[c,0.]) for t,c in [([[0,1],[0,0]],.7),([[1,1],[1,0]],.9),
            ([[0,1],[2,0]],.2),([[2,1],[0,0]],.2),([[1,1],[3,0]],.25),([[3,1],[1,0]],.25),
            ([[0,1],[1,1],[1,0],[0,0]],.3)]])
    save(directory/'base.json',request);meta=dict(n_qubits=4,norb=2,nalpha=1,nbeta=1,hf_id=12)
    session=Session(directory/'base.json',directory/'exporter');rows=[]
    try:
        for arm in ['FULL_N20','FILTERED_SHORT']:
            N=20 if arm=='FULL_N20' else length(session.ready['full_lambda'],session.ready['filtered_lambda'])
            for seed in [571,572]:
                spec=dict(time=1.,num_terms=N,seed=seed,arm=arm);target=session.directory/(arm+'-'+str(seed))
                session.export(arm,spec,target)
                member=load_member(target);state,info=member_state(meta,member)
                q=qpy.load(BytesIO(gzip.decompress((target/'circuit.qpy.gz').read_bytes())))[0]
                prep,sample,_=circuit_views(q);expected=np.zeros(16)
                for x,a in state.items():
                    address=sum(((x>>(3-i))&1)<<member['permutation'][i] for i in range(4));expected[address]=abs(a)**2
                errors=[float(np.sum(abs(abs(Statevector(c).data)**2-expected))) for c in [prep,sample]]
                if max(errors)>1e-8:raise ValueError('Actual native SDK/source probability parity failed')
                rows.append(dict(arm=arm,seed=seed,errors=errors,**info))
    finally:session.close()
    save(directory/'result.json',dict(status='ALL4_NATIVE_FILTERED_TOY_PROBABILITIES_QUALIFIED',rows=rows,
        new_model_samples=0,new_QPU_shots=0,exporter_SHA256=sha256((ROOT/'frontierloop/filtered_SqDRIFT_exporter.py').read_bytes())))
    print('N57_ALL4_SMALL_NATIVE_PARITY_PASSED',flush=True)


def load_member(folder):
    member=json.loads(gzip.decompress((folder/'member.json.gz').read_bytes()))
    if sha256((folder/'circuit.qpy.gz').read_bytes())!=member['circuit_file_sha256']:raise ValueError('Native QPY changed')
    return member


def run():
    p=verified();space()
    if (LOCAL/'source-started.json').exists():raise RuntimeError('Existing source run; do not redraw')
    cpu,wall=cpu_total(),perf_counter();base_bytes=sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())
    initial_local=sum(f.stat().st_size for f in LOCAL.rglob('*') if f.is_file())
    save(LOCAL/'source-started.json',dict(pid=os.getpid(),protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes())))
    ledger=BudgetLedger(LOCAL/'shots.sqlite');completed=[];session=None;checks=0
    def bound():
        space();current=sum(f.stat().st_size for f in LOCAL.rglob('*') if f.is_file())
        if current>p['new_output_bytes'] or base_bytes+current-initial_local>p['research_root_bytes']:raise OSError('N57 registered storage cap')
        if cpu_total()-cpu+(session.live_CPU if session else 0)>p['source_CPU_seconds'] or perf_counter()-wall>p['source_wall_seconds']:
            raise TimeoutError('N57 finite source budget')
    try:
        for c in p['cases']:
            meta=load(c['meta']);meta['case_id']=c['case'];session=Session(c['base'],LOCAL/'exports'/c['case'])
            if abs(session.ready['full_lambda']-c['full_lambda'])>1e-8 or abs(session.ready['filtered_lambda']-c['filtered_lambda'])>1e-8:raise ValueError('Frozen source normalization changed')
            for draw in [d for d in p['draws'] if d['case']==c['case']]:
                folder=LOCAL/'draws'/draw['draw_id'];save(folder/'request.json',draw)
                ledger.register(draw['draw_id'],1000);items=[];members=[]
                for m in draw['members']:
                    bound();target=session.directory/'members'/m['member_id'];reply=session.export(draw['arm'],m['spec'],target)
                    member=load_member(target)
                    if member['spec']!=m['spec'] or member['diagonal_terms_retained']!=(draw['arm']=='FULL_N20'):raise ValueError('Source filter or spec differs')
                    plan=dict(index=m['index'],member_id=m['member_id'],spec=m['spec'],historical_member_shots=m['shots'],
                        artifacts={name:dict(path=str(target/name),sha256=sha256((target/name).read_bytes())) for name in ['member.json.gz','circuit.qpy.gz']})
                    blobs={str(target/name):(target/name).read_bytes() for name in ['member.json.gz','circuit.qpy.gz']}
                    item=prepare_compiler(meta,plan,blobs);items.append(item);members.append(member)
                check=verify_batch(items,wall_limit=60);checks+=len(items)*3
                if any(v['status']!='PASSED' for v in check['results'].values()):raise ValueError('Independent compiler probability verification failed')
                save(folder/'compiler-check.json',check);merged=Counter();records=[]
                for m,member,item in zip(draw['members'],members,items):
                    bound();state,numerics=member_state(meta,member)
                    request=dict(member_id=m['member_id'],problem_id=draw['problem_id'],readout_seed=m['readout_seed'],shots=m['shots'],
                        source_packet_SHA256=sha256((session.directory/'members'/m['member_id']/'member.json.gz').read_bytes()))
                    if ledger.reserve(draw['draw_id'],m['member_id'],request,m['shots'])!='NEW_RESERVATION':raise RuntimeError('Unknown/duplicate model draw; no resample')
                    ids=np.array(sorted(state),np.uint64);prob=np.array([abs(state[int(x)])**2 for x in ids]);prob/=prob.sum()
                    counts=np.random.default_rng(m['readout_seed']).multinomial(m['shots'],prob);keep=counts>0
                    blob=arrays(ids=ids[keep],counts=counts[keep]);write(folder/(m['member_id']+'.npz'),blob)
                    ledger.complete(draw['draw_id'],m['member_id'],int(counts.sum()),sha256(blob))
                    for x,w in zip(ids[keep],counts[keep]):merged[interleaved_to_blocked(int(x),meta['norb'])]+=int(w)
                    r=item['row'];r.update(request=request,counts_SHA256=sha256(blob),state_numerics=numerics)
                    records.append(r)
                    write(folder/(m['member_id']+'-certificate.json.gz'),item['certificate'])
                    for variant,b in item['qpy_blobs'].items():write(folder/(m['member_id']+'-'+variant+'.qpy.gz'),b)
                ids=np.array(sorted(merged),np.uint64);counts=np.array([merged[int(x)] for x in ids],np.uint64)
                mask=(1<<meta['norb'])-1
                if int(counts.sum())!=1000 or any((int(x)&mask).bit_count()!=meta['nalpha'] or (int(x)>>meta['norb']).bit_count()!=meta['nbeta'] for x in ids):raise ValueError('Sample count or particle-sector mismatch')
                blob=arrays(blocked_ids=ids,counts=counts);write(folder/'counts.npz',blob)
                fields=['cnot_count','cnot_depth','total_unitary_depth']
                resources={name:{k:sum(m['shots']*r['resources'][name][k] for m,r in zip(draw['members'],records))/1000 for k in fields}
                    for name in ['NATIVE','RUSTIQ','PROJECTED']}
                resources['SELECTED']={k:sum(m['shots']*r['policy']['selected_resources'][k] for m,r in zip(draw['members'],records))/1000 for k in fields}
                result=dict(status='COMPLETED',draw_id=draw['draw_id'],case=draw['case'],arm=draw['arm'],seed=draw['seed'],
                    problem_id=draw['problem_id'],counts_SHA256=sha256(blob),budget=ledger.snapshot(draw['draw_id']),
                    members=records,resources=resources,coefficient_slots_mean=sum(m['shots']*r['coefficient_slots_original'] for m,r in zip(draw['members'],records))/1000,
                    independent_parameter_rank=None,VQE_optimized_variables=0,protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),
                    new_model_samples=1000,new_QPU_shots=0)
                save(folder/'result.json',result);completed.append({k:result[k] for k in ['draw_id','case','arm','seed','status']})
                save(LOCAL/'source-progress.json',dict(completed=completed,model_samples_committed=1000*len(completed),CPU_seconds_reaped=cpu_total()-cpu))
                print('N57_DRAW',draw['case'],draw['arm'],draw['seed'],len(completed),flush=True)
            session.close();session=None
        save(LOCAL/'source-terminal.json',dict(status='ALL24_FRESH_SOURCE_DRAWS_COMPLETED',completed=completed,
            new_model_samples=24000,new_QPU_shots=0,actual_MPS_circuit_checks=checks,
            CPU_seconds_including_children=cpu_total()-cpu,wall_seconds=perf_counter()-wall,goal_complete=False))
    except Exception as exc:
        save(LOCAL/'source-failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),completed=completed,
            CPU_seconds_reaped=cpu_total()-cpu,unreaped_live_exporter_CPU_seconds=session.live_CPU if session else 0,wall_seconds=perf_counter()-wall))
        raise
    finally:
        if session is not None:session.close()


if __name__=='__main__':{'qualify':qualify,'run':run}[sys.argv[1]]()
