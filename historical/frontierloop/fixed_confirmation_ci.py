"""Common finite-penalty CI for frozen N76 counts, preserving missing endpoints."""
from pathlib import Path
from time import process_time,perf_counter
import os,sys,traceback
import numpy as np
from .paths import ROOT
from .io import load,save,write,sha256,digest,arrays
from .fixed_confirmation_execution import LOCAL,bound,capacity,checked_write
from .paired_lowshot import ranked_half
from .joint_ci_sparse import solve
from .contiguous_CI_audit import audit

def pool(requests,records,n,hf):
    if len(records)!=len(requests):raise ValueError('Incomplete source mixture: do not renormalize partial ensemble')
    expected={r['id']:r for r in requests};seen=set();counts={}
    for rec in records:
        r=rec['request'];rid=r['id']
        if rid not in expected or r!=expected[rid] or rid in seen or rec['status']!='COMPLETED':raise ValueError('Request/result identity mismatch')
        seen.add(rid);f=Path(rec['counts_file'])
        if sha256(f.read_bytes())!=rec['counts_SHA256']:raise ValueError('Counts digest mismatch')
        with np.load(f,allow_pickle=False) as a:
            xx,cc=a['blocked_ids'],a['counts']
            if xx.dtype.kind not in 'iu' or cc.dtype.kind not in 'iu' or len(xx)!=len(cc) or len(np.unique(xx))!=len(xx) or np.any(cc<=0) or int(cc.sum())!=r['shots']:raise ValueError('Invalid actual observations')
            mask=(1<<n)-1
            if any((int(x)&mask).bit_count()!=int(hf).bit_count() or (int(x)>>n).bit_count()!=int(hf).bit_count() or int(x)>>2*n for x in xx):raise ValueError('Observed particle sector mismatch')
            for x,c in zip(xx,cc):counts[int(x)]=counts.get(int(x),0)+int(c)
    ids=np.array(sorted(counts),np.uint64);cc=np.array([counts[int(x)] for x in ids],np.uint64)
    if int(cc.sum())!=sum(r['shots'] for r in requests):raise ValueError('Campaign sampling cost mismatch')
    return ids,cc,ranked_half(ids,cc,n,hf)

def prepare():
    science,b=bound();capacity()
    if (LOCAL/'CI-binding.json').exists():raise RuntimeError('Already CI bound; inspect existing execution')
    if not (LOCAL/'m4-terminal.json').exists() or not (LOCAL/'gpu/terminal.json').exists():raise RuntimeError('Source stage still live or unreconciled')
    if not (LOCAL/'hci/terminal.json').exists():raise RuntimeError('HCI source stage still live or unreconciled')
    records={};bindings={};sources=[]
    for f in sorted((LOCAL/'member-draws').glob('*.json')):
        if f.name.endswith('.intent.json'):continue
        rec=load(f)
        if rec.get('status')=='COMPLETED':records[rec['request']['id']]=rec;bindings[str(f)]=sha256(f.read_bytes());bindings[rec['counts_file']]=rec['counts_SHA256']
    hci=load(LOCAL/'hci/terminal.json');bindings[str(LOCAL/'hci/terminal.json')]=sha256((LOCAL/'hci/terminal.json').read_bytes())
    for cell in science['cells']:
        meta=load(next(c['meta'] for c in science['cases'] if c['case']==cell['case']))
        for rep in science['replicates']:
            row=dict(case=cell['case'],method=cell['method'],replicate=rep)
            if cell['status']=='MISSING_SAME_PROBLEM_SOURCE':row.update(status=cell['status'],rank=None)
            elif cell['method']=='CLASSICAL_HF_HCI':
                native=next((r for r in hci['rows'] if r['case']==cell['case']),None)
                row.update(status='COMPLETED' if native and native['status']=='COMPLETED' else 'MISSING_NATIVE_HCI_SOURCE',rank=native['rank'] if native and native['status']=='COMPLETED' else None,
                    deterministic_source_same_for_both_replicates_not_independent=True,native_source_record=native)
            else:
                req=[r for r in b['requests'] if r['case']==cell['case'] and r['method']==cell['method'] and r['replicate']==rep]
                available=[records[r['id']] for r in req if r['id'] in records]
                row.update(expected_member_requests=len(req),completed_member_requests=len(available),actual_new_model_samples=sum(r['actual_new_model_samples'] for r in available),planned_model_samples=sum(r['shots'] for r in req))
                try:
                    xx,cc,rank=pool(req,available,meta['norb'],(1<<meta['nalpha'])-1)
                    file=LOCAL/'pooled-counts'/(digest([cell['case'],cell['method'],rep])+'.npz');blob=arrays(blocked_ids=xx,counts=cc);checked_write(file,blob)
                    row.update(status='COMPLETED',rank=rank,counts_file=str(file),counts_SHA256=sha256(blob));bindings[str(file)]=sha256(blob)
                except Exception as exc:row.update(status='MISSING_OR_INVALID_COMPLETE_SOURCE',rank=None,error=repr(exc))
            sources.append(row)
    endpoints=[]
    for e in science['endpoints']:
        s=next(r for r in sources if (r['case'],r['method'],r['replicate'])==(e['case'],e['method'],e['replicate']))
        row=dict(e,source_status=s['status'])
        if s['status']!='COMPLETED':row.update(status=s['status'],strings=None)
        else:
            count=min(e['R_cap'],len(s['rank']))
            if e['context']=='COMMON_ACTUAL_R32':count=min(32,*(len(r['rank']) for r in sources if r['case']==e['case'] and r['replicate']==e['replicate'] and r['status']=='COMPLETED'))
            row.update(status='FROZEN_READY',strings=sorted(map(int,s['rank'][:count])),actual_R=count,actual_D=count*count)
        endpoints.append(row)
    assert len(sources)==64 and len(endpoints)==320
    from . import joint_ci_sparse,contiguous_CI_audit,portable_penalized_ci,paired_lowshot
    for module in [sys.modules[__name__],joint_ci_sparse,contiguous_CI_audit,portable_penalized_ci,paired_lowshot]:bindings[module.__file__]=sha256(Path(module.__file__).read_bytes())
    tests=ROOT/'tests/test_fixed_confirmation_ci.py';bindings[str(tests)]=sha256(tests.read_bytes())
    import xml.etree.ElementTree as ET
    testout=ROOT/'qualification-fixed-confirmation-CI-v1.xml';suites=list(ET.parse(testout).getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)>=3 and not any(int(s.get(k,0)) for s in suites for k in ['errors','failures'])
    bindings[str(testout)]=sha256(testout.read_bytes())
    save(LOCAL/'CI-binding.json',dict(science_SHA256=sha256((Path(b['science_file'])).read_bytes()),bindings=bindings,sources=sources,endpoints=endpoints,planned_new_model_samples=sum(r['shots'] for r in b['requests']),
        actual_persisted_new_model_samples=sum(r['actual_new_model_samples'] for r in records.values()),execution_binding_SHA256=sha256((LOCAL/'execution-binding.json').read_bytes())))
    print('CI_BOUND320_FIELDS',sum(e['status']=='FROZEN_READY' for e in endpoints),'AVAILABLE',flush=True)

def run():
    science,b=bound();p=load(LOCAL/'CI-binding.json');capacity()
    if (LOCAL/'CI-started.json').exists():raise RuntimeError('Existing CI intent; no duplicate run')
    for f,h in p['bindings'].items():
        if sha256(Path(f).read_bytes())!=h:raise RuntimeError('Frozen CI input/code drift '+f)
    cpu,wall=process_time(),perf_counter();save(LOCAL/'CI-started.json',dict(pid=os.getpid(),binding_SHA256=sha256((LOCAL/'CI-binding.json').read_bytes())))
    rows=[];attempts=[];cache={};physical={}
    order={'R32':0,'R8':1,'R16':2,'R64':3,'COMMON_ACTUAL_R32':4}
    for e in sorted(p['endpoints'],key=lambda e:order[e['context']]):
        row=dict(e)
        if e['status']!='FROZEN_READY':rows.append(row);continue
        c=next(c for c in science['cases'] if c['case']==e['case']);meta=load(c['meta']);strings=e['strings']
        key=digest([meta['problem_id'],strings,science['common_equation'],science['library_SHA256']]);fresh=key not in cache
        if fresh:
            tick=process_time();remaining=min(science['CI']['CPU_cap']-(tick-cpu),science['CI']['wall_cap']-(perf_counter()-wall));entry=dict(status='NOT_RUN_GLOBAL_CI_CAP')
            if remaining>0:
                try:
                    capacity()
                    if e['case'] not in physical:
                        with np.load(c['integrals'],allow_pickle=False) as a:physical[e['case']]=(a['h1'],a['eri'])
                    h,g=physical[e['case']];pairs=np.array([(a,b) for a in strings for b in strings],np.uint64)
                    result,v=solve(h,g,meta['core_energy'],(meta['nalpha'],meta['nbeta']),pairs,Path(science['library']),seconds=min(remaining,science['CI']['root_wall_cap']),triangle_cap=science['CI']['triangle_cap'])
                    independent=audit(h,g,meta['core_energy'],(meta['nalpha'],meta['nbeta']),strings,v,result)
                    file=LOCAL/'CI-roots'/(key+'.npz');blob=arrays(joint_pairs=pairs,coefficients=v);checked_write(file,blob)
                    entry=dict(status='COMPLETED',result=result,independent_RDM_audit=independent,root_file=str(file),root_SHA256=sha256(blob))
                except Exception as exc:entry=dict(status='FAILED_CI_PRESERVED',error=repr(exc),traceback=traceback.format_exc())
            entry['CPU_seconds']=process_time()-tick;cache[key]=entry;attempts.append(dict(basis_id=key,status=entry['status'],CPU_seconds=entry['CPU_seconds']));save(LOCAL/'CI-roots'/(key+'.json'),entry)
        root=cache[key];row.update(basis_id=key,status=root['status'],root=root,new_CI_attempt=fresh,energy_Ha=root.get('result',{}).get('energy_Ha'))
        rows.append(row);save(LOCAL/'CI-progress.json',dict(rows=rows,attempts=attempts,CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall))
        print('CI',len(rows),e['case'],e['method'],e['context'],root['status'],flush=True)
    save(LOCAL/'CI-terminal.json',dict(status='ALL320_ENDPOINT_STATUSES_TERMINAL',rows=rows,attempts=attempts,CPU_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
        actual_new_CI_roots=sum(a['status']=='COMPLETED' for a in attempts),new_QPU_shots=0,new_source_search_authorized=False,goal_complete=False))
    print('CONFIRMATION_CI_TERMINAL320',sum(r['status']=='COMPLETED' for r in rows),flush=True)

if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
