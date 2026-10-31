"""Derived verifier for the frozen N17-style worker, with source-specific inputs.

Never selects a new circuit, runs an eigensolver, or changes a frozen request.
"""
import gzip,hashlib
from pathlib import Path
import numpy as np
from .io import load,sha256
from .native_dice_binary import decode
from .joint_ci_plan import joint_pairs
from .analyze_joint_ci import verify_row
from .analyze_prefix_performance_ci import basis_digest,check_observation


def audit(local,source_check):
    spec=load(local/'protocol.json');c=local/'collected';t=load(c/'terminal.json');start=load(c/'started.json')
    psha=sha256((local/'protocol.json').read_bytes())
    if start['protocol_SHA256']!=psha or start['pyscf_threads']!=1 or set(start['thread_env'].values())!={'1'}:raise RuntimeError('Actual protocol/thread binding differs')
    if sha256((local/'worker.py').read_bytes())!=spec['worker_SHA256'] or spec['worker_SHA256']!=spec['original_N17_matched_worker_SHA256']:raise RuntimeError('Exact common worker changed')
    for name,value in spec['pipeline_SHA256'].items():
        if sha256((local/'pipeline'/name).read_bytes())!=value:raise RuntimeError('Qualified math changed')
    if load(c/'qualification/result.json')['status']!='SAME_NATIVE_EXPORT_PHYSICAL_PARITY_PASSED':raise RuntimeError('Actual native export qualification failed')
    states={s['task_id']:s for s in t['states']};plans=spec['plans']
    if len(states)!=spec['planned'] or len(t['states'])!=len(plans) or set(states)!={p['task_id'] for p in plans}:raise RuntimeError('All planned terminal states required')
    observations=load(c/'observations.json');attempts=load(c/'ci-attempts.json');byid={o['id']:o for o in observations}
    done=[a for a in attempts if a['status']=='COMPLETED']
    if len(byid)!=len(observations) or len(done)!=len(observations) or len(observations)!=t['actual_new_CI_roots'] or len(attempts)!=t['ci_attempts'] or {a['observation_id'] for a in done}!=set(byid):raise RuntimeError('Actual CI attempt accounting differs')
    if any(a['status'] not in ('COMPLETED','FAILED_PRESERVED') for a in attempts):raise RuntimeError('Nonterminal numerical attempt')
    for o in observations:
        verify_row(o['row']);path=c/o['vector_file']
        if sha256(path.read_bytes())!=o['vector_SHA256']:raise RuntimeError('Saved actual root changed')
        with np.load(path,allow_pickle=False) as z:pairs=z['joint_pairs'];v=z['coefficients']
        h=basis_digest(pairs)
        if len(pairs)!=len(v) or not np.isfinite(v).all() or abs(v@v-1)>1e-10 or h!=o['basis_digest'] or o['id']!=hashlib.sha256((o['case']+h+psha).encode()).hexdigest():raise RuntimeError('Root/problem/basis identity differs')
        if o['row']['binary_SHA256']!=spec['qualified_external_SHA256'][spec['sparse_library']]:raise RuntimeError('Unregistered numerical kernel')
    bundle=load(local/'bundle-collection-v1.json');rows=[];seen=set();native_cpu=0.
    for p in plans:
        state=states[p['task_id']];case=p['case'];d=spec['physical'][case];folder=c/'cells'/p['task_id'];spaces={}
        if load(folder/'result.json')!=state or state['case']!=case or state['method']!=p['method'] or state['source']!=p['source'] or state['selection_mode']!=p['selection_mode']:raise RuntimeError('Actual plan/result linkage differs')
        if p['source']['expected_problem_id']!=d['problem_id']:raise RuntimeError('Source physical problem differs')
        pool=None
        if p['source']['kind']!='CLASSICAL_HF_ONLY':
            source,strings,pool=source_check(p,d)
            if source!=p['source'] or strings!=p['frequency_strings'] or pool!=p['allowed_strings']:raise RuntimeError('Observed source rule not independently reproduced')
            spaces['SOURCE_FREQUENCY_R128']=[(a,b) for a in strings for b in strings]
        if state.get('native_attempt'):
            native_cpu+=state['native_attempt']['CPU_seconds']
            if load(folder/'native-attempt.json')!=state['native_attempt']:raise RuntimeError('Native work attribution differs')
            for spin in ('Alpha','Beta'):
                if (folder/(spin+'Dets.bin')).read_bytes()!=p['HF_string'].to_bytes(16,'big'):raise RuntimeError('Native start was not HF')
        if 'native' in state:
            raw=gzip.decompress((folder/'dets.bin.gz').read_bytes());h=sha256(raw)
            if h!=state['native']['raw_determinants_SHA256'] or bundle['native_raw_SHA256'].get('cells/'+p['task_id']+'/dets.bin')!=h:raise RuntimeError('Native raw/transport hash differs')
            n,pairs=decode(raw);joint,selection=joint_pairs(pairs,n,tuple(d['nelec']),p['HF_string'])
            if n!=d['norb'] or selection!=state['selection']:raise RuntimeError('Native physical sector/selection differs')
            if pool is not None:
                if (folder/'AllowedHalfDets.bin').read_bytes()!=b''.join(int(x).to_bytes(16,'big') for x in pool) or [int(x) for x in (folder/'CEO-sample-pool-v1.txt').read_text().split()]!=pool or any(a not in pool or b not in pool for a,b,_ in pairs):raise RuntimeError('Actual native search escaped the observed pool')
                if 'CEO_SAMPLE_POOL_V1 ' not in (folder/'native.log').read_text():raise RuntimeError('Restricted-kernel marker missing')
            elif (folder/'CEO-sample-pool-v1.txt').exists():raise RuntimeError('Restriction leaked into unrestricted classical control')
            spaces['JOINT_TOP_D16384']=joint
        for arm in p['requested_readouts']:
            item=state['arms'].get(arm)
            result=check_observation(item,case,spaces[arm],byid,seen) if item else None
            rows.append(dict(case=case,method=p['method'],arm=arm,task_id=p['task_id'],source=p['source'],
                status='COMPLETED' if result else 'READOUT_FAILED_PRESERVED',result=result,
                actual_CI_dimension=result['actual_CI_dimension'] if result else None,
                observation_id=item['observation_id'] if item else None,new_CI_solve=item['new_CI_solve'] if item else None,
                observed_half_pool_size=len(pool) if pool is not None else None,
                native_CPU_seconds=state.get('native_attempt',{}).get('CPU_seconds'),
                standalone_native_plus_CI_CPU_seconds=(result['CPU_seconds']+(state.get('native_attempt',{}).get('CPU_seconds',0.) if arm=='JOINT_TOP_D16384' else 0.)) if result else None,
                error=None if result else state.get('errors',{}).get(arm,state.get('error'))))
    if len(rows)!=spec['readout_endpoints'] or len(seen)!=len(observations) or native_cpu+sum(a['CPU_seconds'] for a in attempts)>t['CPU_seconds']+1e-5:raise RuntimeError('Unattributed or double-counted numerical work')
    return dict(rows=rows,terminal=t,actual_new_CI_roots=len(observations),failed_CI_attempts=sum(a['status']=='FAILED_PRESERVED' for a in attempts),
        protocol_SHA256=psha,terminal_SHA256=sha256((c/'terminal.json').read_bytes()))
