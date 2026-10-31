"""N18 actual terminal status, evidence collection and all28 readout audit."""
import argparse
import gzip
import hashlib
from pathlib import Path
import numpy as np
from .paths import ROOT,OLD
from .io import load,save,write,sha256
from .half_source_native_screen import LOCAL,REMOTE,half_pool
from .native_source_control_plan import initial_strings
from .sample_universe_audit import allowed_pool
from .native_dice_binary import decode
from .joint_ci_plan import joint_pairs
from .analyze_joint_ci import verify_row
from .analyze import reference
from .analyze_prefix_performance_ci import basis_digest,check_observation
from .audit_prefix_sampling import all_control_outputs
from benchmark.gpu_remote import ssh


def status():
    job=load(LOCAL/'job.json')['job_id'];q=ssh('squeue -h -u reiwatanabe -o "%i|%T|%j|%R"',timeout=30)
    live='\n'.join(x for x in q.splitlines() if x.split('|')[0]==job)
    account=ssh('sacct -X -n -j '+job+' --format=JobID,State,Elapsed,ExitCode,AllocTRES -P',timeout=30)
    save(LOCAL/'status.json',dict(job_id=job,live=live,accounting=account));print(live or account,flush=True)


def collect():
    from . import collect_small_pool_bundle as collector
    collector.LOCAL=LOCAL;collector.REMOTE=REMOTE;collector.collect()


def analyze():
    spec=load(LOCAL/'protocol.json');c=LOCAL/'collected';terminal=load(c/'terminal.json');started=load(c/'started.json')
    protocol_sha=sha256((LOCAL/'protocol.json').read_bytes())
    if started['protocol_SHA256']!=protocol_sha or started['pyscf_threads']!=1 or set(started['thread_env'].values())!={'1'}:
        raise RuntimeError('Actual N18 thread/protocol binding differs')
    if sha256((LOCAL/'worker.py').read_bytes())!=spec['worker_SHA256'] or sha256((ROOT/'frontierloop/half_source_native_screen.py').read_bytes())!=spec['preparer_SHA256']:
        raise RuntimeError('Frozen N18 implementation changed')
    for name,h in spec['pipeline_SHA256'].items():
        if sha256((LOCAL/'pipeline'/name).read_bytes())!=h:raise RuntimeError('Qualified math helper changed')
    if load(c/'qualification/result.json')['status']!='SAME_NATIVE_EXPORT_PHYSICAL_PARITY_PASSED':raise RuntimeError('Actual export qualifier failed')
    states={s['task_id']:s for s in terminal['states']};plans=spec['plans']
    if len(plans)!=16 or len(states)!=16 or set(states)!={p['task_id'] for p in plans}:raise RuntimeError('All16 actual terminal states required')
    observations=load(c/'observations.json');attempts=load(c/'ci-attempts.json');byid={o['id']:o for o in observations}
    completed=[a for a in attempts if a['status']=='COMPLETED']
    if len(byid)!=len(observations) or len(completed)!=len(observations) or len(observations)!=terminal['actual_new_CI_roots'] or len(attempts)!=terminal['ci_attempts'] or {a['observation_id'] for a in completed}!=set(byid):
        raise RuntimeError('Actual root/attempt accounting mismatch')
    for o in observations:
        verify_row(o['row']);path=c/o['vector_file']
        if sha256(path.read_bytes())!=o['vector_SHA256']:raise RuntimeError('Saved actual root changed')
        with np.load(path,allow_pickle=False) as a:pairs=a['joint_pairs'];v=a['coefficients']
        h=basis_digest(pairs)
        if len(v)!=len(pairs) or not np.isfinite(v).all() or abs(v@v-1)>1e-10 or h!=o['basis_digest'] or o['id']!=hashlib.sha256((o['case']+h+protocol_sha).encode()).hexdigest():
            raise RuntimeError('Saved root physical/basis/vector identity mismatch')
        if o['row']['binary_SHA256']!=spec['qualified_external_SHA256'][spec['sparse_library']]:raise RuntimeError('Unregistered numerical kernel')
    raw_n15=load(ROOT/'runs/N15-SLATER-PREFIX-ROUNDING-PREFLIGHT-v1/result.json')['rows']
    parameters=load(ROOT/'analysis/N15-declared-coordinate-bindings.json')['rows'];seen=set();rows=[];native_cpu=0.
    bundle=load(LOCAL/'bundle-collection-v1.json');source_controls={}
    for p in plans:
        s=states[p['task_id']];case=p['case'];d=spec['physical'][case];source=p['source'];folder=c/'cells'/p['task_id']
        if load(folder/'result.json')!=s or s['case']!=case or s['method']!=p['method'] or s['source']!=source:
            raise RuntimeError('Actual task/source metadata differs')
        if source['expected_problem_id']!=d['problem_id']:raise RuntimeError('Source problem mismatch')
        spaces={}
        if source['kind']=='QUANTUM_FULL_COUNTS':
            if sha256(Path(source['payload']).read_bytes())!=source['payload_SHA256']:raise RuntimeError('Original full counts changed')
            strings=initial_strings(source,d['norb'],tuple(d['nelec']));pool=sorted(allowed_pool(source,d['norb'],tuple(d['nelec'])))
            with np.load(source['payload'],allow_pickle=False) as a:
                if int(a['counts'].sum())!=100000:raise RuntimeError('Actual full observation budget differs')
        elif source['kind']=='QUANTUM_HALF_COUNTS':
            if sha256(Path(source['payload']).read_bytes())!=source['payload_SHA256']:raise RuntimeError('Original half counts changed')
            with np.load(source['payload'],allow_pickle=False) as a:strings,pool=half_pool(a['half_strings'],a['counts'],d['norb'],d['nelec'][0],p['HF_string'])
        else:strings,pool=None,None
        if strings!=p['frequency_strings'] or pool!=p['allowed_strings']:raise RuntimeError('Source-only frequency/pool rule not reproduced')
        if strings is not None:spaces['SOURCE_FREQUENCY_R128']=[(a,b) for a in strings for b in strings]
        if s.get('native_attempt'):
            native_cpu+=s['native_attempt']['CPU_seconds']
            if load(folder/'native-attempt.json')!=s['native_attempt']:raise RuntimeError('Native attempt accounting differs')
            for spin in ['Alpha','Beta']:
                if (folder/(spin+'Dets.bin')).read_bytes()!=p['HF_string'].to_bytes(16,'big'):raise RuntimeError('Actual native start was not HF')
        if 'native' in s:
            raw=gzip.decompress((folder/'dets.bin.gz').read_bytes());h=sha256(raw)
            if h!=s['native']['raw_determinants_SHA256'] or bundle['native_raw_SHA256'].get('cells/'+p['task_id']+'/dets.bin')!=h:
                raise RuntimeError('Native raw/transport binding mismatch')
            n,pairs=decode(raw);joint,selection=joint_pairs(pairs,n,tuple(d['nelec']),p['HF_string'])
            if n!=d['norb'] or selection!=s['selection']:raise RuntimeError('Native sector/norm/joint selection differs')
            if pool is not None:
                if (folder/'AllowedHalfDets.bin').read_bytes()!=b''.join(int(x).to_bytes(16,'big') for x in pool) or [int(x) for x in (folder/'CEO-sample-pool-v1.txt').read_text().split()]!=pool:
                    raise RuntimeError('Actual constrained half-pool input differs')
                domain=set(pool)
                if any(a not in domain or b not in domain for a,b,_ in pairs):raise RuntimeError('Native expansion escaped observed half-pool')
            elif (folder/'CEO-sample-pool-v1.txt').exists():raise RuntimeError('Restriction leaked to unrestricted control')
            spaces['JOINT_TOP_D16384']=joint
        for arm in p['requested_readouts']:
            item=s['arms'].get(arm);r=check_observation(item,case,spaces[arm],byid,seen) if item else None
            hf,ref=reference(case);meta=load(OLD/'data/evaluation-only'/(case+'.json'))
            if meta['problem_id']!=d['problem_id']:raise RuntimeError('Evaluation-only reference problem mismatch')
            resources=source.get('sampler_resources',{});source_coordinates=None
            if p['method']!='UNRESTRICTED_HF_HCI':
                source_coordinates=next(x['original_source_ffsim_declared_coordinates'] for x in parameters if x['case']==case and x['method'].endswith('L1'))
                if case not in source_controls:source_controls[case]=all_control_outputs(case,'C5_C_LUCJ_COMPRESSED_L1',raw_n15)
                if p['method']=='FULL_LUCJ_L1':
                    measured=source_controls[case][0]['resources'];current=source['sampler_resources']
                    if measured['CNOT']!=current['cnot_count'] or measured['CNOT_Depth']!=current['cnot_depth'] or measured['Total_Depth']!=current['total_unitary_depth']:
                        raise RuntimeError('Full historical resources do not match actual common opt1 QPY evidence')
            row=dict(case=case,method=p['method'],arm=arm,status='COMPLETED' if r else s['status'],energy_Ha=r['energy_Ha'] if r else None,
                penalized_energy_Ha=r['penalized_energy_Ha'] if r else None,spin_square=r['spin_square'] if r else None,
                recovery_pct=100*(hf-r['energy_Ha'])/(hf-ref) if r else None,reference_gap_mHa=1000*(r['energy_Ha']-ref) if r else None,
                reference_kind='FCI_ARCHIVE' if meta['energies_Ha'].get('fci_energy') is not None else 'SCI_ARCHIVE_NOT_FCI',
                reference_precision='VERIFIED' if meta.get('sector_and_reference_precision_verified') is True else 'UNVERIFIED',
                CNOT=resources.get('cnot_count'),CNOT_Depth=resources.get('cnot_depth'),Total_Depth=resources.get('total_depth',resources.get('total_unitary_depth')),
                sampler_qubits=source.get('sampler_qubits'),source_model_input_coordinates=source_coordinates,independent_ansatz_parameter_count=None,
                source_input_coordinate_count_not_quantum_parameter_rank=True,
                model_shots_per_source=source.get('actual_historical_model_shots'),half_configuration_count=source.get('half_configuration_count'),
                sampler_member_count=source.get('member_count'),actual_CI_dimension=r['actual_CI_dimension'] if r else None,
                actual_new_CI_solve=item['new_CI_solve'] if item else None,
                standalone_numeric_CPU_seconds=(r['CPU_seconds']+(s.get('native_attempt',{}).get('CPU_seconds',0.) if arm=='JOINT_TOP_D16384' else 0.)) if r else None,
                source_pattern_or_compilation_CPU_seconds=source.get('source_pattern_or_compilation_CPU_seconds'),
                source_sampling_CPU_seconds=source.get('source_sampling_CPU_seconds'),
                all_source_compiler_points=source_controls.get(case),
                source_observations_reused_not_independent_confirmation=True,new_model_shots=0,new_QPU_shots=0,
                error=None if r else s.get('errors',{}).get(arm,s.get('error')))
            rows.append(row)
    if len(rows)!=28 or len(seen)!=len(observations) or native_cpu+sum(a['CPU_seconds'] for a in attempts)>terminal['CPU_seconds']+1e-5:
        raise RuntimeError('Full endpoint/work attribution mismatch')
    comparisons=[]
    for r in rows:
        if r['method'] not in ['F2','H1']:continue
        parent=next(x for x in rows if (x['case'],x['method'],x['arm'])==(r['case'],'FULL_LUCJ_L1',r['arm']))
        comparisons.append(dict(case=r['case'],method=r['method'],arm=r['arm'],
            physical_energy_delta_mHa=1000*(r['energy_Ha']-parent['energy_Ha']) if r['energy_Ha'] is not None and parent['energy_Ha'] is not None else None,
            CNOT_reduction_pct=100*(parent['CNOT']-r['CNOT'])/parent['CNOT'],CNOT_Depth_reduction_pct=100*(parent['CNOT_Depth']-r['CNOT_Depth'])/parent['CNOT_Depth'],
            Total_Depth_reduction_pct=100*(parent['Total_Depth']-r['Total_Depth'])/parent['Total_Depth'],same_shots_not_same_half_count=True,
            statistical_superiority_certified=False))
    out=dict(status='N18_ALL16_TASKS28_READOUTS_AUDITED',rows=rows,parent_comparisons=comparisons,
        actual_new_CI_roots=len(observations),actual_native_attempts=sum(bool(s.get('native_attempt')) for s in states.values()),
        failed_CI_attempts=sum(a['status']=='FAILED_PRESERVED' for a in attempts),CPU_seconds=terminal['CPU_seconds'],wall_seconds=terminal['wall_seconds'],
        new_model_shots=0,new_QPU_shots=0,GPU_allocations=0,independent_quantum_confirmation=False,
        full_HCI_HSQD_paper_reproduction=False,full_paper_superiority=False,goal_complete=False,
        protocol_SHA256=protocol_sha,terminal_SHA256=sha256((c/'terminal.json').read_bytes()))
    save(ROOT/'analysis/N18-half-source-native-summary.json',out)
    print(out['status'],len(rows),'CI CPU',out['CPU_seconds'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['status','collect','analyze']);a=p.parse_args();globals()[a.action]()
