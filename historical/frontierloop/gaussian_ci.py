"""N22 same-opportunity readout of every Gaussian ablation draw.

Copies the exact qualified N17 frequency-before-native worker. Frozen inputs
are HF plus observed configurations, never evaluation-only references.
"""
import argparse
import csv
import gzip
import hashlib
import math
from collections import Counter
from io import BytesIO, StringIO
from pathlib import Path
import shutil
import statistics
import numpy as np
from qiskit import qpy
from .paths import ROOT, OLD
from .io import load,save,write,sha256,digest
from .gaussian_source import LOCAL as SAMPLES,checked_counts
from .prefix_performance_sampling import capacity
from .native_source_control_plan import initial_strings
from .sample_universe_audit import allowed_pool

LOCAL=ROOT/'runs/N22-GAUSSIAN-COMMON-CI-v1'
REMOTE='/share/reiwatanabe/raacs-frontier-loop-20261025/N22-gaussian-common-ci-v1'


def rebuild_source(draw, definition, sample_protocol_sha, terminal_sha):
    path=Path(draw['payload'])
    if sha256(path.read_bytes())!=draw['payload_SHA256'] or draw['actual_samples']!=100000:
        raise RuntimeError('Actual100k immutable observations required')
    with np.load(path,allow_pickle=False) as a:
        ids,counts=a['blocked_ids'],a['counts']
        if len(ids)!=len(set(map(int,ids))) or any(int(x)<=0 for x in counts):
            raise RuntimeError('Positive unique actual counts required')
        checked_counts(np.repeat(ids,counts),definition['norb'],tuple(definition['nelec']),100000)
    source=dict(status='AVAILABLE',kind='QUANTUM_FULL_COUNTS',
        semantics='EXACT_MODEL_EQUIVALENT_CLASSICAL_GAUSSIAN_DRAWS_NOT_QPU_EVENTS',
        expected_problem_id=definition['problem_id'],payload=str(path),payload_SHA256=draw['payload_SHA256'],
        draw_id=draw['draw_id'],replicate=draw['replicate'],actual_model_equivalent_samples=100000,
        sampling_protocol_SHA256=sample_protocol_sha,sampling_terminal_SHA256=terminal_sha,
        actual_QPU_shots=0,sampling_CPU_seconds=draw['sampling_CPU_seconds'])
    return source, initial_strings(source,definition['norb'],tuple(definition['nelec'])), sorted(allowed_pool(source,definition['norb'],tuple(definition['nelec'])))


def prepare():
    if LOCAL.exists():raise RuntimeError('Existing N22 CI; never overwrite')
    capacity()
    result=load(SAMPLES/'terminal.json');sampling=load(SAMPLES/'protocol.json')
    if result['status']!='N22_ALL12_GAUSSIAN_DRAW_STATES_TERMINAL' or len(result['draws'])!=12 or any(d['status']!='COMPLETED' for d in result['draws']):
        raise RuntimeError('Close all twelve draw states; handle failures explicitly rather than silently dropping')
    if result['protocol_SHA256']!=sha256((SAMPLES/'protocol.json').read_bytes()) or result['code_SHA256']!=sampling['code_SHA256']:
        raise RuntimeError('Actual sampling protocol binding changed')
    oldroot=ROOT/'runs/N17-INDEPENDENT-PREFIX-CI-A100-v1';old=load(oldroot/'protocol.json')
    sources={s['case']:s for s in result['sources']};plans=[]
    psha=sha256((SAMPLES/'protocol.json').read_bytes());tsha=sha256((SAMPLES/'terminal.json').read_bytes())
    for draw in result['draws']:
        d=old['physical'][draw['case']];s=sources[draw['case']]
        if s['problem_source']['norb']!=d['norb'] or s['nelec']!=d['nelec']:
            raise RuntimeError('Same orbital count and particle sector required')
        # The N15 source binding already has the actual physical source identity.
        original=load(s['problem_source']['construction'])
        if original['problem_id']!=d['problem_id'] or sha256(Path(s['problem_source']['construction']).read_bytes())!=s['problem_source']['construction_SHA256']:
            raise RuntimeError('Gaussian source belongs to a different Hamiltonian')
        source,strings,pool=rebuild_source(draw,d,psha,tsha);hf=(1<<d['nelec'][0])-1
        row=dict(case=draw['case'],method='GAUSSIAN_PREFIX:'+str(draw['replicate']),source=source,
            selection_mode='SAMPLE_CONSTRAINED',initial_strings=[hf],allowed_strings=pool,
            frequency_strings=strings,HF_string=hf,requested_readouts=['SOURCE_FREQUENCY_R128','JOINT_TOP_D16384'])
        row['task_id']=digest(row);plans.append(row)
    for case,d in sorted(old['physical'].items()):
        hf=(1<<d['nelec'][0])-1
        row=dict(case=case,method='UNRESTRICTED_HF_HCI',source=dict(status='AVAILABLE',kind='CLASSICAL_HF_ONLY',expected_problem_id=d['problem_id']),
            selection_mode='UNRESTRICTED_HF_CONTROL',initial_strings=[hf],allowed_strings=None,frequency_strings=None,HF_string=hf,requested_readouts=['JOINT_TOP_D16384'])
        row['task_id']=digest(row);plans.append(row)
    if len(plans)!=16 or len({p['task_id'] for p in plans})!=16:raise RuntimeError('All12draws plus4classicalcontrols required')
    LOCAL.mkdir();(LOCAL/'pipeline').mkdir()
    for name,value in old['pipeline_SHA256'].items():
        path=oldroot/'pipeline'/name
        if sha256(path.read_bytes())!=value:raise RuntimeError('Qualified numerical helper changed')
        shutil.copyfile(path,LOCAL/'pipeline'/name)
    worker=(oldroot/'worker.py').read_bytes()
    if sha256(worker)!=old['worker_SHA256']:raise RuntimeError('Qualified worker changed')
    write(LOCAL/'worker.py',worker)
    spec={k:old[k] for k in ['physical','common_runtime','packages','native_binary','native_lock','qualified_external_SHA256','minimal_qualification','pipeline_SHA256','sparse_library']}
    spec.update(version='N22_ALL12_GAUSSIAN_DRAWS_COMMON_TWO_READOUTS_PLUS4_HF_V1',remote_root=REMOTE,plans=plans,
        policy=dict(old['policy'],global_wall_seconds=3600),worker_SHA256=sha256(worker),
        preparer_SHA256=sha256(Path(__file__).read_bytes()),sampling_protocol_SHA256=psha,sampling_terminal_SHA256=tsha,
        input_tests_SHA256=sha256((ROOT/'qualification-N22-ci-input-v1-tests.xml').read_bytes()),
        planned=16,readout_endpoints=28,new_model_shots=0,new_QPU_shots=0,GPU_allocations=0,
        original_N17_matched_worker_SHA256=old['worker_SHA256'],full_paper_superiority=False,goal_complete=False)
    save(LOCAL/'protocol.json',spec);print('N22_CI_FROZEN_ALL16TASKS28ENDPOINTS',flush=True)


def submit():
    if sha256(Path(__file__).read_bytes())!=load(LOCAL/'protocol.json')['preparer_SHA256']:
        raise RuntimeError('Frozen N22 dispatcher changed')
    from . import half_source_native_screen as shared
    shared.LOCAL=LOCAL;shared.REMOTE=REMOTE;shared.submit()


def status():
    from . import nested_ci_study as shared
    shared.LOCAL=LOCAL;shared.REMOTE=REMOTE;shared.status()


def collect():
    from . import collect_small_pool_bundle as shared
    shared.LOCAL=LOCAL;shared.REMOTE=REMOTE;shared.collect()


def analyze():
    from .native_dice_binary import decode
    from .joint_ci_plan import joint_pairs
    from .analyze_joint_ci import verify_row
    from .analyze_prefix_performance_ci import basis_digest,check_observation
    from .analyze import reference
    spec=load(LOCAL/'protocol.json');c=LOCAL/'collected';term=load(c/'terminal.json');started=load(c/'started.json')
    psha=sha256((LOCAL/'protocol.json').read_bytes())
    if started['protocol_SHA256']!=psha or started['pyscf_threads']!=1 or set(started['thread_env'].values())!={'1'}:
        raise RuntimeError('Actual physical protocol or thread binding differs')
    if sha256((LOCAL/'worker.py').read_bytes())!=spec['worker_SHA256'] or sha256(Path(__file__).read_bytes())!=spec['preparer_SHA256']:
        raise RuntimeError('Frozen worker/preparer changed')
    for name,value in spec['pipeline_SHA256'].items():
        if sha256((LOCAL/'pipeline'/name).read_bytes())!=value:raise RuntimeError('Qualified math helper changed')
    if load(c/'qualification/result.json')['status']!='SAME_NATIVE_EXPORT_PHYSICAL_PARITY_PASSED':raise RuntimeError('Actual native export qualification failed')
    states={s['task_id']:s for s in term['states']};observations=load(c/'observations.json');attempts=load(c/'ci-attempts.json');byid={o['id']:o for o in observations}
    if len(states)!=16 or set(states)!={p['task_id'] for p in spec['plans']} or len(byid)!=len(observations):raise RuntimeError('All actual terminal tasks/unique roots required')
    completed=[a for a in attempts if a['status']=='COMPLETED']
    if len(completed)!=len(observations) or len(observations)!=term['actual_new_CI_roots'] or len(attempts)!=term['ci_attempts'] or {a['observation_id'] for a in completed}!=set(byid):
        raise RuntimeError('Actual attempted/completed CI accounting differs')
    for o in observations:
        verify_row(o['row']);path=c/o['vector_file']
        if sha256(path.read_bytes())!=o['vector_SHA256']:raise RuntimeError('Actual saved root changed')
        with np.load(path,allow_pickle=False) as a:pairs=a['joint_pairs'];v=a['coefficients']
        h=basis_digest(pairs)
        if len(pairs)!=len(v) or not np.isfinite(v).all() or abs(v@v-1)>1e-10 or h!=o['basis_digest'] or o['id']!=hashlib.sha256((o['case']+h+psha).encode()).hexdigest():
            raise RuntimeError('Actual saved root/basis/problem identity differs')
        if o['row']['binary_SHA256']!=spec['qualified_external_SHA256'][spec['sparse_library']]:raise RuntimeError('Unregistered CI kernel')
    sample=load(SAMPLES/'terminal.json');sources={s['case']:s for s in sample['sources']};draws={d['draw_id']:d for d in sample['draws']}
    if sha256((SAMPLES/'terminal.json').read_bytes())!=spec['sampling_terminal_SHA256'] or sha256((SAMPLES/'protocol.json').read_bytes())!=spec['sampling_protocol_SHA256']:
        raise RuntimeError('Original sampling context changed')
    bundle=load(LOCAL/'bundle-collection-v1.json');rows=[];seen=set();native_cpu=0.
    for p in spec['plans']:
        s=states[p['task_id']];case=p['case'];d=spec['physical'][case];folder=c/'cells'/p['task_id'];spaces={};pool=None
        if load(folder/'result.json')!=s or s['source']!=p['source'] or s['method']!=p['method']:raise RuntimeError('Actual terminal/source mapping differs')
        if p['source']['kind']=='QUANTUM_FULL_COUNTS':
            source,strings,pool=rebuild_source(draws[p['source']['draw_id']],d,spec['sampling_protocol_SHA256'],spec['sampling_terminal_SHA256'])
            if source!=p['source'] or strings!=p['frequency_strings'] or pool!=p['allowed_strings']:raise RuntimeError('Actual observed pool/frequency rule differs')
            spaces['SOURCE_FREQUENCY_R128']=[(a,b) for a in strings for b in strings]
        if s.get('native_attempt'):
            native_cpu+=s['native_attempt']['CPU_seconds']
            if load(folder/'native-attempt.json')!=s['native_attempt']:raise RuntimeError('Native work attribution differs')
            for spin in ['Alpha','Beta']:
                if (folder/(spin+'Dets.bin')).read_bytes()!=p['HF_string'].to_bytes(16,'big'):raise RuntimeError('Native initial state is not HF')
        if 'native' in s:
            raw=gzip.decompress((folder/'dets.bin.gz').read_bytes());h=sha256(raw)
            if h!=s['native']['raw_determinants_SHA256'] or bundle['native_raw_SHA256'].get('cells/'+p['task_id']+'/dets.bin')!=h:raise RuntimeError('Actual native raw binding differs')
            n,pairs=decode(raw);joint,selection=joint_pairs(pairs,n,tuple(d['nelec']),p['HF_string'])
            if n!=d['norb'] or selection!=s['selection']:raise RuntimeError('Native sector/selection differs')
            if pool is not None:
                if (folder/'AllowedHalfDets.bin').read_bytes()!=b''.join(int(x).to_bytes(16,'big') for x in pool) or [int(x) for x in (folder/'CEO-sample-pool-v1.txt').read_text().split()]!=pool or any(a not in pool or b not in pool for a,b,_ in pairs):raise RuntimeError('Native expansion escaped observed pool')
                if 'CEO_SAMPLE_POOL_V1 ' not in (folder/'native.log').read_text():raise RuntimeError('No actual restricted-kernel evidence')
            elif (folder/'CEO-sample-pool-v1.txt').exists():raise RuntimeError('Observed restriction leaked into classical control')
            spaces['JOINT_TOP_D16384']=joint
        for arm in p['requested_readouts']:
            item=s['arms'].get(arm)
            r=check_observation(item,case,spaces[arm],byid,seen) if item else None
            hf,ref=reference(case);meta=load(OLD/'data/evaluation-only'/(case+'.json'))
            if meta['problem_id']!=d['problem_id']:raise RuntimeError('Reporting reference belongs to different problem')
            state=sources[case] if pool is not None else None
            resources=state['compiler_outputs'][0]['resources'] if state else {}
            rows.append(dict(case=case,method='GAUSSIAN_PREFIX' if state else 'UNRESTRICTED_HF_HCI',replicate=p['source'].get('replicate'),
                arm=arm,status='COMPLETED' if r else 'READOUT_FAILED_PRESERVED',energy_Ha=r['energy_Ha'] if r else None,
                penalized_energy_Ha=r['penalized_energy_Ha'] if r else None,spin_square=r['spin_square'] if r else None,
                recovery_pct=100*(hf-r['energy_Ha'])/(hf-ref) if r else None,reference_gap_mHa=1000*(r['energy_Ha']-ref) if r else None,
                reference_kind='FCI_ARCHIVE' if meta['energies_Ha'].get('fci_energy') is not None else 'SCI_ARCHIVE_NOT_FCI',
                reference_precision='VERIFIED' if meta.get('sector_and_reference_precision_verified') is True else 'UNVERIFIED',
                CNOT=resources.get('CNOT'),CNOT_Depth=resources.get('CNOT_Depth'),Total_Depth=resources.get('Total_Depth'),
                declared_parameters=state['declared_shared_prefix_coordinates'] if state else None,
                actual_CI_dimension=r['actual_CI_dimension'] if r else None,observed_half_pool_size=len(pool) if pool is not None else None,
                observed_half_pool_fraction_of_particle_sector=len(pool)/math.comb(d['norb'],d['nelec'][0]) if pool is not None else None,
                half_pool_is_complete_sector=(len(pool)==math.comb(d['norb'],d['nelec'][0])) if pool is not None else None,
                standalone_native_plus_CI_CPU_seconds=(r['CPU_seconds']+(s.get('native_attempt',{}).get('CPU_seconds',0.) if arm=='JOINT_TOP_D16384' else 0.)) if r else None,
                model_equivalent_samples=100000 if state else None,new_QPU_shots=0,
                final_CI_state_not_prepared_by_sampler=True,task_id=p['task_id'],
                error=None if r else s.get('errors',{}).get(arm,s.get('error'))))
    if len(rows)!=28 or len(seen)!=len(observations) or native_cpu+sum(a['CPU_seconds'] for a in attempts)>term['CPU_seconds']+1e-5:raise RuntimeError('All endpoint/root/work accounting required')
    old=load(ROOT/'analysis/N17-whole-study-summary.json');groups=[]
    for case in sorted(sources):
        for arm in ('SOURCE_FREQUENCY_R128','JOINT_TOP_D16384'):
            rs=[r for r in rows if r['case']==case and r['method']=='GAUSSIAN_PREFIX' and r['arm']==arm];good=[r for r in rs if r['energy_Ha'] is not None]
            if len(rs)!=3:raise RuntimeError('All three planned replicates required')
            parent=next(g for g in old['groups'] if (g['case'],g['source'],g['method'],g['arm'])==(case,'C5_C_LUCJ_COMPRESSED_L1','ORIGINAL',arm))
            mean=statistics.mean(r['energy_Ha'] for r in good) if good else None
            paired=len(good)==3 and parent['complete_replicates']==3
            source=sources[case];portfolio=[]
            for point in source['compiler_outputs']:
                level=point['resources']['optimization_level'];pr=parent['original_opt1_resources' if level==1 else 'original_opt3_resources'];actual=point['resources']
                qpath=Path(point['sampling']['path'])
                if sha256(qpath.read_bytes())!=point['sampling']['SHA256']:raise RuntimeError('Actual source QPY changed')
                qc=qpy.load(BytesIO(qpath.read_bytes()))[0]
                recount=dict(CNOT=int(qc.count_ops().get('cx',0)),CNOT_Depth=int(qc.depth(filter_function=lambda x:x.operation.name=='cx')),Total_Depth=int(qc.depth()))
                if any(actual[k]!=v for k,v in recount.items()):raise RuntimeError('Independent QPY recount differs')
                portfolio.append(dict(level=level,parent=pr,gaussian=actual,reduction_pct={k:100*(pr[k]-actual[k])/pr[k] for k in ('CNOT','CNOT_Depth','Total_Depth')}))
            groups.append(dict(case=case,arm=arm,completed_replicates=len(good),planned_replicates=3,energy_mean_Ha=mean,
                energy_SD_mHa=1000*statistics.stdev(r['energy_Ha'] for r in good) if len(good)>1 else None,
                recovery_mean_pct=statistics.mean(r['recovery_pct'] for r in good) if good else None,
                original_full_LUCJ_mean_Ha=parent['energy_Ha_mean'],energy_mean_delta_vs_full_LUCJ_mHa=1000*(mean-parent['energy_Ha_mean']) if paired else None,
                parent_comparison_all3_available=paired,matched_compiler_portfolio=portfolio,
                declared_parameters=source['declared_shared_prefix_coordinates'],parent_declared_parameters=parent['declared_parameters'],
                actual_CI_dimension_mean=statistics.mean(r['actual_CI_dimension'] for r in good) if good else None,
                native_plus_CI_CPU_mean=statistics.mean(r['standalone_native_plus_CI_CPU_seconds'] for r in good) if good else None,
                original_native_plus_CI_CPU_mean=parent['standalone_numeric_CPU_seconds_mean'],
                sampling_CPU_all3=sum(d['sampling_CPU_seconds'] for d in sample['draws'] if d['case']==case),
                statistical_equivalence_or_superiority_certified=False))
    out=dict(status='N22_ALL16_TASKS28_READOUTS_AND_ALL12_SOURCES_AUDITED',rows=rows,groups=groups,
        actual_new_CI_roots=len(observations),failed_CI_attempts=sum(a['status']=='FAILED_PRESERVED' for a in attempts),
        CI_campaign_CPU_seconds=term['CPU_seconds'],CI_campaign_wall_seconds=term['wall_seconds'],
        new_exact_model_equivalent_samples=sample['actual_new_exact_model_equivalent_samples'],
        sampling_and_compile_CPU_seconds=sample['CPU_seconds'],sampling_and_compile_wall_seconds=sample['wall_seconds'],
        new_QPU_shots=0,GPU_allocations=0,cold_total_matched_work=False,classically_efficient_control=True,
        original_initialization_costs_retained_and_additional=True,statistical_or_best_paper_superiority_established=False,
        protocol_SHA256=psha,terminal_SHA256=sha256((c/'terminal.json').read_bytes()),goal_complete=False)
    save(ROOT/'analysis/N22-gaussian-CI-summary.json',out)
    for name,data in [('all28-readouts',rows),('all8-group-comparisons',groups)]:
        stream=StringIO();writer=csv.DictWriter(stream,fieldnames=sorted(set().union(*(r.keys() for r in data))));writer.writeheader();writer.writerows(data)
        write(ROOT/f'analysis/N22-{name}.csv',stream.getvalue().encode('utf-8-sig'))
    def fmt(x,n=3):return '未取得' if x is None else f'{x:.{n}f}'
    lines=['# N22：Gaussian prefixだけで、LUCJの複雑な相関層を代替できるか','',
        f"全28readout：{dict(Counter(r['status'] for r in rows))}。新CI根{len(observations)}、CI失敗{out['failed_CI_attempts']}。",
        f"新規exact-model-equivalent draws {out['new_exact_model_equivalent_samples']:,}、QPU Shot0、GPU割当0。sampling/compile {sample['CPU_seconds']:.3f} CPU秒、CI campaign {term['CPU_seconds']:.3f} CPU秒。",'',
        '| 条件 | readout | 完了/3 | Gaussian energy平均 Ha | SD mHa | full LUCJ energy平均 Ha | Gaussian−full mHa | Recovery平均 % | CNOT opt1 | CX Depth opt1 | Total Depth opt1 | Parameters（同chart親→子） | 実D平均 | native+CI CPU平均秒 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        res=g['matched_compiler_portfolio'][0]['gaussian']
        lines.append('| '+' | '.join([g['case'],g['arm'],str(g['completed_replicates']),fmt(g['energy_mean_Ha'],9),fmt(g['energy_SD_mHa']),fmt(g['original_full_LUCJ_mean_Ha'],9),fmt(g['energy_mean_delta_vs_full_LUCJ_mHa']),fmt(g['recovery_mean_pct']),str(res['CNOT']),str(res['CNOT_Depth']),str(res['Total_Depth']),str(g['parent_declared_parameters'])+'→'+str(g['declared_parameters']),fmt(g['actual_CI_dimension_mean'],1),fmt(g['native_plus_CI_CPU_mean'])])+' |')
    lines+=['','## 限界・判断の境界','',
        'これは既知のGaussian/Slater状態のablationであり、新しい量子優位アルゴリズムではない。元LUCJの相関層を捨てるので状態保存・energy保存は保証しない。古典DPPで正確にsamplingでき、原著LUCJのCCSD/CISD/圧縮初期化費用も残る。',
        '新しい全3反復と元LUCJの全3反復を比較。平均が低いだけでは統計的優越・精度同等・最強論文への勝利としない。同じ最大R/Dでもpool数・実D・実CPUは違う。両compiler opt1/opt3の同level比較をJSONへ保存し、一方だけ強いcompilerへしない。',
        'Parametersは同じ共有spin prefix/suffix chartの宣言実座標であり、最小Jacobian rankではない。初期U(n)の表現変更を追加利益にしない。readout最適化変数0とansatz Parameters0は別。',
        '最終energyはCI状態のphysical H期待値で、sampler回路がこのCI状態を量子準備した意味ではない。最適化式H+0.1S²とphysical Hを区別。Recoveryはenergy換算で独立指標ではない。Feは精度未認証SCI、N₂のarchive FCIも精度認証未完。',
        '上記CPUは全cold end-to-end総費用ではなく、source初期化、旧full LUCJ構築/compile/転送/queue、テスト、回収、解析は別。frequencyにはnative費用を足していない。失敗の費用・欠測も全28CSVに保持。',
        '全4 unrestricted HF-HCI対照は同じCSVへ記載。古典CI結果を0 CNOTの量子状態準備とは呼ばない。原著最高TN設定・SqDRIFT原著全予算の再現ではなく、開発診断。']
    write(ROOT/'analysis/N22_GAUSSIAN_RESULTS_JA.md',('\n'.join(lines)+'\n').encode())
    print(out['status'],out['CI_campaign_CPU_seconds'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','submit','status','collect','analyze']);a=p.parse_args();globals()[a.action]()
