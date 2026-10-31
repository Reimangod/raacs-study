"""N76 confirmation input inventory only. No new source policy or measurements."""
from pathlib import Path
from io import BytesIO
import gzip,json
from qiskit import qpy
from .paths import ROOT,OLD
from .io import load,save,write,sha256,space
from .state_activity_preflight import LOCAL as N72
from .activity_resource_portfolio import LOCAL as N73B
from .prefix_performance_sampling import LOCAL as N17,verified_protocol
from .ensemble_breadth_plan import LOCAL as N65
from .full20_breadth_plan import LOCAL as N66

LOCAL=ROOT/'runs/N76-FIXED-CONFIRMATION-INVENTORY-v1'

def run():
    if (LOCAL/'inventory.json').exists():raise RuntimeError('Existing fixed-source inventory; no replace')
    space();closed=load(ROOT/'analysis/EXPLORATION_SEARCH_CLOSED_2026-10-27.json');assert closed['new_source_search_authorized'] is False
    p72=load(N72/'protocol.json');t72=load(N72/'terminal.json');t73b=load(N73B/'terminal.json');p17=verified_protocol();rows=[];bindings={}
    for f in [ROOT/'analysis/EXPLORATION_SEARCH_CLOSED_2026-10-27.json',ROOT/'EXPLORATION_CLOSEOUT_DIRECTIVE_2026-10-27.md',
        N72/'protocol.json',N72/'terminal.json',N73B/'terminal.json',N17/'protocol.json',Path(__file__)]:bindings[str(f)]=sha256(f.read_bytes())
    for s in p72['sources']:
        case=s['case'];meta=load(s['meta']);bindings[s['meta']]=sha256(Path(s['meta']).read_bytes())
        active=[r for r in t72['selected'] if r['case']==case];points=[]
        for r in active:
            f=Path(r['state_file']);assert sha256(f.read_bytes())==r['state_SHA256'];bindings[str(f)]=r['state_SHA256']
            c=next(c for c in t73b['rows'] if c['case']==case and c['time']==r['time'])['selected'];qfile=c['files']['sampling'];qblob=Path(qfile['file']).read_bytes();assert sha256(qblob)==qfile['SHA256']
            bindings[qfile['file']]=qfile['SHA256'];q=qpy.load(BytesIO(gzip.decompress(qblob)))[0];assert q.num_qubits==2*meta['norb']
            points.append(dict(time=r['time'],source_file=str(f),source_SHA256=r['state_SHA256'],QPY=qfile,resources=c['resources']))
        rows.append(dict(case=case,method='ACTIVE_GROUP_FIXED_LIBRARY',role='FROZEN_PROPOSAL',status='LOCAL_VERIFIED_SPARSE_SOURCE_AVAILABLE',members=points,
            historical_construction_model_shots=0,backend='M4_EXISTING_PAIRED_PLUS_GROUP',independent_parameter_rank=None))
        f=ROOT/'runs/B4'/case/'74001/result.json';b=load(f);assert b['status']=='COMPLETED' and b['problem_id']==meta['problem_id'];bindings[str(f)]=sha256(f.read_bytes());members=[]
        assert b['pilot_model_shots']+b['final_model_shots']==b['budget']==100000
        for m in b['members']:
            qfile=f.parent/'final-circuits'/(m['id']+'.qpy');blob=qfile.read_bytes();q=qpy.load(BytesIO(blob))[0]
            assert q.num_qubits==meta['n_qubits'] and q.metadata['paper_exact_sparse_actions']==m['actions'];bindings[str(qfile)]=sha256(blob)
            members.append(dict(id=m['id'],actions=m['actions'],QPY_file=str(qfile),QPY_SHA256=sha256(blob),
                sampling_resources=m['sampling_resources'],old_final_shots=m['shots']))
        rows.append(dict(case=case,method='B4_FIXED_SELECTED_ENSEMBLE',role='FROZEN_PROPOSAL_OR_REFERENCE_NOT_MANDATORY_WINNER',status='LOCAL_EXISTING_EXACT_SPARSE_QE_SOURCE_AVAILABLE',members=members,
            historical_construction_model_shots=b['pilot_model_shots'],historical_final_model_shots=b['final_model_shots'],
            historical_construction_CPU_seconds=b['construction_CPU_seconds'],backend='M4_EXISTING_AUTHOR_QE_SPARSE',
            confirmation_scope='CONDITIONAL_ON_FROZEN_FINAL_ENSEMBLE_NOT_NEW_ADAPTIVE_CONSTRUCTION',independent_parameter_rank=None))
        for layer in [1,2,4]:
            method=f'C5_C_LUCJ_COMPRESSED_L{layer}';cells=[c for c in p17['cells'] if c['case']==case and c['source']==method and c['method']=='ORIGINAL'];assert len(cells)==3
            if cells[0]['status']=='MISSING_SAME_PROBLEM_SOURCE':
                rows.append(dict(case=case,method=method,role='STRONG_LUCJ_CONTROL',status='MISSING_SAME_PROBLEM_SOURCE',independent_parameter_rank=None));continue
            circuit=next(c for c in p17['circuits'] if c['circuit_id']==cells[0]['circuit_id']);d=N17/'circuits'/circuit['circuit_id']
            for name,key in [('config.json','config_SHA256'),('arrays.npz','arrays_SHA256')]:
                file=d/name;assert sha256(file.read_bytes())==circuit[key];bindings[str(file)]=circuit[key]
            rows.append(dict(case=case,method=method,role='STRONG_LUCJ_CONTROL',status='EXISTING_QUALIFIED_SECTOR_EXECUTION_INPUT_AVAILABLE_REMOTE_NOT_RECHECKED',
                circuit=circuit,config_file=str(d/'config.json'),arrays_file=str(d/'arrays.npz'),backend=circuit['backend'],
                resources=cells[0]['selected_resources'],compiler_evidence=cells[0]['compiled_evidence'],historical_construction_model_shots=0,independent_parameter_rank=None))
        for base,arm,method in [(N65,'FULL_N5','SQDRIFT_N5_FIXED30'),(N66,'FULL_N20','SQDRIFT_N20_FIXED30')]:
            p=load(base/'protocol.json');d=next(d for d in p['draws'] if d['case']==case and d['arm']==arm);members=[]
            for m in d['members']:
                file=base/'exports'/case/'members'/m['member_id']/'member.json.gz';blob=file.read_bytes();r=json.loads(gzip.decompress(blob));assert r['spec']==m['spec'] and r['diagonal_terms_retained'] is True;bindings[str(file)]=sha256(blob)
                members.append(dict(member_id=m['member_id'],time=m['spec']['time'],source_packet=str(file),source_packet_SHA256=sha256(blob)))
            draw=base/'draws'/d['draw_id']/'result.json';bindings[str(draw)]=sha256(draw.read_bytes())
            rows.append(dict(case=case,method=method,role='SHORT_OR_STANDARD_SQDRIFT_CONTROL',status='EXISTING_VERIFIED_FIXED_PACKET_SOURCE_AVAILABLE',members=members,
                backend='M4_EXISTING_SPARSE_GROUP',historical_construction_model_shots=0,
                resources=load(draw)['resources'],independent_parameter_rank=None,paper_1500_circuit_setting_reproduced=False))
        n4=ROOT/'runs/N4-NATIVE-HCI-PROPOSALS-v1/protocol.json';build=ROOT/'runs/N1-NATIVE-DICE-BUILD-repair-v2/result.json';qual=ROOT/'analysis/N2-native-Dice-toy-summary.json'
        for f in [n4,build,qual,ROOT/'runs/N3-PRIVATE-COMMON-CI-v1/result.json']:bindings[str(f)]=sha256(f.read_bytes())
        assert load(qual)['passed']==7 and load(ROOT/'runs/N3-PRIVATE-COMMON-CI-v1/result.json')['status']=='PRIVATE_CPU_RUNTIME_FOUR_CASE_PARITY_PASSED'
        rows.append(dict(case=case,method='CLASSICAL_HF_HCI',role='STRONG_CLASSICAL_CONTROL',status='EXISTING_QUALIFIED_REMOTE_EXECUTOR_NOT_RECHECKED',
            backend='PRIVATE_CPU_CLUSTER_NATIVE_DICE',protocol_file=str(n4),binary=load(build)['binary'],binary_SHA256=load(build)['binary_SHA256'],
            final_CI_state_prepared_as_zero_CNOT_quantum_state=False,independent_parameter_rank='NOT_APPLICABLE_CLASSICAL_METHOD'))
    assert len(rows)==32 and sum(r['status']=='MISSING_SAME_PROBLEM_SOURCE' for r in rows)==2
    save(LOCAL/'inventory.json',dict(status='FIXED_CONFIRMATION_INPUT_INVENTORY_ONLY_NOT_EXECUTION_AUTHORIZATION',rows=rows,bindings=bindings,
        proposals=['ACTIVE_GROUP_FIXED_LIBRARY','B4_FIXED_SELECTED_ENSEMBLE'],maximum_proposals=2,new_source_search_authorized=False,
        source_policies_frozen_not_molecule_best_found=True,new_model_samples=0,new_QPU_shots=0,new_CI_roots=0,
        confirmation_sampling_started=False,remote_readiness_verified=False,matched_compiler_frontier_complete=False,goal_complete=False))
    lines=['# 固定確認N76：入力の棚卸し（まだ確認実験は開始していない）','',
        '新案探索は2ループで閉鎖済み。新規方策を作らず、既存2方策と強い対照の実行入力を点検した。',
        'B4は旧adaptive構築後の固定ensembleの確認であり、B4の新しい構築反復ではない。旧pilot費用を別途計上する。','',
        '| 条件 | 方策・対照 | 入力状況 | 実行先 | 旧構築pilot model Shot |','|---|---|---|---|---:|']
    for r in rows:lines.append(f"| {r['case']} | {r['method']} | {r['status']} | {r.get('backend','—')} | {r.get('historical_construction_model_shots','—')} |")
    lines+=['','ナフタレンLUCJ L2/L4は同じ物理問題のsourceが無い。別basisや別geometryで代用しない。',
        'A100/remote CPUの現環境・queueはまだ再確認していない。新しい独立counts、HCI根、GPU jobは0。',
        '次に確認用seed・総Shot（B4 pilot込み）・R/D・全費用上限・compiler機会を固定してから実行する。新たな候補探索には戻らない。',
        '独立Parameters、全cold費用、原著最高予算再現とcompiler frontierの未解決は残っている。確認結果の主張範囲に明記する。','']
    write(ROOT/'analysis/N76_FIXED_CONFIRMATION_INPUTS_JA.md','\n'.join(lines).encode());print('N76_INVENTORY32_2MISSING_NO_NEW_OUTCOMES',flush=True)

if __name__=='__main__':run()
