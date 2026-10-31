"""All fixed T3 outcomes, same-count seed stability and honest CI costs."""
from collections import Counter
import csv,json,sqlite3
from io import StringIO
from pathlib import Path
import numpy as np
from .paths import ROOT
from .io import load,save,write,sha256
from .readout_seed_screen import FREEZE,RUN,binding
from .readout_seed_plan import evaluation_key
from .analyze import reference
from .pool_ranking import selections


def sign(x,tol=1e-8):return 'LOWER' if x< -tol else 'HIGHER' if x>tol else 'EQUAL_WITHIN_TOLERANCE'


def audit_cell_selection(plan,result,request_record,selection_record,db,require_complete=True):
    """Reconstruct the registered policy from saved roots, without new CI."""
    if request_record!=plan or result['cell_id']!=plan['cell_id']:
        raise RuntimeError('Completed cell request/source identity mismatch')
    for field,expected in [('request_digest',plan['request']['request_digest']),
            ('physical_problem_id',plan['context']['problem_id']),('source_counts_SHA256',plan['source']['payload_SHA256'])]:
        if (require_complete or field in result) and result.get(field)!=expected:
            raise RuntimeError('Completed cell request/source identity mismatch')
    expected_names={'GLOBAL_CI_MARGINAL','SAME_UNION_FREQUENCY','NATIVE_TRIM_RATIO1'}
    rows=result['rows']
    names=[x['arm'] for x in rows]
    if len(names)!=len(set(names)) or not set(names)<=expected_names or (
            require_complete and set(names)!=expected_names):
        raise RuntimeError('Missing or duplicate completed arm')
    calls=result['calls'];screens=[];screen_keys=[]
    expected_purposes={'FINAL_'+name for name in expected_names}
    if plan['request']['screening_required']:
        expected_purposes|={'SCREEN_'+str(i) for i in range(4)}
    purposes=[x['purpose'] for x in calls]
    if len(purposes)!=len(set(purposes)) or not set(purposes)<=expected_purposes or (
            require_complete and set(purposes)!=expected_purposes):
        raise RuntimeError('Missing or duplicate registered root call')
    for call in calls:
        found=db.execute('SELECT lookup,record,origin FROM observations WHERE id=?',
            (call['observation_id'],)).fetchone()
        if found is None or found[0]!=call['key'] or found[2]!=call['origin']:
            raise RuntimeError('Call observation/context/origin mismatch')
        root=json.loads(found[1])
        if evaluation_key(plan['context'],call['CI_strings'])!=found[0] or (
                root['CI_strings']!=call['CI_strings']):
            raise RuntimeError('Call basis mismatch')
        if call['purpose'].startswith('SCREEN_'):
            i=int(call['purpose'].split('_')[1])
            if call['CI_strings']!=plan['request']['screens'][i] or not call['needs_marginals']:
                raise RuntimeError('Native screen request mismatch')
            screens.append({**root,'round':0,'batch':i});screen_keys.append(call['key'])
    if not rows and not require_complete:
        return True  # Preserved partial screens are not endpoint outcomes.
    if screens:
        if len(screens)!=4:raise RuntimeError('Endpoint without complete registered screens')
        screens.sort(key=lambda x:x['batch'])
        with np.load(plan['source']['payload'],allow_pickle=False) as a:
            expected=selections(screens,a['blocked_ids'],a['counts'],plan['context']['norb'],plan['HF_string'])
        if selection_record!=expected:
            raise RuntimeError('Saved selection is not registered marginal/native/frequency policy')
        expected_arms=expected['arms']
    else:
        expected_arms={name:plan['request']['universe'] for name in expected_names}
        if selection_record.get('arms')!=expected_arms or not selection_record.get('screening_skipped'):
            raise RuntimeError('Skipped-screen identical-basis policy mismatch')
    for row in rows:
        if row['CI_strings']!=expected_arms[row['arm']]:
            raise RuntimeError('Endpoint basis not selected by registered policy')
        call=next(x for x in calls if x['purpose']=='FINAL_'+row['arm'])
        if call['CI_strings']!=row['CI_strings'] or call['observation_id']!=row['observation_id']:
            raise RuntimeError('Endpoint/call observation mismatch')
        required=set([call['key']]+(screen_keys if row['arm']!='SAME_UNION_FREQUENCY' else []))
        ids=row['required_work']['canonical_observation_ids']
        keys=[db.execute('SELECT lookup FROM observations WHERE id=?',(oid,)).fetchone()[0] for oid in ids]
        if len(keys)!=len(required) or set(keys)!=required or (
                row['required_work']['unique_CI_evaluations']!=len(required)):
            raise RuntimeError('Required CI basis set omitted or counted twice')
    if (require_complete or 'all_arms_identical_basis' in result) and result.get('all_arms_identical_basis')!=(len({tuple(x) for x in expected_arms.values()})==1):
        raise RuntimeError('Identical-basis classification mismatch')
    return True


def run():
    frozen=load(FREEZE);terminal=load(RUN/'terminal.json')
    if binding()!=frozen:raise RuntimeError('T3 input/implementation drift')
    dbpath=RUN/'evaluations.sqlite'
    if sha256(dbpath.read_bytes())!=terminal['catalog_SHA256']:raise RuntimeError('Terminal observation catalog changed')
    db=sqlite3.connect('file:'+str(dbpath)+'?mode=ro',uri=True)
    if db.execute('PRAGMA integrity_check').fetchall()!=[('ok',)]:raise RuntimeError('Catalog integrity')
    if len(frozen['plans'])!=48 or len({p['cell_id'] for p in frozen['plans']})!=48:
        raise RuntimeError('Planned48 unique cells required')
    if len(terminal['states'])!=48 or {x['cell_id'] for x in terminal['states']}!={p['cell_id'] for p in frozen['plans']}:
        raise RuntimeError('Terminal coverage differs from frozen48 cells')
    rows=[];contrasts=[];statuses=[]
    for p in frozen['plans']:
        source=p['source'];case=source['case'];method=source['method_id'];seed=p['request']['seed']
        target=RUN/'cells'/p['cell_id'];done=(target/'result.json').exists()
        r=load(target/('result.json' if done else 'failure.json')) if done or (target/'failure.json').exists() else {}
        status=r.get('status','NOT_EXECUTED_PRESERVED')
        recorded=next(x for x in terminal['states'] if x['cell_id']==p['cell_id'])
        if recorded['status']!=status:raise RuntimeError('Terminal/cell status mismatch')
        if r:
            audit_cell_selection(p,r,load(target/'request.json'),
                load(target/'selection.json') if (target/'selection.json').exists() else None,db,require_complete=done)
        statuses.append(dict(case=case,method=method,seed=seed,status=status,cell_id=p['cell_id']))
        arms={a['arm']:a for a in r.get('rows',[])};hf,ref=reference(case)
        baseline=load(source['R2_baseline'])
        oldkey=evaluation_key(p['context'],baseline['CI_strings'])
        old_cpu=db.execute('SELECT cpu FROM observations WHERE lookup=? ORDER BY seq LIMIT 1',(oldkey,)).fetchone()[0]
        for arm in load(ROOT/'protocol-T3-READOUT-SEEDS.json')['arms']:
            a=arms.get(arm)
            row=dict(case=case,method=method,seed=seed,arm=arm,status='COMPLETED' if a else status,
                source_kind=source['kind'],source_final_model_shots=source['source_final_model_shots'],
                source_total_model_shot_budget=source['source_total_budget'],source_classical_draws=source.get('classical_draws',0),
                candidate_union_size=len(p['request']['universe']),screening_required=p['request']['screening_required'],
                R_cap=128,actual_CI_dimension=a['actual_CI_dimension'] if a else None,
                energy_Ha=a['energy_Ha'] if a else None,penalized_energy_Ha=a['penalized_energy_Ha'] if a else None,
                recovery_pct=100*(hf-a['energy_Ha'])/(hf-ref) if a else None,S2=a['S2'] if a else None,
                required_unique_CI_evaluations=a['required_work']['unique_CI_evaluations'] if a else None,
                canonical_CI_CPU_proxy=a['required_work']['canonical_CI_CPU_proxy'] if a else None,
                canonical_matvec_proxy=a['required_work']['canonical_matvec_proxy'] if a else None,
                old_R2_energy_Ha=baseline['energy_Ha'],old_R2_actual_CI_dimension=baseline['actual_CI_dimension'],
                old_R2_canonical_CPU_proxy=old_cpu,old_R2_historical_CPU_seconds=baseline['incremental_CPU_seconds'],
                new_actual_cell_CI_solves=r.get('new_CI_solves'),new_actual_cell_CPU_seconds=r.get('new_actual_CPU_seconds'),
                actual_cell_cost_shared_across_three_arm_rows=True,observation_id=a['observation_id'] if a else None,
                evaluation_origin=a['evaluation_origin'] if a else None,new_model_shots=0,independent_quantum_confirmation=False)
            if a:
                key=evaluation_key(p['context'],a['CI_strings'])
                found=db.execute('SELECT lookup,record,origin FROM observations WHERE id=?',(a['observation_id'],)).fetchone()
                if found is None or found[0]!=key:raise RuntimeError('Endpoint observation/context alias')
                root=json.loads(found[1])
                if root['energy_Ha']!=a['energy_Ha'] or root['penalized_energy_Ha']!=a['penalized_energy_Ha'] or (
                        root['spin_square']!=a['S2'] or found[2]!=a['evaluation_origin']):
                    raise RuntimeError('Endpoint energy not bound to catalog observation')
                if a['actual_CI_dimension']!=p['request']['final_actual_CI_dimension'] or abs(a['energy_Ha']+.1*a['S2']-a['penalized_energy_Ha'])>1e-8:
                    raise RuntimeError('Dimension/spin consistency violation')
                if arm=='SAME_UNION_FREQUENCY' and a['required_screen_keys']:raise RuntimeError('Frequency charged unnecessary screens')
                if not p['request']['screening_required'] and a['required_screen_keys']:raise RuntimeError('Unnecessary screens when union fits')
                # Cost must be the canonical (first compatible) observation of each
                # UNIQUE required basis; never count a shared root twice.
                cost=0.;mv=0;unknown_cpu=False;unknown_mv=False
                required=a['required_work']['canonical_observation_ids']
                for oid in required:
                    data=db.execute('SELECT lookup,cpu,record FROM observations WHERE id=?',(oid,)).fetchone()
                    first=db.execute('SELECT id FROM observations WHERE lookup=? ORDER BY seq LIMIT 1',(data[0],)).fetchone()[0]
                    if first!=oid:raise RuntimeError('Noncanonical cost cherry-pick')
                    if data[1] is None:unknown_cpu=True
                    else:cost+=data[1]
                    value=json.loads(data[2]).get('matvec_calls')
                    if value is None:unknown_mv=True
                    else:mv+=value
                if (None if unknown_cpu else cost)!=a['required_work']['canonical_CI_CPU_proxy'] or (
                        None if unknown_mv else mv)!=a['required_work']['canonical_matvec_proxy']:
                    raise RuntimeError('Required work double accounting')
            rows.append(row)
        if {'GLOBAL_CI_MARGINAL','SAME_UNION_FREQUENCY'}<=set(arms):
            g,f=[arms[k] for k in ['GLOBAL_CI_MARGINAL','SAME_UNION_FREQUENCY']]
            n=arms.get('NATIVE_TRIM_RATIO1')
            if len({a['actual_CI_dimension'] for a in arms.values()})!=1:raise RuntimeError('Paired dimension mismatch')
            same=g['CI_strings']==f['CI_strings']
            contrasts.append(dict(case=case,method=method,seed=seed,identical_global_frequency_basis=same,
                identical_all_bases=r.get('all_arms_identical_basis'),all_three_endpoint_energies_available=len(arms)==3,
                screening_required=p['request']['screening_required'],
                global_minus_frequency_Ha=g['energy_Ha']-f['energy_Ha'],
                global_minus_native_ratio1_Ha=g['energy_Ha']-n['energy_Ha'] if n else None,
                global_minus_old_R2_Ha=g['energy_Ha']-baseline['energy_Ha'],
                global_minus_frequency_penalized_Ha=g['penalized_energy_Ha']-f['penalized_energy_Ha'],
                global_minus_frequency_S2=g['S2']-f['S2'],
                Global_CPU_proxy=g['required_work']['canonical_CI_CPU_proxy'],
                frequency_CPU_proxy=f['required_work']['canonical_CI_CPU_proxy'],
                Global_required_CI_roots=g['required_work']['unique_CI_evaluations'],
                frequency_required_CI_roots=f['required_work']['unique_CI_evaluations']))
    new_count=db.execute('SELECT count(*) FROM observations WHERE is_new=1').fetchone()[0]
    attempts=db.execute('SELECT status,count(*),sum(cpu) FROM attempts GROUP BY status').fetchall()
    db.close()
    if new_count!=terminal['new_completed_CI_roots']:raise RuntimeError('Actual new-root work mismatch')
    sources=[]
    for key in dict.fromkeys((p['source']['case'],p['source']['method_id']) for p in frozen['plans']):
        group=[x for x in contrasts if (x['case'],x['method'])==key]
        item=dict(case=key[0],method=key[1],completed_paired_seeds=len(group),planned_seeds=3,
            identical_basis_seeds=sum(x['identical_global_frequency_basis'] for x in group),
            Global_vs_frequency=dict(Counter('IDENTICAL_BASIS' if x['identical_global_frequency_basis'] else sign(x['global_minus_frequency_Ha']) for x in group)),
            Global_vs_old_R2=dict(Counter(sign(x['global_minus_old_R2_Ha']) for x in group)))
        if group:
            item['complete_three_arm_seeds']=sum(x['all_three_endpoint_energies_available'] for x in group)
            for field in ['global_minus_frequency_Ha','global_minus_native_ratio1_Ha','global_minus_old_R2_Ha']:
                values=[1000*x[field] for x in group if x[field] is not None]
                item[field.replace('_Ha','_mHa')]=None if not values else dict(n=len(values),median=float(np.median(values)),minimum=min(values),maximum=max(values))
            cpu_g=[x['Global_CPU_proxy'] for x in group];cpu_f=[x['frequency_CPU_proxy'] for x in group]
            item['median_Global_CPU_proxy']=None if None in cpu_g else float(np.median(cpu_g))
            item['median_frequency_CPU_proxy']=None if None in cpu_f else float(np.median(cpu_f))
        sources.append(item)
    summary=dict(status='FIXED48_SAME_COUNTS_CLASSICAL_SEED_STABILITY_NOT_INDEPENDENT_QUANTUM_CONFIRMATION',
        planned_cells=48,cell_statuses=dict(Counter(x['status'] for x in statuses)),planned_endpoint_rows=144,
        completed_endpoint_rows=sum(x['status']=='COMPLETED' for x in rows),paired_cells=len(contrasts),
        sources=sources,new_completed_CI_roots=new_count,actual_attempts=attempts,
        new_campaign_CPU_seconds=terminal['CPU_seconds'],new_campaign_wall_seconds=terminal['wall_seconds'],
        Global_vs_frequency=dict(Counter('IDENTICAL_BASIS' if x['identical_global_frequency_basis'] else sign(x['global_minus_frequency_Ha']) for x in contrasts)),
        Global_vs_old_R2=dict(Counter(sign(x['global_minus_old_R2_Ha']) for x in contrasts)),
        new_model_shots=0,new_QPU_shots=0,source_observations_reused=True,goal_complete=False)
    save(ROOT/'analysis/T3-readout-seed-summary.json',summary)
    save(ROOT/'analysis/T3-paired-contrasts.json',contrasts)
    for filename,data in [('T3-readout-seed-all.csv',rows),('T3-readout-seed-cell-statuses.csv',statuses)]:
        out=StringIO();w=csv.DictWriter(out,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
        write(ROOT/'analysis'/filename,out.getvalue().encode('utf-8-sig'))
    text=['# T3：配置選択の改善は抽出seedを変えても残るか','',
        f"固定48セル（16source×3 classical seeds）の状態：{summary['cell_statuses']}。144予定endpoint中{summary['completed_endpoint_rows']}数値結果。新規CI計算{new_count}回、追加model/QPU Shot0。",'',
        '同じ保存countsを使い、4つのdisjoint screenの分割だけを変えた。量子回路の再実行、独立Quantum replicate、48独立分子の実証ではない。物理系は4条件/3分子で、N₂の2距離も同じ分子に属する。','',
        '## 全sourceの結果：負の差はGlobalのenergyが低い','',
        '| 条件 | source | 比較完了seed / 予定 | Global対同union頻度順 | 差の中央値 mHa | 差の範囲 mHa | 元全R2との差中央値 mHa | CPU proxy Global / 頻度順 |',
        '|---|---|---:|---|---:|---:|---:|---:|']
    for x in sources:
        if not x['completed_paired_seeds']:
            text.append(f"| {x['case']} | {x['method']} | 0 / 3 | 未比較 | — | — | — | — |");continue
        d=x['global_minus_frequency_mHa'];old=x['global_minus_old_R2_mHa'];cg=x['median_Global_CPU_proxy'];cf=x['median_frequency_CPU_proxy']
        counts=', '.join(f'{k}:{v}' for k,v in x['Global_vs_frequency'].items())
        text.append(f"| {x['case']} | {x['method']} | {x['completed_paired_seeds']} / 3 | {counts} | {d['median']:.3f} | {d['minimum']:.3f}〜{d['maximum']:.3f} | {old['median']:.3f} | {'—' if cg is None or cf is None else f'{cg:.3f} / {cf:.3f}'} |")
    text+=['','IDENTICAL_BASISは同じCI空間なのでGlobalの成功に数えない。異なるbasisでも差1e−8Ha以内は数値的同等。全4条件・全16source・全seedを残し、良いseedだけを選んでいない。Native ratio1との比較もcontrasts JSONと全144行CSVに残す。','',
        '各行の中央値・範囲は数値結果が得られたGlobal/頻度順のprimary pairだけの記述統計。Native armだけが後で失敗しても、既に得られたprimary pairを捨てない。Native比較の有効数はJSON内のnに別記。未完了seedを精度の勝ち・負け・0差へ補完しない。失敗・時間上限は全48セルstatus表とattempts費用に残す。現producerではscreenを先に実行するため、screenの時間上限で止まると、独立に実行できる頻度順armも未実行になり得る。この欠測は頻度順自身の計算失敗を意味しない。','',
        f"全paired cellのGlobal対同union頻度順：{summary['Global_vs_frequency']}。元全観測R2頻度順との比較：{summary['Global_vs_old_R2']}。",'',
        '## 何をそろえ、何はそろえていないか','',
        '- 同じsource、同じ観測counts、同じ原軌道、同じ有限0.1Ha S²罰則、同じC7 solver/k2/seed20261025。3armは同じ候補unionと同じmin(128,union数)²の実CI次元。OO/新しいrecovery/新しい外部配置は無し。',
        '- Globalは必要な4screenのCI情報を使う。候補が128以内なら全候補を残せるので、不要なscreenを省き、marginal weightsを捏造しない。頻度順には不要なscreen費用を負担させない。',
        '- 元R2は全観測からの頻度順なので、今回の小さいcandidate unionとは異なる場合がある。Globalが同union対照より良くても、元R2より悪い場合を隠さない。',
        '- physical energyとpenalized energy/S²を別々に出した。厳密S=0、大域最低根の数学証明、原著native HCI/DICE/full TN再現とはしない。', '',
        '## 費用の読み方','',
        f"新campaignは{terminal['CPU_seconds']:.2f}CPU秒/{terminal['wall_seconds']:.2f}wall秒。実際の新CIは{new_count}回、失敗を含むattempts集計は{attempts}。過去の同一問題・同一CI空間の観測を再利用し、古いenergy/evaluation originを上書きしていない。",'',
        '表のCPUは単独方式が必要とするunique CI空間のcanonical数値費用proxy。各空間に最初の互換観測の費用を統一して使うため、同じ根を使ったarm間で都合のよい履歴時間を選んでいない。同じCI空間が複数screenやfinalに出た場合は1回分。CIは元から固有vectorを生成するので、古いartifactにmarginalが残っていない場合の再計算は新campaignへ実費計上し、単独アルゴリズムに必要な別の根として二重請求しない。', '',
        'これはcold-start end-to-end時間ではない。request生成、freeze、実装/テスト、旧source構築/CCSD/TN/pilot、昔のShot/CI費用は別。CPUproxyだけでmatched-work優位性としない。全144行CSVのcell実費は3armで共有なので3重加算しない。元R2の当時の実測CPUも別列に保持。未知CPU/matvecはnullで、0にしていない。', '',
        '## 資源側との関係と次の判断','',
        '今回変更したのはCIへ渡す配置であり、sampler回路・CNOT・Depth・独立Parametersの変更ではない。SqDRIFT20の全bank圧縮資源はG5の証明済み結果へ参照できるが、G6 opt3はprefixのみなので全bank率へ置き換えない。CI出力の相関状態をsampling回路が量子準備したという意味でもない。', '',
        'Globalの効果が複数seedで残っても、追加screen費用と元全観測R2からのpool損失を含めて投資先を判断する。新しい比率/閾値を結果に合わせて変更しない。小poolのRAACSに順位を付けるだけでは観測支持は増えない。次は有望な部品を保持しつつ、強いsource/native HCI/TNと独立Quantum確認へ進む。最強論文への総合優越は依然未達。', '',
        '出典：[公式Trim SQD guide](https://qiskit.github.io/qiskit-addon-sqd/guides/trim_sqd.html)（公式APIを使用、Merz et al.への帰属はT1資料参照）。新しいGlobal marginal原理としての新規性は主張しない。', '',
        '[元T2](T2_POOLED_SELECTION_RESULTS_JA.md)、[G5全bank](G5_WHOLE_N20_RESULTS_JA.md)、[G6小対照](G6_OPT3_PREFIX_RESULTS_JA.md)。全行CSV、全セルstatus CSV、paired contrasts JSONを同じフォルダーへ保存。','']
    write(ROOT/'analysis/T3_READOUT_SEED_RESULTS_JA.md','\n'.join(text).encode())
    print(summary['cell_statuses'],'NEW_ROOTS',new_count,'PAIRED',summary['Global_vs_frequency'])


if __name__=='__main__':run()
