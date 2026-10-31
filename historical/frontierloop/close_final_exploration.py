"""N75 terminal audit and owner-authorized end of new-source exploration."""
from collections import Counter
from pathlib import Path
import numpy as np
from pyscf.cc.addons import spatial2spin
from .paths import ROOT
from .io import load,save,write,sha256
from .chemical_activity_preflight import LOCAL,choose,classify_target,component,DIRECTIVE
from paperstudy.lucj import amplitudes_from_saved

def run():
    p=load(LOCAL/'protocol.json');t=load(LOCAL/'terminal.json')
    for f,h in p['bindings'].items():assert sha256(Path(f).read_bytes())==h,'Frozen input drift '+f
    assert len(t['rows'])==120 and len(t['selected'])==12 and not t['all4_applicable']
    for s in p['sources']:
        meta=load(s['meta']);n,k=meta['norb'],meta['nalpha'];t1,t2,kind=amplitudes_from_saved(s['amplitudes'],n,k)
        spin1,spin2=spatial2spin(t1),spatial2spin(t2)
        for r in [r for r in t['rows'] if r['case']==s['case']]:
            assert all(len(label)==2*n for label,_ in r['group'])
            target=classify_target(r['group'],n,k);assert target==r['target'] and kind==r['amplitude_kind']
            value=component(target,spin1,spin2);assert value==r['spin_component']
            expected=float(value*value*r['score']['probability']) if value is not None and abs(value)>=p['component_abs_floor'] and r['score']['probability'] is not None else None
            assert expected==r['priority']
        for r in [r for r in t['selected'] if r['case']==s['case']]:
            assert r['selection']==choose([x for x in t['rows'] if x['case']==s['case'] and x['time']==r['time']])
            if r['selection']:
                f=Path(r['state_file']);assert sha256(f.read_bytes())==r['state_SHA256']
                with np.load(f,allow_pickle=False) as a:assert abs(np.vdot(a['amplitudes'],a['amplitudes']).real-1)<1e-10
    assert len([r for r in t['selected'] if r['selection'] is None])==1
    missing=next(r for r in t['selected'] if r['selection'] is None);assert (missing['case'],missing['time'])==('FE2S2-40-P2',3)
    taxonomy=Counter(r['target']['status'] for r in t['rows']);frozen=dict(owner_directive_SHA256=sha256(DIRECTIVE.read_bytes()),
        final_preflight_protocol_SHA256=sha256((LOCAL/'protocol.json').read_bytes()),final_preflight_terminal_SHA256=sha256((LOCAL/'terminal.json').read_bytes()),
        exploration_loops_consumed=2,new_source_search_authorized=False,third_loop_authorized=False,
        final_loop_disposition='REJECTED_ALL4_POLICY_NOT_APPLICABLE_FIXED_FE_TIME3_LIBRARY',
        last_loop_CPU_seconds=t['CPU_seconds'],last_loop_wall_seconds=t['wall_seconds'],last_loop_new_model_samples=0,last_loop_new_QPU_shots=0,last_loop_new_CI_roots=0,
        further_chemical_source_pilot_authorized=False,frozen_confirmation_and_conclusion_authorized=True,
        scientific_superiority_goal_achieved=False,study_confirmation_complete=False,
        source_taxonomy=dict(taxonomy),all120_rows_recomputed=True)
    target=ROOT/'analysis/EXPLORATION_SEARCH_CLOSED_2026-10-27.json'
    if target.exists():assert load(target)==frozen
    else:save(target,frozen)
    lines=['# 最後の探索ループN75：棄却して新案探索を終了','',
        '最新owner directiveに従い、N72–N74を第1ループ、N75を最後の第2ループとして数えた。新しい案・候補pool・分子・角度・seedへの救済探索は開始しない。','',
        '| 条件 | 3時間枠への適用 | 適格枠の予定数/全予定数 | 適格枠の期待非pair観測小計 | 振幅の由来 |','|---|---|---|---:|---|']
    for c in t['cases']:
        known=sum(r['shots'] for r in t['selected'] if r['case']==c['case'] and r['selection'] is not None)
        lines.append(f"| {c['case']} | {'全3枠可' if c['applicable_all3times'] else '欠測枠あり'} | {known}/1000 | {c['expected_nonpair_observations_per1000']:.6f} | {c['amplitude_kind']} |")
    lines+=['','## 棄却の理由','',
        'Fe₂S₂の時間3の10候補は全て、HFを基準にした粒子数保存single/double targetに対応しなかった。固定したproxyの適格候補がなく、1枠だけ選択不能だった。',
        'rawのexpected_nonpair_observations_per1000というfield名は、Feでは適格な時間1/2の667予定サンプル分の小計495.188309を保持している。未定義の時間3の333サンプルを0として埋めた全1000回の期待値ではない。rawは変更せず、本表では小計と対象数を明示する。実際のsamplingは行っていない。',
        '他の3条件だけを成功集合にせず、Feだけstate-activityへfallbackせず、全4条件の同一規則という事前条件により新方策を棄却した。これは既存120候補と本proxyの適用性の結果で、化学情報を用いる全ての方法が無効という結論ではない。',
        'PySCFのpinned spin変換・同spin反対称化・CISD登録fallbackは5事前テストで確認。全120分類・振幅component・priorityを再計算して照合した。',
        f"実費CPU {t['CPU_seconds']:.6f}秒、wall {t['wall_seconds']:.6f}秒。新model samples/QPU Shot/energy/CI/VQE/FCIはいずれも0。",'',
        '## 次に許可される作業','',
        '既存の最大2方策を全4条件で固定し、強いLUCJ・短いSqDRIFT・classical HCI・compiler対照を含む有限確認と結論作成へ移る。B4は参照可能だが採用・勝利を義務にしない。',
        '新しい独立読出しと、過去counts/subset/根の再利用を区別する。構築pilotと初期化費用、実R/D、欠測・上限を表示する。原著最高設定が未再現ならその範囲への優越を主張しない。',
        '新案探索は終了したが、確認試験と最終結論はまだ未完了。最強論文への優越という元科学目標を達成済みに変更しない。','']
    write(ROOT/'analysis/N75_FINAL_EXPLORATION_CLOSEOUT_JA.md','\n'.join(lines).encode())
    print('NEW_SOURCE_SEARCH_CLOSED_LOOP2_REJECTED_120_AUDITED',flush=True)

if __name__=='__main__':run()
