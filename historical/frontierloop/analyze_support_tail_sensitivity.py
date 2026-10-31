"""Audit/report N64B; numeric zero-support artifacts cannot be claimed as wins."""
from pathlib import Path
import numpy as np
from .paths import ROOT
from .io import load,save,write,sha256
from .support_tail_sensitivity import LOCAL
from .support_saturation import LOCAL as N64

def run():
    p=load(LOCAL/'protocol.json');t=load(LOCAL/'terminal.json');prior=load(ROOT/'analysis/N64-support-summary.json')
    assert sha256((LOCAL.parents[1]/'frontierloop/support_tail_sensitivity.py').read_bytes())==p['code_SHA256']
    assert sha256((N64/'terminal.json').read_bytes())==p['prior_terminal_SHA256']
    assert len(t['rows'])==24 and all(r['status']=='COMPLETED' for r in t['rows'])
    assert t['CPU_seconds']<p['CPU_cap'] and t['wall_seconds']<p['wall_cap']
    groups=[];finite_identical=0
    # Obtain original roots from the source-bound cache files, without path guesses.
    original=[]
    for f in load(N64/'protocol.json')['cache_files'][:2]:original+=load(f)['states']
    details=[]
    for r in t['rows']:
        root=r['root'];assert sha256(Path(root['root_file']).read_bytes())==root['root_SHA256']
        assert root['result']['residual_norm']<=1e-8 and max(root['independent_RDM_audit']['errors'].values())<=1e-7
        matches=[x for x in prior['rows'] if (x['case'],x['arm'],x['seed'])==(r['case'],r['arm'],r['seed'])]
        assert len(matches)==1
        prev=matches[0]
        finite=next(x for x in original if x['draw_id']==r['draw_id'] and x['R_cap']==128 and x.get('context','MAIN')=='MAIN')
        same=r['basis_id']==finite['basis_id']
        if r['arm']=='FULL_N5':assert same;finite_identical+=int(same)
        d=dict(case=r['case'],seed=r['seed'],arm=r['arm'],finite_energy_Ha=prev['finite_energy_Ha'],
            full_positive_energy_Ha=prev['full_support_energy_Ha'],tail_removed_energy_Ha=root['result']['energy_Ha'],
            meaningful_R=r['actual_R'],numeric_positive_R=r['prior_numerical_R'],finite_R=prev['finite_R'],
            removed_half_mass=r['removed_half_mass'],union_bound_1000_samples=min(1,2000*r['removed_half_mass']),
            delta_vs_finite_mHa=1000*(root['result']['energy_Ha']-prev['finite_energy_Ha']),finite_identical_basis=same)
        details.append(d)
    for case in sorted({r['case'] for r in details}):
        for arm in ['FULL_N5','FILTERED_SHORT']:
            rows=[r for r in details if r['case']==case and r['arm']==arm]
            groups.append(dict(case=case,arm=arm,finite_R=[r['finite_R'] for r in rows],meaningful_R=[r['meaningful_R'] for r in rows],
                finite_energy_Ha=float(np.mean([r['finite_energy_Ha'] for r in rows])),tail_removed_energy_Ha=float(np.mean([r['tail_removed_energy_Ha'] for r in rows])),
                delta_vs_finite_mHa=float(np.mean([r['delta_vs_finite_mHa'] for r in rows])),max_removed_mass=max(r['removed_half_mass'] for r in rows)))
    s=dict(status='ALL24_SENSITIVITY_ROOTS_AUDITED',groups=groups,details=details,actual_new_CI_roots=len(t['attempts']),
        all12_N5_meaningful_bases_already_observed=finite_identical==12,new_model_samples=0,new_QPU_shots=0,
        CPU_seconds=t['CPU_seconds'],wall_seconds=t['wall_seconds'],posthoc_diagnostic=True,not_an_equivalent_rewrite=True,
        finite_Shot_results_changed=False,goal_complete=False)
    save(ROOT/'analysis/N64B-tail-sensitivity-summary.json',s)
    lines=['# N64B：全支持oracleの改善を、そのまま測定価値と見なしてはいけない','',
        'N64の数値全支持には確率1e-35以下の配置が含まれた。全24drawに共通の1e-12半配置確率閾値を適用した事後感度検査を実施した。閾値は本番sampling/compiler規則に採用しておらず、元の有限Shot結果は不変。',
        '極小確率の配置を外すだけでCI energyが大きく変わった。数値全支持の改善幅は、追加Shotで得られる改善予測には使えない。CIは小さいsource振幅に係数を固定せず再対角化するため、配置を一つ加える効果と、その配置が測定される確率は別の量である。','',
        '| 条件 | 方法 | 実1,000回のR | 尾部除去後R | 実1,000回 E Ha | 尾部除去後 E Ha | 差 mHa | 除去half確率質量の最大 |',
        '|---|---|---|---|---:|---:|---:|---:|']
    for g in groups:lines.append(f"| {g['case']} | {g['arm']} | {g['finite_R']} | {g['meaningful_R']} | {g['finite_energy_Ha']:.9f} | {g['tail_removed_energy_Ha']:.9f} | {g['delta_vs_finite_mHa']:.3f} | {g['max_removed_mass']:.3e} |")
    lines+=['','## 判断','',
        '通常N5は12draw全てで、尾部除去後のCI basisが既存1,000回のbasisと完全一致した。固定9回路へのShot追加を救済策にしない。これは登録モデルの数値診断で、全てのN5回路への不可能性定理ではない。',
        'SHORTではナフタレンとN₂にまだ観測可能な配置の未取得が残る一方、Feの3drawはすでに同じbasisだった。Feを増Shotだけで救済する根拠も弱い。',
        '除去半配置の平均確率質量をmとすると、登録済み配分で1,000回のどれかにその配置が出る確率はunion boundで2,000m以下。本検査のmは最大でも1e-32未満で、測定で容易に回収できる尾部ではない。数値source誤差や実機noiseの別の分布まで保証する式ではない。',
        'N64のrawと結果は保全するが、数値全支持の改善を成功実験として採用しない。今の有限Shot表・N63の結論は変わらない。次の候補はN65の同総Shot・ensemble幅比較であり、数値尾部を使ったsampler優越主張ではない。','',
        f"実費：新CI根{len(t['attempts'])}、全24状態数値化、body CPU {t['CPU_seconds']:.6f}秒、wall {t['wall_seconds']:.6f}秒、新Shot0。古い26根と二重計上しない。",
        '派生report作成時に旧context欄欠測、変数名衝突、N64Bの旧schema ID参照の3件を修正した。失敗test XML v1/v3を保全し、raw/source/CIの再実行はしていない。','']
    write(ROOT/'analysis/N64B_NUMERICAL_TAIL_RESULTS_JA.md','\n'.join(lines).encode())
    print('N64B_AUDITED24',len(t['attempts']),flush=True)

if __name__=='__main__':run()
