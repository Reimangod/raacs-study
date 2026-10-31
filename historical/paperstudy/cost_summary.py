"""Readable S6 cost subtotals, never an overlapping grand total.

Only persisted cost evidence is read. No circuit, count, energy or experiment
setting is changed. Unknown CPU times and lower bounds stay visible.
"""
import csv,json,math
from collections import defaultdict
from .io import ROOT,atomic_bytes,save_json,sha256


def summarize(rows):
    groups=defaultdict(list)
    for row in rows:groups[row['category']].append(row)
    result=[]
    for category,items in sorted(groups.items()):
        measured=[];unknown=lower=duplicates=0
        for row in items:
            if str(row.get('duplicate_compatibility_record_not_additive','')).lower()=='true':
                duplicates+=1;continue
            value=row.get('CPU_seconds')
            if value in(None,''):unknown+=1;continue
            value=float(value)
            if not math.isfinite(value)or value<0:raise ValueError('Invalid measured CPU cost')
            measured.append(value)
            lower+=str(row.get('CPU_value_is_lower_bound','')).lower()=='true'
        result.append({'category':category,'evidence_records':len(items),'measured_CPU_records':len(measured),
            'unknown_CPU_records':unknown,'lower_bound_CPU_records':lower,
            'duplicate_compatibility_records_not_added':duplicates,
            'known_CPU_seconds_subtotal':sum(measured)if measured else None,
            'known_CPU_hours_subtotal':sum(measured)/3600 if measured else None,
            'subtotal_not_complete_if_unknown_or_lower_bound':bool(unknown or lower),
            'different_categories_not_additive_without_coverage_reconciliation':True})
    return result


def section(category):
    if any(x in category for x in('storage','offline_index')):return '保存・圧縮などの管理作業（手法の優劣に使わない）'
    if category.startswith(('P1','P2','H_')):return '論文接続・公開データ再解析（主比較とは別）'
    if category.startswith('S3'):return '実装pilot（主比較には加えない）'
    if category.startswith('TN'):return 'TN調整と最終測定（失敗・時間切れも残す）'
    if 'compiler'in category or 'recompile'in category:return 'SqDRIFTの別compilerによる資源検算'
    if category.startswith('classical_'):return '古典HCI対照'
    if category.startswith('fixed20q'):return '20q固定状態のMPS誤差診断'
    if 'CI'in category:return '保存countsを読み直す追加CI（追加Shotなし）'
    if category.startswith(('shared_','S5_compressed','S5_shared')):return '共有した初期化・回路準備（一回払い）'
    if category.startswith(('local_primary','remote_shared')):return '主比較のworker・共有状態準備'
    return '層数・長さ・軌道の追加対照'


def run():
    folder=ROOT/'analysis-v1';source=folder/'canonical-cost-evidence-no-overlap-within-category.csv'
    with source.open()as stream:rows=list(csv.DictReader(stream))
    groups=summarize(rows)
    output={'source':source.name,'source_SHA256':sha256(source.read_bytes()),'category_subtotals':groups,
        'new_model_shots':0,'new_energy_evaluations':0,'grand_total_CPU_seconds':None,
        'grand_total_reason':'Worker, child/CI, diagnostic and Slurm evidence can cover overlapping work; categories are not blindly summed',
        'unknown_cost_not_zero':True,'GPU_allocation_is_not_kernel_time_or_speedup':True}
    save_json(folder/'cost-subtotals-with-unknowns.json',output)
    lines=['# 計算費用を大まかに読む','',
        '同じShot数でも、計算全体の費用は同じではありません。RAACSは短い回路を選ぶ途中で測定とCIを使い、LUCJはCCSD/CISDや初期化、SqDRIFTは多数の回路生成を使います。','',
        '以下は保存されたCPU秒の**種類別小計**です。worker全体とその内部CIなど、種類をまたぐ重なりがあるため、この表を縦に足した「研究総CPU時間」は出しません。wall時間も並列・待ち時間・共有状態を含むためCPU秒と交換できません。','',
        '数値は既知の部分だけです。未知欄がある小計や下限値を、完全なend-to-end費用として比較しません。共有の初期化は実消費では一回だけです。各設定を単独で使う場合の費用は、別の[共有・単独運用表](LUCJ-shared-versus-standalone-preprocessing.csv)を参照してください。','']
    headings=list(dict.fromkeys(section(r['category'])for r in groups))
    for heading in headings:
        lines+=['## '+heading,'','| 費用の出典分類 | 既知CPU時間 h | CPU記録あり | CPU不明 | 下限記録 | 二重加算しない参照 |','|---|---:|---:|---:|---:|---:|']
        for r in groups:
            if section(r['category'])!=heading:continue
            value='不明'if r['known_CPU_hours_subtotal']is None else f"{r['known_CPU_hours_subtotal']:.4f}"
            lines.append('| '+' | '.join(str(x)for x in(r['category'],value,r['measured_CPU_records'],r['unknown_CPU_records'],r['lower_bound_CPU_records'],r['duplicate_compatibility_records_not_added']))+' |')
        lines.append('')
    lines+=['## 何に使える表か','',
        '- 遅さの所在、初期化の一回払い、追加CIや失敗したTN試行の負担を見つけるための表です。','- CPU機種、並列数、cache、元入力の作成費用が違うため、これだけで全手法の厳密な速度順位は決めません。','- 各runのShot・pilot/final・CNOT適用総数は[run費用表](per-run-paid-costs-counted-once.csv)、元のCPU・wall証拠は[正規費用台帳](canonical-cost-evidence-no-overlap-within-category.csv)にあります。','- [Slurmのjob別割当](slurm-job-costs-counted-once.csv)は、実CPU時間・割当CPU時間・割当GPU時間を分けています。GPU割当の重複は、GPU kernel利用率や高速化の証明ではありません。','- 実QPUへの新規実行は0です。model ShotとCPU/GPUシミュレーション費用を混同しません。','',
        'この集計は現在ある証拠だけを読みます。数値実験の完了・原論文の再現成功は、この費用表だけでは認定しません。']
    atomic_bytes(folder/'COST_SUMMARY_JA.md',('\n'.join(lines)+'\n').encode())
    print(json.dumps({'cost_categories':len(groups),'unknown_CPU_records':sum(r['unknown_CPU_records']for r in groups),
        'grand_total_CPU_seconds':None,'new_model_shots':0}),flush=True)


if __name__=='__main__':run()
