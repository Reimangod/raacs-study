"""Export B4 and paper-method comparison data without running experiments."""
from pathlib import Path
import csv,io,json,hashlib,zipfile,os,time,re
from collections import Counter,deque

W=Path(__file__).resolve().parent
P=W/'outputs/paper-aligned-confirmation-2026-10-28'
N=P/'naphthalene-extension-v1'
C=N/'conference-delivery-final-v2'
OUT=W/'exports/B4_paper_comparison_2026-10-28'
OUT.mkdir(parents=True,exist_ok=True)
METHODS={'B4','LUCJ1','LUCJ2','LUCJ4','SqDRIFT5','SqDRIFT20','SqDRIFT50'}
CASES={'NAPH20-BLOCK-BOYS','N2-32-R1p2-P2','N2-32-R2p4-P2','FE2S2-40-P2'}
sources={};generated={};provenance=[];table_counts={};missing=[];queue=deque();visited=set()
def sha(b):return hashlib.sha256(b).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def eligible_path(p):
    return p.is_relative_to(W)and 'ACTIVE'not in p.parts and not any(re.match(r'^(LiH|BeH2|H2O|H2|H4|H6|NH3|HF)(?:_|-|$)',part)for part in p.parts)
def filtered(obj):
    if isinstance(obj,dict):
        # Native author implementations use aliases such as
        # C5_C_LUCJ_COMPRESSED_L2. Preserve those descriptions and the CCSD/CISD
        # initialization records, rather than deleting valid paper-method input.
        if 'method'in obj:
            method=str(obj['method']).upper()
            allowed=obj['method']in METHODS or 'LUCJ'in method or 'SQDRIFT'in method or bool(re.search(r'(^|_)B4($|_)',method))or method in {'CCSD','CISD'}
            if not allowed:return None
        if 'case'in obj and isinstance(obj['case'],str)and obj['case']not in CASES:return None
        return {k:v2 for k,v in obj.items()if not k.startswith('ACTIVE')and k!='ACTIVE'and(v2:=filtered(v))is not None}
    if isinstance(obj,list):return [v2 for v in obj if(v2:=filtered(v))is not None]
    if obj=='ACTIVE':return None
    return obj
def strings(obj):
    if isinstance(obj,dict):
        for k,v in obj.items():
            if isinstance(k,str):yield k
            yield from strings(v)
    elif isinstance(obj,list):
        for v in obj:yield from strings(v)
    elif isinstance(obj,str):yield obj
def add(p):
    p=Path(p)
    if not p.is_absolute():p=W/p
    p=p.resolve()
    if not eligible_path(p)or p.suffix in {'.pyc','.lock'}or '__pycache__'in p.parts or '.venv'in p.parts:return
    if not p.is_file():return
    if p not in sources:sources[p]='raw/'+p.relative_to(W).as_posix();queue.append(p)
def tree(p):
    if not p.is_dir():return
    for base,dirs,files in os.walk(p):
        dirs[:]=[d for d in dirs if d not in {'__pycache__','.git','.venv'}]
        for name in files:
            if name!='.DS_Store':add(Path(base)/name)
def table(src,name):
    data=src.read_bytes();rows=list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))
    kept=[r for r in rows if r.get('method')in METHODS and (not r.get('case')or r['case']in CASES)]
    out=io.StringIO();writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(kept)
    generated['tables/'+name]=out.getvalue().encode('utf-8-sig');table_counts[name]=len(kept)
    provenance.append({'source':str(src),'source_SHA256':sha(data),'export':'tables/'+name,'operation':'row selection only; numbers and missingness unchanged','selected_rows':len(kept),'original_rows':len(rows)})
    for r in kept:
        for v in r.values():
            if isinstance(v,str)and v.startswith(str(W)+'/'):add(v)
    return kept
def view(src,name):
    data=src.read_bytes();d=filtered(json.loads(data.decode('utf-8-sig')))
    generated['conditions/'+name]=json.dumps(d,ensure_ascii=False,indent=2).encode()
    provenance.append({'source':str(src),'source_SHA256':sha(data),'export':'conditions/'+name,'operation':'selected-method export view, not a replacement frozen protocol'})
    for value in strings(d):
        if value.startswith(str(W)+'/'):add(value)
    return d

old=table(P/'matched-shot-D-v1/all-endpoints.csv','existing_85120shots_all_endpoints.csv')
table(P/'matched-shot-D-v1/group-summary.csv','existing_85120shots_group_summary.csv')
new=table(C/'new-cohort-all-endpoints.csv','new_naphthalene_all_endpoints.csv')
table(C/'new-cohort-group-summary.csv','new_naphthalene_group_summary.csv')
table(C/'cost-components.csv','new_naphthalene_cost_components.csv')
table(C/'existing-resource-points-with-missing.csv','existing_common_D_resource_points.csv')
table(C/'existing-sample-capacity-runs.csv','existing_shot_sweep_sample_capacity.csv')
op=view(P/'matched-shot-D-v1/protocol.json','existing_matched_shot_D_protocol_VIEW.json')
np=view(N/'readouts/protocol.json','new_naphthalene_readout_protocol_VIEW.json')
view(N/'protocol.json','new_naphthalene_construction_protocol_VIEW.json')
view(N/'common-resources-v2.json','new_naphthalene_authoritative_resources_v2_VIEW.json')
proto=view(P/'protocol-v1.json','existing_construction_protocol_VIEW.json')

for r in op['inputs']:
    if r['method']=='B4':tree(P/'runs/B4-v2'/r['case'])
    if r.get('source_counts'):
        f=Path(r['source_counts']);add(f.with_suffix('.json'))
for m in sorted(METHODS):
    tree(N/'sources'/m);tree(N/'counts'/m);tree(N/'native-resource-successor'/m)
    tree(N/'readouts/subsets'/m)
for r in new:
    eid=r['id'];add(N/'readouts/endpoints'/f'{eid}.json')
for r in old:add(P/'matched-shot-D-v1/endpoints'/f'{r["id"]}.json')

# Follow file-level provenance only. Never traverse a referenced directory or
# another experiment namespace wholesale.
start=time.monotonic()
while queue:
    p=queue.popleft()
    if p in visited:continue
    visited.add(p)
    if p.suffix=='.json'and p.stat().st_size<3*1024**2:
        try:d=load(p)
        except(ValueError,OSError):continue
        clean=filtered(d)
        if clean!=d:
            sources[p]='views/'+p.relative_to(W).as_posix()
            generated[sources[p]]=json.dumps(clean,ensure_ascii=False,indent=2).encode()
            provenance.append({'source':str(p),'source_SHA256':sha(p.read_bytes()),'export':sources[p],'operation':'mixed record filtered to requested methods/cases'})
        for s in strings(clean):
            if s.startswith(str(W)+'/'):add(s)
        for ext in ['.npz','.qpy']:
            sibling=p.with_suffix(ext)
            if sibling.exists():add(sibling)

# Include the physical problems and initial amplitudes for the four registered
# large cases, independently of whether a method reached a target dimension.
base=W/'outputs/raacs-lucj-sqdrift-paper-aligned-plan-2026-10-25/execution-v1'
for case in CASES:
    tree(base/'data/physical'/case);tree(base/'data/initialization-fresh'/case)
for p in [W/'outputs/raacs-frontier-research-loop-2026-10-25/frontierloop/beam.py',W/'outputs/raacs-frontier-research-loop-2026-10-25/frontierloop/spin_beam.py',W/'outputs/raacs-frontier-research-loop-2026-10-25/frontierloop/singlet_cache.py',P/'matched_readout.py',P/'sdk_readout_v2.py',P/'naphthalene_resources.py',P/'naphthalene_resource_successor.py']:add(p)

index=[]
for cohort,rows in [('existing_85120_final_shots',old),('new_naphthalene',new)]:
    for r in rows:
        index.append({'cohort':cohort,'case':r.get('case','NAPH20-BLOCK-BOYS'),'method':r['method'],'seed':r['seed'],'final_shots':r.get('used_final_shots',r.get('final_shots')),'context':r.get('context','EXACT_D'),'target_D':r['target_D'],'actual_D':r.get('actual_D'),'status':r['status'],'energy_Ha':r.get('energy_Ha'),'Recovery_pct':r.get('Recovery_pct'),'CNOT':r.get('CNOT',r.get('cnot_count')),'CNOT_Depth':r.get('CNOT_Depth',r.get('cnot_depth')),'Total_Depth':r.get('Total_Depth',r.get('total_depth')),'CI_CPU_seconds':r.get('actual_new_CI_CPU_seconds',r.get('CI_CPU_seconds')),'source_root':r.get('root',''),'source_counts':r.get('source_counts','')})
s=io.StringIO();writer=csv.DictWriter(s,fieldnames=list(index[0]));writer.writeheader();writer.writerows(index);generated['tables/START_HERE_all_endpoint_index.csv']=s.getvalue().encode('utf-8-sig')
generated['provenance/EXPORT_TRANSFORMS.json']=json.dumps(provenance,ensure_ascii=False,indent=2).encode()

readme='''# B4と既存論文手法の比較データ

## 対象
- ナフタレン：20 qubit。
- N₂：32 qubit、結合長1.2 Åと2.4 Å。
- Fe₂S₂：40 qubit。
- B4、LUCJ 1/2/4層、SqDRIFT 5/20/50。SqDRIFT50は新ナフタレン群のみ。

小規模分子、別の提案手法、過去のCEO圧縮・Transformer実験は収録しません。直近の比較で使用した保存済みの表、生サンプル、選択配置、CI結果、回路、構築記録、物理問題、計数規則を収録しています。全ての旧探索版・未採用分岐の総バックアップではありません。

## 最初に開くファイル
tables/START_HERE_all_endpoint_index.csv：条件・手法・seed・Shot・D・Energy・Recovery・回路資源・CI CPU秒をまとめた一覧。
tables/new_naphthalene_group_summary.csv：新ナフタレン3回の集計。
tables/existing_85120shots_group_summary.csv：既存4条件の集計。
tables/new_naphthalene_cost_components.csv：探索・最終測定・古典CPUの費用。

## データを混同しないための注意
1. 既存群の最終85,120 model Shotと、新ナフタレン群の最終753,120 model Shotは別の実験群です。contextがPREFIX_CRNの行はさらに別の測定量の系列です。母数は各行に記録しています。
2. Shotは無雑音モデルからのサンプル数で、実QPU shotは0です。
3. Energyは測定回路のVQE期待値ではなく、サンプル由来の空間で計算したCI後の物理Hの値です。共通最終CIはH+0.1S²を解いて物理Hを評価し、厳密S=0射影とは区別します。
4. Recoveryは相関エネルギー回収率です。配置回復処理の成功率ではありません。Fe₂S₂の参照値は未認証であり、そのRecoveryを認証済みの値として扱わないでください。
5. 本実験のRはスピン別文字列数、D=R²です。原著によってRの意味が異なります。
6. 未到達、欠測、失敗はそのまま残しています。D未到達を0 Energyや成功回だけの平均で補っていません。
7. 最終Shot/Dが同じでも、構築・初期化・pilot・古典計算まで全て同じ費用とは限りません。LUCJはcompressed-t2初期化を基本とする登録実装で、最強のVQE/TN最適化版を再現したという意味ではありません。
8. 新ナフタレンのLUCJ資源はauthor-native simplificationを適用したresource v2（1/2/4層のCNOT=322/544/988）を採用します。raw内の古い一般コンパイラー値や取得時メタデータを、最新の資源表へ上書きして混ぜないでください。既存群は当時の登録資源値を保持しています。
9. 数値は我々の条件で行ったシミュレーション結果で、原著論文の数値の転記や完全なbyte-identical再現ではありません。

## フォルダー
- tables/：指定手法の行だけを抜き出したCSV。数値は変更していません。
- conditions/：元の条件ファイルから指定手法だけを取り出した閲覧用VIEW。新しいfreezeや元プロトコルの改変ではありません。予算・閾値は元のままです。
- raw/：選択した実験の元ファイルをbyte-identicalで格納。
- views/：複数手法を含む元記録から対象外レコードを除いた派生ビュー。
- provenance/：元ファイルのハッシュ、抽出操作、元パスとZIP内パスの対応。
- MANIFEST.csv：ZIP内ファイルのサイズとSHA-256。

元の絶対パスがJSON内に残っている場合は、provenance/SOURCE_PATH_MAP.csvでZIP内の場所を確認してください。未保存のCIベクトルや係数を捏造して補ってはいません。元のvector_SHA256だけが残る記録は、その状態で収録します。

元データの変更・削除や、新規の実験・最適化は行っていません。未公開のデータを含むので外部共有前に内容を確認してください。
'''
generated['README_JA.md']=readme.encode()

archive=OUT.parent/'B4_vs_LUCJ_SqDRIFT_experiment_data_2026-10-28.zip'
if archive.exists():raise RuntimeError('Archive already exists')
manifest=[];pathmap=[];raw_total=0
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True)as z:
    for p,name in sorted(sources.items(),key=lambda kv:kv[1]):
        if name in generated:continue
        if not p.exists():missing.append(str(p));continue
        before=p.stat();data=p.read_bytes();after=p.stat()
        if(before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('Source changed during capture: '+str(p))
        z.writestr(name,data);manifest.append({'path':name,'bytes':len(data),'SHA256':sha(data),'kind':'original'});pathmap.append({'original_path':str(p),'zip_path':name,'SHA256':sha(data),'kind':'original'});raw_total+=len(data)
    for name,data in sorted(generated.items()):
        z.writestr(name,data);manifest.append({'path':name,'bytes':len(data),'SHA256':sha(data),'kind':'export_view'})
    for p,name in sources.items():
        if name in generated:pathmap.append({'original_path':str(p),'zip_path':name,'SHA256':sha(generated[name]),'kind':'filtered_view'})
    def write_csv(name,rows):
        s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);z.writestr(name,s.getvalue().encode('utf-8-sig'))
    write_csv('provenance/SOURCE_PATH_MAP.csv',pathmap);write_csv('MANIFEST.csv',manifest)
    report={'cases':sorted(CASES),'methods':sorted(METHODS),'tables':table_counts,'endpoint_rows':len(index),'payload_files':len(manifest),'raw_bytes':raw_total,'missing_files':missing,'new_experiments':0,'source_data_changed':False}
    z.writestr('provenance/PACKAGE_REPORT.json',json.dumps(report,ensure_ascii=False,indent=2))

# Verify every payload from the finished ZIP; CRC is checked by zipfile too.
with zipfile.ZipFile(archive)as z:
    for row in manifest:
        data=z.read(row['path']);assert len(data)==row['bytes']and sha(data)==row['SHA256'],row['path']
    for name in table_counts:
        rows=list(csv.DictReader(io.StringIO(z.read('tables/'+name).decode('utf-8-sig'))))
        assert all(r['method']in METHODS for r in rows)
        assert len(rows)==table_counts[name]
    assert not any('/ACTIVE/'in n for n in z.namelist())
    assert z.testzip()is None
digest=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest();report.update(zip_bytes=archive.stat().st_size,zip_SHA256=digest,all_payload_hashes_verified=True)
(OUT/'PACKAGE_REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));(OUT/'README_JA.md').write_text(readme);archive.with_suffix('.zip.sha256').write_text(digest+'  '+archive.name+'\n')
print(json.dumps({'archive':str(archive),**report},ensure_ascii=False,indent=2),flush=True)
