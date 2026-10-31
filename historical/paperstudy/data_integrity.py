"""Independent read-only integrity checks over actual completed molecular rows.

Not a new scientific approval gate. Unstarted/capped rows remain visible and
are not silently passed as numerical results. No additional energy evaluations.
"""
import json,sqlite3
from collections import Counter,defaultdict
from pathlib import Path
from time import time
import numpy as np
from .io import ROOT,save_json,sha256,atomic_bytes
from .common_execution import run_identity


def result_checks(result,expected,counts,counts_sha,problem_id):
    errors=[]
    def check(condition,message):
        if not condition:errors.append(message)
    case,method,B,seed,lane=expected
    check(result.get('status')=='COMPLETED','not completed result')
    check(result.get('case_id')==case and result.get('method_id')==method,'case/method mismatch')
    check(result.get('budget')==B and result.get('seed')==seed,'budget/seed mismatch')
    check(result.get('lane')==lane and result.get('run_id')==run_identity(case,method,B,seed,lane),'lane/run identity mismatch')
    check(result.get('problem_id')==problem_id,'physical problem mismatch')
    check(result.get('actual_QPU_shots')==0,'simulation unexpectedly recorded new QPU shots')
    paid=result.get('budget_ledger',{})
    check(paid.get('known_actual_shots')==B and paid.get('unknown_or_inflight_draws')==0,'unreconciled paid shots')
    final=result.get('independent_final_model_shots');pilot=result.get('pilot_model_shots')
    check(isinstance(final,int)and isinstance(pilot,int)and final+pilot==B,'pilot/final total mismatch')
    check(int(np.sum(counts,dtype=np.uint64))==final,'final counts do not match paid final shots')
    ci=result.get('final_CI',[])
    check(sorted(c.get('r_cap',-1)for c in ci)==[8,16,32],'missing/duplicate CI capacity')
    for c in ci:
        r=c.get('r_cap',0);d=c.get('actual_d',-1)
        check(0<d<=r*r,'actual CI dimension exceeds registered bound')
        check(np.isfinite(c.get('energy_Ha',np.nan)),'nonfinite CI energy')
        check(abs(c.get('valid_sector_fraction',0)-1)<1e-12,'ideal sampling has invalid sector counts')
        if 'problem_id'in c:check(c['problem_id']==problem_id,'CI problem binding mismatch')
        if 'final_counts_sha256'in c:check(c['final_counts_sha256']==counts_sha,'CI/final counts binding mismatch')
    ordered=sorted(ci,key=lambda c:c['r_cap'])
    for a,b in zip(ordered,ordered[1:]):
        if a.get('converged')and b.get('converged'):
            check(b['energy_Ha']<=a['energy_Ha']+1e-8,'converged nested CI energy increases')
    if 'member_records'in result:
        members=result['member_records'];check(sum(m['shots']for m in members)==final,'ensemble allocation mismatch')
        expected_gates=sum(m['shots']*m['resources']['Z_sampling_equivalent']['cnot_count']for m in members)
        check(result.get('total_model_cnot_applications')==expected_gates,'ensemble CNOT work mismatch')
    else:
        resources=result.get('sampling_resources',{})
        minimum=resources.get('cnot_count',0)*(final or 0)
        check(result.get('total_model_cnot_applications',-1)>=minimum,'total CNOT work less than independent final work')
    return errors


def extension_checks(extension,result,result_sha,counts_sha,source_path,hash_only_source_binding=False):
    """Check a persisted readout without rerunning its solver or adding a gate."""
    errors=[]
    def check(condition,message):
        if not condition:errors.append(message)
    check(extension.get('status')=='COMPLETED','extension not completed')
    for key in('case_id','method_id','budget','seed'):
        if key in extension or not hash_only_source_binding:
            check(extension.get(key)==result.get(key),'extension '+key+' mismatch')
    if 'lane'in extension:check(extension['lane']==result['lane'],'extension lane mismatch')
    if 'problem_id'in extension:check(extension['problem_id']==result['problem_id'],'extension problem mismatch')
    if 'source_result'in extension or not hash_only_source_binding:
        check(extension.get('source_result')==source_path,'extension source path mismatch')
    check(extension.get('source_result_sha256')==result_sha,'extension source hash mismatch')
    check(extension.get('source_counts_sha256')==counts_sha,'extension counts hash mismatch')
    check(extension.get('new_model_shots')==0 and extension.get('actual_QPU_shots')==0,'extension unexpectedly adds shots')
    ci=extension.get('CI',{})
    check(ci.get('r_cap')==64 and 0<ci.get('actual_d',-1)<=4096,'extension capacity mismatch')
    check(np.isfinite(ci.get('energy_Ha',np.nan)),'extension nonfinite energy')
    check(abs(ci.get('valid_sector_fraction',0)-1)<1e-12,'extension invalid sector counts')
    check(ci.get('total_raw_model_shots')==result['independent_final_model_shots'],'extension observation count mismatch')
    smaller=next((c for c in result['final_CI']if c['r_cap']==32),None)
    if smaller is None:errors.append('extension missing r32 source')
    else:
        for spin in('alpha','beta'):
            key='selected_'+spin+'_strings';strings=ci.get(key,[])
            check(len(strings)==len(set(strings))and len(strings)==ci.get('actual_d_'+spin),'extension '+spin+' basis dimension mismatch')
            check(key in smaller and set(smaller[key]).issubset(strings),'extension basis is not nested')
        check(ci.get('actual_d')==ci.get('actual_d_alpha',-1)*ci.get('actual_d_beta',-1),'extension Cartesian dimension mismatch')
        if smaller.get('converged')and ci.get('converged'):
            check(ci['energy_Ha']<=smaller['energy_Ha']+1e-8,'extension converged energy increases')
    return errors


def numerical_summary(rows):
    """Describe recorded convergence; residuals are not full-space error bars."""
    groups=defaultdict(list)
    for lane,ci in rows:groups[(lane,ci.get('r_cap'))].append(ci)
    answer=[]
    for(lane,r),values in sorted(groups.items()):
        residuals=[float(c['residual_norm'])for c in values if c.get('residual_norm')is not None and np.isfinite(c['residual_norm'])]
        answer.append({'lane':lane,'r_cap':r,'readouts':len(values),
            'recorded_converged':sum(c.get('converged')is True for c in values),
            'recorded_not_converged':sum(c.get('converged')is False for c in values),
            'convergence_unknown':sum(c.get('converged')not in(True,False)for c in values),
            'finite_residuals':len(residuals),'residual_min':min(residuals)if residuals else None,
            'residual_max':max(residuals)if residuals else None,
            'unknown_or_nonfinite_residuals':len(values)-len(residuals),
            'spin_purity_certified':False,'residual_is_not_full_CI_energy_error_bound':True})
    return answer


def run():
    db=sqlite3.connect(ROOT/'run-queue.sqlite');rows=[];states={}
    for table,lane in(('planned_cells','C_PRIMARY'),('sensitivity_cells','S5_SENSITIVITY'),('basis_cells','S_ORBITAL_BASIS')):
        if not db.execute('SELECT 1 FROM sqlite_master WHERE name=?',(table,)).fetchone():continue
        states[lane]=dict(db.execute('SELECT status,COUNT(*) FROM '+table+' GROUP BY status').fetchall())
        fields='case_id,method_id,budget,seed,result_path'if table!='sensitivity_cells'else'case_id,method_id,100000,seed,result_path'
        rows.extend(((*r[:4],lane),r[4])for r in db.execute('SELECT '+fields+' FROM '+table+' WHERE status="COMPLETED"'))
    db.close()
    for lane in('S_TN_INDEPENDENT_FINAL','S_TN_HIGH_BUDGET_COMPLETION'):
        for p in ROOT.glob('common-runs/'+lane+'/*/*/*/result.json'):
            r=json.loads(p.read_text());rows.append(((r['case_id'],r['method_id'],r['budget'],r['seed'],r['lane']),str(p.relative_to(ROOT))))
    problems={};violations=[];checked=0;counts_total=0;readouts=[];extensions_checked=0;hash_only_extensions=0
    ledger=sqlite3.connect(ROOT/'common-budget.sqlite')
    for expected,relative in rows:
        case,method,B,seed,lane=expected;p=ROOT/relative
        if case not in problems:
            m=json.loads((ROOT/'data/physical'/case/'problem.json').read_text())
            with np.load(ROOT/'data/physical'/case/'integrals.npz',allow_pickle=False)as a:
                actual={k:sha256(a[k].astype('<f8').tobytes())for k in('h1','eri')}
            if actual!=m['physical_arrays_sha256']:raise RuntimeError('Actual integral bytes changed: '+case)
            problems[case]=m['problem_id']
        result=json.loads(p.read_text());filename=p.parent/'final-counts.npz';blob=filename.read_bytes()
        with np.load(filename,allow_pickle=False)as a:ids,counts=a['blocked_ids'],a['counts']
        errors=result_checks(result,expected,counts,sha256(blob),problems[case])
        if len(ids)!=len(counts)or len(np.unique(ids))!=len(ids)or np.any(counts==0):errors.append('noncanonical final counts representation')
        actual=ledger.execute('SELECT SUM(actual),SUM(state!="COMPLETED") FROM draws WHERE run=?',(result['run_id'],)).fetchone()
        if actual!=(B,0):errors.append('live ledger does not reconcile to result')
        readouts.extend((lane,ci)for ci in result['final_CI'])
        extension_path=(ROOT/'CI-extension-v1'/case/method/(str(B)+'-'+str(seed))/'CI-r64.json'if lane=='C_PRIMARY'else
            ROOT/'control-CI-extension-v1'/lane/case/method/(str(B)+'-'+str(seed))/'CI-r64.json')
        if lane=='S_ORBITAL_BASIS':
            extension_path=ROOT/'basis-sensitivity-v1/readout-r64'/method/(str(seed)+'.json')
        if extension_path.exists():
            extension=json.loads(extension_path.read_text())
            hash_binding=lane=='S_ORBITAL_BASIS'
            errors.extend(extension_checks(extension,result,sha256(p.read_bytes()),sha256(blob),relative,hash_binding))
            readouts.append((lane,extension['CI']));extensions_checked+=1
            hash_only_extensions+=int(hash_binding)
        if errors:violations.append({'source':relative,'errors':errors})
        checked+=1;counts_total+=int(counts.sum())
    ledger.close()
    frozen=json.loads((ROOT/'main-execution-freeze-v1.json').read_text())
    frozen_changed=[p for p,value in frozen['code_sha256'].items()if sha256(Path(p).read_bytes())!=value]
    report={'observed_at_unix':time(),'completed_runs_checked':checked,'final_observations_checked':counts_total,
        'capacity_extensions_checked':extensions_checked,'basis_sidecars_bound_by_exact_source_and_counts_hash_not_copied_metadata':hash_only_extensions,
        'CI_numerical_summary':numerical_summary(readouts),
        'status_counts_including_unstarted':states,'physical_integral_problems_checked':problems,
        'violations':violations,'frozen_scientific_files_changed':frozen_changed,
        'completed_rows_consistent':not violations and not frozen_changed,
        'new_energy_evaluations':0,'new_model_shots':0,'actual_QPU_shots':0,
        'whole_experiment_complete':False,'unstarted_or_failed_rows_not_counted_as_numeric_passes':True}
    save_json(ROOT/'analysis-v1/actual-molecular-data-integrity.json',report)
    lines=['# 保存済み分子データの整合性と数値収束','',
        f'完了run {checked}件と、既存countsからのr64追加読出し{extensions_checked}件を確認しました。主比較だけの件数ではなく、S5・局在軌道・TNの完了runも含みます。追加の測定・energy計算はしていません。','',
        f'記録間の不整合：{len(violations)}件。凍結科学ファイルの変更：{len(frozen_changed)}件。未実行セルや上限停止を数値検証成功へ数えません。','',
        '| レーン | r上限 | 読出し数 | solver収束 | solver未収束 | 収束不明 | 最大記録残差 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for g in report['CI_numerical_summary']:
        residual='不明'if g['residual_max']is None else f"{g['residual_max']:.3g}"
        lines.append('| '+' | '.join(str(v)for v in(g['lane'],g['r_cap'],g['readouts'],g['recorded_converged'],g['recorded_not_converged'],g['convergence_unknown'],residual))+' |')
    lines+=['','収束は保存された有限CI空間でのsolver判定です。残差やsolver設定値を、全FCIに対するenergy誤差・統計的同等性marginへ読み替えません。粒子数sectorの検査とS²純粋性の保証も別です。',
        'r64は元counts/resultのSHA、case/method/予算/seed、基底の包含関係、r32とのenergy単調性を確認しました。未生成の追加読出しの残件はREQUIREMENT_COVERAGE_JA.mdを参照してください。',
        f'局在軌道の追加{hash_only_extensions}件は、sidecarに重複のcase/seed等を保存していません。元resultとcountsの完全SHA一致からその実行情報へ結合し、存在しないmetadataを過去記録へ追加していません。',
        '研究全体・原論文の数値再現の完了認定ではありません。']
    atomic_bytes(ROOT/'analysis-v1/ACTUAL_DATA_INTEGRITY_JA.md',('\n'.join(lines)+'\n').encode())
    print(json.dumps({k:report[k]for k in('completed_runs_checked','completed_rows_consistent','violations','frozen_scientific_files_changed')}),flush=True)


if __name__=='__main__':run()
