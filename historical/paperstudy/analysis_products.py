"""All observed grid points, explicit coverage, resources and honest costs.

Read-only with respect to scientific artifacts. Plots are descriptive; no
interpolation, best-seed selection, method tuning or confirmatory p values.
"""
from collections import Counter,defaultdict
import csv
from io import StringIO
import json
from pathlib import Path
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .io import ROOT,atomic_bytes,save_json,sha256
from .research_report import reference


def execution_open_scopes(planned):
    terminal={'COMPLETED','FAILED_ENGINEERING_PRESERVED','RESOURCE_CAPPED',
        'ENGINEERING_OR_RESOURCE_LIMIT_NO_FINAL_STATE','INITIALIZATION_CAPPED'}
    scopes=[]
    for lane,expected,label in(('C_PRIMARY',480,'All primary cells'),('S5_SENSITIVITY',125,'Layer/length sensitivities'),
        ('S_ORBITAL_BASIS',40,'Orbital-basis stratum'),('S_TN_INDEPENDENT_FINAL',20,'TN independent final readouts')):
        group=[x for x in planned if x['lane']==lane]
        if len(group)!=expected or any(x['status']not in terminal for x in group):scopes.append(label)
    high=[x for x in planned if x['lane']=='S_TN_HIGH_BUDGET_COMPLETION']
    if high and(len(high)!=20 or any(x['status']not in terminal for x in high)):scopes.append('Original-plan high-budget TN readout completion')
    return scopes


def write_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    stream=StringIO();w=csv.DictWriter(stream,keys);w.writeheader();w.writerows(rows)
    atomic_bytes(path,stream.getvalue().encode())


def percentile(values,weights,q=.95):
    order=np.argsort(values,kind="stable");v=np.asarray(values)[order];w=np.asarray(weights)[order]
    return float(v[np.searchsorted(np.cumsum(w),q*sum(w),side="left")])


def resource_rows(result):
    if "member_records"in result:
        members=result["member_records"];weights=[m["shots"] for m in members]
        rows=[]
        for view in ("preparation","Z_sampling_equivalent"):
            for metric in ("cnot_count","cnot_depth","total_unitary_depth"):
                values=[m["resources"][view][metric] for m in members]
                rows.append({"view":view,"metric":metric,"shot_weighted_mean":float(np.average(values,weights=weights)),
                             "maximum":max(values),"shot_weighted_p95":percentile(values,weights),
                             "members":len(members),"model_shot_weighted_applications":int(sum(v*w for v,w in zip(values,weights)))})
        values=[m["coefficient_slots"] for m in members]
        rows.append({"view":"member_coefficients","metric":"coefficient_slots","shot_weighted_mean":float(np.average(values,weights=weights)),
                     "maximum":max(values),"shot_weighted_p95":percentile(values,weights),"members":len(members),
                     "model_shot_weighted_applications":None})
        return rows
    rows=[]
    for view,key in (("preparation","preparation_resources"),("Z_sampling_equivalent","sampling_resources")):
        for metric in ("cnot_count","cnot_depth","total_unitary_depth"):
            values=result[key];value=values.get(metric,values.get("total_depth") if metric=="total_unitary_depth" else None)
            rows.append({"view":view,"metric":metric,"shot_weighted_mean":value,"maximum":value,"shot_weighted_p95":value,
                         "members":1,"model_shot_weighted_applications":value*result["independent_final_model_shots"]})
    return rows


def resource_plot_label(method,budget):
    # Late import: frontiers also uses this module's CSV writer.
    from .frontiers import short_label
    compact_budget={10000:'10k',100000:'100k',1000000:'1M'}.get(int(budget),str(budget))
    return short_label(method)+'/'+compact_budget


QUALITY_PANELS=(('energy_minus_HF_mHa',8),('energy_minus_HF_mHa',16),
    ('energy_minus_HF_mHa',32),('energy_minus_HF_mHa',64),
    ('correlation_recovery_pct',32),('correlation_recovery_pct',64))


def products():
    out=ROOT/"analysis-v1";out.mkdir(exist_ok=True)
    db=sqlite3.connect(ROOT/"run-queue.sqlite")
    planned=[]
    for case,method,B,seed,status,reason,path in db.execute("SELECT case_id,method_id,budget,seed,status,reason,result_path FROM planned_cells"):
        planned.append({"lane":"C_PRIMARY","case_id":case,"method_id":method,"budget":B,"seed":seed,"status":status,"reason":reason,"result_path":path})
    if db.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='sensitivity_cells'").fetchone()[0]:
        for _,case,method,seed,status,reason,path in db.execute("SELECT * FROM sensitivity_cells"):
            planned.append({"lane":"S5_SENSITIVITY","case_id":case,"method_id":method,"budget":100000,"seed":seed,"status":status,"reason":reason,"result_path":path})
    if db.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='basis_cells'").fetchone()[0]:
        for _,case,method,B,seed,status,reason,path in db.execute("SELECT * FROM basis_cells"):
            planned.append({"lane":"S_ORBITAL_BASIS","case_id":case,"method_id":method,"budget":B,"seed":seed,"status":status,"reason":reason,"result_path":path})
    db.close()
    # Independent final readouts share ONE optimizer per case. They are not
    # five independent optimizations, and never enter the primary lane.
    for case in ("NAPH20-P1","N2-32-R1p2-P2","N2-32-R2p4-P2","FE2S2-40-P2"):
        for seed in (21001,21011,21019,21023,21031):
            p=ROOT/"common-runs/S_TN_INDEPENDENT_FINAL"/case/"E_LUCJ_TN_COMMON_L1"/("100000-"+str(seed))/"result.json"
            attention=ROOT/"TN-controls-v1"/case/"terminal-attention.json"
            successor=ROOT/"TN-controls-v2"/case/"job.json"
            exhausted=ROOT/"TN-controls-v2"/case/"no-final-state.json"
            final_failed=ROOT/"TN-controls-v1"/case/"final-readout/terminal-attention.json"
            tn_status="COMPLETED"if p.exists()else("ENGINEERING_OR_RESOURCE_LIMIT_NO_FINAL_STATE"if exhausted.exists()or final_failed.exists()else("ENGINEERING_SUCCESSOR_IN_PROGRESS"if successor.exists()else("ENGINEERING_OR_RESOURCE_LIMIT_NO_FINAL_STATE"if attention.exists()else"NOT_YET_OBSERVED")))
            planned.append({"lane":"S_TN_INDEPENDENT_FINAL","case_id":case,"method_id":"E_LUCJ_TN_COMMON_L1",
                "budget":100000,"seed":seed,"status":tn_status,
                "reason":None if p.exists()else(str(attention.relative_to(ROOT))if attention.exists()else"Await bounded shared optimizer and exact independent final sampling"),
                "result_path":str(p.relative_to(ROOT))if p.exists()else None})
    readouts=[];resources=[];run_costs=[]
    high_spec=ROOT/'TN-high-final-v1/spec.json'
    if high_spec.exists():
        high=json.loads(high_spec.read_text())
        for case in high['cases']:
            for seed in high['seeds']:
                p=ROOT/'common-runs'/high['lane']/case/high['method_id']/(str(high['model_shots_per_readout'])+'-'+str(seed))/'result.json'
                jobroot=ROOT/'TN-high-final-v1'/case
                status='COMPLETED'if p.exists()else('FAILED_ENGINEERING_PRESERVED'if(jobroot/'terminal-attention.json').exists()else('RUNNING'if(jobroot/'job.json').exists()else'NOT_STARTED'))
                planned.append({'lane':high['lane'],'case_id':case,'method_id':high['method_id'],'budget':high['model_shots_per_readout'],
                    'seed':seed,'status':status,'reason':None if p.exists()else'Post-start completion of original high-budget TN scope; no new optimizer',
                    'result_path':str(p.relative_to(ROOT))if p.exists()else None})
    for row in planned:
        if row["status"]!="COMPLETED" or not row["result_path"]:continue
        filename=ROOT/row["result_path"];result=json.loads(filename.read_text())
        if result["status"]!="COMPLETED" or result["budget_ledger"]["known_actual_shots"]!=row["budget"]:
            raise RuntimeError("Result/ledger/planned mismatch")
        m=json.loads((ROOT/"data/physical"/row["case_id"]/"problem.json").read_text());Eref,quality=reference(row["case_id"])
        counts_file=filename.parent/"final-counts.npz"
        identity={k:row[k] for k in ("lane","case_id","method_id","budget","seed")}
        resource=resource_rows(result)
        resources += [{**identity,**x,"result_sha256":sha256(filename.read_bytes())} for x in resource]
        sampling={x["metric"]:x["shot_weighted_mean"] for x in resource if x["view"]=="Z_sampling_equivalent"}
        slots=result.get("coefficient_slots")
        if slots is None and "member_records"in result:
            slots=next(x["shot_weighted_mean"] for x in resource if x["metric"]=="coefficient_slots")
        extended=ROOT/"CI-extension-v1"/row["case_id"]/row["method_id"]/(str(row["budget"])+"-"+str(row["seed"]))/"CI-r64.json"
        cis=list(result["final_CI"])
        if row["lane"]=="C_PRIMARY"and extended.exists():
            extension=json.loads(extended.read_text())
            if extension["source_result_sha256"]!=sha256(filename.read_bytes()):raise RuntimeError("CI capacity source binding mismatch")
            cis.append(extension["CI"])
        basis_extension=ROOT/"basis-sensitivity-v1/readout-r64"/row["method_id"]/(str(row["seed"])+".json")
        if row["lane"]=="S_ORBITAL_BASIS"and basis_extension.exists():
            extension=json.loads(basis_extension.read_text())
            if extension["source_result_sha256"]!=sha256(filename.read_bytes())or extension["source_counts_sha256"]!=sha256(counts_file.read_bytes()):
                raise RuntimeError("Localized-basis CI extension source mismatch")
            cis.append(extension["CI"])
        control_extension=ROOT/"control-CI-extension-v1"/row["lane"]/row["case_id"]/row["method_id"]/(str(row["budget"])+"-"+str(row["seed"]))/"CI-r64.json"
        if row["lane"]in("S5_SENSITIVITY","S_TN_INDEPENDENT_FINAL","S_TN_HIGH_BUDGET_COMPLETION")and control_extension.exists():
            extension=json.loads(control_extension.read_text())
            if extension["source_result_sha256"]!=sha256(filename.read_bytes())or extension["source_counts_sha256"]!=sha256(counts_file.read_bytes()):
                raise RuntimeError("Control capacity extension source mismatch")
            cis.append(extension["CI"])
        compiler=ROOT/"rustiq-control-v1/common-readout-resource-joins"/row["case_id"]/(str(row["budget"])+"-"+str(row["seed"])+".json")
        alternative={}
        if row["method_id"]=="D_SQDRIFT_N20"and row["lane"]=="C_PRIMARY"and compiler.exists():
            comp=json.loads(compiler.read_text())
            if comp["source_result_sha256"]!=sha256(filename.read_bytes()):raise RuntimeError("Compiler/source mismatch")
            alternative={"Rustiq_"+x["metric"]:x["shot_weighted_mean"]for x in comp["resources"]if x["view"]=="Z_sampling_equivalent"}
            alternative["Rustiq_final_model_CNOT_applications"]=next(x["model_shot_weighted_applications"]for x in comp["resources"]if x["view"]=="Z_sampling_equivalent"and x["metric"]=="cnot_count")
        for ci in cis:
            denominator=None if Eref is None else m["hf_energy_Ha"]-Eref
            readouts.append({**identity,"r_cap":ci["r_cap"],"D_cap":ci["r_cap"]**2,"actual_d":ci["actual_d"],
                "energy_Ha":ci["energy_Ha"],"energy_minus_HF_mHa":1000*(ci["energy_Ha"]-m["hf_energy_Ha"]),
                "energy_error_mHa":None if Eref is None else 1000*(ci["energy_Ha"]-Eref),
                "correlation_recovery_pct":None if denominator is None or denominator<=0 else 100*(m["hf_energy_Ha"]-ci["energy_Ha"])/denominator,
                "reference_quality":quality,"CNOT":sampling["cnot_count"],"CNOT_depth":sampling["cnot_depth"],
                "total_depth":sampling["total_unitary_depth"],"coefficient_slots_or_member_mean":slots,
                "slots_are_not_nonredundant_parameter_rank":True,"nonredundant_parameters":None,
                "VQE_optimized_variables":result["VQE_optimized_variables"],"CI_converged":ci["converged"],
                "CI_residual":ci["residual_norm"],"CI_matvec_calls":ci["matvec_calls"],
                "CI_wall_seconds":ci["classical_wall_seconds"],"spin_square":ci["spin_square"],
                "valid_sector_fraction":ci["valid_sector_fraction"],"configuration_recovery_enabled":False,
                "counts_sha256":sha256(counts_file.read_bytes()),"result_path":row["result_path"],
                "capacity_extension_new_model_shots":0 if ci["r_cap"]==64 else None,**alternative})
        run_costs.append({**identity,"pilot_model_shots":result["pilot_model_shots"],"final_model_shots":result["independent_final_model_shots"],
            "paid_model_shots":result["budget_ledger"]["known_actual_shots"],"actual_QPU_shots":result["actual_QPU_shots"],
            "total_model_CNOT_applications":result["total_model_cnot_applications"],
            "attempt_wall_seconds":result["this_attempt_wall_seconds"],"attempt_CPU_seconds":result["this_attempt_CPU_seconds"],
            "local_time_CI_only":result.get("local_attempt_time_is_CI_only",result.get("detail",{}).get("local_attempt_time_is_CI_only_not_end_to_end",False)),
            "whole_run_simulation_includes_shared_preprocessing":False,"result_path":row["result_path"]})
        run_costs[-1]["Rustiq_equivalent_circuit_final_CNOT_applications"]=alternative.get("Rustiq_final_model_CNOT_applications")
        run_costs[-1]["Rustiq_is_resource_recompilation_same_counts_not_additional_sampling"]=bool(alternative)
    write_csv(out/"all-planned-cells.csv",planned);write_csv(out/"all-observed-readouts.csv",readouts)
    write_csv(out/"all-circuit-resource-views.csv",resources);write_csv(out/"per-run-paid-costs-counted-once.csv",run_costs)
    groups=defaultdict(list)
    for row in readouts:groups[(row["lane"],row["case_id"],row["method_id"],row["budget"],row["r_cap"])].append(row)
    summary=[]
    for key,rows in sorted(groups.items()):
        lane,case,method,B,r=key;item={"lane":lane,"case_id":case,"method_id":method,"budget":B,"r_cap":r,
             "observed_runs":len(rows),"planned_runs":10 if lane in("C_PRIMARY","S_ORBITAL_BASIS") else 5}
        for field in ("energy_Ha","energy_minus_HF_mHa","correlation_recovery_pct","CNOT","CNOT_depth","total_depth","coefficient_slots_or_member_mean","actual_d"):
            values=[x[field] for x in rows if x[field]is not None]
            item["mean_"+field]=float(np.mean(values)) if values else None
            item["median_"+field]=float(np.median(values)) if values else None
            item["std_"+field]=float(np.std(values,ddof=1)) if len(values)>1 else None
        compiled=[r for r in rows if r.get("Rustiq_cnot_count")is not None]
        item["Rustiq_observed_runs"]=len(compiled)
        for field in("cnot_count","cnot_depth","total_unitary_depth"):
            item["mean_Rustiq_"+field]=float(np.mean([r["Rustiq_"+field]for r in compiled]))if len(compiled)==len(rows)else None
        summary.append(item)
    write_csv(out/"all-group-statistics.csv",summary)
    plots=[]
    colors={"A":"#1b6ca8","B":"#b7472a","C":"#26834a","D":"#864b98","E":"#d18b00"}
    from .frontiers import short_label,place_labels,AXIS_LABELS
    for case in sorted(set(row["case_id"] for row in readouts if row["lane"]=="C_PRIMARY")):
        fig,axes=plt.subplots(2,3,figsize=(12,8.5),layout="constrained")
        for ax,(field,r)in zip(axes.flat,QUALITY_PANELS):
            plotted=False
            for method in sorted(set(x["method_id"] for x in summary if x["case_id"]==case and x["lane"]=="C_PRIMARY")):
                values=[x for x in summary if x["case_id"]==case and x["lane"]=="C_PRIMARY" and x["method_id"]==method and x["r_cap"]==r]
                values.sort(key=lambda x:x["budget"])
                values=[x for x in values if x["mean_"+field] is not None]
                if not values:continue
                ax.errorbar([x["budget"] for x in values],[x["mean_"+field] for x in values],
                    yerr=[x["std_"+field] or 0 for x in values],fmt="o",capsize=3,label=short_label(method),color=colors[method[0]])
                plotted=True
                for x in values:
                    if x['observed_runs']<x['planned_runs']:
                        ax.annotate('n='+str(x['observed_runs']),(x['budget'],x['mean_'+field]),
                                    xytext=(4,4),textcoords='offset points',fontsize=7)
            ax.set_xscale("log");ax.set_xlabel("Total model shots (construction included)")
            ax.set_ylabel("Correlation recovery (%)" if field.startswith("correlation") else "CI energy - HF (mHa); lower is better")
            ax.set_title("Common r="+str(r)+", D <= "+str(r*r)+(' (conditional)'if r==64 else''),fontsize=9);ax.grid(alpha=.2)
            if not plotted:
                ax.text(.5,.5,'No eligible saved readout yet\nNot a zero-energy/recovery result',
                        transform=ax.transAxes,ha='center',va='center',fontsize=8)
        axes[0,0].legend(fontsize=7)
        fig.suptitle(case+" | primary observed grid; r64 = registered capacity sensitivity\n"
            "A RAACS; B1 naive LUCJ; C1 compressed LUCJ; D20 SqDRIFT. Whiskers = sample SD, not confidence intervals. Full groups n=10; partial n shown.",fontsize=9)
        for ext in ("png","svg"):fig.savefig(out/(case+"-shots-CI."+ext),dpi=200)
        plt.close(fig);plots.append(case+"-shots-CI.png")
        fig,axes=plt.subplots(2,2,figsize=(11,8),layout="constrained")
        points=[x for x in summary if x["case_id"]==case and x["r_cap"]==32
                and x['observed_runs']==x['planned_runs']]
        label_specs=[]
        for ax,field in zip(axes.flat,("CNOT","CNOT_depth","total_depth","coefficient_slots_or_member_mean")):
            specs=[]
            for x in points:
                ax.scatter(x["mean_"+field],x["mean_energy_minus_HF_mHa"],color=colors[x["method_id"][0]],
                    marker="o" if x["lane"]=="C_PRIMARY" else "^",s=28)
                specs.append((resource_plot_label(x['method_id'],x['budget']),
                              (x['mean_'+field],x['mean_energy_minus_HF_mHa'])))
            ax.set_xlabel(AXIS_LABELS[field]);ax.set_ylabel("CI energy - HF (mHa); lower is better");ax.grid(alpha=.2)
            ax.margins(x=.12,y=.18);label_specs.append((ax,specs))
        fig.suptitle(case+" | r=32 | retrospective observed grid | sampling resources\n"
            "Labels: method/budget. A RAACS; B/C LUCJ layers; D SqDRIFT length; E TN. Complete seed groups only: main n=10, controls n=5.",fontsize=9)
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        for ax,specs in label_specs:place_labels(ax,specs,renderer)
        for ext in ("png","svg"):fig.savefig(out/(case+"-resources-quality."+ext),dpi=200)
        plt.close(fig);plots.append(case+"-resources-quality.png")
    primary_counts=Counter(x["status"] for x in planned if x["lane"]=="C_PRIMARY")
    sensitivity_counts=Counter(x["status"] for x in planned if x["lane"]=="S5_SENSITIVITY")
    tn_counts=Counter(x["status"]for x in planned if x["lane"]=="S_TN_INDEPENDENT_FINAL")
    high_tn_counts=Counter(x['status']for x in planned if x['lane']=='S_TN_HIGH_BUDGET_COMPLETION')
    basis_counts=Counter(x["status"]for x in planned if x["lane"]=="S_ORBITAL_BASIS")
    open_scope=execution_open_scopes(planned)
    triggers=list((ROOT/'CI-extension-v1').glob('*/trigger.json'))
    if len(list((ROOT/'CI-extension-v1').glob('*/*/*/CI-r64.json')))!=120*len(triggers):open_scope.append('Conditional CI4096')
    if len(list((ROOT/'rustiq-control-v1/common-readout-resource-joins').glob('*/*.json')))!=120:open_scope.append('SqDRIFT resource compiler control')
    open_scope.append('Final full-scope cost/quality/resource synthesis and audit')
    evidence={"primary":dict(primary_counts),"sensitivity":dict(sensitivity_counts),"orbital_basis":dict(basis_counts),"TN_independent_final_readouts":dict(tn_counts),"TN_high_budget_completion":dict(high_tn_counts),"observed_readouts":len(readouts),"figures":plots,
              "paid_shots_observed_completed_runs_only":sum(x["paid_model_shots"] for x in run_costs),
              "actual_QPU_shots":0,"whole_experiment_complete":False,
              "open_scope":open_scope,"terminal_paper_attempts_are_not_certified_reproductions":True,
              "parameter_rank_is_not_coordinate_count":True,"P_and_H_not_mixed_with_common_C":True}
    save_json(out/"coverage.json",evidence)
    stage_note=("登録した数値実行・追加CI・compiler対照は終了しました。研究全体の監査と判断は上位のFINAL_REPORT_JA.md・FULL_PLAN_AUDIT_JA.mdを参照してください。原論文の完全再現とは別です。"
        if open_scope==['Final full-scope cost/quality/resource synthesis and audit']else
        "研究全体はまだ実行中です。以下の図は保存済みの点だけを描きます。")
    lines=["# 公平比較：全データと図の読み方","",stage_note, "",
       "| 区分 | 状態 | 件数 |","|---|---|---:|"]
    for lane,c in (("主比較",primary_counts),("層・長さ感度",sensitivity_counts),("局在軌道副比較",basis_counts),("TN中予算の独立最終測定",tn_counts),("TN高予算の計画補完",high_tn_counts)):
        lines += [f"| {lane} | {k} | {v} |" for k,v in sorted(c.items())]
    lines += ["","## まず見る点","","同じ分子・総Shot・rの中でenergyが低い方法が、同じ測定予算とCI上限では良い結果です。小さい回路でもenergyが悪ければ、無条件の勝ちではありません。",
       "LUCJの色BとCの差は初期化の差です。Cは圧縮初期化で、VQEを追加したものではありません。三角は層数などを変えた副試験で、後から一番良い点だけ選ぶと探索費用と選択バイアスが入ります。",
       "資源比較図の短いlabelは、B1/C1などがLUCJ層数、D5/D20/D50が励起数、E1がTN調整1層、後ろの10k/100k/1Mが総測定予算です。主10seed・追加対照5seedがそろった群だけを描き、未完了群は全予定CSVとShot別の進捗図に残します。図のlabel配置・表示範囲の改善は計算値を変えません。",
       "rは採用する各スピンの文字列上限、実際のCI次元dは別です。同じcountsをr違いで読み直すので、表の行数を独立した測定数として足してはいけません。",
       "Shot別quality図は、energyのr8/16/32/64とRecoveryのr32/64を同じ6panelへ示します。r64は登録された容量到達条件による主比較の感度解析で、追加Shotは0、追加CI費用は別計上です。全seed群はn=10、途中の群だけnを点に記し、低いenergyから都合のよいrを選ぶ図にはしません。",
       "係数スロット数は、数学的に非冗長なparameter数の証明ではありません。SqDRIFTの回転角とLUCJ/RAACSの係数を、同じ意味の最適化変数だと扱いません。",
       "Recoveryは相関energy回収率です。配置の修復率ではありません。著者報告参照の精度・sectorを独立確認できていない場合、その制約をCSVのreference_qualityに残しています。",
       "費用CSVのattempt時間は前処理を含むend-to-end時間とは限りません。リモートLUCJの同じ状態を5/10回読む場合、状態構築費用は共有1回です。CIの行ごとにshotsや状態構築費用を重複加算してはいけません。", ""]
    for plot in plots:lines += [f"![{plot}]({plot})",""]
    lines += ["## 表と解説の入口","",
        "- [分子別の平易な比較](CASE_INTERPRETATION_JA.md)",
        "- [論文設定への接続とCI容量](PAPER_BRIDGES_JA.md)",
        "- [P2全anchor・公開実機CRの結果と欠測](SOURCE_REANALYSIS_JA.md)",
        "- [TN調整費用と独立finalの精度](TN_CONTROL_JA.md)",
        "- [計算費用の種類別小計と不明値](COST_SUMMARY_JA.md)",
        "- [全計画の充足状況](REQUIREMENT_COVERAGE_JA.md)",
        "- [保存済み分子結果・counts・予算・物理IDの整合検査](actual-molecular-data-integrity.json)",
        "- [数値収束とr64追加読出しの確認を平易に読む](ACTUAL_DATA_INTEGRITY_JA.md)","",
        "## 未完了と制約の違い","",
        "現在の残作業はREQUIREMENT_COVERAGE_JA.mdを参照してください。時間・容量上限で終了した対照は、未収集の作業と分け、欠測・限界を報告します。原論文を再現できたという意味にはしません。図の存在を、研究全体の完了や手法の一般的優越性と読み替えないでください。"]
    for case in('P3-N2-ratio1p00','P3-N2-ratio2p00','P3-FE2S2-40'):
        p=case+'-public-CR-capacity-cost.png'
        if(out/p).exists():lines +=['',f'![{p}]({p})']
    atomic_bytes(out/"README_JA.md",("\n".join(lines)+"\n").encode())
    print(json.dumps(evidence),flush=True)


if __name__=="__main__":products()
