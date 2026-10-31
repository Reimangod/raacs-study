"""Non-destructive source-cost recovery, not a new molecular computation.

Never sum inclusive source attempts with nested components. A known initializer
subtotal is not a cold total. Shared component IDs refer to exact artifact and
field, not to method/row count.
"""
import csv,math
from io import StringIO
from pathlib import Path
from .paths import ROOT,OLD
from .io import load,save,write,sha256,digest


def cpu(value):
    if value is None:return None
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
        raise ValueError('CPU must be a finite nonnegative measured value or unknown')
    return float(value)


def unique_subtotal(components):
    seen={}
    for c in components:
        if c['CPU_seconds'] is None:continue
        key=c['id'];v=cpu(c['CPU_seconds'])
        if key in seen and seen[key]!=v:raise ValueError('Shared cost identity has inconsistent values')
        seen[key]=v
    return math.fsum(seen.values()) if seen else None


def run():
    manifest=ROOT/'runs/R2-COMMON-CI-v1/freeze.json';f=load(manifest);components={};rows=[];evidence={}
    def read(p):
        p=Path(p);raw=p.read_bytes();value=sha256(raw);evidence[str(p)]=value
        return load(p),value
    def component(path,field,value,phase):
        p=Path(path);_,h=read(p);v=cpu(value);cid=digest([str(p.resolve()),h,field])
        c=dict(id=cid,artifact=str(p),artifact_SHA256=h,field=field,phase=phase,CPU_seconds=v)
        if cid in components and components[cid]!=c:raise RuntimeError('Cost component collision')
        components[cid]=c;return c
    for s in f['sources']:
        row=dict(case=s['case'],method=s['method_id'],source_status=s['status'],
            initializer_component_ids=[],known_disjoint_initializer_CPU_seconds=None,
            recorded_source_construction_CPU_seconds=cpu(s.get('source_construction_CPU_seconds')),
            compilation_or_preparation_CPU_seconds=None,source_attempt_CPU_seconds=None,
            source_attempt_scope=None,remote_shared_simulation_CPU_seconds=None,
            all_initializer_stages_converged=None,initializer_failed_stages=None,
            initializer_optimizer_messages=[],TN_optimizer_returned=None,
            TN_convergence_claimed=None,full_cold_source_CPU_seconds=None,
            new_quantum_evaluations=0,new_numeric_CI_evaluations=0)
        if s['status']!='AVAILABLE':
            row['missing_reason']='NO_SAME_PROBLEM_SOURCE';rows.append(row);continue
        p=Path(s['source_result']);r,h=read(p)
        if h!=s['source_result_SHA256'] or sha256(Path(s['payload']).read_bytes())!=s['payload_SHA256']:
            raise RuntimeError('Frozen source or payload drift')
        construction=r.get('detail',{}).get('construction') or r.get('construction') or {}
        if r.get('shared_optimization_result'):
            tp=OLD/r['shared_optimization_result'];tr,_=read(tp)
            construction=tr.get('configuration',{}).get('initialization',{})
            row['TN_optimizer_returned']=tr.get('optimizer_returned')
            row['TN_convergence_claimed']=not tr.get('optimizer_convergence_not_claimed',True)
            row['TN_terminal_accounting']=tr.get('original_terminal_accounting')
            row['TN_original_terminal_accounting']=tr.get('configuration',{}).get('original_terminal_accounting')
            row['TN_full_optimization_CPU_seconds']=None
        init=construction.get('initialization_cost',{});selected=[]
        if init.get('source_result'):
            ip=OLD/init['source_result'];ir,_=read(ip)
            # These two initializers start their timers AFTER loading fresh
            # amplitudes. Their CPU field excludes amplitude generation.
            if 'CPU_process_seconds' in ir:
                if 'original_amplitude_creation_CPU_seconds' in ir:
                    selected.append(component(ip,'CPU_process_seconds',ir['CPU_process_seconds'],'compressed_initializer_only'))
                    ap=OLD/'data/initialization-fresh'/s['case']/'result.json';ar,_=read(ap)
                    expected=cpu(ir['original_amplitude_creation_CPU_seconds'])
                    if expected is not None and abs(expected-cpu(ar['CPU_process_seconds']))>1e-9:
                        raise RuntimeError('Initializer amplitude timing mismatch')
                    if expected is not None:selected.append(component(ap,'CPU_process_seconds',ar['CPU_process_seconds'],'fresh_amplitudes'))
                else:selected.append(component(ip,'CPU_process_seconds',ir['CPU_process_seconds'],'fresh_amplitudes'))
            row['all_initializer_stages_converged']=ir.get('all_stages_optimizer_success')
            stages=ir.get('stages')
            if stages is not None:
                row['initializer_failed_stages']=sum(st.get('success') is False for st in stages)
                row['initializer_optimizer_messages']=sorted({str(st.get('message')) for st in stages if st.get('success') is False})
                row['initializer_last_loss']=ir.get('last_loss')
            # Factorization CPU is inline, separate from fresh amplitude file.
            # Compilation/attempt may include it: keep separate, never add both.
            if init.get('naive_factorization_CPU_seconds') is not None:
                row['recorded_naive_factorization_CPU_seconds']=cpu(init['naive_factorization_CPU_seconds'])
        row['initializer_component_ids']=[c['id'] for c in selected]
        row['known_disjoint_initializer_CPU_seconds']=unique_subtotal(selected)
        row['compilation_or_preparation_CPU_seconds']=cpu(construction.get('compilation_CPU_seconds',construction.get('preparation_CPU_seconds')))
        row['source_attempt_CPU_seconds']=cpu(r.get('this_attempt_CPU_seconds'))
        if row['source_attempt_CPU_seconds'] is not None:
            row['source_attempt_scope']='OLD_CI_ONLY' if r.get('local_attempt_time_is_CI_only') else 'INCLUSIVE_OLD_ATTEMPT_SIMULATION_CI_POSSIBLY_PREPARATION_NOT_PURE_CONSTRUCTION'
        if r.get('remote_shared_simulation_result'):
            rp=OLD/r['remote_shared_simulation_result'];rr,_=read(rp)
            row['remote_shared_simulation_CPU_seconds']=cpu(rr.get('CPU_seconds'))
            row['remote_shared_simulation_scope']='SHARED_STATE_BUILD_AND_MULTIPLE_READOUT_SEEDS_NOT_PER_SELECTED_SOURCE'
        if r.get('remote_result'):
            rp=OLD/r['remote_result']
            if rp.is_file():
                rr,_=read(rp);row['remote_result_CPU_seconds']=cpu(rr.get('CPU_seconds'))
                row['remote_result_scope']='SHARED_REMOTE_RESULT_KEEP_SEPARATE_FROM_LOCAL_CI'
            else:row['remote_result_missing_locally']=str(rp)
        row['source_result']=str(p);row['source_result_SHA256']=h
        rows.append(row)
    for name in ['common_execution.py','author_compressed.py','sensitivity_initialization.py']:
        p=OLD/'paperstudy'/name;evidence[str(p)]=sha256(p.read_bytes())
    result=dict(status='ALL56_SOURCE_INVENTORY_COST_COMPONENTS_RECOVERED_WITH_UNKNOWNS',rows=rows,
        components=list(components.values()),evidence_SHA256=evidence,source_manifest_SHA256=sha256(manifest.read_bytes()),
        initializer_subtotal_rows=sum(r['known_disjoint_initializer_CPU_seconds'] is not None for r in rows),
        unique_initializer_CPU_seconds=unique_subtotal(list(components.values())),
        summed_source_rows_for_campaign_cost=False,full_cold_cost_reconstructed=False,
        inclusive_attempts_never_added_to_nested_components=True,
        new_model_shots=0,new_QPU_shots=0,new_CI_solves=0,goal_complete=False)
    save(ROOT/'analysis/SOURCE-cost-components-v1.json',result)
    fields=list(dict.fromkeys(k for r in rows for k in r));stream=StringIO();w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(rows)
    write(ROOT/'analysis/SOURCE-cost-components-v1.csv',stream.getvalue().encode('utf-8-sig'))
    lines=['# source構築費用の非破壊回収','',
        f"元56予定sourceを確認し、{result['initializer_subtotal_rows']}行で初期化費用の小計を復元した。同じ初期化artifactを重複せず数えた小計は{result['unique_initializer_CPU_seconds']:.3f} CPU秒。",'',
        '**これは完全な総費用ではない。** 初期化・compile・sampling simulation・古いCIが異なる範囲で測られているため、全部を単純に足すと二重計上になる。元sourceのinclusive attemptやCI-only費用は別列に保持し、cold totalはunknownのまま。今回の費用回収で量子評価やCI再計算は行っていない。','',
        '| 条件 | 手法 | 初期化小計 CPU秒 | 全stage収束 | 上限等で不成功のstage数 |',
        '|---|---|---:|---|---:|']
    for r in rows:
        if r['known_disjoint_initializer_CPU_seconds'] is not None:
            lines.append(f"| {r['case']} | {r['method']} | {r['known_disjoint_initializer_CPU_seconds']:.3f} | {r['all_initializer_stages_converged']} | {r['initializer_failed_stages']} |")
    lines+=['','LUCJ initializerの回数上限は、保存回路・energyが無効という意味ではない。収束した最強baselineという主張もできない。有限回数のこの設定の結果として扱う。TNについてもoptimizerが正常returnしていない場合、その事実と旧timeout費用の未回収を残す。',
        'shared componentはartifact SHAとfieldで一意化。分子ごとの別ジョブ費用や同じ初期化の再利用を、表の行数で重複加算しない。実機測定費用・GPU時間・CPU秒は別単位である。']
    write(ROOT/'analysis/SOURCE_COST_RECOVERY_JA.md',('\n'.join(lines)+'\n').encode())
    print(result['status'],result['initializer_subtotal_rows'],len(components),flush=True)


if __name__=='__main__':run()
