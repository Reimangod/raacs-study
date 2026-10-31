"""Finite new-direction preflight after terminal N55, no legacy mutation."""
from pathlib import Path
import os,subprocess,sys
from .paths import ROOT,OLD,PROJECT
from .io import load,save,sha256,space,write
LOCAL=ROOT/'runs/N56-DIAGONAL-FILTER-SQDRIFT-PREFLIGHT-v1'


def run():
    if LOCAL.exists():raise RuntimeError('Existing N56; reconcile not recreate')
    space();a=load(ROOT/'analysis/N55-independent-root-audit.json')
    assert len(a['coverage'])==2 and all(x['status']=='LIMIT_OR_ENGINEERING_FAILURE_PRESERVED' for x in a['coverage'])
    cases=[]
    for case in ['NAPH20-BLOCK-BOYS','N2-32-R1p2-P2','N2-32-R2p4-P2','FE2S2-40-P2']:
        source=OLD/'native-SqDRIFT'/case/'base.json'
        cases.append(dict(case=case,source=str(source),SHA256=sha256(source.read_bytes())))
    worker=ROOT/'frontierloop/diagonal_SqDRIFT_preflight.py'
    save(LOCAL/'protocol.json',dict(version='N56_OUTCOME_FREE_NATIVE_SDK_DIAGONAL_FILTER_INVENTORY',cases=cases,
        CPU_cap_seconds=30,wall_cap_seconds=60,worker_SHA256=sha256(worker.read_bytes()),
        source_circuits_counts_and_CI_unchanged=True,no_sampling_or_source_outcomes=True,
        no_old_N55_cap_extension_or_restart=True,original_physical_H_must_remain_CI_target=True,
        distribution_preservation_claim_prohibited=True,goal_complete=False))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        PYTHONPATH=os.pathsep.join([str(PROJECT/'.runtime-overlays/sqdrift-qiskit25-v1'),str(OLD)]))
    command=['rtk','proxy',sys.executable,'-u',str(worker),str(LOCAL)]
    save(LOCAL/'process-intent.json',dict(command=command))
    with (LOCAL/'stdout.txt').open('wb') as out,(LOCAL/'stderr.txt').open('wb') as err:
        child=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=out,stderr=err)
        save(LOCAL/'owner.json',dict(pid=child.pid,command=command))
        code=child.wait(timeout=75)
    if code!=0:raise RuntimeError('N56 failed; preserve evidence and inspect logs')
    result=load(LOCAL/'terminal.json');assert len(result['rows'])==4
    save(ROOT/'analysis/N56-diagonal-filter-inventory.json',result)
    lines=['# N56：公式SqDRIFTの強い設定への不足を実測','',
        '元SDKの全4問題を同じnormal-order/groupingで確認し、公式filterが対角項だけを除去することを検査した。',
        '新回路・Shot・CI・energyは0。これは対照設定の点検であり、圧縮や精度の改善結果ではない。','',
        '| 対象 | qubit | 対角だけのgroupを引く確率 | 全group数→filter後 | 除去した対角term数 |',
        '|---|---:|---:|---:|---:|']
    for r in result['rows']:lines.append(f"| {r['case']} | {r['n_qubits']} | {100*r['pure_diagonal_draw_probability']:.2f}% | {r['original_groups']}→{r['filtered_groups']} | {r['removed_diagonal_terms']} |")
    lines+=['','対角項は単独では配置を移さないが、途中の相対位相はその後の遷移へ影響する。',
        '従ってfilterは分布保存compilerではない。小系の反例も確認し、既存energyの再利用を禁止する。',
        '同じステップ数ならoff-diagonal操作が増えるため、CNOTが自動的に下がる保証もない。',
        '次は有限のpaired-source試験として、同物理Hで新sampling/CI評価が必要。元結果と混ぜない。',
        '[公式説明](https://quantum.cloud.ibm.com/docs/en/tutorials/sqdrift)。',
        f"今回のCPU {result['CPU_seconds']:.6f}秒、wall {result['wall_seconds']:.6f}秒。全目標は未達。",'']
    write(ROOT/'analysis/N56_DIAGONAL_FILTER_PREFLIGHT_JA.md','\n'.join(lines).encode())
    print(result['status'],[(r['case'],r['pure_diagonal_draw_probability']) for r in result['rows']],flush=True)


if __name__=='__main__':run()
