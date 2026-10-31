"""Pin author TN source and document precise paper/code/runtime differences.

Fetches licensed source only; never executes the downloaded author task or
changes any active optimizer. Existing molecular artifacts remain immutable.
"""
import importlib.util,json,urllib.request
from .io import ROOT,atomic_bytes,save_json,sha256

COMMIT='b52bca05650e60d890b06ac0a3b6d167303d0a90'
FILES=('scripts/quimb/fe2s2_30e20o/lucj_compressed_t2_nomad.py',
    'src/lucj/quimb_task/lucj_sqd_quimb_task_nomad.py',
    'src/lucj/quimb_task/lucj_sqd_quimb_task_sci_nomad.py','LICENSE')


def run():
    root=ROOT/'vendor/TN-source-v1';records=[];texts={}
    for name in FILES:
        p=root/name;url='https://raw.githubusercontent.com/kevinsung/lucj/'+COMMIT+'/'+name
        if p.exists():blob=p.read_bytes()
        else:
            blob=urllib.request.urlopen(url,timeout=30).read();atomic_bytes(p,blob)
        texts[name]=blob.decode();records.append({'path':name,'url':url,'SHA256':sha256(blob),'modified':False})
    runtime={}
    for name in('qiskit_addon_dice_solver','ffsim.variational.util','ffsim.linalg.util','molecules_catalog.util'):
        try:
            spec=importlib.util.find_spec(name)
            runtime[name]={'available':spec is not None,'origin':None if spec is None else spec.origin}
        except(ModuleNotFoundError,ImportError)as exc:runtime[name]={'available':False,'error':repr(exc)}
    nominal=texts[FILES[1]];sci=texts[FILES[2]]
    def occurrences(text,needles):
        return [{'line':i,'text':line.strip()}for i,line in enumerate(text.splitlines(),1)if any(s in line for s in needles)]
    v={'author_repository_commit':COMMIT,'same_as_paper_executed_commit_verified':False,'sources':records,
        'paper':'https://arxiv.org/html/2511.22476v1','paper_section':'4.1 Tensor network simulation',
        'paper_stated':{'MPS_bond':50,'cutoff':1e-10,'samples_per_objective':10000,'objective_evaluation_limit':500,
            'variables_per_subproblem':20,'threads':4,'evaluations_per_subproblem':20,
            'number_of_subproblems_stated_in_text':None},
        'pinned_driver':{'n_reps':1,'heavy_hex':True,'shots':10000,'samples_per_batch':4000,'batches':10,
            'max_iterations':1,'source_file':FILES[0]},
        'pinned_Dice_task_options':occurrences(nominal,['MAX_BB_EVAL','PSD_MADS_NB_VAR_IN_SUBPROBLEM','PSD_MADS_NB_SUBPROBLEM','PSD_MADS_SUBPROBLEM_MAX_BB_EVAL','qiskit_addon_dice_solver']),
        'pinned_PySCF_exploration_task':{'source_file':FILES[2],
            'contains_pre_optimizer_return':sci.index('        return\n        lb = []')<sci.index('result = PyNomad.optimize'),
            'evidence':occurrences(sci,['MAX_BB_EVAL','PSD_MADS_NB_SUBPROBLEM','init_mps_sqd_result_filename'])},
        'current_local_runtime_dependencies':runtime,
        'our_S_control':{'identity':'P2_INSPIRED_BOUNDED_COMMON_r32_TN_VARIANT',
            'objective':'common spin-symmetrized Cartesian frequency CI, r32, not author10x4000/Dice',
            'number_of_subproblems':3,'MAX_BB_EVAL':500,'new_seed':31001,
            'starting_amplitudes':'measured fresh CCSD or registered CISD fallback, not necessarily author saved amplitudes',
            'final_exact_shots_per_readout':100000,'final_independent_readouts':5,'optimizer_replicates':1},
        'legacy_configuration_key_semantics':{'paper_subproblem_count20_reproduced':
            'Legacy name refers to pinned code subproblem count20; paper text itself gives20 VARIABLES per subproblem, not a count of20 subproblems. Existing artifacts are not rewritten.'},
        'original_paper_TN_reproduced':False,'new_molecular_outcome_evaluations':0,
        'unmodified_author_task_not_executable_in_this_pinned_runtime':not all(v['available']for v in runtime.values()),
        'absence_in_this_runtime_is_not_global_unavailability_proof':True,
        'claim_boundary':'Do not claim superiority over author Fig7 optimized LUCJ; source/code/runtime differences are explicit, not tuned away',
        'no_active_worker_or_result_modified':True}
    save_json(root/'source-crosswalk.json',v)
    lines=['# TN対照と原論文・公開コードの対応','',
        '実行済みSレーンを原論文Fig.7の完全再現と誤認しないための照合です。新しい分子計算や設定変更はしていません。','',
        '| 項目 | 原論文本文 | 取得commitの著者コード | 今回のS対照 |','|---|---|---|---|',
        '| MPS | χ50、cutoff1e-10 | driverも同じ | 同じ |',
        '| surrogate samples | 10,000 | 10,000 | 同じ |',
        '| objective評価上限 | 500 | Dice NOMAD taskでは100 | 500＋登録wall上限 |',
        '| subproblem | 20変数、4threads、各20評価 | Dice taskでは20 subproblems。本文にはsubproblem個数の明記なし | 3 subproblems、4threads |',
        '| QSCI | 配置からのQSCI | 10batch×4000、Dice solver | 共通r32のfrequency Cartesian CI |',
        '| 最終評価 | exact stateから100k | 同様の別final sampling経路あり | exact100k×5独立readout、共通CI |','',
        '同repositoryのPySCF版探索taskには、初期MPS評価を保存してoptimizer前にreturnする経路もあります。任意のファイルを実行しただけで論文の最適化を再現したとは言えません。取得commitと論文実行commitの同一性も未認証です。',
        '現在のlocal runtimeでは一部の著者task用依存が見つかりません。これは原著環境が世界中で入手不能という意味ではありません。既存runtimeを変更せず、現在の実行は明示的なcommon-readout variantとして保存します。',
        '原論文の強いTN対照を再現したという主張や、それを上回ったという主張はしません。独立finalのenergy、実際の費用、未完了・失敗を通常どおり報告します。','',
        '[原論文](https://arxiv.org/html/2511.22476v1) / [著者Dice task](https://github.com/kevinsung/lucj/blob/'+COMMIT+'/'+FILES[1]+')。取得原文・license・全SHAはvendor/TN-source-v1に保存。']
    atomic_bytes(ROOT/'TN_SOURCE_CROSSWALK_JA.md',('\n'.join(lines)+'\n').encode())
    print(json.dumps({'author_files_pinned':len(records),'runtime_dependencies':runtime,'paper_TN_reproduced':False}),flush=True)


if __name__=='__main__':run()
