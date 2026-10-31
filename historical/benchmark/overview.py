"""Japanese reader-facing synthesis from finished evidence, not a win template."""
import argparse
import json
from pathlib import Path
import pandas as pd
from .common import save_binary


def write(root):
    root=Path(root);audit=json.loads((root/"requirements-audit.json").read_text())
    analysis=json.loads((root/"analysis/analysis-status.json").read_text())
    matched=pd.read_csv(root/"sqd-matched-dimension/summary.csv")
    operational=pd.read_csv(root/"sqd-selected-v1/pair-summary.csv")
    optima=pd.read_csv(root/"analysis/checkpoint_optima.csv")
    hardware=json.loads((root/"gpu-production-v1/observed-utilization.json").read_text())
    lines=["# CEO–QSCI/SQDベンチマーク：結果の読み方","",
           "実行設定を記録した30条件の実行を一通り終了した。CEOが常に最良だった、実機で優位だった、論文化が保証された、という結論ではない。以下はSTO-3G、固定活性空間、共通readout条件での古典simulationの結果である。外部事前登録ではなく、後追加の同次元化・recovery再比較はretrospective secondary解析として区別する。", "",
           "## 何を実行したか","",
           "| 項目 | 結果 |","|---|---:|",
           f"| 物理条件（6分子×5距離） | {audit['reference_cases']}/30 |",
           "| 手法セル | 330予定、324で回路保存、6でCCSD初期化失敗 |",
           f"| 保存checkpoint | {audit['checkpoints']:,} |",
           "| ADAPT軌道 | 150：勾配基準達成95、64-step上限55 |",
           f"| 独立multinomial draw | {audit['independent_raw_draws']:,} |",
           f"| 上記drawの合計仮想shots | {audit['simulated_raw_shots']:,} |",
           f"| 最終checkpoint公式recovery readout | {audit['sqd_readouts']:,} |",
           f"| 合成readout-error試験 | {audit['synthetic_noise_readouts']:,} |",
           "| 実機量子shots | 0 |", "",
           "H₂/H₄/H₆/LiH/BeH₂/N₂の距離・活性空間・計数規則は[科学プロトコル](../../STUDY_PROTOCOL.md)に記載。H₆2.2ÅとN₂2.2ÅのCCSD初期化3 variantsずつが欠測。ランダムLUCJはCCSD依存を外して全30条件で評価した。既存24例の係数が変わらないことを検証し、過去の失敗記録は保持した。", "",
           "## 研究の中心8問", "",
           "### 1. 重要な非HF配置を安く発見できたか", "",
           "全checkpointでFCI重み付きcoverage・top-K recall・sector率を計算した。固定CCSD初期化LUCJより良い条件はあるが、QEB/GSD/qubit-ADAPTに対する普遍的優位性ではない。FCI重みは評価用のoracle情報であり、選択・学習・停止には使っていない。詳細はanalysis/coverage_at_qsci_selected_points.csvと各条件のcoverage図。", "",
           "### 2. QSCI向けcheckpointはVQE最終点より早かったか", "",
           "M=1000で、有限shot平均QSCI誤差が最良となる点と最終保存点を比較する。同率はCNOT、次にstepで選んでいる。", "",
           "| Pool | より早い最良QSCI点 | 比較可能条件 |","|---|---:|---:|"]
    for method in ("ceo_ovp","ceo_mvp","qeb","qubit","gsd"):
        early=total=0
        for _,g in optima[(optima.shots==1000)&(optima.method==method)].groupby("case_id"):
            c=g[g.criterion=="C_min_qsci_error"];e=g[g.criterion=="E_final_saved_vqe_checkpoint"]
            if len(c) and len(e):total+=1;early+=int(c.iloc[0].step<e.iloc[0].step)
        lines.append(f"| {method} | {early} | {total} |")
    lines += ["", "これは同じ評価データで選んだretrospective最適点である。plateauやshot揺らぎも含むため、追加費用なしのearly-stopping法を実証したことにはならない。", "",
              "### 3・7. 他手法との比較は、recovery後も同じか", "",
              "以下はM=1000。元の比較点でCEOのCNOTとCNOT Depthが相手以下という制約を置く。負の差はCEOの平均エネルギー誤差が小さい。表は条件間の記述的中央値であり、統計的優越性・未知分子一般化ではない。", "",
              "まず、同じM・R上限32という運用上の予算比較。実際に多くの配置を発見する利点も含むが、実CI次元は異なり得る。", "",
              "| 相手 | CEO | 条件数 | 同R上限：前 mHa | 同R上限：後 mHa |","|---|---|---:|---:|---:|"]
    for (baseline,ceo),g in operational[operational.shots==1000].groupby(["baseline","ceo"]):
        lines.append(f"| {baseline} | {ceo} | {len(g)} | {g.delta_error_mHa.median():.6g} | {g.delta_after_recovery_mHa.median():.6g} |")
    lines += ["","次に、各試行で **CEO/相手×recovery前/後の4者の最終CI次元を同じにした** controlled diagnostic。これは多くの配置を発見する利点も取り除くため、上の運用比較に代わる唯一の『公平な結果』ではない。両方を併記する。", "",
              "| 相手 | CEO | 条件数 | 同実次元：前 mHa | 同実次元：後 mHa |","|---|---|---:|---:|---:|"]
    for (baseline,ceo),g in matched[matched.shots==1000].groupby(["baseline","ceo"]):
        lines.append(f"| {baseline} | {ceo} | {len(g)} | {g.matched_delta_before_mHa.median():.6g} | {g.matched_delta_after_mHa.median():.6g} |")
    lines += ["", "固定CCSD LUCJ1/2層に対する差は残る一方、他ADAPT poolに対する普遍的優位性はない。同次元化では小さなd、場合によってHFだけのd=1になり、差が0になり得る。小さな中央値を統計的な同等性の証明とはしない。実次元はmean_common_dで確認できる。recoveryの中間計算は登録R上限32のままであり、最終次元一致を全古典計算量の一致とは呼ばない。", "",
              "最終checkpointでは公式recoveryを実行済み。さらに元のretrospective比較876組の早い点も同じ設定で評価し、既存counts・既存recoveryを適合する場合だけ再利用した。追加量子shotsは0。公式configuration recoveryとexplicit-determinant qubit-H solverの接続であり、full fermionic SQD全ワークフローの完全再現ではない。", "",
              "### 4. 小さいCNOT/Depthで化学精度へ到達したか", "",
              "M=1000、R上限32において、いずれかの保存点で平均誤差が1kcal/mol以下だった条件数。成功率の信頼保証ではない。", "",
              "| 手法 | 到達条件 | データのある条件 | 予定条件 |","|---|---:|---:|---:|"]
    eligible=optima[(optima.shots==1000)&(optima.criterion=="A_min_cnot_mean_chemical")]
    for method,g in eligible.groupby("method"):
        lines.append(f"| {method} | {sum(g.attained)} | {len(g)} | 30 |")
    lines += ["", "最小CNOT/最小Depthの具体値はanalysis/checkpoint_optima.csv。未到達は0コストや0誤差に置換していない。H₆/N₂等ではR=32というCI容量が制約になり得るため、未達をansatzの一般的な能力限界とはしない。", "",
              "### 5. 距離・相関の強さに依存したか", "",
              "全30点のHF重み・相関エネルギー、距離別heatmap、連続scatterを保存した。ただし同じ分子の5距離を5つの独立分子とは数えず、相関の強さだけで性能差が生じたという因果主張もしない。参照データとfigures/cross_case/correlation_dependenceを参照。", "",
              "### 6. 構築費用まで含めて得だったか", "",
              f"保存されたoptimizer objective callsは合計{analysis['construction_objective_calls']:,}。CPU/GPUのdevice、optimizer時間、CCSD初期化、readout shots、CI・recovery時間を分離している。古典シミュレーションの評価回数を量子Measurement Costへ変換するモデルは未確立であり、end-to-endの量子測定費用優位は主張できない。CCSDは共有cacheなので、各variantの初期化proxy秒数を合算して実作業量としてはいけない。", "",
              "### 8. 次に何を変えるべきか", "",
              "新しいheuristicを増やす前に、保存済みcountsからCI上限・配置選択の感度を切り分けるのが妥当である。VQE energy、entropy、unique配置数だけをsampling性能の代理にしない。今回のretrospective点から新policyを作る場合は、別の評価seedと探索費用の計上が必要になる。この次実験の優位性はまだ実証していない。", "",
              "## 計算機・検証と読むときの注意", "",
              f"- 主計算のN₂5条件は実際にA100 kernelを使用。設定上限2台、観測された主計算同時稼働は{hardware['observed_max_concurrent_primary_A100']}台。別プロジェクトの既存jobが利用者の2GPU枠を1つ使用していたためで、停止していない。独立N₂2条件の64-step認定試験ではA100×2の同時実行を確認済み。",
              "- M4は司令・化学計算・独立回路検証・CI・集計を担当。GPU側もSciPy BFGSの制御はCPUで、状態/energy/gradient kernelがA100。M4のGPUや量子実機を使ったという意味ではない。",
              "- GPU経路は5 poolの固定入力一致、64-step OVP演算子列一致、energy差約4.4e-13Ha、起動/転送/独立検証込みの保守的1.39倍を測定。共有pool cacheを持つこの実行経路に限る観測値で、全pool・全サイズの普遍的speedupではない。",
              "- 95軌道の収束勾配を保存状態から再検査し全件通過。55軌道はstep上限であり収束とは呼ばない。",
              "- CNOT等はZ測定専用の等確率回路の値。VQE energyは元の準備状態の値。両者を同じ量子状態の圧縮点と解釈しない。summary.csvに両方の資源を出している。",
              "- native資源は合成ECR線形target。noiseは共通1%独立readout bit-flipのみ。実機較正・Aer gate-noise・hardware耐性・量子優位の証拠ではない。",
              "- fixed-controlの旧revisionは保全したが、主frontier/最適点選択には最新だけを使う。Paretoのenergy差1e-5mHa以内は数値丸めの範囲として扱い、元energyは丸めず保存する。これは統計的同等性marginではない。",
              "- 主実験はcheckpoint独立readout。別設定の累積readoutはH₂/H₄で実行確認済みで、累積予算M×消費checkpoint数を別記し、30条件の主結果へ混ぜていない。", "",
              "## 主要ファイル", "",
              "- [詳細8問レポート](report.md)","- [全330セル](all-cell-status.csv)","- [全checkpoint・準備/測定資源](summary.csv)",
              "- [有限shot統計](qsci-summary.csv)","- [4者同次元のrecovery比較](sqd-matched-dimension/summary.csv)",
              "- [要件監査](REQUIREMENTS_AUDIT.md)","- [最終実行manifest](execution-manifest-final.json)",
              "- 図：figures/<case>/、横断図：figures/cross_case/。PNG/SVG/PDF。CNOT/Depth/iteration曲線はM=1000/10000/100000を別出力。", "",
              "この研究は『条件を限定した再現可能なサンプラーベンチマーク』として完了した。論文の新規性、最適化済みLUCJへの一般的優越性、end-to-end Measurement Cost、実機価値は別の検証課題である。"]
    save_binary(root/"REPORT_JA.md",("\n".join(lines)+"\n").encode());print(root/"REPORT_JA.md")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);write(p.parse_args().output)
