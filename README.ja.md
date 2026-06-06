# OpenPipe ART x W&B x Weave Retail Hands-on

このリポジトリは、OpenPipe ARTをW&B ModelsとWeaveに連携させて学ぶためのハンズオン教材です。英語版READMEをメインの仕様書として管理しています。

- Main README: [README.md](README.md)
- Course blueprint: [OPENPIPE_ART_WANDB_COURSE_BLUEPRINT.md](OPENPIPE_ART_WANDB_COURSE_BLUEPRINT.md)

## 題材

オープンなretail tool-callingデータセットを使い、カスタマーサポートAgentをSFTとRLで改善します。

- SFTとワークフロー確認: `lefft/tau-dev-task-retail-v1`
- 追加SFT warm start: `amityco/tau-bench-retail-train-next-action-all-step-score-v0.2` と `inclusionAI/AReaL-tau2-data`
- RL rollout / reward / eval設計: tau-bench / tau2-bench retailの考え方を簡略化して利用

## 推奨モデル

H100を使うメインハンズオンでは `ART_MODEL_PROFILE=standard` を推奨します。現在のstandardは `LiquidAI/LFM2.5-8B-A1B` です。

小さいGPUやセットアップ確認では `ART_MODEL_PROFILE=tiny` を使えます。ただし、小規模モデルは学習の流れを確認する用途で、性能改善の説得力はLFM 8Bなどの大きめのモデルで確認する前提です。

## 講師向け再実行runbook

baseline eval、next-action SFT、SFTからのGRPO/GSPO独立分岐、eval、W&B横持ち比較表までを再実行する入口を追加しています。詳細は英語版READMEのQuick Startと `course/09_runbooks/run_retail_agentic_sequence.py` を参照してください。

runbookはtau-style RL向けに `--continue-on-invalid` をデフォルトで有効化しています。最終レポート用には `--eval-rollouts-per-scenario 4 --eval-temperature 0.2` を加えると、pass@kとreward varianceも比較できます。
比較結果は監査用の全列版 `checkpoint_eval_comparison.md/.csv` と、README/スライド向けの簡潔版 `checkpoint_eval_summary.md/.csv` の両方に出力されます。
SFT warm startを強める講師向けフル検証では、英語版READMEの `--include-teacher-sft` と `--include-areal-sft` 例を使うと、公開teacher next-actionデータとAReaL tau2 SFTデータをbridgeデータに混ぜられます。
比較後には各JSONL結果をcached predictionとしてWeave Evaluationにも流し、trace単位とstage単位の両方で確認できるようにしています。

SUNK/Slurm環境では次のラッパーを使えます。

```bash
sbatch course/09_runbooks/sunk_h100_retail_agentic_sequence.sbatch
```

## 診断用の過去結果

下の表は、strictな参照trajectory再現に近い古い診断runです。最終的なワークショップ用の期待結果表ではありません。現在の教材はtau-bench / tau2-benchに近い考え方に寄せ、`outcome_success`、state-changing actionの正しさ、communication successを主指標として再検証しています。

| Stage | Reward | Task success | Tool F1 | Tool order | Invalid calls avg | Final text F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.5102 | 0.1458 | 0.7268 | 0.5542 | 0.6667 | 0.2439 |
| SFT anchor, lr=3e-5 | 0.5012 | 0.2083 | 0.7255 | 0.5750 | 0.7708 | 0.2541 |
| SFT -> GRPO, 8 steps | 0.5165 | 0.1875 | 0.7283 | 0.5740 | 0.6667 | 0.2576 |
| SFT -> GSPO, 3 steps | 0.5043 | 0.2083 | 0.7268 | 0.5792 | 0.7292 | 0.2405 |
| SFT -> RULER-GRPO, 3 steps | 0.5026 | 0.1875 | 0.7254 | 0.5750 | 0.7292 | 0.2405 |

最終版では、固定済みのtau-style報酬でbaseline、SFT、GRPO、GSPO、RULERを再実行し、W&B ArtifactsとWeave traceを含む横持ち比較表に差し替えます。

runbookは横持ち比較に加えて `checkpoint_acceptance.md/.json` も書きます。SFTがbaselineのtask/outcomeを落としていないこと、RLがSFTに対してrewardとagentic metric、state-action errorの両方で改善していることを確認してから、READMEの期待結果として採用します。
