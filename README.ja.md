# OpenPipe ART x W&B x Weave Retail Hands-on

このリポジトリは、OpenPipe ARTをW&B ModelsとWeaveに連携させて学ぶためのハンズオン教材です。英語版READMEをメインの仕様書として管理しています。

- Main README: [README.md](README.md)
- Course blueprint: [OPENPIPE_ART_WANDB_COURSE_BLUEPRINT.md](OPENPIPE_ART_WANDB_COURSE_BLUEPRINT.md)

## 題材

オープンなretail tool-callingデータセットを使い、カスタマーサポートAgentをSFTとRLで改善します。

- SFTとワークフロー確認: `lefft/tau-dev-task-retail-v1`
- RL rollout / reward / eval設計: tau-bench / tau2-bench retailの考え方を簡略化して利用

## 推奨モデル

H100を使うメインハンズオンでは `ART_MODEL_PROFILE=standard` を推奨します。現在のstandardは `LiquidAI/LFM2.5-8B-A1B` です。

小さいGPUやセットアップ確認では `ART_MODEL_PROFILE=tiny` を使えます。ただし、小規模モデルは学習の流れを確認する用途で、性能改善の説得力はLFM 8Bなどの大きめのモデルで確認する前提です。

## 実測済みの期待結果

H100上で `LiquidAI/LFM2.5-8B-A1B` を使い、SFT、GRPO、GSPO、RULER-GRPOまで実行してW&B ArtifactとWeave traceの生成を確認しています。

| Stage | Reward | Task success | Tool F1 | Tool order | Invalid calls avg | Final text F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.5102 | 0.1458 | 0.7268 | 0.5542 | 0.6667 | 0.2439 |
| SFT anchor, lr=3e-5 | 0.5012 | 0.2083 | 0.7255 | 0.5750 | 0.7708 | 0.2541 |
| SFT -> GRPO, 8 steps | 0.5165 | 0.1875 | 0.7283 | 0.5740 | 0.6667 | 0.2576 |
| SFT -> GSPO, 3 steps | 0.5043 | 0.2083 | 0.7268 | 0.5792 | 0.7292 | 0.2405 |
| SFT -> RULER-GRPO, 3 steps | 0.5026 | 0.1875 | 0.7254 | 0.5750 | 0.7292 | 0.2405 |

この表は「全指標が単調改善する」という意味ではありません。SFTはtask successと応答品質を改善し、GRPOはscalar rewardを改善し、GSPOとRULERはRL設計の比較対象として扱います。
