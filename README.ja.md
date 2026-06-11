# OpenPipe ART x W&B x Weave Retail Hands-on

このリポジトリは、OpenPipe ARTをW&B ModelsとWeaveに連携させて学ぶためのハンズオン教材です。英語版READMEをメインの仕様書として管理しています。

- Main README: [README.md](README.md)
- Course blueprint: [OPENPIPE_ART_WANDB_COURSE_BLUEPRINT.md](OPENPIPE_ART_WANDB_COURSE_BLUEPRINT.md)

## 題材

オープンなretail tool-callingデータセットを使い、カスタマーサポートAgentをSFTとRLで改善します。

- SFTとワークフロー確認: `lefft/tau-dev-task-retail-v1`
- 追加SFT warm start: `amityco/tau-bench-retail-train-next-action-all-step-score-v0.2`、`inclusionAI/AReaL-tau2-data`、`KermitCO/qwen3.5-9B-tau2bench-retail-traces`
- 大規模appendix用: `fuvty/tau-bench-synthetic`
- RL rollout / reward / eval設計: tau-bench / tau2-bench retailの考え方を簡略化して利用

`success mix`、`teacher mix`、`bridge-only` は実験条件名です。主な評価指標は `retail_task_success`、`reference_tool_sequence_exact_match`、`reward` です。詳しい定義は英語版READMEの "Task and Metric Terminology" を参照してください。

## 推奨モデル

H100を使うメインハンズオンでは `course/09_runbooks/config.yaml` の `model_profile: standard` を推奨します。現在のstandardは `LiquidAI/LFM2.5-8B-A1B` です。

小さいGPUやセットアップ確認では `model_profile: tiny` を使えます。ただし、小規模モデルは学習の流れを確認する用途で、性能改善の説得力はLFM 8Bなどの大きめのモデルで確認する前提です。

## 実行runbook

baseline eval、next-action SFT、SFTからのGRPO/GSPO独立分岐、eval、W&B横持ち比較表までを再実行する入口を追加しています。通常は `course/09_runbooks/run_retail_agentic_sequence.py` をprofile指定で実行します。

実行条件は `course/09_runbooks/config.yaml` で切り替えます。このファイルだけを参加者が触る前提です。

```yaml
run_profile: workshop_fast_h100
model_profile: standard
gpu_memory_preset: standard
overrides: {}
```

デフォルトはワークショップ用の SFT -> GRPO 実行です。フル検証に切り替える場合は `run_profile: validated_h100` に変えます。小さいGPUで確認する場合は `model_profile: tiny`、vLLMのメモリ使用を下げる場合は `gpu_memory_preset: low` にします。詳細なprofile定義は `course/09_runbooks/base_config.yaml` にありますが、通常は触りません。

`.env` はW&B/Weave/OpenAIなどの認証・プロジェクト設定、`config.yaml` はモデルサイズ・実行規模・GPUメモリ設定です。CLIで明示した値はYAMLより優先されます。

### 何を実行するか

| Job | Profile | 内容 |
| --- | --- | --- |
| 事前確認 | なし | Python環境、ART API、W&B/Weave接続を確認 |
| コマンド確認 | `no_gpu_walkthrough` | 実際の学習はせず、runbookが発行するコマンド列を表示 |
| 最小GPU smoke | `smoke_tiny` | 小型モデルでdata、SFT、GRPO、evalの配線だけを短時間確認 |
| ハンズオン本体 | `workshop_fast_h100` | LFM2.5-8Bで短いSFT -> GRPOを実行 |
| フル検証 | `validated_h100` | LFM2.5-8Bでbaseline、SFT、GRPO、candidate eval、比較表まで実行 |
| 大規模appendix SFT | `appendix_tau_synthetic_sft_h100` | `fuvty/tau-bench-synthetic`でSFT anchorを作成 |
| 大規模appendix data | `appendix_tau_synthetic_wide_data_h100` | GRPO/GSPO用のwide splitを生成 |
| 大規模appendix GRPO | `appendix_tau_synthetic_wide_balanced_grpo_h100` | wide split上でbalanced rewardのGRPOを実行 |
| 大規模appendix GSPO | `appendix_tau_synthetic_wide_balanced_gspo_h100` | wide split上でbalanced rewardのGSPOを実行 |

### ターミナルから直接実行

最初に接続確認を実行します。

```bash
set -a
source .env
set +a
export PATH="$PWD/.art/venv311/bin:$PWD/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1
python course/00_setup/env_check.py
python course/00_setup/art_api_smoke.py
python course/00_setup/wandb_weave_smoke.py
```

GPUを使わずにrunbookの中身だけ確認します。

```bash
set -a
source .env
set +a
export PATH="$PWD/.art/venv311/bin:$PWD/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1
python course/09_runbooks/run_retail_agentic_sequence.py --run-profile no_gpu_walkthrough
```

最小GPU smokeです。

```bash
set -a
source .env
set +a
export PATH="$PWD/.art/venv311/bin:$PWD/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1
python course/09_runbooks/run_retail_agentic_sequence.py --run-profile smoke_tiny
```

ハンズオン本体の短めのGPU実行です。

```bash
set -a
source .env
set +a
export PATH="$PWD/.art/venv311/bin:$PWD/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1
python course/09_runbooks/run_retail_agentic_sequence.py --run-profile workshop_fast_h100
```

フル検証です。

```bash
set -a
source .env
set +a
export PATH="$PWD/.art/venv311/bin:$PWD/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1
python course/09_runbooks/run_retail_agentic_sequence.py --run-profile validated_h100
```

同じcloneで清書用に完全に分離して再実行する場合は、ART保存先とrun slugを変えます。

```bash
set -a
source .env
set +a
export PATH="$PWD/.art/venv311/bin:$PWD/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1
python course/09_runbooks/run_retail_agentic_sequence.py \
  --run-profile validated_h100 \
  --run-slug lfm25-8b-a1b-retail-validated-clean \
  --art-path .art/course_runs_clean
```

runbookはtau-style RL向けに `--continue-on-invalid` をデフォルトで有効化しています。stochastic evalには `--eval-rollouts-per-scenario 4 --eval-temperature 0.2` を加えると、pass@kとreward varianceも比較できます。
tau風プロファイルでは `RETAIL_ALLOW_REFERENCE_STATE_ACTION_JUMPS=true` もデフォルトです。これは、正しいstate-changing actionだけは先行するread-only再現順序を飛ばして受理する設定で、誤った状態変更や未知ツールは引き続き失敗として記録します。これにより、講座の主成功指標 `retail_task_success` はread-only toolの順序ではなく、結果に効く状態変更と最終応答を中心に見ます。
比較結果は監査用の全列版 `checkpoint_eval_comparison.md/.csv` と、README/スライド向けの簡潔版 `checkpoint_eval_summary.md/.csv` の両方に出力されます。
SFT warm startを強めるフル検証では、英語版READMEの `--include-teacher-sft`、`--include-areal-sft`、`--include-success-trace-sft` 例を使えます。runbookでは公開teacher next-actionデータも同じtask-hash splitのSFT foldに絞ってから混ぜます。
比較後には各JSONL結果をcached predictionとしてWeave Evaluationにも流し、trace単位とstage単位の両方で確認できるようにしています。

大規模appendixは、SFT anchorを作ってからGRPO/GSPOを分岐します。

```bash
set -a
source .env
set +a
export PATH="$PWD/.art/venv311/bin:$PWD/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1
python course/09_runbooks/run_retail_agentic_sequence.py --run-profile appendix_tau_synthetic_sft_h100
python course/09_runbooks/run_retail_agentic_sequence.py --run-profile appendix_tau_synthetic_wide_data_h100
python course/09_runbooks/run_retail_agentic_sequence.py --run-profile appendix_tau_synthetic_wide_balanced_grpo_h100
python course/09_runbooks/run_retail_agentic_sequence.py --run-profile appendix_tau_synthetic_wide_balanced_gspo_h100
```

### Slurmから実行

Slurmは任意です。SUNKのようなSlurm H100環境では、同じprofileをsbatch wrapperで投入できます。`ART_COURSE_ROOT="$PWD"` を付けているので、今いるrepoディレクトリをジョブ側でも使います。

接続確認です。

```bash
mkdir -p .art/sunk_runs/logs
sbatch \
  --job-name=art-setup-check \
  --partition=h100 \
  --gres=gpu:h100:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=00:30:00 \
  --output=.art/sunk_runs/logs/%x-%j.out \
  --error=.art/sunk_runs/logs/%x-%j.err \
  --wrap='cd "$SLURM_SUBMIT_DIR" && set -a && source .env && set +a && export PATH="$PWD/.art/venv311/bin:$PWD/.venv/bin:$PATH" && export PYTHONUNBUFFERED=1 && python course/00_setup/env_check.py && python course/00_setup/art_api_smoke.py && python course/00_setup/wandb_weave_smoke.py'
```

GPUを使わずにrunbookの中身だけ確認します。

```bash
ART_COURSE_ROOT="$PWD" sbatch course/09_runbooks/sunk_h100_retail_config_run.sbatch \
  course/09_runbooks/config.yaml \
  course/09_runbooks/base_config.yaml \
  no_gpu_walkthrough
```

最小GPU smokeです。

```bash
ART_COURSE_ROOT="$PWD" sbatch course/09_runbooks/sunk_h100_retail_config_run.sbatch \
  course/09_runbooks/config.yaml \
  course/09_runbooks/base_config.yaml \
  smoke_tiny
```

ハンズオン本体の短めのGPU実行です。

```bash
ART_COURSE_ROOT="$PWD" sbatch course/09_runbooks/sunk_h100_retail_config_run.sbatch \
  course/09_runbooks/config.yaml \
  course/09_runbooks/base_config.yaml \
  workshop_fast_h100
```

フル検証です。

```bash
ART_COURSE_ROOT="$PWD" sbatch course/09_runbooks/sunk_h100_retail_config_run.sbatch \
  course/09_runbooks/config.yaml \
  course/09_runbooks/base_config.yaml \
  validated_h100
```

同じcloneで清書用に完全に分離して再実行する場合です。

```bash
ART_COURSE_ROOT="$PWD" sbatch course/09_runbooks/sunk_h100_retail_config_run.sbatch \
  course/09_runbooks/config.yaml \
  course/09_runbooks/base_config.yaml \
  validated_h100 \
  --run-slug lfm25-8b-a1b-retail-validated-clean \
  --art-path .art/course_runs_clean
```

大規模appendixは、SFT anchor完了後にGRPO/GSPOを同じ8xH100ノード内で並列投入します。

```bash
mkdir -p .art/sunk_runs/logs
jid_sft=$(ART_COURSE_ROOT="$PWD" sbatch --parsable course/09_runbooks/sunk_h100_retail_config_run.sbatch \
  course/09_runbooks/config.yaml \
  course/09_runbooks/base_config.yaml \
  appendix_tau_synthetic_sft_h100)
jid_data=$(ART_COURSE_ROOT="$PWD" sbatch --parsable --dependency=afterok:${jid_sft} course/09_runbooks/sunk_h100_retail_config_run.sbatch \
  course/09_runbooks/config.yaml \
  course/09_runbooks/base_config.yaml \
  appendix_tau_synthetic_wide_data_h100)
ART_COURSE_ROOT="$PWD" sbatch --dependency=afterok:${jid_data} course/09_runbooks/sunk_h100_parallel_profiles.sbatch \
  appendix_tau_synthetic_wide_balanced_grpo_h100 \
  appendix_tau_synthetic_wide_balanced_gspo_h100
```

清書用にappendixも過去checkpointから分離する場合です。

```bash
mkdir -p .art/sunk_runs/logs
jid_sft=$(ART_COURSE_ROOT="$PWD" sbatch --parsable course/09_runbooks/sunk_h100_retail_config_run.sbatch \
  course/09_runbooks/config.yaml \
  course/09_runbooks/base_config.yaml \
  appendix_tau_synthetic_sft_h100 \
  --run-slug lfm25-8b-a1b-tau-synthetic-retail-clean \
  --art-path .art/course_runs_clean)
jid_data=$(ART_COURSE_ROOT="$PWD" sbatch --parsable --dependency=afterok:${jid_sft} course/09_runbooks/sunk_h100_retail_config_run.sbatch \
  course/09_runbooks/config.yaml \
  course/09_runbooks/base_config.yaml \
  appendix_tau_synthetic_wide_data_h100 \
  --run-slug lfm25-8b-a1b-tau-synthetic-retail-wide-data-clean \
  --art-path .art/course_runs_clean)
RUNBOOK_EXTRA_ARGS="--run-slug lfm25-8b-a1b-tau-synthetic-retail-clean --art-path .art/course_runs_clean" \
ART_COURSE_ROOT="$PWD" sbatch --dependency=afterok:${jid_data} course/09_runbooks/sunk_h100_parallel_profiles.sbatch \
  appendix_tau_synthetic_wide_balanced_grpo_h100 \
  appendix_tau_synthetic_wide_balanced_gspo_h100
```

`sunk_h100_parallel_profiles.sbatch` はprofileごとにGPUを1枚ずつ割り当てます。複数のGRPO/GSPO条件を速く比較するための導線であり、単一GRPO更新そのものを8GPU分散するものではありません。

## 結果の確認

現在の推奨条件は `LiquidAI/LFM2.5-8B-A1B`、task-disjoint bridge next-action SFT、SFT foldに絞ったpublic teacher next-action SFT mix、GRPOです。RL checkpointは最新stepを自動採用せず、候補checkpointをforkしてheld-out validationで確認します。

結果はrunbookが出力する `checkpoint_eval_summary.md/.csv` とW&Bの横持ち比較表で確認します。

## 評価表

checkpoint比較は横持ち表として出力します。1行が1つのstage、各列が評価指標です。

講座の主指標は `retail_task_success` です。tau-style evaluationの「結果を重視する」考え方に寄せつつ、ARTのrollout内で高速に使えるように、必要なstate-changing action、最終応答、invalid action、truncationを見ます。

評価は2つに分けて読みます。

| 文脈 | 環境 | 指標 | 目的 |
| --- | --- | --- | --- |
| 学習中rollout | 軽量なローカルretail環境 | `data/step_retail_task_success_mean` | GRPO/GSPO/RULER中の高速な学習信号 |
| checkpoint validation | 同じ軽量環境のheld-out scenario | `retail_task_success` | Baseline、SFT、RL checkpointの比較 |

| Column | 意味 |
| --- | --- |
| `model` | base modelまたはcheckpoint family |
| `stage` | Baseline、SFT、GRPO、GSPO、RULER、official tau2 importなど |
| `model_artifact_path` | 評価したcheckpointのW&B Artifact path |
| `reward` | 選択したcourse reward profileでの平均reward |
| `retail_task_success` | 講座の主成功指標。必要な状態変更、最終応答、invalid actionなし、truncationなしを確認 |
| `tau2_official_success` | optionalなofficial tau2 runtime結果をimportした場合の成功率 |
| `reference_tool_sequence_exact_match` | 参照解法と同じtool pathを辿ったか |
| `state_action_sequence_match` | 必要なstate-changing actionを正しく呼べたか |
| `communication_success` | 最終応答で結果をユーザーに伝えられたか |
| `bad_state_action` / `missing_state_action` | Weave traceで確認するstate-changing actionエラー |

W&B Artifactsはdataset、SFT checkpoint、各RL branch checkpointをつなぎ、Weaveはrollout traceとcached checkpoint Evaluationを保存します。Run名とRun IDはW&Bに生成させ、Runの意味は `stage:sft-train`、`kind:training`、`algo:grpo`、`split:validation` などのtags、notes、configに入れます。Weave tracingを使うRunでは、active W&B Runの `run.id` をWeave clientに明示的に渡します。official tau2評価を行う場合は拡張フローとして別列にimportします。
