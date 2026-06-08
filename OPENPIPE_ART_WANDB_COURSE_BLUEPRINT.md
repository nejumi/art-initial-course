# OpenPipe ART x W&B Models x Weave Training Course Blueprint

作成日: 2026-05-29  
最終更新: 2026-06-08
想定: エンタープライズ利用者向け。ローカルGPU / Dedicated Cloud / Customer Managed (W&B公式名称では Self-Managed) を主軸にし、Multi-tenant SaaS と Serverless RL は比較・補足として扱う。

## 1. コースの北極星

このコースは「ARTでRLエージェントを訓練する」だけではなく、受講者が自分の組織で次の一連の流れを再現できることを目標にする。

1. 既存のPythonエージェントをARTの `Scenario -> rollout -> Trajectory -> TrajectoryGroup -> reward -> backend.train` に接続する。
2. W&B Modelsで学習メトリクス、データ、LoRAチェックポイント、Registryの昇格履歴を追えるようにする。
3. Weaveでrollout、tool call、RULER judge、評価失敗の実例をトレースとして辿れるようにする。
4. SFTで形式・ツール利用・初期成功率を底上げし、GRPO系RLでタスク性能を改善する。
5. RULERで報酬設計の初速を上げ、必要に応じて手作り報酬と併用する。
6. GSPO、rollout logprobs、KL penalty、importance sampling、checkpoint forking/deletionなどを、単なる引数一覧ではなく「いつ使うか」で理解する。`precalculate_logprobs` はARTバージョンやbackendで利用可否が変わるadvanced topicとして扱う。
7. Dedicated Cloud / Self-Managed / Customer Managed環境におけるセキュリティ、データ所在、Registry運用、サービスアカウント運用まで含めて設計できる。

短く言うと、受講後の姿は「ARTのサンプルを動かせる人」ではなく、「自社のエージェント学習基盤を設計・デバッグ・説明できる人」。

## 2. 調査で確認した一次情報

主要リンク:

- OpenPipe ART GitHub: https://github.com/OpenPipe/ART
- ART docs index: https://art.openpipe.ai/llms.txt
- ART overview: https://art.openpipe.ai/getting-started/about
- Installation and setup: https://art.openpipe.ai/getting-started/installation-setup
- Training loop: https://art.openpipe.ai/fundamentals/training-loop
- ART client: https://art.openpipe.ai/fundamentals/art-client
- ART backend: https://art.openpipe.ai/fundamentals/art-backend
- SFT training: https://art.openpipe.ai/fundamentals/sft-training
- RULER: https://art.openpipe.ai/fundamentals/ruler
- Tracking metrics: https://art.openpipe.ai/features/tracking-metrics
- Additional histories: https://art.openpipe.ai/features/additional-histories
- Checkpoint forking: https://art.openpipe.ai/features/checkpoint-forking
- Checkpoint deletion: https://art.openpipe.ai/features/checkpoint-deletion
- GSPO: https://art.openpipe.ai/experimental/gspo
- Supported models: https://art.openpipe.ai/resources/models
- W&B Serverless RL: https://docs.wandb.ai/serverless-rl
- W&B Serverless RL available models: https://docs.wandb.ai/serverless-rl/available-models
- W&B Serverless RL trained model inference: https://docs.wandb.ai/serverless-rl/use-trained-models
- W&B Registry overview: https://docs.wandb.ai/models/registry
- W&B Weave quickstart: https://docs.wandb.ai/weave/quickstart
- Weave Evaluations: https://docs.wandb.ai/weave/tutorial-eval
- Weave tracing concepts: https://docs.wandb.ai/weave/guides/tracking/tracing
- Weave trace your code: https://docs.wandb.ai/weave/guides/tracking/create-call
- Link Weave traces to W&B runs: https://docs.wandb.ai/weave/guides/tracking/trace-to-run
- Use Weave with W&B training runs: https://docs.wandb.ai/weave/guides/tools/weave-in-workspaces
- W&B hosting options: https://docs.wandb.ai/platform/hosting
- W&B Dedicated Cloud: https://docs.wandb.ai/platform/hosting/hosting-options/dedicated-cloud
- tau-bench / tau2-bench GitHub: https://github.com/sierra-research/tau2-bench
- tau-dev-task-retail-v1 dataset: https://huggingface.co/datasets/lefft/tau-dev-task-retail-v1
- AReaL tau2 data: https://huggingface.co/datasets/inclusionAI/AReaL-tau2-data

APIバージョン互換性:

- OpenPipe ARTは開発が速いため、ハンズオンでは `openpipe-art` のバージョンをpinし、READMEと `.env.example` に明示する。
- RL学習の基本形は `backend.train(model, trajectory_groups, ...)` と `await model.log(..., metrics=result.metrics, step=result.step, split="train")` を組み合わせる形に統一する。
- 古いnotebookやblog exampleと差がある箇所は、`Version compatibility` として明示する。

## 3. コース全体像

コース形式:

API互換性チェック:

- 受講者の設定は `.env.example` -> `.env` と `course/09_runbooks/config.yaml` に分ける。`.env` はW&B/Weave/OpenAIなどの接続情報、`config.yaml` はrun scale、model size、GPU memory presetを扱う。
- ハンズオンで使う公開API名・引数名は、pinした `openpipe-art` バージョンで `course/00_setup/art_api_smoke.py` のimport/signature smoke testを通してから使う。
- 特に `backend.train` kwargs、RULER kwargs、SFT helper、checkpoint/state/config系APIはdocsだけでなくsource/installed packageの両方で確認する。

- フルワークショップ + 事前セットアップ + 事後capstone
- 各章は「座学 -> ハンズオン -> W&B/Weaveで観察 -> 設計判断ディスカッション」の型で進める。
- 受講者はGPUとネットワーク状況に応じて演習トラックを選ぶ。

### 3.1 短時間デリバリー版

コースのゴールは、W&B Models / Weave / ARTでagentic SFT/RLをどう設計・観測・検証するかを理解し、短い実行で初期信号を確認し、評価ワークフローとして妥当な採用判断を行えるようにすること。構成はセッション時間、GPU、事前準備状況に合わせて調整する。

ワークショップ設定の基本:

受講者が直接編集する設定ファイルは2つに絞る。詳細presetは参照専用として分ける。

| ファイル | 役割 | 触るタイミング |
| --- | --- | --- |
| `.env` | W&B / Weave / OpenAI key、W&B project、entity、ART保存先などの環境依存設定 | 受講開始前または最初のsetup |
| `course/09_runbooks/config.yaml` | run profile、model size、GPU memory preset、必要最小限のoverride | ハンズオン中にGPUや実行時間に合わせて変更 |
| `course/09_runbooks/base_config.yaml` | SFT/RL/eval/vLLMの詳細preset | 通常は触らない。advanced labで扱う |

受講者が見る `config.yaml` の基本形:

```yaml
run_profile: workshop_fast_h100
model_profile: standard
base_model:
gpu_memory_preset: standard
overrides: {}
```

変更シナリオ:

| やりたいこと | 変更する行 | 意味 |
| --- | --- | --- |
| 標準ワークショップを実行 | `run_profile: workshop_fast_h100` | SFT -> GRPOの流れを短めに実行する |
| フル検証に近い実行 | `run_profile: validated_h100` | SFT/RL/evalを増やし、checkpoint候補選択とheld-out評価まで行う |
| 小さいGPUで流れだけ確認 | `model_profile: tiny` | 小型モデルでsetup/SFT/RL smokeを確認する |
| H100想定の標準モデル | `model_profile: standard` | `LiquidAI/LFM2.5-8B-A1B` を使う |
| 任意のHF/vLLM互換モデル | `model_profile: custom` と `base_model: ...` | 指定したモデルを直接使う |
| vLLMのメモリ圧を下げる | `gpu_memory_preset: low` | context長やbatchingを抑え、OOMリスクを下げる |
| 一時的にRLだけ短くする | `overrides: {rl_steps: 1}` | preset全体は保ちつつ、特定項目だけ上書きする |

設定のメンタルモデル:

- `.env` は「どこにログを送るか、どのkeyを使うか」。
- `config.yaml` は「何を、どの大きさで、どれくらい学習するか」。
- `base_config.yaml` は「標準profileの中身」。初回ハンズオンでは開かない。
- CLI引数は一時的な上書き用。スライドとハンズオンでは `config.yaml` を主導線にする。

推奨セグメント:

| セグメント | 内容 | その場で動かすもの | 見せる証拠 |
| --- | --- | --- | --- |
| Opening | W&B Models / Weave復習 | W&B project, Artifact, Weave traceを開く | dataset artifact, SFT checkpoint artifact, trace例 |
| ART primitives | ART概念マップ | `Scenario -> rollout -> TrajectoryGroup -> train` の最小コード確認 | W&B metricsとWeave traceの対応 |
| Task and evals | Retail taskと評価指標 | cached eval JSONL / Weave Evaluationを見る | Retail Task Successとtrace診断の読み方 |
| SFT warm start | SFT warm start | 小モデルまたはdry-runでSFT commandを実行 | SFT loss curve、checkpoint artifact lineage |
| Agentic RL | GRPO/GSPO/RULERの考え方 | 可能なら短いGRPO smoke | group reward range、winner-minus-loser、dropped no-signal groups |
| Validation workflow | checkpoint評価 | W&B table / Weave tracesを読む | SFTとRL checkpointを同じheld-out条件で比較する流れ |
| Enterprise wrap-up | Enterprise運用設計 | Dedicated Cloud / Self-Managed / LocalBackend比較 | Registry昇格、Artifact lineage、再現性チェックリスト |

実行トラック:

- No GPU / ネットワーク制限あり: `.env`、data artifact、cached eval、Weave trace、runbook dry-runを中心にする。
- 小さめGPU: `config.yaml` で `model_profile: tiny` にし、setup、SFT、GRPO smokeの流れを確認する。
- H100 1枚: `config.yaml` で `model_profile: standard`、`run_profile: workshop_fast_h100` または `workshop_standard_h100` を使い、短いSFTと数stepのGRPOを実行する。
- H100複数枚: SFT parentからGRPO/GSPO/RULERを独立分岐で並列実行し、checkpoint candidate selectionとheld-out evalまで行う。

Key takeaways:

- 短時間runでは、操作手順、W&B/Weave連携、初期学習信号の読み方を体験する。
- 性能改善は、checkpoint validation、W&B Artifact lineage、Weave traceが揃った評価で確認する。
- 長期RLは単調改善を仮定しない。`select_checkpoint_candidate.py` で中間checkpointを選び、fresh validation evalで採用する。
- train rewardのbest rowは採用根拠にしない。checkpoint validationを通った結果だけを比較対象にする。

評価ワークフローの読み方:

- Baselineで、base modelの初期難易度、state action error、truncationを把握する。
- SFT anchorで、tool-call形式、状態変更行動、最終応答がRLの出発点として安定しているかを見る。
- GRPO selected checkpointで、SFT anchorと同じheld-out条件に対する改善、失敗理由、artifact lineage、Weave traceを確認する。

読み方:

- SFTでtool-call形式、状態変更行動、初期成功率を底上げする。
- GRPOで `bad_state_action` / `missing_state_action` を下げ、`state_action_sequence_match` と `retail_task_success` を改善する。
- 長いRLは必ずしも単調改善しない。W&Bの学習曲線で候補を選び、held-out validationとWeave traceで採用checkpointを確認する。

トラック:

- Primary: LocalBackend on local GPU / enterprise-managed GPU
- Enterprise: W&B Dedicated Cloud or Self-Managed with local/customer-managed training
- Optional: ServerlessBackend on W&B Training, 比較デモ

モデル選択:

- `model_profile: tiny`: `Qwen/Qwen3-0.6B`。小さなGPUやCPU寄り環境でのsetup/SFT/RL smoke test用。性能改善の説得力ではなく、教材の操作手順を低コストに確認するためのプロファイル。
- `model_profile: standard`: `LiquidAI/LFM2.5-8B-A1B`。H100を想定したメインハンズオンの基準モデル。next-action SFTとtau-style RLを扱う標準プロファイル。
- `model_profile: openpipe`: `OpenPipe/Qwen3-14B-Instruct`。OpenPipe/Qwen系の互換性比較やmanaged trainingの話題に使う。
- `model_profile: serverless`: `OpenPipe/Qwen3-14B-Instruct`。W&B Serverless RLの軽い比較デモ用。
- `model_profile: moe`: `Qwen/Qwen3-30B-A3B-Instruct-2507`。Serverless/Megatron/MoEの発展説明用。
- `model_profile: custom` と `base_model: ...` を指定すると、環境に応じて任意のHF/vLLM互換モデルへ差し替えられる。

SFT/RL設計の実務的学び:

- Baseline LFM2.5-8B-A1Bはretail tool callingをある程度こなせるため、初期モデルが完全に壊れているtoy exampleにならない。
- SFTは有効だが、full trajectoryをそのまま全assistant turnで模倣させるだけでは、重要な意思決定が「それっぽい会話の再現」に埋もれやすい。
- Agentic SFTでは、状態を変えるtool callや最終応答などの意思決定点を `next-action` 形式で切り出し、直前contextから「次に何をすべきか」を学ばせる。
- SFTはagentic RLの代替ではなく、tool-call dialect、policy adherence、初期成功率、rolloutの安定性を整えるwarm startとして位置づける。
- RLの報酬設計は、final successだけの疎な報酬では信号が弱く、dense rewardを足しすぎると本来の成功方向とずれることがある。verifiable outcome、state-changing action correctness、communication quality、安全ペナルティを分けて設計する。
- 報酬設計は一度で決めない。既存研究のレシピを初期仮説にし、reward profile、penalty weight、learning rate、checkpoint selection metricを複数水準で比較する。

Agentic RLの報酬設計を分解して考える:

| 観点 | 目的 | W&B / Weaveで確認するもの |
| --- | --- | --- |
| Verifier reward | DB state、tool action、task completionなど、厳密に検証できる正しさを守る | `retail_task_success`, `state_action_sequence_match`, reward component table |
| Shaping reward | sparse successだけでは弱いので、途中行動にも学習信号を与える | group内reward variance、winner-minus-loser差分、state-action reached/attempted rate |
| Penalty design | 不正tool、余計なmutation、missing action、truncationを抑える | `bad_state_action`, `missing_state_action`, invalid tool trace, truncated rollout |
| RULER | 会話品質、説明の自然さ、policy alignmentなど厳密検証しにくい品質を見る | RULER judge trace、hybrid reward比率別のheld-out eval |
| Checkpoint metric | train reward最大ではなく、held-outで伸びる候補を選ぶ | checkpoint comparison table、model artifact lineage、acceptance gate |
| Weave trace | なぜ成功/失敗したかをtrajectory単位で説明可能にする | rollout trace、tool call trace、reward function trace、failure examples |

報酬設計の従来知見との接続:

| 教材で扱う観点 | 関連する既存知見 |
| --- | --- |
| SFT -> RLの段階設計 | InstructGPT/RLHFはdemonstration SFT、preference/reward model、RLという段階設計を採用する。Agentic RLでもSFTをwarm startとして使い、RLで実際のrolloutから改善する構図は近い。 |
| Verifiable reward | DeepSeekMath/DeepSeek-R1系のGRPO/RLVRでは、正誤判定やformat rewardなど検証可能な報酬が中核になる。tool-using agentではDB state、tool action、policy complianceがこれに相当する。 |
| Process / shaping reward | reward shapingは古典的RLから知られる設計課題で、途中状態への報酬は学習を助ける一方、最適化対象をずらす危険がある。process supervisionは最終結果だけでなく中間ステップを評価する流れと対応する。 |
| Reward hackingへの警戒 | 報酬を増やしても本来の目的を満たさないspecification gamingは、RL安全性の古典的問題。agentでは「もっともらしい応答」「余計なtool call」「状態を壊す近道」を検出する必要がある。 |
| LLM-as-judge / RLAIF | Constitutional AI/RLAIFやRULERは、厳密検証しにくい品質をAI judgeで評価する系譜にある。教材ではverifiable rewardを主軸にし、RULERは会話品質やpolicy説明の補助として使う。 |
| Trace-driven eval | tau-bench/tau2-benchやinteractive tool-agent研究では、最終応答だけでなくmulti-turn trajectory、tool use、state transition、policy adherenceを評価する。W&B/Weaveはこの評価を実験管理と失敗分析に接続する。 |

RLVRの位置づけ:

- Verifiable Rewardの発想自体は新しくない。数学の正答照合、コードのunit test、ゲームスコア、定理証明器、tool実行結果など、機械的に検証できる報酬は従来からRLの強い設定だった。
- DeepSeekMathはGRPOとrule-based rewardを組み合わせ、LLM reasoning post-trainingでVerifiable Rewardが大きく効くことを示した代表例になった。
- DeepSeek-R1以降、RLVRは「人間の好みを毎回ラベルする代わりに、検証可能なタスクで大規模にrolloutし、正解・形式・制約で報酬を返す」post-training recipeとして広く注目されている。
- Agentic workflowでは、検証対象がfinal answerだけでなく、tool call、DB state、policy compliance、user communication、turn-level processに広がる。

RLVRのpitfall:

| Pitfall | 何が起きるか | 教材での対策 |
| --- | --- | --- |
| 代理指標の限界 | verifierが測れるものだけを最適化し、本来の顧客価値とずれる | outcome、state action、communication、policyを分解して見る |
| Sparse reward | 最後の成功/失敗だけでは、どの行動が良かったか学習信号が弱い | state-changing action correctnessやprocess metricsを入れる |
| Reward hacking | verifierの穴、format、固定task分布、unit testの癖に過適合する | held-out eval、negative examples、trace inspectionで確認する |
| Process blind spot | 最終状態は正しいが、途中で余計なtool callや危険なmutationをする | bad/missing state action、invalid mutation、truncationをpenalty化する |
| Quality blind spot | DB stateは合っているが、顧客対応として不親切、説明不足、brand tone不一致になる | RULERやWeave traceで会話品質を補助評価する |
| Simulator noise | multi-turn agentではユーザー役や環境のノイズが報酬を歪める | deterministic env、seed固定、rollout trace、複数rollout evalで分離する |
| Distribution shift | train verifierでは勝つが、新しい言い回し・長い会話・異なるpolicyで崩れる | train/holdout分割、stochastic eval、scenario-level failure analysisを行う |

RULER / LLM-as-judgeの使いどころ:

| 注意点 | 何が起きるか | 教材での扱い |
| --- | --- | --- |
| Relative judgeでありverifierではない | もっともらしい会話を高く評価し、DB stateやtool correctnessを取り違えることがある | strict verifierを主軸にし、RULERは低めの重みで補助する |
| Position bias | trajectoryの提示順によってscoreが揺れることがある | trajectory順序をshuffleし、必要なら複数judge passを比較する |
| Long-context confusion | trajectoryが長いと、trajectory IDと内容の対応や失敗箇所の記憶が混ざる | 短いcurriculum、要約、failure trace確認、max turns管理を使う |
| Rubric sensitivity | rubricの曖昧さで、何を良いtrajectoryとみなすかが変わる | rubricをtask success、tool use、policy、communicationに分解する |
| Score calibration | 0-1 scoreの絶対値はjudgeやpromptに依存する | GRPOのgroup-relative signalとして使い、held-out evalで採用判断する |

Key message:

> Verifiable Reward is necessary, but not sufficient. In agentic workflows, the trajectory itself is part of the product.

Reference anchors:

- Ouyang et al., "Training language models to follow instructions with human feedback", 2022: https://arxiv.org/abs/2203.02155
- Ng, Harada, Russell, "Policy invariance under reward transformations: Theory and application to reward shaping", 1999: https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf
- Lightman et al., "Let's Verify Step by Step", 2023: https://arxiv.org/abs/2305.20050
- Amodei et al., "Concrete Problems in AI Safety", 2016: https://arxiv.org/abs/1606.06565
- Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", 2024: https://arxiv.org/abs/2402.03300
- Guo et al., "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", 2025: https://arxiv.org/abs/2501.12948
- Bai et al., "Constitutional AI: Harmlessness from AI Feedback", 2022: https://arxiv.org/abs/2212.08073
- AReaL/SEA, "From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents", 2026: https://arxiv.org/abs/2601.22607
- tau2-bench repository: https://github.com/sierra-research/tau2-bench
- ART RULER documentation: https://art.openpipe.ai/fundamentals/ruler

- RULERは、正解DB stateのような厳密検証を置き換えるものではなく、会話品質、ポリシー説明、簡潔さ、安全な顧客対応など、verifierだけでは表現しにくい軸を補助するために使う。
- Retail taskでは、`verifier reward` を主軸にし、RULERを低めの重みでhybrid rewardに混ぜる水準と、RULERを強める水準を比較する。
- 短いbridge curriculumで `next-action SFT -> GRPO branch / GSPO branch` を独立比較し、`tau_irc` 系報酬、state-changing action correctness、communication success、Retail Task Successを横持ち表で検証する。
- SFT checkpointはlossだけでは採用しない。baseline/SFT/RLを同じholdoutでWeave evalし、SFTが少なくともtool-call形式とstate-changing action指標を改善していることを確認してからRL parentにする。
- RLは「エラーなく回る」では合格にしない。group内reward variance、winner-minus-loser差分、zero-variance group filter、state-action attempt/reached rate、bad/missing state-action rateをW&Bに出し、GRPO/GSPOが実際に学習信号を受けていることを確認する。
- 長いRL runは単調改善を仮定しない。`train_metrics_<algo>_<suffix>.jsonl` と `select_checkpoint_candidate.py` で候補stepを選び、中間checkpointをheld-out eval、W&B Artifact lineage、Weave traceで確認してから採用する。
- 採用判断はtrain rewardの最大値ではなく、held-out evalとacceptance gateを通ったcheckpointだけで行う。

SFT設計で巨人の肩に乗るポイント:

- AReaL/SEA系の前例に合わせ、`SFT -> verifier-based GRPO/GSPO` を基本形にする。SFTはagentic RLの代替ではなく、tool-call dialect、policy、初期成功率を整えるためのpolicy initializationとして扱う。
- CoVeの考え方に合わせ、ただの成功ログではなく、制約・検証器・canonical reward・judge qualityでcleanに確認できる軌跡を優先する。教材では `KermitCO/...retail-traces` の reward-1 / non-memory / blind-strict filter と、unknown tool除去がこの役割を担う。
- TopoCurateの考え方に合わせ、SFTは「成功しているが多様で、必要なら回復行動も含む」軌跡を使い、RLは「まだ失敗分岐が残り、group-relative advantageが立つ」タスク集合を選ぶ。bridge curriculumは簡単にしすぎず、no-signal group diagnosticsで難易度を監視する。
- 次アクションSFTを主軸にし、full-dialog SFTは教育用baselineに留める。長いfull-dialogをそのまま全assistant turnでmaskすると、前のassistant行動まで重複監督して、W&B上のloss改善と実際のagentic改善がズレやすい。
- SFT採用条件はlossではなく、Weave eval上の `retail_task_success`, `state_action_sequence_match`, `bad_state_action`, `missing_state_action`, `reference_tool_sequence_exact_match`, tool-call F1/argument matchで判断する。

学習題材:

- メイン題材は「Retail Customer Support Agent」
- オープンデータ: `lefft/tau-dev-task-retail-v1` をSFT/形式理解に使い、tau-bench/tau2-bench retailの考え方を評価/RL rollout設計に使う。SFT warm start強化では `amityco/tau-bench-retail-train-next-action-all-step-score-v0.2` を使い、Advancedでは `inclusionAI/AReaL-tau2-data` をnext-action SFT/RLデータ設計の比較対象にする。
- 入力: 顧客からの注文キャンセル、返品、交換、住所変更、注文状況確認、商品情報確認などの問い合わせ。
- 出力: 顧客への自然文応答と、必要なOpenAI tool-calling形式の関数呼び出し。
- ツール例: `get_user_details`, `get_order_details`, `modify_pending_order_address`, `cancel_pending_order`, `return_delivered_order`, `exchange_delivered_order`, `calculate`。

受講者向け導入スライド: 今回解くタスク

このハンズオンでは、オンライン小売店のカスタマーサポートAgentを作る。ユーザーは注文キャンセル、返品、交換、住所変更、注文状況確認などを依頼する。Agentは会話だけで答えるのではなく、必要に応じてtoolを呼び、架空の小売DBの状態を正しく更新する。

Agentが行うこと:

| Step | Agentの仕事 | 例 |
| --- | --- | --- |
| 1. 依頼を理解する | ユーザーが何をしたいかを読む | 「注文をキャンセルしたい」「配送先を変更したい」 |
| 2. 状態を確認する | 顧客、注文、商品、ポリシーをread-only toolで確認する | `get_user_details`, `get_order_details` |
| 3. 判断する | その依頼を実行してよいかをpolicyと状態から判断する | 出荷前ならキャンセル可、出荷後なら返品案内 |
| 4. 状態を変える | 必要な場合だけstate-changing toolを呼ぶ | `cancel_pending_order`, `modify_pending_order_address` |
| 5. 結果を伝える | 実行結果、できない理由、次に必要な情報をユーザーに伝える | 「注文をキャンセルしました」「この注文は既に発送済みです」 |

具体例:

| ユーザー依頼 | Agentが確認すること | 正しいtool action | 正しい最終応答 |
| --- | --- | --- | --- |
| 「注文Aをキャンセルしたい」 | 注文Aが本人の注文か、まだキャンセル可能か | キャンセル可能なら `cancel_pending_order` | キャンセル完了、またはキャンセル不可の理由を伝える |
| 「配送先住所を変えたい」 | 注文がまだ発送前か、新住所が指定されているか | 変更可能なら `modify_pending_order_address` | 変更完了、または変更不可の理由を伝える |
| 「届いた商品を返品したい」 | 配送済みか、返品期間内か、商品が返品対象か | 条件を満たせば `return_delivered_order` | 返品受付、返金や返送に必要な情報を伝える |

何を成功とみなすか:

| 評価軸 | 意味 | 成功例 | 失敗例 |
| --- | --- | --- | --- |
| Retail Task Success | 必要な状態変更を正しく行い、結果をユーザーに伝え、危険なtool actionや途中打ち切りがない | 正しい注文だけをキャンセルし、完了と返金見込みを伝える | 別注文を変更した、必要な状態変更をしない、結果を伝えない |
| State Action Match | 結果に効くstate-changing toolを正しい引数で呼べたか | `cancel_pending_order` を正しいorder_idとreasonで呼ぶ | `return_delivered_order` を誤って呼ぶ、order_idを間違える |
| Communication Success | 最終応答でユーザーが必要とする結果を伝えたか | 完了、不可理由、次の手順を簡潔に伝える | DBは更新したが、ユーザーに結果を伝えない |
| Reference Tool Sequence Exact Match | 参照解法と同じtool pathを辿ったか | 模範解答と同じ順番でtoolを呼んだ | 違う順番だが正しく解けた場合も0になりうる |

重要な区別:

- 講座の主指標は `retail_task_success`。これはARTのrollout内で使える軽量な成功判定。
- 参照解法と同じread-only tool順序を辿ること自体は主成功条件ではない。
- 結果に効くstate-changing action、final response、invalid action、truncationを分けて見る。
- official tau2 runtimeでの最終確認は拡張扱い。必要な場合にだけ、別環境で同じcheckpointを評価する。

学習中に使う報酬:

- Retail Task Success
- state-changing action correctness and argument correctness
- communication success
- invalid/unknown tool call penalty
- bad or missing state-changing action penalty
- excessive turn / truncation penalty
- RULERによる「顧客対応の自然さ」「ポリシー説明のよさ」「簡潔さ」

この題材がよい理由:

- 実際に入手できるオープンデータがある。`lefft/tau-dev-task-retail-v1` はOpenAI tool-calling wire formatで915件、train/validation/test split付き。
- Retailはsingle-controlなので、Telecomのようなuser-side state mutation/user simulator設計を初回ハンズオンに持ち込まずに済む。
- tool call、multi-turn、policy compliance、verifiable reward、Weave Evals、W&B Models/Registryの意味が自然に出る。
- SFTでtool-call形式と次アクション分布を覚え、RLでstate-changing outcomeとcommunicationを改善する流れが説明しやすい。
- Telecomは発展編として「production agentでは人間がtrajectoryに入るが、RLではuser simulator/environmentで近似する」話に回す。

## 4. 受講者ペルソナ

主対象:

- ML engineer / AI platform engineer
- LLM agentを社内システムに組み込むapplication engineer
- W&Bを扱うMLOps / platform team
- Field engineer / solution architect

前提知識:

- Python asyncの基礎
- OpenAI Chat Completions形式
- LoRA/SFT/RLHF/GRPOの概念を聞いたことがある程度
- W&B Experiments/Artifactsの初歩

前提にしないこと:

- RLの数式を一から導出できること
- vLLM/Unsloth/torchtuneを内部実装まで理解していること
- W&B Dedicated CloudやSelf-Managedの運用経験

## 5. 学習成果

受講後にできること:

- ARTの `TrainableModel`, `LocalBackend`, `ServerlessBackend`, `Trajectory`, `TrajectoryGroup`, `gather_trajectory_groups`, `ruler_score_group` を説明し、最低限の訓練ループを書ける。
- `model.openai_client()` と `model.get_inference_name()` で常に最新LoRAへ推論する設計を理解できる。
- `Choice` と `dict` assistant messageの違い、logprobsがRLでなぜ重要かを説明できる。
- SFT用JSONLを作り、`train_sft_from_file` または `model.train_sft` で形式・初期方策をwarmupできる。
- GRPO系のgroup-relative reward設計を行い、報酬のばらつきがないgroupがtrainableにならない理由を説明できる。
- RULERを `after_each` で組み込み、judge failureを `swallow_exceptions=True` で運用上処理できる。
- GSPOを `importance_sampling_level="sequence"` として試し、MoE/long-contextでの意味と実験的性格を説明できる。
- W&B Models/Artifacts/RegistryとWeave tracesを「学習の証跡」として設計できる。
- `precalculate_logprobs`, `allow_training_without_logprobs`, `packed_sequence_length`, `logprob_calculation_chunk_size`, `kl_penalty_coef`, `checkpoint_forking`, `delete_checkpoints` をいつ触るべきか判断できる。

## 6. ARTの概念マップ

最初の30分で受講者に刻む図:

```text
Scenario
  -> rollout(model, scenario)
    -> model.openai_client().chat.completions.create(...)
    -> Trajectory(messages_and_choices=[prompt..., Choice], reward, metrics, metadata)
  -> TrajectoryGroup([same scenario, multiple attempts])
  -> reward / RULER / hybrid reward
  -> backend.train(model, groups)
  -> model.log(groups, metrics=result.metrics)
  -> W&B run metrics + local history.jsonl + trajectory parquet
  -> LoRA checkpoint
  -> W&B Artifact / Registry / Weave trace links
```

設計上のキーワード:

- Scenario: 学習対象の実務ケース。環境、入力、期待されるアウトカムを持つ。
- Rollout: agentが1回試行する実行ログ。
- Trajectory: 学習可能な会話履歴 + reward + metrics + metadata。
- TrajectoryGroup: 同じScenarioに対する複数試行。GRPO/RULERの相対比較単位。
- Backend: vLLM推論、LoRA更新、GPU memory管理、checkpoint保存を担う。
- Model.log: ARTのメトリクスとtrajectoryをW&B/ローカルへ送る境界。

### 6.1 ART仕様徹底詳解で扱うこと

このセクションは、単なるAPIリファレンスではなく「Retail agentをARTに接続するとき、どの概念がどの行に現れるか」を分解する章にする。受講者が最後に説明できるべきことは、`TrainableModel -> register -> openai_client/get_inference_name -> Trajectory/Choice/logprobs -> TrajectoryGroup -> gather -> backend.train -> model.log -> checkpoint/Registry` の一連の契約である。

#### A. Model identity and lifecycle

解説すること:

- `art.Model` は比較用のprompted/external modelにも使える推論・ログ単位。
- `art.TrainableModel` は `base_model` を持つtrainableな方策で、LoRA checkpointの系列を持つ。
- `name`, `project`, `entity`, `base_path`, `report_metrics`, `config` はW&Bとローカル保存先の設計に直結する。
- `update_wandb_config()` はrun configを固定するので、dataset id、base model、ART version、reward versionを最初に入れる。

重要API:

```python
model = art.TrainableModel(
    name="retail-agent-qwen3-14b",
    project="openpipe-art-retail",
    entity=ENTITY,
    base_model="OpenPipe/Qwen3-14B-Instruct",
    base_path="./.art",
)
model.update_wandb_config({
    "dataset": "lefft/tau-dev-task-retail-v1",
    "reward_version": "retail_reward_v1",
})
```

落とし穴:

- `TrainableModel` は `register` 前に推論clientを持たない。
- 個人entityで作ったartifactを組織Registryへ昇格できないことがあるため、enterprise演習ではteam entity/service accountを最初から使う。
- `config` はJSON serializableである必要がある。

#### B. Backend contract

解説すること:

- ARTはclient/backend分離。rolloutと環境はアプリ側、vLLM推論、LoRA更新、GPU memory、checkpoint保存はbackend側。
- `LocalBackend` はローカルGPUまたはcustomer-managed GPUでの本命。
- `ServerlessBackend` は同じ抽象のmanaged版として比較デモに留める。
- `await model.register(backend)` が、推論URL/API key/model nameをモデルへ注入する境界。

重要API:

```python
backend = LocalBackend(path="./.art", gpu_cost_per_hour_usd=2.25)
await model.register(backend)
```

落とし穴:

- `backend.train(...)` は自動で `model.log(...)` しない。
- LocalBackendのcheckpointは自動でW&B Registryに入らない。`LocalTrainResult.checkpoint_path` から教材側でartifact化する。
- PipelineTrainerのdedicated modeなど、複数GPU構成はadvanced appendixに隔離する。

#### C. OpenAI-compatible inference

解説すること:

- ARTの強みは、既存のOpenAI Chat Completions型agent loopに差し込めること。
- `model.openai_client()` と `model.get_inference_name()` を必ずセットで使う。
- Retailでは、tau-benchのtool schemaを `tools=` に渡し、assistantのtool callを環境で実行する。

重要API:

```python
completion = await model.openai_client().chat.completions.create(
    model=model.get_inference_name(),
    messages=trajectory.messages(),
    tools=trajectory.tools,
    temperature=0.8,
    max_completion_tokens=512,
    logprobs=True,
    top_logprobs=0,
)
choice = completion.choices[0]
```

落とし穴:

- `messages_and_choices` をそのままAPIへ渡さない。`Choice` が混ざるので `trajectory.messages()` を使う。
- live rolloutでは可能な限り `logprobs=True, top_logprobs=0` を要求する。
- backendやmodelによってlogprobs/tool calling対応が違うため、setup smokeで検出する。

#### D. Trajectory data model

解説すること:

- `Trajectory` は「学習可能な会話履歴 + 報酬 + metrics + metadata + logs」。
- `messages_and_choices` には、system/user/tool messageのdictと、trainable assistant出力のOpenAI `Choice` が混在する。
- `tools` は会話で使用可能なtool schema。Retailでは各trajectoryにtau-bench retail toolsを付ける。
- `metadata` には `scenario_id`, `task_id`, `domain`, `split`, `difficulty`, `weave_trace_id`, `checkpoint_step` を入れる。

重要API:

```python
trajectory = art.Trajectory(
    messages_and_choices=[
        {"role": "system", "content": retail_policy},
        {"role": "user", "content": scenario.user_message},
    ],
    tools=retail_tools,
    metadata={"scenario_id": scenario.id, "domain": "retail"},
)
trajectory.messages_and_choices.append(choice)
trajectory.reward = reward.total
trajectory.metrics.update(reward.metrics)
trajectory.log(reward.explanation)
trajectory.finish()
```

落とし穴:

- 普通のassistant dictはログ上はassistantだが、通常のRL trainable unitとしては `Choice` と意味が違う。
- `Choice.logprobs` が無いとLocalBackendで `There are no assistant logprobs to train on` に到達しやすい。
- Weave表示用にはlogprobsを落とし、ART学習用trajectoryには保持する。

#### E. Tool calling and environment stepping

解説すること:

- Retailはsingle-controlなので、環境状態を変えるのはエージェント側tool callだけ。
- rolloutは「assistant choice -> tool calls -> tool messages -> next user/message or done」のループ。
- tool結果もtrajectoryに残すが、学習対象ではなく次のassistant出力の文脈になる。

Retail最小ループ:

```python
for _ in range(max_turns):
    completion = await client.chat.completions.create(
        model=model.get_inference_name(),
        messages=trajectory.messages(),
        tools=trajectory.tools,
        logprobs=True,
        top_logprobs=0,
    )
    choice = completion.choices[0]
    trajectory.messages_and_choices.append(choice)

    env_step = retail_env.step(choice.message)
    trajectory.messages_and_choices.extend(env_step.tool_messages)
    if env_step.user_message:
        trajectory.messages_and_choices.append(env_step.user_message)
    if env_step.done:
        break
```

落とし穴:

- tool call argumentはJSON文字列として来ることがある。structured parserで検証する。
- 不正tool名、不足argument、許可されていないmutationをrewardに入れる。
- tau-bench由来のpolicyをsystem promptに入れないと、RULERもagentもtask目的を誤解しやすい。

#### F. TrajectoryGroup and gather

解説すること:

- `TrajectoryGroup` は同じscenarioに対する複数試行。GRPO-style loopはこの相対比較からadvantageを作る。
- group内reward varianceが学習信号になる。全員成功/全員失敗では弱い。
- `gather_trajectory_groups` はparallel rollout、progress、例外集約、`after_each` hookを担う。

重要API:

```python
groups = await art.gather_trajectory_groups(
    (
        art.TrajectoryGroup(
            rollout(model, scenario) for _ in range(rollouts_per_scenario)
        )
        for scenario in batch
    ),
    max_exceptions=0.1,
    after_each=maybe_score_with_ruler,
)
```

落とし穴:

- 異なるscenarioを同じgroupに混ぜない。
- group sizeを大きくすると良い比較は増えるが、rollout costとlatencyが上がる。Retail初回は4、余裕があれば8。
- `after_each` はRULERだけでなく、group filtering、metadata付与、reward normalizationにも使える。

#### G. Training API and RL config

解説すること:

- 現行の中心APIは `result = await backend.train(model, groups, ...)`。
- LocalBackend defaultは `loss_fn="cispo"`、`loss_fn="ppo"` も選択可能。授業では「GRPO-style grouped RL」として直感を教え、実装詳細としてCISPO/PPOを補足する。
- GSPOは `importance_sampling_level="sequence"` で有効化するexperimental機能。

設定の教え方:

| レイヤー | 最初に触る | Advancedへ送る |
| --- | --- | --- |
| 学習率 | `learning_rate=5e-6` | reward std連動、trajectory数連動 |
| reward scaling | `scale_rewards=True` | `advantage_balance` |
| clipping/IS | default | `epsilon`, `epsilon_high`, `truncated_importance_sampling` |
| GSPO | 比較実験のみ | MoE/long trajectoryでの検証 |
| logprobs | rolloutで取得 | backend対応時の `precalculate_logprobs`, `allow_training_without_logprobs` |
| memory | default | `packed_sequence_length`, `logprob_calculation_chunk_size` |
| stability | reward/learning rate | `kl_penalty_coef`, KL reference |

落とし穴:

- `precalculate_logprobs` は、追加forward pass、stale logprobs、ART/backendの対応状況を理解してから使うadvanced設定として扱う。
- `allow_training_without_logprobs=True` は便利だが、importance samplingの理解を曖昧にするのでadvanced扱い。
- `backend.train` 直後に `result.step`, `result.metrics`, `checkpoint_path` または `artifact_name` を観察する。

#### H. SFT versus RL data contract

解説すること:

- SFT JSONLは `messages` と任意の `tools` を持ち、最後のmessageがassistantである必要がある。
- SFTは形式、tool-call dialect、ポリシー文体をwarmupする。RLはcurrent policyに実際にRetail taskを解かせてrewardから更新する。
- `lefft/tau-dev-task-retail-v1` はOpenAI tool-calling wire formatなので、SFT教材に向く。

重要API:

```python
from art.utils.sft import train_sft_from_file

await train_sft_from_file(
    model=model,
    backend=backend,
    file_path="data/retail_bridge_state1/sft_next_action.jsonl",
    config=art.TrainSFTConfig(learning_rate=5e-5, batch_size="auto"),
)
```

落とし穴:

- SFTの成功はRLの成功ではない。Weave Evalsでbaseline/SFT/RLを同じholdoutで比較する。
- tool callだけのassistant turnは `content: null` になりうる。OpenAI形式として処理する。

#### I. Logging, metrics, W&B, and Weave

解説すること:

- `model.log` はART観測性の境界。`history.jsonl`, trajectory parquet, W&B run metricsがここで揃う。
- W&Bは時系列とlineage、Weaveは各rollout/reward/evalの実行木を見る。
- Weave traceとART trajectoryは同じものではない。W&B Run ID、trace id、artifact uriをmetadataで結ぶ。
- W&B RunとWeave Traceは同じprojectに保存するだけではなく、同じrun contextに関連付ける。これにより、W&B Workspace上で学習曲線、checkpoint、rollout trace、cached Evaluationを同じ実験単位で読める。
- W&B Run IDはRunの識別子。実験条件のラベル付けには `tags`、`notes`、`config` を使う。

重要API:

```python
import wandb
import weave

run = wandb.init(
    entity=ENTITY,
    project=PROJECT,
    job_type="grpo",
    tags=["stage:grpo-train", "kind:training", "algo:grpo", "profile:validated_h100"],
    notes="SFT checkpointからGRPOを実行し、rollout trace、checkpoint lineage、held-out evalを確認するrun。",
    config={
        "base_model": BASE_MODEL,
        "art_model_name": ART_MODEL_NAME,
        "dataset": DATASET_ID,
        "reward_profile": REWARD_PROFILE,
    },
)
weave_client = weave.init(f"{ENTITY}/{PROJECT}")
weave_client.set_wandb_run_context(run_id=run.id)

result = await backend.train(model, groups, learning_rate=5e-6)
await model.log(groups, metrics=result.metrics, step=result.step, split="train")

@weave.op()
async def score_reward(scenario_id: str, trajectory_summary: dict) -> dict:
    ...
```

落とし穴:

- `model.log` を忘れると、訓練は進んでもW&B/ローカル履歴に証跡が残らない。
- `model._get_wandb_run()` は内部API。教材の一般コードでは直接依存しない。
- Weaveにlogprobsをそのまま出すと重いので `strip_logprobs` を使う。
- Weave TraceがW&B Runに見えないときは、project一致だけでなく、Weave clientにactive W&B Run IDが渡っているかを確認する。Run IDはW&Bが生成した `run.id` を使い、条件名やモデル名をIDとして渡さない。

#### J. Checkpoints, state, and registry

解説すること:

- LocalBackendは`.art`にLoRA checkpointを保存し、`LocalTrainResult.checkpoint_path` を返す。
- ServerlessBackendはW&B Artifact名を返す。
- `model.read_state()`, `merge_state()`, `overwrite_state()` は長時間runの進捗やdataset cursorに使える。
- Registryはcheckpoint置き場ではなく、評価済み候補の昇格先。

落とし穴:

- checkpoint削除はRegistry昇格やrollback方針とセットで扱う。
- artifact metadataには、base model、ART version、dataset artifact、reward version、Weave eval URL、best metricを入れる。

#### K. API drift and source reading

解説すること:

- ARTは活発に変化している。教材ではversion pinとAPI drift noteを明示する。
- 古いnotebookの `model.train(...)` 形式と、現行の `backend.train(...)` + `model.log(...)` を比較して、受講者が古い情報に引っかからないようにする。
- `_internal_config` やexperimental APIは「読めるが、基本教材では触らない」領域として扱う。

Retailハンズオンでの最小説明コード:

```python
async def train_one_step(model, backend, scenarios):
    groups = await art.gather_trajectory_groups(
        art.TrajectoryGroup(rollout_retail(model, s) for _ in range(4))
        for s in scenarios
    )
    result = await backend.train(model, groups, learning_rate=5e-6)
    await model.log(groups, metrics=result.metrics, step=result.step, split="train")
    return result
```

この短いコードを分解して、`Scenario`, `rollout`, `Trajectory`, `Choice`, `logprobs`, `TrajectoryGroup`, `gather`, `train`, `log`, `checkpoint` の全仕様を順に説明する。

## 7. デッキ構成案

Claude Designで作るスライドは、説明を長文で詰めるより、図とコードの最小断片で進める。

### Chapter 0 - Opening: From Prompting to On-the-job Training

Slide 1: Title

- H1: OpenPipe ART with W&B Models and Weave
- Subtitle: Enterprise-ready agent reinforcement training from local GPUs to governed model lineage
- Visual: agent loop, GPU, W&B run, Weave trace, Registryの4要素を1枚で

Slide 2: Why RL for agents

- SFTだけでは「正解例の模倣」に寄る
- RLでは同じscenarioに複数試行させ、成功した振る舞いを増やす
- agentはtool useやmulti-step decisionがあるため、trajectory全体が学習単位になる

Slide 3: Course promise

- Write the loop
- See the loop
- Improve the loop
- Govern the loop

### Chapter 1 - ART Mental Model

Slide 4: Client/backend split

- Client: application/agent/environment side
- Backend: vLLM inference + trainer + checkpoints
- LocalBackend vs ServerlessBackendの比較

Slide 5: The one loop that matters

```python
train_groups = await art.gather_trajectory_groups(
    art.TrajectoryGroup(rollout(model, scenario) for _ in range(8))
    for scenario in batch_scenarios
)
result = await backend.train(model, train_groups, learning_rate=5e-6)
await model.log(train_groups, metrics=result.metrics, step=result.step, split="train")
```

Slide 6: What gets logged where

- `.art/<project>/models/<name>/history.jsonl`
- `.art/.../trajectories/<split>/<step>.parquet`
- W&B run metrics when `WANDB_API_KEY` is set
- LoRA checkpoint under `.art` for LocalBackend
- W&B Artifact for ServerlessBackend or explicit deploy/upload

### Chapter 2 - Environment and Enterprise Topology

Slide 7: Local GPU first

- `pip install openpipe-art[backend]`
- Python 3.11+
- CUDA-compatible GPU
- W&B service account key
- Weave project matching W&B project

Slide 8: Deployment options

- Multi-tenant Cloud: easiest, trial and lower-governance workflows
- Dedicated Cloud: single-tenant, W&B-managed, isolation, BYOB, IP allowlist/private connectivity, OIDC/LDAP
- Self-Managed / Customer Managed: customer runs W&B Server, best for strict on-prem or regulatory constraints

Slide 9: Data boundaries

- prompts, completions, traces, trajectories, checkpoints
- what can go to W&B
- what stays in customer VPC
- redaction and `strip_logprobs`

### Chapter 3 - Hands-on Task: Retail Customer Support Agent

Slide 10: Open-data task definition

- Source data: `lefft/tau-dev-task-retail-v1` and tau-bench/tau2-bench retail tasks
- Customer asks to cancel, return, exchange, change address, or inspect an order
- Agent must follow retail policy and use tools only when appropriate
- Output is a normal support conversation plus OpenAI tool calls

Slide 11: Scenario schema

```python
class RetailScenario(BaseModel):
    id: str
    user_request: str
    task_id: str | None = None
    expected_actions: list[str]
    forbidden_actions: list[str] = []
    policy_tags: list[str] = []
    split: Literal["train", "val", "test"]
```

Slide 12: Reward design

- verifier task success
- correct tool name and arguments
- policy compliance
- invalid or forbidden mutation penalty
- excessive turn/tool-call penalty
- RULER soft quality score for customer-facing explanation

### Chapter 4 - Writing ART Code

Slide 13: `TrainableModel`

- `name`, `project`, `entity`, `base_model`
- `base_path`, `report_metrics`
- `_internal_config` as advanced/unstable escape hatch

Slide 14: Inference

- `model.openai_client()`
- `model.get_inference_name()`
- `logprobs=True`, `top_logprobs=0` in rollout when backend supports it

Slide 15: Trajectory details

- `messages_and_choices`
- assistant `Choice` is trainable when logprobs exist
- `metrics`, `metadata`, `logs`
- `finish()` and `track_duration`

### Chapter 5 - W&B and Weave Observability

Slide 16: W&B run metrics

- `train/reward`
- `train/reward_std_dev`
- `train/exception_rate`
- `loss/train`, `loss/entropy`, `loss/kl_div`
- `data/step_num_trajectories`
- `time/step_trainer_s`
- `costs/*`

Slide 17: Weave trace tree

- `@weave.op()` on `rollout`, `call_tool`, `score_reward`, `evaluate`
- automatic OpenAI tracing
- link traces to W&B run
- artifact URL as trace attribute

Slide 18: Debugging pattern

- W&B chart shows reward collapse
- click step
- inspect Weave traces
- identify reward hacking or malformed JSON
- fork checkpoint before collapse
- select candidate steps from per-step metrics and validate them on held-out rollouts before promotion

### Chapter 6 - SFT Warmup

Slide 19: Why SFT before RL

- format contract
- tool call shape
- initial success rate
- reduced zero-reward phase

Slide 20: JSONL format

- `messages`
- optional `tools`
- last message must be assistant
- only assistant response tokens contribute to loss

Slide 21: Distillation

- teacher model creates structured target
- student learns task dialect
- W&B dataset artifact records source and generated examples

### Chapter 7 - GRPO-style Training

Slide 22: Grouped advantage

- same scenario
- multiple attempts
- reward mean/std within group
- relative learning signal

Slide 23: Reward variance

- if all rewards equal, no trainable advantage
- scenario diversity and group size matter

Slide 24: Core knobs

- `learning_rate`
- `scale_rewards`
- `advantage_balance`
- `epsilon`
- `importance_sampling_level`
- `kl_penalty_coef`

### Chapter 8 - RULER

Slide 25: RULER idea

- LLM judge ranks trajectories relative to each other
- GRPO only needs relative differences
- no labeled data required

Slide 26: Integration

```python
groups = await art.gather_trajectory_groups(
    (
        art.TrajectoryGroup(rollout(model, s) for _ in range(4))
        for s in scenarios
    ),
    after_each=lambda g: ruler_score_group(
        g,
        judge_model="openai/gpt-5.5",
        rubric=RUBRIC,
        extra_litellm_params={"reasoning_effort": "medium"},
        swallow_exceptions=True,
    ),
)
```

Slide 27: RULER risks

- judge cost
- inconsistent criteria
- common-prefix token cost
- additional histories not supported yet
- use `debug=True` during development

### Chapter 9 - GSPO and Advanced RL Settings

Slide 28: GRPO vs GSPO intuition

- token-level ratio vs sequence-level ratio
- MoE stability
- experimental API

Slide 29: Enabling GSPO

```python
result = await backend.train(
    model,
    train_groups,
    learning_rate=5e-6,
    importance_sampling_level="sequence",
)
```

Slide 30: rollout logprobs and stale-policy diagnostics

- rollout-time logprobs are requested with `logprobs=True, top_logprobs=0`
- current-policy logprob recomputation is backend/version dependent
- extra forward-pass cost
- useful for stale/off-policy diagnostics when the installed ART path exposes it

### Chapter 10 - Checkpoints, Registry, and Deployment

Slide 31: Local checkpoint lifecycle

- `.art` directory
- `checkpoint_path` from `LocalTrainResult`
- upload LoRA as W&B Artifact
- link to Registry collection

Slide 32: Serverless checkpoint lifecycle

- W&B Artifact automatically
- inference endpoint uses `wandb-artifact:///[entity]/[project]/[model]:stepN`

Slide 33: Governance

- registry collection per task
- aliases: `candidate`, `staging`, `production`, `rollback`
- artifact lineage and audit history
- restricted projects / service accounts

### Chapter 11 - Capstone

Slide 34: Capstone task

- improve retail holdout task success by at least N% over the prompted baseline
- produce W&B report with run, artifact, Weave eval, and trace examples
- promote best checkpoint to Registry with eval evidence

Slide 35: Review rubric

- code correctness
- reward quality
- observability quality
- governance readiness
- failure analysis

## 8. ハンズオン教材構成

推奨は `.py` scripts を主教材にし、notebookは短い可視化/探索用にする。エンタープライズ環境ではnotebookが許可されないことがあるため。

```text
course/
  00_setup/
    env_check.py
    wandb_weave_smoke.py
  01_art_primitives/
    retail_schema.py
    retail_tools.py
    dry_run_rollout.py
  02_weave_evals/
    make_weave_dataset.py
    eval_prompted_baseline.py
    scorers.py
    compare_checkpoints.py
  03_sft_warmup/
    download_tau_retail.py
    make_sft_jsonl.py
    train_sft_local.py
    inspect_sft_run.ipynb
  04_grpo_local/
    train_grpo_local.py
    evaluate_checkpoint.py
  05_ruler/
    train_with_ruler.py
    ruler_debug_examples.ipynb
  06_gspo_and_configs/
    config_matrix.py
    train_gspo_sequence.py
    compare_runs.ipynb
  07_models_registry_weave/
    upload_lora_to_wandb.py
    link_to_registry.py
    weave_model_wrapper.py
  08_enterprise_ops/
    fork_checkpoint.py
    delete_checkpoints.py
    deployment_topology.md
  shared/
    scenarios.py
    retail_env.py
    tools.py
    rewards.py
    tracing.py
    config.py
```

### Lab 00 - Setup and Observability Smoke Test

目的:

- GPU、CUDA、Python、ART、W&B、Weaveの疎通を確認。
- Dedicated Cloud/Self-Managedでは `WANDB_BASE_URL` と service account keyを使う。

内容:

- `python -m pip install "openpipe-art[backend]" weave wandb python-dotenv`
- `wandb login`
- `.env` の `WANDB_ENTITY`, `WANDB_PROJECT`, `WEAVE_PROJECT` を確認
- `course/00_setup/env_check.py` で環境変数と認証状態を確認
- `course/00_setup/wandb_weave_smoke.py` でW&B RunとWeave Traceの連携を確認
- `course/00_setup/art_api_smoke.py` でART API互換性を確認

期待成果:

- W&B Runが1つ作られる
- Weave Traceが1つ作られる
- Weave callの `wb_run_id` がW&B Runを指し、Run画面からtraceへ移動できる
- W&B Project内で、Run、Artifact、Traceを同じ実験単位として読める

### Lab 01 - ART Primitives Without Training

目的:

- `Scenario`, `rollout`, `Trajectory`, `TrajectoryGroup`, reward, metrics, metadataを理解。

内容:

- `lefft/tau-dev-task-retail-v1` から10件のretail conversationをロード
- `RetailScenario`, `RetailEnv`, `Trajectory`, `TrajectoryGroup` を作る
- prompted baseline modelでdry-run rollout
- `logprobs=True, top_logprobs=0` を要求
- `dense`, `tau_sparse`, `tau_irc` のreward profileを切り替えて、reference-path rewardとRetail Task Successの違いを見る
- `await model.log(groups, split="val")`

期待成果:

- W&Bに `val/reward`, `val/retail_task_success`, `val/policy_violation`, `val/invalid_tool_call` が出る
- Weaveにrollout/tool/reward traceが出る
- parquet trajectoryが `.art/.../trajectories/val/0000.parquet` に保存される

### Lab 02 - Weave Evals Baseline

目的:

- RL前に、評価dataset、scorer、model wrapper、leaderboardの基本形を作る。

内容:

- retail holdout taskを `weave.Dataset(name="tau-retail-holdout", rows=...)` としてpublish
- `weave.Model` または `@weave.op` wrapperでprompted baselineを評価
- scorerを実装: `retail_task_success`, `reference_tool_sequence_exact_match`, `policy_violation`, `correct_tool_name`, `correct_tool_args`, `turn_count`, `unsafe_mutation`
- `weave.Evaluation(dataset=dataset, scorers=[...])` を実行
- aggregate結果をW&B run summary/tableにも戻す

期待成果:

- Weave Evaluation UIでbaselineの失敗例を開ける
- W&B runに `eval/retail_task_success`, `eval/policy_violation`, `eval/avg_turns` が出る
- 以後のSFT/RL checkpointを同じdataset/scorersで比較できる

### Lab 03 - SFT Warmup

目的:

- SFTがRL前の形式学習として効くことを理解。
- next-action SFTをデフォルトにし、full-dialog SFTは比較用baselineとして扱う。

内容:

- `lefft/tau-dev-task-retail-v1` のtrain splitからSFT JSONLを生成
- `tools` 付きmulti-turn conversationを含め、OpenAI tool-calling wire formatを保持
- `make_next_action_sft_jsonl.py` でper-turn next-action rowsを生成
- `train_sft_local.py --sft-mask-mode last-assistant` をLocalBackendで実行
- public teacher / AReaL / success-trace rowsを混ぜる場合もsource metadataを保持してW&B Data Artifactへ入れる
- W&Bにloss curveを出す

注意:

- SFT JSONLは最後のmessageがassistantである必要がある。
- assistant response tokenのみloss対象で、system/userはmaskされる。
- next-action rowsでは過去assistant turnsはcontextであり、loss対象は最後のassistant actionだけにする。
- full-dialog SFTはtool schema/wire formatの説明には便利だが、RL parentとしては過去turnを過剰に模倣しやすい。

期待成果:

- baselineよりJSON validityが上がる
- Registryに `sft-warmup-candidate` として任意登録できる

### Lab 04 - GRPO-style Local RL

目的:

- same scenarioに対する複数rolloutからgroup-relative signalを作る。

内容:

- 1 stepあたり `num_scenarios=8`, `rollouts_per_scenario=4 or 8`
- main pathは `--reward-profile tau_irc --continue-on-invalid` で学習
- `backend.train(..., learning_rate=5e-6)` 実行
- `model.log(..., split="train")`

観察:

- `train/reward`
- `train/reward_std_dev`
- `data/step_num_groups_trainable`
- `loss/train`
- `time/step_trainer_s`

設計ディスカッション:

- rewardが全部同じgroupはなぜ学習されないか
- group sizeを4/8/16でどう選ぶか
- replay exact rewardだけだと何を見落とすか
- `tau_sparse` と `tau_irc` を比較し、shapingがwinner-loser gapを本当に広げているかを見る

Reward profile matrix:

| Profile | Primary use | What it teaches | Caveat |
| --- | --- | --- | --- |
| `dense` | early mechanics | reference tool names, arguments, final text similarity | replay-style reward for learning mechanics |
| `strict_success` | failure demo | exact reference trajectory success/failure | too brittle for agentic RL |
| `agentic` | transition profile | state-changing action and final communication diagnostics | still tied to replay data |
| `tau_sparse` | tau-style baseline | sparse outcome / communication anchor | lower reward density |
| `tau_irc` | main RL lab | calibrated outcome + state-action shaping | must be validated against held-out eval |
| optional official tau2 runtime | benchmark-grade extension | final database state and task-specific assertions pass in the official simulator | requires separate tau2 runtime |

### Lab 05 - RULER Hybrid Reward

目的:

- hand-crafted rewardとRULERを組み合わせる。

内容:

- `ruler_score_group` を `after_each` で利用
- custom rubricを作成
- original rewardを `independent_reward` として保存
- judge scoreをauxiliary metric, tie-breaker, or risk demoとして扱う
- judge costをART metricsで記録

Weave:

- RULER judge callをtrace
- judge explanationをtrajectory.logsに残す
- failed judge groupを `swallow_exceptions=True` でskip

注意:

- RULERは現時点で `additional_histories` を含むtrajectoryをサポートしない。
- 開発初期は `debug=True` でjudge reasoningを確認する。
- tau-style trainingではverifiable outcome/state-changing-action rewardをanchorにする。judge rewardでDB、NL assertion、communicationなどの検証可能信号を置き換えない。

### Lab 06 - GSPO and Config Matrix

目的:

- advanced knobsを「表で覚える」ではなく、実験として理解する。

実験:

- A: default token-level importance sampling
- B: `importance_sampling_level="sequence"` (GSPO)
- C: `importance_sampling_level="geometric_average"`
- D: `kl_penalty_coef=0.01`
- E: rollout logprob / stale-policy diagnostic, with `precalculate_logprobs` only when supported by the installed ART/backend path

比較:

- reward slope
- KL / entropy
- trainable token count
- wall-clock
- instability cases

受講者への問い:

- dense 7B/14BではGSPOの差が見えるか
- MoEのQwen3 30B A3Bでは何が変わるか
- sequence-level objectiveは長い回答やtool-heavy trajectoryにどう効くか

### Lab 07 - W&B Models, Artifacts, Registry, Weave Model

目的:

- LocalBackendで作ったLoRAをW&B governance workflowに乗せる。

内容:

- `LocalTrainResult.checkpoint_path` を使う
- `wandb.Artifact(type="model" or "lora")` でcheckpoint dirをlog
- Registry `Model` / collection `retail-support-agent` にlink
- alias運用: `candidate`, `best-val`, `staging`
- Weave traceに `wandb-artifact:///<entity>/<project>/<artifact>:<version>` をattributeとして付ける
- `weave.Model` wrapperでartifact + prompt config + inference adapterをversioning

期待成果:

- W&B Registry collectionにモデル候補が見える
- artifact lineageがrunとつながる
- Weave traceからartifactへclickable linkができる

### Lab 08 - Enterprise Ops

目的:

- 長時間学習、失敗、ロールバック、コスト、権限を扱う。

内容:

- `delete_checkpoints(best_checkpoint_metric="val/reward")`
- `_experimental_fork_checkpoint` でcollapse前に戻る
- `gpu_cost_per_hour_usd` をLocalBackendに設定
- service account keyとrestricted project運用
- Dedicated Cloud/Self-Managedでの `WANDB_BASE_URL`
- ServerlessBackendの比較デモ

Serverless補足:

- `ServerlessBackend` はW&B API keyで利用
- LoRA checkpointはW&B Artifactsへ自動保存
- trained model endpointは `wandb-artifact:///[ENTITY]/[PROJECT]/[MODEL-NAME]:[STEP]`
- 2026-05-29時点のW&B docsではServerless RLのtraining対応モデルに `OpenPipe/Qwen3-14B-Instruct` と `Qwen/Qwen3-30B-A3B-Instruct-2507` が掲載されている

## 9. 設定項目の教材化方針

### Public-facing first

最初に教える値:

| 項目 | 役割 | 初期教材での扱い |
| --- | --- | --- |
| `learning_rate` | LoRA更新の強さ | まず `5e-6`、SFTは別 |
| `scale_rewards` | group内reward stdで正規化 | 初期はTrue |
| `advantage_balance` | 正/負advantageの学習バランス | advanced |
| `importance_sampling_level` | token/sequence単位のIS | GSPO章で扱う |
| `epsilon`, `epsilon_high` | clipping範囲 | PPO/IS章で扱う |
| `kl_penalty_coef` | referenceから逸脱しすぎるtokenのadvantage補正 | 安定化章 |
| rollout logprobs / `precalculate_logprobs` | rollout時logprobsの取得と、対応backendでのcurrent policy logprobs再計算 | off-policy/stale logprobs章 |
| `allow_training_without_logprobs` | logprobsなしassistant dictも訓練対象にする | 注意付きadvanced |
| `packed_sequence_length` | packing後のsequence長 | VRAM/long context章 |
| `logprob_calculation_chunk_size` | logprob計算chunk | memory tuning章 |
| `save_checkpoint` | LocalBackend checkpoint保存 | ops章 |

### rollout logprobs / `precalculate_logprobs` の説明

教材での言い方:

- ARTのRLでは、assistant出力のtoken logprobsが重要。
- `Choice` にlogprobsがある場合、tokenizerがそれをold policy logprobsとして使う。
- LocalBackendでは、logprobsが全く無く、`allow_training_without_logprobs=False` の場合、学習対象データなしとしてskipされることがある。
- `precalculate_logprobs` は、対応しているART/backend pathで学習直前に現在のpolicy logprobsを再計算するadvanced設定として扱う。利用前にpinしたARTバージョンのdocs/source/smoke testで確認する。
- 追加forward passが必要なので、最初のlabでは使わない。
- off-policy data、stale logprobs、truncated importance samplingの実験、特定のstability issueの調査で扱う。

受講者に植え付けるルール:

- live rolloutでは、可能なら `logprobs=True, top_logprobs=0` を要求する。
- Weaveに巨大なlogprobsを載せすぎないため、公式例のように `strip_logprobs` をglobal postprocessに入れる。
- `allow_training_without_logprobs=True` は、importance samplingを使わない簡易設定としてadvancedで扱う。

### GRPO/GSPO/PPO/CISPOの表現

ART docsは大きな学習ループをGRPOとして説明している。一方、現行LocalBackendの実装APIでは `loss_fn="cispo"` がdefaultで、`loss_fn="ppo"` も選べる。教材では次のように整理する。

- 初学者向け: 「同じscenarioの複数trajectoryの相対rewardで学習するGRPO-style loop」
- 実装詳細: LocalBackend defaultはCISPO-style lossで、PPO clipping pathもある
- GSPO: `importance_sampling_level="sequence"` によるsequence-level importance ratio
- 研究者/advanced向け: loss実装とimportance samplingの違いをコードで読む

## 10. W&B Models / Weave 連携設計

### W&B Models

教材で扱う責務:

- experiment tracking: ARTのmetrics rowをW&B runに記録
- artifacts: dataset, SFT JSONL, LoRA checkpoint, evaluation table
- registry: organization-wide model lifecycle
- lineage: dataset -> SFT -> RL -> evaluation -> promoted checkpoint

LocalBackend時:

- ARTは `.art` にcheckpointを保存する。
- `WANDB_API_KEY` があると `model.log` 経由でW&B run metricsが記録される。
- LoRA checkpointをW&B Artifact/Registryに乗せる処理は教材側で明示的に実装する。

ServerlessBackend時:

- checkpointsはW&B Artifactsとして自動保存される。
- inference endpointは `wandb-artifact://` schemaで参照する。

### Weave

教材で扱う責務:

- rollout trace
- tool call trace
- reward function trace
- RULER judge trace
- evaluation trace
- artifact version attribute

推奨instrumentation:

```python
import weave
from art.utils.strip_logprobs import strip_logprobs

weave.init(
    f"{ENTITY}/{PROJECT}",
    global_postprocess_output=strip_logprobs,
)

@weave.op()
async def rollout(model_name: str, scenario_id: str) -> dict:
    ...
```

W&B runとの関連付け:

- `wandb.init()` context内で `@weave.op()` を呼ぶと、Weaveはactive runを検出できる。
- 学習runや評価runでは `set_wandb_run_context(run_id=run.id)` を明示し、traceを特定のW&B Runに関連付ける。
- Run名とRun IDはW&Bに生成させる。検索・比較・再現に必要な情報は `tags`、`notes`、`config`、Artifact lineageに入れる。
- ARTの `model._get_wandb_run()` は内部APIなので教材の一般コードでは直接依存しない。
- `step` を渡すと特定stepのtraceとして見られる。step未指定でもRun単位の関連付けは残る。

## 11. Enterprise deployment guidance

### Multi-tenant Cloud

用途:

- PoC
- 社内制約が軽い検証
- 受講者個人のハンズオン

教材での扱い:

- 一番簡単な動作確認パス
- ただし本コースの主役ではない

### Dedicated Cloud

用途:

- single-tenant W&B platform
- W&B-managedで運用負荷を抑えたい
- データレジデンシ、ネットワーク分離、SSO、audit、restricted projectsが必要

教材での扱い:

- enterprise reference topologyとして詳しく扱う
- `WANDB_BASE_URL`, service account, secure storage connector, IP allowlisting/private connectivity, OIDC/LDAPを設計図に入れる

### Self-Managed / Customer Managed

用途:

- on-prem
- 規制上Dedicated Cloudでも満たせない要件
- 顧客がW&B Serverを運用する体制がある

教材での扱い:

- platform team向けappendix
- Helm/operator、アップグレード、Registry/Weave対応version、object storage、backup/retentionのチェックリストを置く

## 12. Capstone rubric

受講者は最後に次を提出する。

1. ART training script
2. SFT dataset artifact
3. RL run with at least 3 training steps
4. W&B workspace screenshot or report URL
5. Weave trace examples: success, failure, reward-hacking-like case
6. Best checkpoint artifact linked to Registry collection
7. Short architecture memo for enterprise deployment

採点観点:

| 観点 | 満点条件 |
| --- | --- |
| ART correctness | trajectory group, reward, backend.train, model.logが正しく接続されている |
| Reward design | exact rewardとRULER/soft rewardの役割が分離されている |
| Observability | W&B metricsとWeave tracesから失敗原因を追える |
| Governance | artifact, registry, aliases, provenanceが説明できる |
| Engineering judgement | configをむやみに触らず、実験仮説とセットで変更している |

## 13. 作成する教材成果物

最終的なdeliverables:

- `slides/openpipe-art-wandb-weave.md`
  - Claude Design入力用
  - Chapterごとに `visual`, `layout`, `speaker_notes`, `demo_checkpoint` を含める
- `labs/`
  - `.py` scriptsを主軸
  - notebookは観察・比較だけ
- `data/retail/`
  - downloaded tau retail source metadata
  - SFT JSONL derived from `lefft/tau-dev-task-retail-v1`
  - Weave eval holdout rows
  - optional tau2 retail task cache
- `configs/`
  - local tiny
  - local 7B
  - local 14B/30B
  - serverless demo
- `docs/`
  - enterprise topology
  - config glossary
  - troubleshooting guide
  - API drift notes

## 14. 次の実装順

1. `README.md` を作る
   - course overview
   - prerequisites
   - setup matrix
2. `shared/` の題材コードを作る
   - `Scenario`
   - deterministic tools
   - reward functions
   - Weave tracing helpers
3. Lab 00/01を作る
   - GPU不要のdry runとW&B/Weave smoke
4. Lab 02 Weave Evalsを作る
   - holdout dataset
   - baseline evaluation
   - custom scorers
5. Lab 03 SFTを作る
   - tau retail JSONL conversion
   - local GPU script
6. Lab 04 GRPOを作る
   - short run
   - W&B metrics
7. Lab 05 RULERを作る
   - judge providerをenvで差し替え
   - cost tracking
8. Lab 06 advanced config
   - config matrix runner
9. Lab 07 Registry/Weave Model
   - local checkpoint upload
   - registry link
10. Claude Design slide Markdown
   - このblueprintをslide-nativeな形式に変換

## 15. トラブルシューティング章に入れるべき内容

- `There are no assistant logprobs to train on`
  - `Choice` をtrajectoryにappendしているか
  - completionで `logprobs=True` を要求したか
  - `allow_training_without_logprobs` を使うべき状況か
- rewardが上がらない
  - group内reward varianceがあるか
  - scenarioが難しすぎないか
  - open-source modelが初期で30%程度成功できるか
  - rewardが形式だけ見ていないか
- RULERが変
  - system promptが曖昧
  - rubricが絶対評価に寄りすぎ
  - group sizeが大きすぎる
  - judge modelが弱い
- W&B metricsが出ない
  - `WANDB_API_KEY`
  - `report_metrics`
  - `model.log` 呼び忘れ
  - Dedicated/Self-Managedの `WANDB_BASE_URL`
- Weave traceがrunに紐づかない
  - `weave.init` projectとW&B projectの一致
  - `wandb.init` context内でopを呼んだか
  - `set_wandb_run_context(run_id=wandb.run.id)` が呼ばれているか
- checkpointが増えすぎる
  - `delete_checkpoints`
  - best metricがhistoryに存在するか
  - Registryに昇格済みか
- Qwen3 multi-turnでthinking tokenが消える
  - `additional_histories`
  - RULERはadditional histories未対応なので評価設計を分ける

## 16. このコースの雰囲気

エンタープライズ向けなので、教材は「派手なRLデモ」よりも「静かに強い実務基盤」に寄せる。

- 受講者に見せる画面は、W&B run chart、Weave trace、Registry lineageを中心にする。
- すべてのコードに「どのログを見れば成功か」を添える。
- 各configは「defaultから変える理由」がある時だけ登場させる。
- Serverlessは魔法の近道としてではなく、Local/Dedicated/Self-Managedと同じART abstractionの別backendとして扱う。
- 最後は「このモデルはどのデータで、どのrewardで、どのcheckpointから、誰が昇格し、どのtraceで失敗分析できるか」を説明できる状態にする。
