# ART Backend Scaling Notes

This appendix is intentionally provider-neutral. It applies to local workstations,
customer-managed GPU boxes, and single-node cloud GPU instances.

## Recommended Course Positioning

Use three levels in the course:

1. **Intro hands-on:** `LocalBackend` shared mode on one GPU.
2. **Enterprise single-node scale:** `LocalBackend` dedicated mode with separate
   training and inference GPUs.
3. **Advanced appendix:** `MegatronBackend` for larger dense/MoE models and
   multi-GPU training on one node.

Multi-node training should be discussed as architecture, not as the default
hands-on path. In the current ART source inspected for this course, the
Megatron runtime launches `torch.distributed.run --nproc_per_node ...`, so the
safe public lab target is single-node multi-GPU.

## Backend Selection Matrix

| Scenario | Recommended backend | Why |
| --- | --- | --- |
| First successful RL run | `LocalBackend` shared mode | Smallest moving parts; easiest to debug. |
| SFT warmup | `LocalBackend` shared mode | SFT is supported and straightforward. |
| Rollout while training | `LocalBackend` dedicated mode + `PipelineTrainer` | Keeps vLLM inference on a separate GPU while training updates LoRA. |
| Larger dense model on one machine | `MegatronBackend` | Uses Megatron distributed runtime across local GPUs. |
| MoE / Qwen3 A3B-style experiments | `MegatronBackend` or Serverless demo | Megatron has explicit MoE support paths; Serverless is simpler operationally. |
| Strict enterprise governance | Local/customer-managed GPU + W&B Dedicated/Self-Managed | Data and checkpoint movement remain explicit. |
| Multi-node training | Architecture discussion / future advanced module | Do not promise as an introductory lab unless validated in the target environment. |

## LocalBackend Shared Mode

This is the default for the course.

```python
from art.local import LocalBackend

backend = LocalBackend(path="./.art")
```

Use it when:

- the model fits on one GPU;
- training pauses during inference are acceptable;
- you want the simplest SFT and GRPO loop;
- students are learning `Trajectory`, `TrajectoryGroup`, `backend.train`, and
  `model.log` for the first time.

Caveats:

- shared mode can pause inference during training;
- it is not the right mode for asynchronous pipeline training;
- it is the right default for the course's first Retail labs.

For a real workshop environment, verify the rollout-to-train handoff before the
full lab. The health probe uses a separate model name by default, runs a tiny
rollout, optionally forces a non-zero reward gap, and prints GPU memory plus the
vLLM sleep state if the installed ART version exposes it:

```bash
python course/08_enterprise_ops/localbackend_health_probe.py \
  --data-dir data/retail_bridge_state1 \
  --split train \
  --rollouts 2 \
  --max-turns 8 \
  --max-completion-tokens 256
```

If this probe shows that vLLM is not sleeping around training, or GPU memory
does not drop enough for the trainer, reduce vLLM memory settings first:

```bash
export ART_VLLM_GPU_MEMORY_UTILIZATION=0.70
export ART_VLLM_MAX_MODEL_LEN=16384
export ART_VLLM_MAX_NUM_BATCHED_TOKENS=16384
export ART_VLLM_MAX_NUM_SEQS=8
```

## LocalBackend Dedicated Mode

Dedicated mode separates training GPUs from vLLM inference GPUs.

```python
import art
from art.local import LocalBackend

backend = LocalBackend(path="./.art")
model = art.TrainableModel(
    name="retail-support-agent-dedicated",
    project="openpipe-art-retail",
    base_model="OpenPipe/Qwen3-14B-Instruct",
    _internal_config=art.dev.InternalModelConfig(
        trainer_gpu_ids=[0],
        inference_gpu_ids=[1],
    ),
)
await model.register(backend)
```

Use it when:

- you have at least two GPUs;
- you want rollouts/evals to keep running while training updates happen;
- you introduce `PipelineTrainer` and asynchronous actor/trainer patterns;
- you need a more production-like agent training loop.

Current constraints to teach explicitly:

- `trainer_gpu_ids` and `inference_gpu_ids` must both be set or both unset;
- the two sets must not overlap;
- ART currently requires exactly one inference GPU for this path;
- trainer GPUs must be contiguous starting from 0 in the process view;
- `PipelineTrainer + LocalBackend` requires dedicated mode;
- dedicated-mode SFT is not the safe intro path, so run SFT in shared mode first.

## PipelineTrainer

`PipelineTrainer` is not necessary for the first hands-on. It becomes useful when
rollout collection and training should be pipelined instead of run as a strict
`gather -> train -> log` loop.

Course use:

- mention in the ART specification section;
- add a short demo or code reading exercise in enterprise ops;
- avoid making it a required lab unless all participants have multi-GPU access.

## MegatronBackend

`MegatronBackend` subclasses `LocalBackend` and swaps the training service to a
Megatron distributed runtime. It requires explicit sequence packing settings and
additional dependencies.

```python
import art
from art.megatron import MegatronBackend

backend = MegatronBackend(path="./.art")
model = art.TrainableModel(
    name="retail-support-agent-megatron",
    project="openpipe-art-retail",
    base_model="OpenPipe/Qwen3-14B-Instruct",
)
await model.register(backend)

result = await backend.train(
    model,
    trajectory_groups,
    learning_rate=5e-6,
    packed_sequence_length=8192,
)
await model.log(trajectory_groups, metrics=result.metrics, step=result.step, split="train")
```

Use it when:

- a single-GPU Unsloth path is too small or unstable;
- you want to use multiple GPUs on one node for the trainer;
- you are experimenting with larger dense or MoE models;
- explicit packing and distributed runtime setup are acceptable complexity.

Do not use it for the first Retail lab because:

- dependencies are heavier: install with ART backend plus Megatron extras;
- `packed_sequence_length` must be chosen deliberately;
- distributed logs and failures add noise before students understand ART's core
  abstractions.

## Torchtune Note

The current public ART docs mention that `LocalBackend` can run Unsloth or
torchtune. In the source inspected for this course, the actively visible local
training service path is Unsloth, with Megatron available as a separate backend.
For course material, avoid building a hands-on around Torchtune unless the target
ART version exposes and validates that path directly.

## How To Cover Multi-GPU Without Overfitting To One Cluster

Keep the course generic:

- show GPU allocation as an environment concern, not ART-specific code;
- teach that ART sees logical GPU IDs after the scheduler/container sets
  `CUDA_VISIBLE_DEVICES`;
- show dedicated mode with `[0]` for training and `[1]` for inference;
- show Megatron as single-node multi-GPU advanced material;
- keep Slurm, Kubernetes, Ray, or cloud-specific launchers in optional deployment
  notes, not the core labs.

## Suggested Slide Framing

- Slide: "One ART loop, three execution modes"
- Slide: "Shared LocalBackend: simplest path"
- Slide: "Dedicated LocalBackend: separate actor and trainer resources"
- Slide: "MegatronBackend: when the trainer needs the whole node"
- Slide: "Multi-node is an infrastructure project, not a beginner lab"
