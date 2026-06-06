# Enterprise Deployment Topology

Primary course assumption:

- Training runs on local or customer-managed GPUs with `LocalBackend`.
- Metrics, artifacts, eval evidence, and registry lifecycle live in W&B.
- Weave stores traces and evaluations for rollout-level debugging.

Environment variables:

```bash
export WANDB_BASE_URL=https://<dedicated-or-self-managed-host>
export WANDB_ENTITY=<team-entity>
export WANDB_PROJECT=openpipe-art-retail
export WANDB_API_KEY=<service-account-key>
```

Operational checklist:

- Use team entities and service accounts from the beginning.
- Keep raw customer payloads out of public projects.
- Attach dataset artifact, reward version, scorer version, and Weave eval URL to every model artifact.
- Log local ART LoRA checkpoint directories as W&B `model` artifacts; keep large base model weights external and reference them by `base_model` metadata.
- Call `use_artifact` for the dataset and parent checkpoint before each SFT/RL/eval stage so W&B shows SFT -> GRPO/GSPO/RULER lineage.
- Promote only evaluated checkpoints to Registry.
- Prune local checkpoints after Registry promotion and rollback verification.
