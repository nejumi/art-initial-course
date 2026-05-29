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
- Promote only evaluated checkpoints to Registry.
- Prune local checkpoints after Registry promotion and rollback verification.
