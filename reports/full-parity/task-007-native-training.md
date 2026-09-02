# TASK-007 native MLX training evidence

Status: **pass**.

The native runtime implements the official conditioning sequence, CFG text/voice dropout,
teacher-forced audio, EOS loss, latent-stat EMA, OT Flow Matching, and the released-model LSD
objective. LSD includes both the diagonal term and the JVP-based self-distillation term with
classic or input-gradient-only stop-gradient behavior. AdamW, gradient accumulation, finite
gradient guard, global-norm clipping, weight EMA, atomic checkpoint retention, and full
model/optimizer/EMA resume are covered by tests.

## M3 Max gates

- Real pretrained FlowLM plus real Mimi latents: one full LSD update passed in 0.7464 seconds;
  loss 1.2404, unclipped gradient norm 207.384, peak process RSS 2.116 GB.
- Atomic model/AdamW/EMA save took 1.1517 seconds; a new load restored step 1 in 0.1021 seconds.
- Interrupted overfit: save at step 10, reconstruct all objects, restore identical weights, and
  continue to optimizer step 30 passed.
- Fixed-seed evaluation loss fell from 0.7665 to 0.4659 (39.2%); first-five versus last-five
  mean train loss fell from 1.8110 to 0.3637.
- Unit/static gates: 29 tests pass; Ruff lint and format pass. The sole warning is an existing
  Starlette TestClient/httpx deprecation and does not affect training.

Evidence: `task-007-real-step.json`, `task-007-overfit.json`, and the retained local checkpoints
under `outputs/full-parity/training/`.

