# TASK-009 native distillation evidence

Status: **pass** for the required CFG and depth smoke paths and inference exports.

The MLX trainer now supports the official guidance-baking target
`z_null + cfg * (z_conditioned - z_null)`, frozen teachers outside the optimizer/checkpoint tree,
and frozen student flow/EOS/objective heads. Depth distillation seeds a shallow student from the
bottom and top of a deeper teacher; the 24→6 test selected layers 0, 1, 2, 21, 22, and 23.

On M3 Max, CFG=3 distillation completed a real update in 0.176 seconds and its direct inference
export produced a WER 0.0 sample. German 24→6 depth distillation completed a real update in 0.104
seconds and its export strictly loaded and generated finite audio.

The single-step depth sample had WER 1.0. This is expected to be an initialization/smoke artifact,
not a converged distilled model, and is explicitly **fail** for release quality. The task acceptance
is implementation smoke plus inference-loadable export; no quality-parity claim is made.

