# TASK-010 isolated legacy checkpoint import evidence

Status: **pass**.

The public `checkpoint` option now launches an isolated importer subprocess. Only that module
imports PyTorch and it uses `torch.load(..., weights_only=True)`. It overlays EMA parameters on
the raw training state, removes training-only objective tensors, converts BF16 at the boundary,
and emits a safe NumPy container. The MLX runtime validates names and shapes, overlays the state
on the config model, and materializes a complete safetensors inference artifact. The same artifact
can then enter the normal 8-bit quantization path.

An official-layout full FlowLM fixture with 127 tensors passed strict import and was equal to the
source model parameter by parameter. Its generated sentence had WER 0.0. The combined
checkpoint-plus-8-bit path also strictly loaded and generated finite audio. The temporary `.pt`
fixture was removed automatically; no PyTorch model was retained.

Static isolation proves that no production or native-training module imports `torch`; only
`legacy_importer.py` contains that dependency boundary.

