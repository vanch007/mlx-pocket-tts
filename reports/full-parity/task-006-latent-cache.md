# TASK-006 native MLX training data and latent cache

Status: `pass`

- Strict JSONL schema reports manifest line and exact invalid field.
- Relative audio paths resolve against the manifest.
- Cache keys include audio bytes, window, transcript, alignments and Mimi encoder fingerprint.
- Latent safetensors and manifests are atomically replaced.
- Unit fixture proved first encode, full reuse, and invalidation after audio bytes changed.
- Real gated Mimi encoder on M3 Max encoded two 24 kHz files in 0.0817 s.
- Immediate replay encoded zero files and reused both entries in 0.0342 s.
- Encoder fingerprint: `17266fc420afbd3d22279e9b62672fe1fc66b3711b0d955b62cf86d6e9fc65c8`.

Evidence: `task-006-latent-cache.json`, the annotated manifest, metadata and per-entry
safetensors under `outputs/full-parity/training/cache/`.
