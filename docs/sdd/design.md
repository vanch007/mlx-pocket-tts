# MLX Pocket TTS implementation design v1

The canonical design is
`docs/spark/2026-09-01-full-mlx-parity-design.md`. This file maps requirements to components.

| Component | Requirements | Location |
|---|---|---|
| Compatibility API and defaults | REQ-001, REQ-003 | `src/mlx_pocket_tts/__init__.py`, `defaults.py`, `loader.py` |
| CLI | REQ-002, REQ-003 | `src/mlx_pocket_tts/cli.py` |
| Web/service | REQ-004 | `src/mlx_pocket_tts/server.py`, `web/` |
| MLX training data | REQ-005 | `src/mlx_pocket_tts/training/data/` |
| MLX training engine | REQ-006 | `src/mlx_pocket_tts/training/` |
| Distillation | REQ-007 | `src/mlx_pocket_tts/training/distillation.py` |
| Legacy importer | REQ-008 | `scripts/import_pytorch_checkpoint.py` |
| Evidence and docs | REQ-009 | `reports/`, `README.md`, `docs/` |

All user entry points call one MLX inference core. Native checkpoints contain model, optimizer,
EMA, step, RNG, data cursor, and resolved config; inference exports contain safetensors plus config.
Errors fail early with the exact config, record, key, shape, or resource that caused the failure.
