# MLX Pocket TTS requirements v1

Authority: `docs/spark/2026-09-01-full-mlx-parity-design.md`.

## Functional requirements

- REQ-001: The package shall expose the official public Python API. Acceptance: the mapped
  upstream API tests pass with MLX-native return values.
- REQ-002: The CLI shall reproduce official `generate`, `serve`, and `export-voice` behavior.
  Acceptance: every upstream CLI behavior has a passing local test.
- REQ-003: The runtime shall load official languages, configs, revisions, checkpoints, and MLX
  quantized artifacts. Acceptance: strict load/audit and real generation pass for every published
  12/24-layer artifact.
- REQ-004: The service shall provide the official-purpose Web UI and streaming HTTP API.
  Acceptance: UI and endpoint tests plus one real browser/server replay pass.
- REQ-005: The project shall implement data preparation and latent caching natively for MLX.
  Acceptance: cache reuse and invalidation tests pass on an authorized tiny dataset.
- REQ-006: The project shall train and resume Pocket TTS natively with MLX, including EMA.
  Acceptance: tiny-set overfit, interrupted resume, EMA reload, and inference export pass.
- REQ-007: The training runtime shall implement depth and CFG distillation. Acceptance: both
  distillation smoke runs complete and export inference-loadable artifacts.
- REQ-008: The project shall import legacy PyTorch checkpoints offline without a PyTorch runtime
  dependency. Acceptance: importer fixture converts and strict MLX load passes.
- REQ-009: Documentation shall match observed capability evidence. Acceptance: README contains the
  approved compatibility matrix, workflows, measurements, and explicit non-applicable features.

## Non-functional requirements

- NFR-001: Runtime and native training imports shall succeed with PyTorch absent.
- NFR-002: All target behavior shall run on Apple Silicon macOS; CUDA/DDP is `not applicable`.
- NFR-003: Every release claim shall be backed by `pass/fail/pending/not applicable` evidence.
- NFR-004: Model, audio, training data, secrets, and generated weights shall remain outside Git.
- NFR-005: Checkpoints shall be atomically written and diagnosable after a failed run.

## Users and deployment

The users are Apple Silicon developers and local TTS operators. The stack is Python, MLX,
FastAPI, the official static Web UI, safetensors, Hugging Face Hub, pytest, and Ruff, with no database. The deployment
target is local Apple Silicon macOS. Codex is the active development agent.

## Out of scope

Linux, Windows, x86, CUDA, NCCL/DDP, Docker/cloud swarm execution, browser/WASM ports, and
third-party integrations are outside the runnable parity target.
