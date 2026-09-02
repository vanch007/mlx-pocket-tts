# MLX Pocket TTS implementation tasks v1

## Phase 1 — official inference surface

- [x] **TASK-001** [REQ-001, NFR-001] Implement and test official Python API compatibility.
  Verify: `uv run pytest tests/test_public_api.py -q`.
- [x] **TASK-002** [REQ-003] Implement language/config/quantized model resolution.
  Verify: loader unit tests and strict audits.
- [x] **TASK-003** [REQ-002] Implement official `generate` CLI behavior except the legacy
  checkpoint option, which is completed by TASK-010.
  Verify: mapped CLI tests and one real generation.
- [x] **TASK-004** [REQ-002] Implement file/directory `export-voice` compatibility.
  Verify: export/import fixtures and real reference audio.
- [x] **TASK-005** [REQ-004] Implement official HTTP semantics and Web UI.
  Verify: endpoint tests and real browser replay.

## Phase 2 — native training foundation

- [x] **TASK-006** [REQ-005, NFR-004] Implement manifest validation and latent cache.
  Verify: cache reuse/invalidation fixtures.
- [x] **TASK-007** [REQ-006, NFR-005] Implement MLX loss, optimizer, EMA, atomic checkpoint, resume.
  Verify: tiny overfit and interrupted-resume tests.
- [x] **TASK-008** [REQ-006] Export a native checkpoint for direct inference.
  Verify: strict load and real generated WAV.

## Phase 3 — advanced training and release

- [x] **TASK-009** [REQ-007] Implement depth and CFG distillation.
  Verify: two smoke runs and inference export.
- [x] **TASK-010** [REQ-003, REQ-008, NFR-001] Implement isolated legacy checkpoint importer
  and connect the official `checkpoint` loading option.
  Verify: conversion fixture and no-PyTorch isolation test.
- [x] **TASK-011** [REQ-009, NFR-003] Run full M3 Max matrix and publish evidence-backed docs.
  Verify: JSON/Markdown/listening reports and README audit.
