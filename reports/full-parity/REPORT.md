# MLX Pocket TTS full-parity release report

## Verdict

Status: **pass** for the approved Apple Silicon parity scope.

The pinned official inference/API/CLI/config/Web surface is implemented, including the legacy
training-checkpoint option. Native MLX training now covers data caching, official conditioning,
EOS, LSD and OT flow matching, AdamW, gradient accumulation/clipping, EMA, atomic resume, direct
inference export, CFG guidance baking and 24→6 depth distillation.

## Provenance and target

- Official source: `kyutai-labs/pocket-tts@e244e9a0dad520dad8f61a3c5882e91ea4859439`
- Official gated model tested: `kyutai/pocket-tts@492522650173a0653b7575cdc25ae09810e5d741`
- Baseline community MLX revision: `cbf71d5f6657bbc3f4bc02f85ee408261225bec7`
- Machine: Apple M3 Max, 128 GB unified memory, MLX Metal device
- Runnable target: Apple Silicon macOS; CUDA/DDP/Linux/Windows are not applicable

## Full acceptance matrix

| Requirement | Status | Evidence |
|---|---|---|
| Official Python API | pass | `task-001-api-smoke.*` |
| Local/HF/HTTPS configs and 8-bit | pass | `task-002-*` |
| Generate CLI and stdout/stdin | pass | `task-003-*` |
| Voice file/directory export and reuse | pass | `task-004-*` |
| FastAPI and official-purpose Web UI | pass | `task-005-*`; real in-app browser replay |
| Manifest validation and Mimi cache | pass | `task-006-*` |
| Native loss/optimizer/EMA/resume | pass | `task-007-*` |
| Direct inference export | pass | `task-008-*`; strict 214-tensor audit, WER 0.0 |
| CFG and depth distillation | pass (smoke) | `task-009-*` |
| Isolated legacy `.pt` importer | pass | `task-010-*`; WER 0.0, checkpoint+8-bit |
| Final unit/static gates | pass | 33 pytest; Ruff lint/format; `git diff --check` |
| Runtime without PyTorch/`mlx_audio` | pass | neither module is imported by inference/training |

## Real-device training measurements

- Full LSD update on real pretrained weights/real Mimi latents: 0.746 seconds, peak RSS 2.116 GB.
- Interrupted overfit restored identical weights at step 10 and continued to step 30; fixed-seed
  evaluation loss fell 39.2%.
- Step-1 EMA export: strict load, RTF 0.098 and ASR WER 0.0.
- CFG=3 distillation: update 0.176 seconds, strict export and WER 0.0.
- German 24→6 depth smoke: update 0.104 seconds, peak RSS 2.077 GB, strict export.
- Full legacy `.pt` fixture: 127 FlowLM tensors, parameter equality, WER 0.0; combined 8-bit path
  also generated finite audio.

## Browser evidence

The real 8-bit server was opened in the Codex in-app browser. The official controls rendered,
text submission returned HTTP 200, and the UI displayed a playable two-second audio element plus
a download link. It reported 0.18 seconds to first audio, 0.26 seconds total and 10.2× real time.

## Known failures and limits

- The one-step depth-distillation artifact has ASR WER 1.0. It is only an architecture smoke and
  must not be presented as a converged or release-quality distilled model.
- A 30-step one-sentence overfit artifact has WER 1.0; it is retained only to prove optimization
  and resume, while the step-1 EMA export is the quality-preserving export gate.
- The official README's `hf://kyutai/pocket-tts/config/english_2026-04.yaml` example was absent at
  tested revision `492522650173a0653b7575cdc25ae09810e5d741`; local and other HF/HTTPS config
  paths passed. This is an upstream artifact mismatch, not silently treated as a pass.
- Starlette emits one TestClient/httpx deprecation warning; runtime HTTP and browser replay pass.

Generated model weights, checkpoints, WAV files and voice states remain local and ignored by Git.

