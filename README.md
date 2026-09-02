# MLX Pocket TTS

[![Hugging Face model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-vanch007%2Fmlx--pocket--tts-yellow)](https://huggingface.co/vanch007/mlx-pocket-tts)
[![GitHub source](https://img.shields.io/badge/GitHub-vanch007%2Fmlx--pocket--tts-black)](https://github.com/vanch007/mlx-pocket-tts)

Standalone Apple Silicon MLX port of [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts),
pinned to upstream source `e244e9a0dad520dad8f61a3c5882e91ea4859439`. Inference, serving, data
preparation, training, checkpointing and distillation are native MLX. PyTorch is optional and
confined to a subprocess that imports legacy `.pt` checkpoints offline.

The ready-to-run 8-bit weights are published at
[`vanch007/mlx-pocket-tts`](https://huggingface.co/vanch007/mlx-pocket-tts). The model card links
back to this repository so code, weights, attribution and verification evidence remain connected.

## Compatibility and evidence

| Surface | Status | M3 Max evidence |
|---|---|---|
| Official Python inference API | pass | mapped API tests and real 8-bit generation |
| `generate` CLI, stdin/stdout WAV and controls | pass | default, stdin, stdout and HF-config replay |
| Preset voices and arbitrary voice cloning | pass | BF16/8-bit clone, reusable state and ASR gates |
| Long text and incremental streaming | pass | release-v1 real-time matrix |
| `export-voice` file and directory modes | pass | single/batch export and WER 0.0 reuse |
| Official FastAPI/static Web UI | pass | real browser submit, playable audio and download link |
| Six languages, compact and 24-layer previews | pass | strict conversion and real generation matrix |
| Dynamic 8-bit FlowLM quantization | pass | strict audit, German/Czech/config paths and generation |
| Local/HF/HTTPS YAML configuration | pass | conversion-cache and strict-load tests |
| Legacy training `.pt` checkpoint option | pass | isolated full import, EMA overlay, WER 0.0 and 8-bit |
| Native MLX data/latent cache | pass | real Mimi encode, reuse and content invalidation |
| Native LSD/flow-matching training | pass | finite gradients, real update and 30-step overfit |
| AdamW, clipping, EMA, atomic resume | pass | interrupted step-10→30 restore, identical weights |
| Direct inference export | pass | strict load, RTF 0.098 and WER 0.0 |
| CFG and 24→6 depth distillation | pass (smoke) | both update/export paths pass; depth one-step quality fails |
| CUDA, DDP, Linux/Windows | not applicable | Apple Silicon MLX target only |

The current full-parity evidence is in
[`reports/full-parity/REPORT.md`](reports/full-parity/REPORT.md). Earlier broad inference matrices
remain in [`reports/release-v1/REPORT.md`](reports/release-v1/REPORT.md).

## Install

```bash
git clone https://github.com/vanch007/mlx-pocket-tts.git
cd mlx-pocket-tts
uv sync --extra dev
# Include local Whisper evaluation tools:
uv sync --extra dev --extra eval
```

Python 3.10+ and an Apple Silicon Mac are required for the runnable target. Model weights, audio,
training data and generated checkpoints are intentionally excluded from Git.

## Generate

Official-compatible language/config mode:

```bash
uv run mlx-pocket-tts generate \
  --language english \
  --voice alba \
  --text "Hello from Pocket TTS on Apple Silicon." \
  --output-path outputs/hello.wav
```

Published 8-bit artifact mode (an MLX extension):

```bash
uv run mlx-pocket-tts generate \
  --model vanch007/mlx-pocket-tts \
  --voice alba \
  --text "Streaming locally with MLX." \
  --stream \
  --output outputs/stream.wav
```

`--text -` reads stdin and `--output-path -` writes a streaming WAV to stdout. Generation exposes
temperature, sampler steps, noise clamp, EOS threshold, frames-after-EOS and max-token controls.
Use `--ref-audio reference.wav` with a cloning-capable artifact; public no-cloning weights reject
arbitrary reference audio explicitly.

## Python API

```python
from mlx_pocket_tts import TTSModel, write_audio

model = TTSModel.load_model(language="english", quantize=True)
voice_state = model.get_state_for_audio_prompt("alba")
audio = model.generate_audio(voice_state, "Hello world.")
write_audio("output.wav", audio, model.sample_rate)
```

The compatibility class also provides `generate`, `generate_audio_stream`, device validation,
voice-state export/import and the official configuration/checkpoint arguments.

## Configs, conversion and quantization

Local `.yaml`, HTTPS URLs and `hf://repo/path@revision` configs are converted into deterministic
MLX caches. Converted artifact directories and `config.json` files load directly.

```bash
uv run mlx-pocket-tts convert \
  --language english \
  --config configs/english.yaml \
  --output models/english-public-mlx

uv run mlx-pocket-tts audit models/english-public-mlx

uv run mlx-pocket-tts quantize \
  --model models/english-public-mlx \
  --output models/english-public-mlx-8bit \
  --bits 8
```

Quantization targets the generation FlowLM; Mimi stays at source precision. This matches the
scope of upstream CPU int8 inference.

## Legacy `.pt` training checkpoints

```bash
uv run mlx-pocket-tts generate \
  --config path/to/model-config.yaml \
  --checkpoint path/to/checkpoint.pt \
  --voice path/to/reference.wav \
  --text "Loaded from an old training checkpoint." \
  --quantize \
  --output-path outputs/checkpoint.wav
```

The first use starts the isolated importer and caches a complete MLX safetensors artifact. The
normal inference/training process never imports PyTorch. The importer requires PyTorch only in
that offline environment and uses `weights_only=True`; EMA weights are preferred when present.

## Export reusable voices

```bash
uv run mlx-pocket-tts export-voice reference.wav voices/reference.safetensors \
  --model path/to/voice-cloning-model

uv run mlx-pocket-tts export-voice references/ voices/ \
  --model path/to/voice-cloning-model
```

The resulting safetensors voice state can be passed back through `--voice` without re-encoding
the reference audio.

## Web UI and streaming HTTP

```bash
uv run mlx-pocket-tts serve \
  --model vanch007/mlx-pocket-tts \
  --host 127.0.0.1 \
  --port 8000
```

Open `http://127.0.0.1:8000/`. The official-purpose static UI accepts text, preset/HF/HTTP voice
references and uploaded voice audio. `GET /health` reports readiness; `POST /tts` returns a
streaming mono 24-kHz PCM WAV. Generation is serialized because Mimi decoding is stateful.

## Native MLX training

The training API mirrors the upstream conditioning sequence and objectives while replacing
CUDA/DDP with a single-Mac MLX loop:

```python
from mlx_pocket_tts import load
from mlx_pocket_tts.training import (
    EMA, FlowArgs, LatentDataLoader, OptimArgs, TrainArgs, TrainableTTS,
    Trainer, build_optimizer, export_inference_artifact, precompute_manifest,
    save_checkpoint,
)

base = load("models/english-clone-mlx")
cache = precompute_manifest("data/train.jsonl", base, "runs/demo/cache")
loader = LatentDataLoader(
    cache["manifest"], base.flow_lm.conditioner.tokenizer.sp, batch_size=4
)
model = TrainableTTS(base, TrainArgs(flow=FlowArgs(type="lsd")))
optimizer = build_optimizer(model, OptimArgs())
ema = EMA(model, 0.999)
trainer = Trainer(model, optimizer, max_norm=1.0, ema=ema)

for step, batch in enumerate(loader, start=1):
    metrics = trainer.step(batch)
    if step % 100 == 0:
        save_checkpoint("runs/demo", step, model, optimizer, ema)

export_inference_artifact("runs/demo/export", model, base, ema)
```

`Trainer.step([micro_batch_1, ...])` performs gradient accumulation. Checkpoints atomically retain
model, AdamW and EMA state and can be restored with `latest_checkpoint`/`load_checkpoint`.

For CFG guidance baking, attach a separately loaded teacher with
`attach_distillation(student, teacher, cfg_coef=3)`. For depth distillation, pass
`seed_from_teacher=True`; the student is initialized from the bottom and top teacher layers.
The included one-step 24→6 evidence proves the path runs and exports, not that it has converged.

## Verification

```bash
uv run pytest -q
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
```

Source code is MIT licensed. Weights, tokenizers and voice assets retain their upstream licenses
and access conditions. The published model is derived from
[`kyutai/pocket-tts`](https://huggingface.co/kyutai/pocket-tts), is marked CC BY 4.0, and includes
the required upstream attribution and responsible-use notice in its model card.
