# Pocket TTS comprehensive native MLX parity design

## 1. Decision and objective

This project will expand from the previously approved inference-and-serving port into a
comprehensive Apple Silicon MLX implementation of the official Pocket TTS repository pinned at
`kyutai-labs/pocket-tts@e244e9a0dad520dad8f61a3c5882e91ea4859439`.

“Comprehensive” means that the official public inference, CLI, Python API, local Web UI,
configuration, voice export, quantization and training semantics are covered by native MLX
implementations and auditable tests. The production and training paths must not depend on
PyTorch. A one-time offline importer may read legacy PyTorch checkpoints.

The target platform is Apple Silicon macOS. CUDA, NCCL, distributed GPU training, x86, Linux,
Windows, Docker and cloud swarm deployment are outside the runnable MLX target and must be marked
`not applicable`, with migration notes where useful. Browser/WASM ports and third-party
integrations are not part of the official core parity target.

## 2. Delivery strategy

Use a staged native port rather than a line-by-line translation or a dual-backend abstraction.
Each stage has its own acceptance evidence and may not be reported complete until its gates pass.

1. Exact inference, Python API, CLI, configuration and model-loading compatibility.
2. Gradio Web UI, FastAPI streaming and batch voice export.
3. Native MLX data preparation, latent caching and basic training loop.
4. Checkpointing, exact resume, EMA, tokenizer tooling and inference export.
5. Depth distillation, CFG distillation and legacy PyTorch checkpoint import.
6. Official-test mapping, full M3 Max validation, release report and complete README.

Stages build on the existing verified MLX inference implementation. Existing MLX-only maintenance
commands such as `convert`, `audit` and `quantize` remain available alongside the official-compatible
surface.

## 3. Architecture

### 3.1 Compatibility entry layer

- Expose official-compatible `TTSModel` construction and public methods.
- Match the official public functions, parameter names, defaults and primary return semantics.
- Return MLX arrays from native APIs and provide explicit NumPy conversion interoperability.
- Cover the official `generate`, `serve` and `export-voice` commands.
- Preserve the existing `convert`, `audit` and `quantize` project commands as documented MLX
  extensions.

### 3.2 Inference runtime

- Reuse the validated native MLX FlowLM, Mimi codec, streaming cache, text chunking, preset voice,
  arbitrary cloning and reusable voice-state implementation.
- Add official default text and voices, language selection, local/HTTPS/Hugging Face configuration
  loading, stdin input, stdout audio output, runtime quantization behavior and checkpoint selection.
- Support English, French, German, Italian, Portuguese and Spanish and all published 12- and
  24-layer configurations.
- Keep official-compatible APIs and MLX extensions backed by one core engine so behavior cannot
  silently diverge.

### 3.3 Serving and Web UI

- Provide a Gradio interface with the same user purpose and controls as official `serve`.
- Preserve the existing FastAPI health and streaming WAV endpoints.
- Serialize mutation of stateful codec/model instances or use explicitly isolated instances.
- Support configured default voice, uploaded reference audio, reusable voice states, language and
  generation controls.

### 3.4 Native MLX training runtime

Create a focused training package whose components can be tested independently:

- configuration and argument validation;
- dataset manifest preparation and audio validation;
- Mimi latent precomputation and versioned caching;
- tokenizer training and loading;
- length-aware batching, cropping and collation;
- differentiable MLX training model and losses;
- optimizer, learning-rate schedule, gradient accumulation and clipping;
- EMA lifecycle;
- atomic checkpoint save, load and exact resume;
- validation loss and generated-sample hooks;
- depth distillation and CFG distillation;
- inference artifact export.

Mimi is frozen by default and supplies latent encoding/decoding. FlowLM and its conditioning path
are trainable. The initial validated target is single-machine Apple Silicon training. Multi-device
training may only be added after it has separate MLX evidence.

### 3.5 Conversion and compatibility

- Continue to load converted official safetensors with strict key and shape audits.
- Save native training checkpoints in a documented MLX-safe format containing model, optimizer,
  EMA, step, random state, data position and resolved configuration.
- Export inference artifacts as safetensors plus standard configuration, without optimizer state.
- Provide a one-time legacy PyTorch checkpoint importer isolated from runtime and training
  dependencies.
- Keep official configuration field names where semantics match. Put Apple/MLX-only settings such
  as memory limits, compilation policy and accumulation tuning under an `mlx` namespace.

## 4. Public compatibility requirements

### 4.1 Python API

- Implement official `TTSModel.load_model(...)` behavior and documented defaults.
- Cover preset voice, audio reference, Hugging Face voice URL and safetensors voice-state loading.
- Cover complete and incremental generation.
- Cover model state export using official-compatible naming.
- Expose sample rate and an MLX-appropriate device description.
- Document any unavoidable tensor-type difference and its NumPy bridge.

### 4.2 CLI

The MLX CLI must cover official behavior for:

- default invocation with default text, language and voice;
- `--text`, including stdin through `-`;
- output file and stdout through `-`;
- `--voice` as preset, local audio, remote audio or saved state;
- `--language` and language-specific defaults;
- `--config` from local path, HTTPS URL or `hf://` URI with revision;
- legacy training `--checkpoint` import;
- Apple/MLX device validation;
- runtime quantization selection;
- generation controls and statistics;
- `serve` default voice and Web UI controls;
- `export-voice` for a single file or an input directory.

### 4.3 Training tools

Cover the official training workflow semantics for:

- data preparation;
- latent precomputation and stale-cache detection;
- tokenizer training;
- scratch training;
- resume from checkpoint;
- checkpoint shrinking/export;
- depth distillation;
- CFG distillation;
- validation and inference replay of produced artifacts.

## 5. Training data flow

```text
JSONL audio records
  -> schema/audio validation and resampling
  -> versioned Mimi latent precomputation
  -> tokenizer
  -> length buckets, crop and batch
  -> MLX forward and loss
  -> gradients, accumulation, clipping and optimizer
  -> EMA
  -> checkpoint and validation
  -> inference artifact export
  -> direct load by the MLX inference runtime
```

The latent manifest records content hashes and the relevant encoder, configuration and tool
versions. A change to audio, configuration or encoder invalidates the affected cache rather than
silently reusing it. Training data, voice samples, generated weights and audio outputs remain
untracked and are never uploaded automatically.

## 6. Failure handling

- Reject configuration, revision, tokenizer, weight-key and shape mismatches during loading.
- Identify the exact record for corrupt audio, unsupported sampling rate, empty text or stale cache;
  do not silently skip it.
- Distinguish authentication, network, disk-space and memory failures and provide actionable
  messages.
- Write checkpoints to a temporary path and atomically replace the target only after validation.
- Detect NaN/Inf values, abnormal gradients and loss divergence, preserve diagnostic state and
  stop safely.
- Return `not applicable` for unsupported platform capabilities instead of claiming success.

## 7. Verification and evidence

Every result uses one of `pass`, `fail`, `pending` or `not applicable`. Evidence records commands,
environment, source commit, model revision, configuration, logs and output indexes.

### 7.1 Static gate

- Formatting and lint pass.
- Package and tests import without PyTorch in the runtime/training environment.
- Isolation checks prevent accidental `mlx_audio` production dependencies.

### 7.2 Official compatibility gate

- Every relevant official inference, CLI, API, configuration and training test has an MLX
  equivalent.
- Any non-reusable official test has a recorded reason and an equivalent behavioral assertion.

### 7.3 Numerical gate

- Fixed fixtures compare important layer outputs, loss components and sampling behavior against
  the pinned official implementation using documented per-test tolerances.
- Tolerances distinguish expected floating-point/backend differences from semantic failures.

### 7.4 M3 Max inference gate

- Real generation for all six languages and both 12- and 24-layer architecture paths.
- Preset voice, arbitrary cloning, voice-state export/import, long text and incremental streaming.
- Gradio Web UI and FastAPI streaming endpoint.
- Audio validity, ASR/WER, reference leakage, ECAPA similarity, time to first audio, RTF, duration
  and peak memory.

### 7.5 Native training gate

- Overfit a tiny authorized dataset and show a sustained loss decrease.
- Demonstrate checkpoint interruption and exact semantic resume.
- Validate EMA save/load and inference export.
- Validate latent-cache reuse and invalidation.
- Complete depth-distillation and CFG-distillation smoke runs.
- Load the exported artifact in the production inference runtime and generate valid speech.

The training gate proves a functional training system. It does not claim final model quality from
a tiny dataset. Any large-scale quality claim requires a separately documented dataset, run and
evaluation.

### 7.6 Release quality gate

- Produce JSON, Markdown and listening-test HTML reports.
- Retain generated samples and manifests outside Git.
- Require all core parity rows to be `pass` before using “comprehensive official feature port” in
  release language.

## 8. Documentation deliverables

The README and linked documentation must include:

- platform requirements and installation;
- quick-start inference and official-to-MLX command mapping;
- six-language and 12/24-layer model matrix with pinned revisions;
- preset voice, cloning, voice-state and Hugging Face authorization workflows;
- Python API and complete CLI reference;
- Gradio and FastAPI use;
- native training data preparation, scratch training and resume;
- EMA, checkpoint, distillation and export workflows;
- compatibility matrix and explicit non-applicable platform features;
- measured performance and quality tables linked to evidence;
- troubleshooting for access, network, disk, memory, configuration and data failures;
- licensing and safe handling of voice/data/model artifacts.

## 9. Completion criteria

The project is complete only when:

1. All official core inference and training capabilities defined above are implemented natively in
   MLX and their required gates are `pass`.
2. Runtime and training do not depend on PyTorch; the optional legacy importer is isolated.
3. Training artifacts load directly in the MLX inference runtime.
4. Six languages, 12/24-layer inference, cloning, streaming, Web UI, resume and both distillation
   paths have real M3 Max evidence.
5. The README, compatibility matrix and release report match the observed evidence.
6. Unsupported CUDA/DDP and non-Apple platforms are explicitly `not applicable`, not presented as
   missing MLX features or successful ports.

Until all criteria pass, project status must remain partial and the report must identify the
remaining failed or pending rows.
