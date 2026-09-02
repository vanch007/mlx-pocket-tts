# TASK-002 config and quantization compatibility

Status: `pass`

## Covered behavior

- Official language aliases and language-specific defaults.
- Local converted artifact and local official YAML resolution.
- `hf://` YAML with a pinned revision.
- HTTPS YAML with content-addressed conversion-cache reuse.
- Strict official-safetensors-to-MLX conversion.
- Generic flow-LM 8-bit quantization for artifacts without a prebuilt 8-bit sibling.
- Strict audit and real M3 Max generation after quantization.

## Real evidence

- German official public model: audit pass, RTF 0.0817, peak memory 0.484 GB, ASR WER 0.10.
- Czech community model from the official README: pinned `hf://` config, audit pass,
  RTF 0.3817, peak memory 1.350 GB, ASR WER 0.2727.
- The equivalent Czech HTTPS config reused the same quantized cache and loaded successfully.
- Unit/static result after implementation: 15 tests passed; Ruff check/format passed.

## Upstream discrepancy

The pinned/current `kyutai/pocket-tts` model repository at
`492522650173a0653b7575cdc25ae09810e5d741` does not contain the README example path
`config/english_2026-04.yaml`; both pinned and `main` requests returned 404. This is recorded as an
upstream documentation/repository mismatch, not reported as a successful test. The fixed-revision
Czech config listed by the same official README was used to prove generic remote-config support.

Legacy `.pt` checkpoint import was staged to TASK-010 and subsequently passed.
