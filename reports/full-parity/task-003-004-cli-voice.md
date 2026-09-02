# TASK-003 and TASK-004 CLI/voice compatibility

Status: `pass`. Legacy `.pt` checkpoint loading subsequently passed under TASK-010.

## Generate

- Default text and official output-path default/alias: pass.
- stdin text using `--text -`: pass.
- stdout streaming WAV using `--output-path -`: pass; 76,844 bytes with valid RIFF/WAVE header.
- Language, temperature/sampler controls, device validation and runtime 8-bit: pass.
- Fixed-revision `hf://` custom config: pass.
- MLX `--model`, reports and segmented-stream extensions remain available.

## Export voice

- Official positional single-file export: pass.
- Official positional directory batch export: pass, two inputs/two safetensors outputs.
- Previous `--audio/--output` flags: preserved and tested.
- Exported state import and real generation: pass, 24 kHz mono PCM WAV, RTF 0.0834.
- ASR replay: exact transcript, WER 0.0.

Unit/static result: 18 tests passed; Ruff check and format passed.
