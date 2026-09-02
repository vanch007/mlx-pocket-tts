# TASK-001 official Python API compatibility

Status: `pass`

- Official public exports: `TTSModel`, `export_model_state`.
- Required methods: `load_model`, `generate_audio`, `generate_audio_stream`,
  `get_state_for_audio_prompt`.
- Required properties: `device`, `sample_rate`.
- Static/unit result: 11 tests passed; Ruff check and format passed.
- Real M3 Max result: Metal, 24 kHz, 48,000 finite samples.
- Model load: 0.0580 s; generation: 0.2650 s.
- Audio: `outputs/full-parity/task-001-api-smoke.wav`.
