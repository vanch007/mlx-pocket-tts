# TASK-005 Web and HTTP compatibility

Status: `pass` — implementation, real HTTP replay and visual in-app-browser replay pass.

## Passed

- Pinned official static streaming UI is packaged and served at `/`.
- Default language text replaces the UI placeholder.
- `/health` returns healthy status and model revision.
- `/tts` accepts the preloaded default voice, official named voices, `voice_url`, legacy `voice`,
  and uploaded reference audio.
- Invalid or conflicting voice inputs return HTTP 400.
- Default, named and uploaded voice requests all returned valid streaming RIFF/WAVE data in a real
  Uvicorn + M3 Max replay: 96,044, 61,484 and 69,164 bytes respectively.
- MLX generation and reference encoding remain on the Uvicorn event thread; this fixes the real
  `There is no Stream(cpu, ...) in current thread` failure found during testing.

## Release browser replay

The Codex in-app browser opened the real local 8-bit service after applying a temporary non-zero
viewport (the default webview reported width zero). The rendered page showed the official text,
voice URL and upload controls. Submitting “Browser replay passes with native MLX.” produced a
playable two-second audio element and download link. The UI reported first audio 0.18 seconds,
total 0.26 seconds and 10.2× faster than real time. The browser tab and viewport were restored.

Final unit/static result: 33 tests passed; Ruff check and format passed.
