# MLX Pocket TTS release-v1 evaluation

## Verdict

The approved inference port is `pass` for every acceptance gate. After the Hugging Face terms were
accepted, the newest official arbitrary-reference cloning model passed strict conversion, real
reference encoding, generation, voice-state export/import, ASR, ECAPA speaker similarity and
8-bit flow-LM evaluation.

## Provenance and machine

- Official source: `kyutai-labs/pocket-tts@e244e9a0dad520dad8f61a3c5882e91ea4859439`
- Public official weights: `kyutai/pocket-tts-without-voice-cloning@d29db7978e464fb90cb3359ee0c69a273b9142cc`
- Official cloning weights: `kyutai/pocket-tts@492522650173a0653b7575cdc25ae09810e5d741`
- Community cloning artifact: `mlx-community/pocket-tts@cbf71d5f6657bbc3f4bc02f85ee408261225bec7`
- ASR evaluator: `mlx-community/whisper-large-v3-turbo@a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb`
- Hardware: Apple M3 Max, 16 CPU cores, 128 GB unified memory
- Sample rate: 24 kHz mono

## Acceptance matrix

| Gate | Status | Evidence |
| --- | --- | --- |
| Standalone runtime without `mlx_audio` | pass | isolation test and package import |
| Official tensor conversion | pass | 12/12 artifacts, zero missing/extra/shape mismatches |
| Compact 6-layer and 24-layer paths | pass | real audio from both architecture sizes |
| Six official languages | pass | English, French, German, Italian, Portuguese, Spanish |
| Preset voices | pass | converted official Alba state; all 21 public voice states copied |
| Long text and decimal-safe chunking | pass | 15.28 s real sample plus regression tests |
| Incremental streaming | pass | English TTFA 43 ms; 24-layer TTFA 90–93 ms |
| Generation controls | pass | temperature, sampler steps, noise clamp, EOS threshold, frames after EOS, max tokens |
| Voice-state export/import | pass | newest official cloned voice roundtrip, ASR WER 0.0 |
| Reference-audio cloning engine | pass | newest official model, ASR WER 0.0, ECAPA cosine 0.3803 |
| Newest official cloning weights | pass | gated revision converted and strictly audited after authorization |
| 8-bit flow LM | pass | preset and official cloning artifacts pass strict load and real generation |
| FastAPI | pass | `/health` 200; `/tts` 200 streaming RIFF/WAV |
| Unit/static checks | pass | 9 pytest tests, Ruff check and format |

## Real-device results

The main compact artifacts ran at RTF 0.077–0.095 with roughly 0.82–0.89 GB peak memory. The
24-layer preview artifacts ran at RTF 0.186–0.193 with roughly 1.60 GB peak memory. All are faster
than real time. Streaming time to first audio was about 40–43 ms for the tested English 12-layer
models and 90–93 ms for the 24-layer models.

The English preset 8-bit artifact ran at RTF 0.060 with 0.424 GB peak memory. The newest official
8-bit cloning artifact ran at RTF 0.069; its ECAPA cosine was 0.4138. Only the generation flow LM
is quantized; the reference encoder and stateful Mimi codec remain at source precision. Every
recorded waveform was finite and had a measured clipping ratio of zero.

Whisper large-v3-turbo passed all 17 generated-audio cases. Fifteen cases had WER 0.0; the compact
Spanish sample had WER 0.1429 and the 8-bit official clone had WER 0.0833. Both newest official
BF16 cloning cases had WER 0.0. ECAPA cosine/P10 were 0.3803/0.3790 for direct cloning and
0.3655/0.2822 after voice-state export/import, above the 0.25 usability threshold.

## Evidence files

- `latest/asr.json`: transcripts and WER for every generated sample
- `latest/*_audit.json`: strict checkpoint topology audits
- `latest/*.json`: timing, TTFA, RTF, memory and waveform statistics
- `latest/http_api.json`: live FastAPI response probe
- `latest/speaker_similarity.json`: segment-aware ECAPA speaker similarity
- `gated-clone.json`: resolved gated-model authorization and test result

Generated WAV and voice-state files remain local and are intentionally ignored by Git.
