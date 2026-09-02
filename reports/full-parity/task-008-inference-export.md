# TASK-008 direct inference export evidence

Status: **pass**.

`export_inference_artifact` merges a native training state (or EMA shadow) with Mimi and the
source config, tokenizer, and preset voices. Its output is the same independent directory layout
accepted by the existing strict `mlx_pocket_tts.load()` path.

The step-1 EMA artifact strictly loaded and generated a finite 1.84-second WAV on M3 Max at RTF
0.0980. Local Whisper replay transcribed the requested sentence exactly modulo capitalization,
for WER 0.0.

The step-30 one-sentence overfit artifact also strictly loaded and generated a valid WAV, but its
ASR WER was 1.0. It is explicitly a training/resume stress artifact and is not suitable for
release. This failure is retained in `task-008-export.json` rather than hidden.

