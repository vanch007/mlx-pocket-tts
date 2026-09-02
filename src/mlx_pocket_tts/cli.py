import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

from .audio import write_audio
from .conversion import PUBLIC_MODEL_REVISION, audit_checkpoint, convert_official
from .defaults import get_default_text_for_language, get_default_voice_for_language
from .loader import load
from .pocket_tts import Model as TTSModel
from .quantization import quantize_artifact
from .server import pcm_wav_stream
from .voice import export_voice_state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlx-pocket-tts")
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--model", help="MLX extension: converted artifact path or HF repo")
    generate.add_argument("--revision")
    generate.add_argument("--text")
    generate.add_argument("--voice")
    generate.add_argument("--ref-audio")
    generate.add_argument("--language")
    generate.add_argument("--config")
    generate.add_argument("--checkpoint")
    generate.add_argument("-q", "--quiet", action="store_true")
    generate.add_argument("--temperature", type=float)
    generate.add_argument("--sampler-decode-steps", type=int, default=1)
    generate.add_argument("--lsd-decode-steps", type=int)
    generate.add_argument("--noise-clamp", type=float)
    generate.add_argument("--eos-threshold", type=float, default=-4.0)
    generate.add_argument("--frames-after-eos", type=int)
    generate.add_argument("--max-tokens", type=int, default=50)
    generate.add_argument("--device", default="cpu")
    generate.add_argument("--quantize", action="store_true")
    generate.add_argument("--stream", action=argparse.BooleanOptionalAction, default=False)
    generate.add_argument("--streaming-interval", type=float, default=0.4)
    generate.add_argument(
        "--output-path", "--output", dest="output_path", default="./tts_output.wav"
    )
    generate.add_argument("--report", type=Path)
    serve = sub.add_parser("serve")
    serve.add_argument("--model", help="MLX extension: converted artifact path or HF repo")
    serve.add_argument("--revision")
    serve.add_argument("--default-voice", "--voice", dest="default_voice")
    serve.add_argument("--language")
    serve.add_argument("--config")
    serve.add_argument("--quantize", action="store_true")
    serve.add_argument("--reload", action="store_true")
    serve.add_argument("--host", default="localhost")
    serve.add_argument("--port", type=int, default=8000)
    convert = sub.add_parser("convert")
    convert.add_argument("--language", default="english")
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--config", type=Path, required=True)
    convert.add_argument("--source-repo", default="kyutai/pocket-tts-without-voice-cloning")
    convert.add_argument("--revision", default=PUBLIC_MODEL_REVISION)
    audit = sub.add_parser("audit")
    audit.add_argument("model", type=Path)
    export_voice = sub.add_parser("export-voice")
    export_voice.add_argument("audio_path", nargs="?")
    export_voice.add_argument("export_path", nargs="?")
    export_voice.add_argument("--audio", dest="audio_option")
    export_voice.add_argument("--output", dest="output_option")
    export_voice.add_argument("--model")
    export_voice.add_argument("--revision")
    export_voice.add_argument("--language")
    export_voice.add_argument("--config")
    export_voice.add_argument("-q", "--quiet", action="store_true")
    quantize = sub.add_parser("quantize")
    quantize.add_argument("--model", type=Path, required=True)
    quantize.add_argument("--output", type=Path, required=True)
    quantize.add_argument("--bits", type=int, default=8, choices=(4, 8))
    quantize.add_argument("--group-size", type=int, default=64)
    return parser


def run_generate(args) -> int:
    if args.language is not None and args.config is not None:
        raise ValueError("Cannot specify both config and language, please choose one or the other.")
    text = args.text if args.text is not None else get_default_text_for_language(args.language)
    if text == "-":
        text = sys.stdin.read()
    if not text.strip():
        raise ValueError("No input received from stdin.")

    load_started = time.perf_counter()
    decode_steps = args.lsd_decode_steps or args.sampler_decode_steps
    if args.model:
        model = load(args.model, revision=args.revision)
        if args.temperature is not None:
            model.temp = args.temperature
        model.lsd_decode_steps = decode_steps
        model.noise_clamp = args.noise_clamp
        model.eos_threshold = args.eos_threshold
    else:
        model = TTSModel.load_model(
            language=args.language,
            config=args.config,
            temp=args.temperature,
            sampler_decode_steps=decode_steps,
            noise_clamp=args.noise_clamp,
            eos_threshold=args.eos_threshold,
            quantize=args.quantize,
            checkpoint=args.checkpoint,
        )
    model.to(args.device)
    load_seconds = time.perf_counter() - load_started
    generation_started = time.perf_counter()
    voice = args.voice or get_default_voice_for_language(
        args.language, args.config, args.checkpoint
    )
    generator = model.generate(
        text=text,
        voice=voice,
        ref_audio=args.ref_audio,
        temperature=args.temperature,
        stream=args.stream,
        streaming_interval=args.streaming_interval,
        frames_after_eos=args.frames_after_eos,
        lsd_decode_steps=decode_steps,
        noise_clamp=args.noise_clamp,
        eos_threshold=args.eos_threshold,
        max_tokens=args.max_tokens,
    )
    chunks = []
    first_audio_seconds = None
    for chunk in generator:
        if first_audio_seconds is None:
            first_audio_seconds = time.perf_counter() - generation_started
        chunks.append(chunk)
    audio = mx.concatenate([chunk.audio for chunk in chunks])
    mx.eval(audio)
    generation_seconds = time.perf_counter() - generation_started
    if args.output_path == "-":
        for payload in pcm_wav_stream([audio], model.sample_rate):
            sys.stdout.buffer.write(payload)
        sys.stdout.buffer.flush()
    else:
        write_audio(args.output_path, audio, model.sample_rate)
    report = {
        "model": str(model.model_path),
        "revision": model.model_revision,
        "language": args.language,
        "output": args.output_path
        if args.output_path == "-"
        else str(Path(args.output_path).resolve()),
        "samples": int(audio.shape[0]),
        "sample_rate": model.sample_rate,
        "audio_seconds": int(audio.shape[0]) / model.sample_rate,
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "time_to_first_audio_seconds": first_audio_seconds,
        "rtf": generation_seconds / (int(audio.shape[0]) / model.sample_rate),
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "waveform": {
            "finite": bool(mx.all(mx.isfinite(audio)).item()),
            "minimum": float(mx.min(audio).item()),
            "maximum": float(mx.max(audio).item()),
            "rms": float(mx.sqrt(mx.mean(mx.square(audio))).item()),
            "clipping_ratio": float(mx.mean(mx.abs(audio) >= 0.999).item()),
            "silence_ratio": float(mx.mean(mx.abs(audio) < 1e-4).item()),
        },
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    if not args.quiet and args.output_path != "-":
        print(json.dumps(report, indent=2))
    return 0


def run_export_voice(args) -> int:
    source_text = args.audio_path or args.audio_option
    destination_text = args.export_path or args.output_option
    if not source_text or not destination_text:
        raise ValueError("export-voice requires an audio file/directory and an export path.")
    if args.language is not None and args.config is not None:
        raise ValueError("Cannot specify both config and language, please choose one or the other.")
    if args.model:
        model = load(args.model, revision=args.revision)
    else:
        model = TTSModel.load_model(language=args.language, config=args.config)

    source = Path(source_text).expanduser()
    destination = Path(destination_text).expanduser()
    if source.is_dir():
        destination.mkdir(parents=True, exist_ok=True)
        audio_files = sorted(
            path
            for path in source.iterdir()
            if path.is_file() and path.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        )
        if not audio_files:
            raise ValueError(f"No supported audio files found in {source}.")
        outputs = []
        for audio_path in audio_files:
            state = model.get_state_for_audio_prompt(audio_path, truncate=True)
            outputs.append(
                export_voice_state(state, destination / f"{audio_path.stem}.safetensors")
            )
    else:
        if not source.is_file():
            raise FileNotFoundError(source)
        if destination.is_dir() or destination.suffix != ".safetensors":
            destination = destination / f"{source.stem}.safetensors"
        state = model.get_state_for_audio_prompt(source, truncate=True)
        outputs = [export_voice_state(state, destination)]
    if not args.quiet:
        print("\n".join(str(path) for path in outputs))
    return 0


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "generate":
        raise SystemExit(run_generate(args))
    if args.command == "convert":
        report = convert_official(
            language=args.language,
            output=args.output,
            config_path=args.config,
            source_repo=args.source_repo,
            revision=args.revision,
        )
        print(json.dumps(report, indent=2))
        return
    if args.command == "audit":
        report = audit_checkpoint(args.model)
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["pass"] else 1)
    if args.command == "export-voice":
        raise SystemExit(run_export_voice(args))
    if args.command == "quantize":
        report = quantize_artifact(
            args.model,
            args.output,
            bits=args.bits,
            group_size=args.group_size,
        )
        print(json.dumps(report, indent=2))
        return
    from .server import run_server

    run_server(
        args.model,
        args.revision,
        args.default_voice,
        args.host,
        args.port,
        language=args.language,
        config=args.config,
        quantize=args.quantize,
        reload=args.reload,
    )
