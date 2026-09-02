from mlx_pocket_tts.cli import build_parser


def test_generate_official_defaults_and_aliases():
    args = build_parser().parse_args(["generate"])
    assert args.text is None
    assert args.voice is None
    assert args.output_path == "./tts_output.wav"
    assert args.sampler_decode_steps == 1
    assert args.device == "cpu"

    args = build_parser().parse_args(
        [
            "generate",
            "--language",
            "german",
            "--sampler-decode-steps",
            "2",
            "--output-path",
            "speech.wav",
            "--quantize",
        ]
    )
    assert args.language == "german"
    assert args.sampler_decode_steps == 2
    assert args.output_path == "speech.wav"
    assert args.quantize is True


def test_generate_mlx_output_alias_is_preserved():
    args = build_parser().parse_args(["generate", "--output", "legacy.wav"])
    assert args.output_path == "legacy.wav"


def test_export_voice_supports_official_positionals_and_legacy_flags():
    args = build_parser().parse_args(["export-voice", "voice.wav", "voice.safetensors"])
    assert args.audio_path == "voice.wav"
    assert args.export_path == "voice.safetensors"
    args = build_parser().parse_args(
        ["export-voice", "--audio", "legacy.wav", "--output", "legacy.safetensors"]
    )
    assert args.audio_option == "legacy.wav"
    assert args.output_option == "legacy.safetensors"


def test_serve_official_options_and_voice_alias():
    args = build_parser().parse_args(
        ["serve", "--language", "french_24l", "--default-voice", "estelle", "--quantize"]
    )
    assert args.language == "french_24l"
    assert args.default_voice == "estelle"
    assert args.quantize is True
    args = build_parser().parse_args(["serve", "--voice", "alba"])
    assert args.default_voice == "alba"
