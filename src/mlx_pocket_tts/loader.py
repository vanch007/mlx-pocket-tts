import hashlib
import json
import os
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download

from .defaults import DEFAULT_LANGUAGE, LANGUAGE_ARTIFACTS
from .pocket_tts import Model
from .quantization import apply_quantization
from .utils import download_if_necessary, make_cache_directory

DEFAULT_MODEL = "mlx-community/pocket-tts"
DEFAULT_REVISION = "cbf71d5f6657bbc3f4bc02f85ee408261225bec7"


def resolve_model_path(model: str | Path, revision: str | None = None) -> Path:
    path = Path(model).expanduser()
    if path.exists():
        return path
    return Path(
        snapshot_download(
            str(model),
            revision=revision,
            allow_patterns=["*.json", "*.safetensors", "*.model", "embeddings/*"],
        )
    )


def load(
    model: str | Path = DEFAULT_MODEL,
    *,
    revision: str | None = None,
    lazy: bool = False,
) -> Model:
    model_path = resolve_model_path(model, revision=revision)
    config = json.loads((model_path / "config.json").read_text())
    tokenizer_path = config.get("flow_lm", {}).get("lookup_table", {}).get("tokenizer_path")
    if tokenizer_path and not str(tokenizer_path).startswith(("http://", "https://", "hf://")):
        config["flow_lm"]["lookup_table"]["tokenizer_path"] = str(model_path / tokenizer_path)
    weights = mx.load(str(model_path / "model.safetensors"))
    instance = Model(config)
    apply_quantization(instance, config, weights)
    instance.load_weights(list(weights.items()), strict=True)
    instance.model_path = model_path
    if not lazy:
        mx.eval(instance.parameters())
    instance.model_revision = revision or _snapshot_revision(model_path)
    return instance


def resolve_language_artifact(language: str | None, *, quantize: bool = False) -> Path | str:
    language = language or DEFAULT_LANGUAGE
    if language == "french":
        raise ValueError("Only the official 24-layer French model is available; use 'french_24l'.")
    try:
        artifact_name = LANGUAGE_ARTIFACTS[language]
    except KeyError as error:
        available = ", ".join(sorted(LANGUAGE_ARTIFACTS))
        raise ValueError(
            f"Unknown language {language!r}; available languages: {available}"
        ) from error

    roots = []
    if configured_root := os.environ.get("MLX_POCKET_TTS_MODEL_ROOT"):
        roots.append(Path(configured_root).expanduser())
    roots.extend((Path.cwd() / "models", Path(__file__).resolve().parents[2] / "models"))
    candidates = []
    for root in roots:
        base = root / artifact_name
        candidates.extend(([base.with_name(base.name + "-8bit"), base] if quantize else [base]))
    for candidate in candidates:
        if (candidate / "config.json").is_file() and (candidate / "model.safetensors").is_file():
            return candidate
    if language in {"english", "english_2026-04"} and not quantize:
        return DEFAULT_MODEL
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"No converted MLX artifact for language {language!r}. Searched: {searched}. "
        "Convert the official artifact or set MLX_POCKET_TTS_MODEL_ROOT."
    )


def load_model(
    *,
    language: str | None = None,
    config: str | Path | None = None,
    temp: float | None = None,
    sampler_decode_steps: int = 1,
    noise_clamp: float | None = None,
    eos_threshold: float = -4.0,
    quantize: bool = False,
    checkpoint: str | Path | None = None,
    lsd_decode_steps: int | None = None,
    revision: str | None = None,
) -> Model:
    if config is not None and language is not None:
        raise ValueError("Cannot specify both config and language, please choose one or the other.")
    if lsd_decode_steps is not None:
        sampler_decode_steps = lsd_decode_steps

    if config is None:
        artifact = resolve_language_artifact(language)
    else:
        config_source = str(config)
        artifact = download_if_necessary(config_source)
        if artifact.is_file() and artifact.name == "config.json":
            artifact = artifact.parent
        elif artifact.suffix in {".yaml", ".yml"}:
            local_language = artifact.stem
            if local_language in LANGUAGE_ARTIFACTS:
                try:
                    artifact = resolve_language_artifact(local_language)
                except FileNotFoundError:
                    artifact = _convert_yaml_to_cache(artifact)
            else:
                artifact = _convert_yaml_to_cache(artifact)
        elif not artifact.is_dir():
            raise ValueError(
                "MLX config must currently resolve to a converted artifact directory, config.json, "
                "or an official local YAML whose language artifact has already been converted."
            )

    if checkpoint is not None:
        from .legacy import materialize_checkpoint_artifact

        artifact = materialize_checkpoint_artifact(artifact, checkpoint)

    if quantize:
        artifact = ensure_quantized_artifact(artifact, revision=revision)

    instance = load(artifact, revision=revision)
    if temp is not None:
        instance.temp = temp
    instance.lsd_decode_steps = sampler_decode_steps
    instance.noise_clamp = noise_clamp
    instance.eos_threshold = eos_threshold
    instance.origin = language or (Path(config).stem if config is not None else DEFAULT_LANGUAGE)
    return instance


def _convert_yaml_to_cache(config_path: Path) -> Path:
    from .conversion import convert_config_artifact

    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()[:16]
    output = make_cache_directory() / "converted" / digest
    if (output / "config.json").is_file() and (output / "model.safetensors").is_file():
        return output
    convert_config_artifact(config_path, output)
    return output


def ensure_quantized_artifact(
    artifact: str | Path, *, revision: str | None = None, bits: int = 8, group_size: int = 64
) -> Path:
    """Return an existing or deterministically cached MLX quantized artifact."""
    from .quantization import quantize_artifact

    source = resolve_model_path(artifact, revision=revision)
    config = json.loads((source / "config.json").read_text())
    quantization = config.get("quantization", {})
    if int(quantization.get("bits", 0)) == bits:
        return source

    sibling = source.with_name(f"{source.name}-{bits}bit")
    if (sibling / "config.json").is_file() and (sibling / "model.safetensors").is_file():
        return sibling

    fingerprint = hashlib.sha256()
    fingerprint.update((source / "config.json").read_bytes())
    fingerprint.update(str((source / "model.safetensors").stat().st_size).encode())
    fingerprint.update(f"{bits}:{group_size}".encode())
    output = make_cache_directory() / "quantized" / fingerprint.hexdigest()[:16]
    if (output / "config.json").is_file() and (output / "model.safetensors").is_file():
        return output
    quantize_artifact(source, output, bits=bits, group_size=group_size)
    return output


def _snapshot_revision(path: Path) -> str | None:
    if path.parent.name == "snapshots" and len(path.name) == 40:
        return path.name
    return None
