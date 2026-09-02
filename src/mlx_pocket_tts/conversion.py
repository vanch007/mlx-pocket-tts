from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import mlx.core as mx
import yaml
from huggingface_hub import hf_hub_download, list_repo_files
from mlx.utils import tree_flatten

from .pocket_tts import Model
from .quantization import apply_quantization
from .utils import download_if_necessary

OFFICIAL_SOURCE_REVISION = "e244e9a0dad520dad8f61a3c5882e91ea4859439"
PUBLIC_MODEL_REVISION = "d29db7978e464fb90cb3359ee0c69a273b9142cc"


def _seanet_name(name: str) -> str:
    encoder = {
        "0": "init_conv1d.conv.conv",
        "3": "layers.0.downsample.conv.conv",
        "6": "layers.1.downsample.conv.conv",
        "9": "layers.2.downsample.conv.conv",
        "11": "final_conv1d.conv.conv",
    }
    decoder = {
        "0": "init_conv1d.conv.conv",
        "2": "layers.0.upsample.convtr.convtr",
        "5": "layers.1.upsample.convtr.convtr",
        "8": "layers.2.upsample.convtr.convtr",
        "11": "final_conv1d.conv.conv",
    }
    match = re.fullmatch(
        r"mimi\.(encoder|decoder)\.model\.(\d+)\.(conv|convtr)\.(weight|bias)",
        name,
    )
    if match:
        side, index, _kind, suffix = match.groups()
        table = encoder if side == "encoder" else decoder
        return f"mimi.{side}.{table[index]}.{suffix}"
    match = re.fullmatch(
        r"mimi\.(encoder|decoder)\.model\.(\d+)\.block\.(1|3)\.conv\.(weight|bias)",
        name,
    )
    if match:
        side, index, block, suffix = match.groups()
        layer_index = {
            "encoder": {"1": 0, "4": 1, "7": 2},
            "decoder": {"3": 0, "6": 1, "9": 2},
        }[side][index]
        block_index = {"1": 0, "3": 1}[block]
        return (
            f"mimi.{side}.layers.{layer_index}.residuals.0.block.{block_index}.conv.conv.{suffix}"
        )
    return name


def map_weight_name(name: str) -> str:
    if name == "flow_lm.speaker_proj_weight":
        return "speaker_proj_weight"
    name = _seanet_name(name)
    name = re.sub(
        r"^(mimi\.(?:encoder|decoder)_transformer\.transformer\.layers\.\d+)\.linear([12])\.weight$",
        r"\1.gating.linear\2.weight",
        name,
    )
    if name == "mimi.downsample.conv.conv.weight":
        return "mimi.downsample.conv.conv.conv.weight"
    if name == "mimi.upsample.convtr.convtr.weight":
        return "mimi.upsample.convtr.convtr.convtr.weight"
    return name


def _reshape_for_target(value: mx.array, target_shape: tuple[int, ...], name: str) -> mx.array:
    if tuple(value.shape) == target_shape:
        return value
    if value.ndim == 3:
        for axes in ((0, 2, 1), (1, 2, 0), (2, 0, 1), (2, 1, 0), (1, 0, 2)):
            candidate = value.transpose(*axes)
            if tuple(candidate.shape) == target_shape:
                return candidate
    raise ValueError(f"Cannot map {name}: source {tuple(value.shape)} target {target_shape}")


def _convert_weights(config: dict, source_weights: Path) -> dict[str, mx.array]:
    model = Model(config)
    target_shapes = {name: tuple(value.shape) for name, value in tree_flatten(model.parameters())}
    source = mx.load(str(source_weights))
    converted: dict[str, mx.array] = {}
    unmapped_source: list[str] = []
    for source_name, value in source.items():
        target_name = map_weight_name(source_name)
        target_shape = target_shapes.get(target_name)
        if target_shape is None:
            unmapped_source.append(source_name)
            continue
        converted[target_name] = _reshape_for_target(value, target_shape, source_name)
    missing = sorted(set(target_shapes) - set(converted))
    if missing or unmapped_source:
        raise ValueError(
            f"Strict conversion failed: missing={missing}, unmapped_source={unmapped_source}"
        )
    return converted


def convert_config_artifact(config_path: Path, output: Path) -> dict:
    """Convert the weights referenced by an official-compatible YAML config."""
    config = yaml.safe_load(config_path.read_text())
    config["model_type"] = "pocket_tts"
    primary = config.get("weights_path")
    fallback = config.get("weights_path_without_voice_cloning")
    if primary is None and fallback is None:
        raise ValueError("Config must define weights_path or weights_path_without_voice_cloning.")
    selected = primary
    has_voice_cloning = primary is not None
    try:
        source_weights = download_if_necessary(selected)
        if not source_weights.is_file():
            raise FileNotFoundError(source_weights)
    except (OSError, RuntimeError):
        if fallback is None or selected == fallback:
            raise
        selected = fallback
        has_voice_cloning = False
        source_weights = download_if_necessary(selected)

    tokenizer_source = config.get("flow_lm", {}).get("lookup_table", {}).get("tokenizer_path")
    if not tokenizer_source:
        raise ValueError("Config must define flow_lm.lookup_table.tokenizer_path.")
    tokenizer = download_if_necessary(tokenizer_source)
    config["flow_lm"]["lookup_table"]["tokenizer_path"] = str(tokenizer)
    config["has_voice_cloning"] = has_voice_cloning
    converted = _convert_weights(config, source_weights)

    output.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(output / "model.safetensors"), converted)
    shutil.copy2(tokenizer, output / "tokenizer.model")
    config["flow_lm"]["lookup_table"]["tokenizer_path"] = "tokenizer.model"
    config["weights_path"] = None
    config["weights_path_without_voice_cloning"] = None
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    report = {
        "pass": True,
        "official_source_revision": OFFICIAL_SOURCE_REVISION,
        "config": str(config_path),
        "selected_weights": str(selected),
        "parameter_count": len(converted),
        "has_voice_cloning": has_voice_cloning,
    }
    (output / "conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def convert_official(
    *,
    language: str,
    output: Path,
    config_path: Path,
    source_repo: str = "kyutai/pocket-tts-without-voice-cloning",
    revision: str = PUBLIC_MODEL_REVISION,
) -> dict:
    config = yaml.safe_load(config_path.read_text())
    config["model_type"] = "pocket_tts"
    config["has_voice_cloning"] = "without-voice-cloning" not in source_repo
    tokenizer_remote = f"languages/{language}/tokenizer.model"
    tokenizer = Path(hf_hub_download(source_repo, tokenizer_remote, revision=revision))
    source_weights = Path(
        hf_hub_download(source_repo, f"languages/{language}/model.safetensors", revision=revision)
    )
    config["flow_lm"]["lookup_table"]["tokenizer_path"] = str(tokenizer)
    converted = _convert_weights(config, source_weights)
    output.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(output / "model.safetensors"), converted)
    shutil.copy2(tokenizer, output / "tokenizer.model")
    config["flow_lm"]["lookup_table"]["tokenizer_path"] = "tokenizer.model"
    config["weights_path"] = None
    config["weights_path_without_voice_cloning"] = None
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    embeddings_dir = output / "embeddings"
    embeddings_dir.mkdir(exist_ok=True)
    copied_voices = []
    prefix = f"languages/{language}/embeddings/"
    voice_files = sorted(
        name
        for name in list_repo_files(source_repo, revision=revision)
        if name.startswith(prefix) and name.endswith(".safetensors")
    )
    for remote_name in voice_files:
        voice = Path(remote_name).stem
        voice_path = hf_hub_download(source_repo, remote_name, revision=revision)
        shutil.copy2(voice_path, embeddings_dir / f"{voice}.safetensors")
        copied_voices.append(voice)
    report = {
        "pass": True,
        "official_source_revision": OFFICIAL_SOURCE_REVISION,
        "source_repo": source_repo,
        "source_revision": revision,
        "language": language,
        "parameter_count": len(converted),
        "voices": copied_voices,
        "has_voice_cloning": config["has_voice_cloning"],
    }
    (output / "conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def audit_checkpoint(path: Path) -> dict:
    config = json.loads((path / "config.json").read_text())
    config["flow_lm"]["lookup_table"]["tokenizer_path"] = str(path / "tokenizer.model")
    model = Model(config)
    weights = mx.load(str(path / "model.safetensors"))
    apply_quantization(model, config, weights)
    target_shapes = {name: tuple(value.shape) for name, value in tree_flatten(model.parameters())}
    actual_shapes = {name: tuple(value.shape) for name, value in weights.items()}
    report = {
        "pass": target_shapes == actual_shapes,
        "missing": sorted(set(target_shapes) - set(actual_shapes)),
        "extra": sorted(set(actual_shapes) - set(target_shapes)),
        "shape_mismatches": sorted(
            name
            for name in set(target_shapes) & set(actual_shapes)
            if target_shapes[name] != actual_shapes[name]
        ),
        "parameter_count": len(actual_shapes),
    }
    return report
