from __future__ import annotations

import os
import shutil
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from .checkpoint import EMA


def export_inference_artifact(
    output_dir: str | Path,
    trainable,
    base_model,
    ema: EMA | None = None,
) -> Path:
    """Export a native training state in the exact local inference layout."""
    if base_model.model_path is None:
        raise ValueError("base_model.model_path is required to export config and tokenizer assets")
    source = Path(base_model.model_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    state = {
        f"flow_lm.{name}": value for name, value in tree_flatten(trainable.flow_lm.parameters())
    }
    state["speaker_proj_weight"] = trainable.speaker_proj_weight
    state.update(
        {f"mimi.{name}": value for name, value in tree_flatten(base_model.mimi.parameters())}
    )
    if ema is not None:
        for name, value in ema.shadow.items():
            if name.startswith("flow_lm.") or name == "speaker_proj_weight":
                state[name] = value

    weights = output / "model.safetensors"
    temporary = output / f".model.tmp.{os.getpid()}.safetensors"
    mx.save_safetensors(str(temporary), state)
    temporary.replace(weights)

    for name in ("config.json", "tokenizer.model"):
        source_file = source / name
        if not source_file.is_file():
            raise FileNotFoundError(f"Required inference asset not found: {source_file}")
        temporary_asset = output / f".{name}.tmp.{os.getpid()}"
        shutil.copy2(source_file, temporary_asset)
        temporary_asset.replace(output / name)
    source_embeddings = source / "embeddings"
    if source_embeddings.is_dir():
        destination_embeddings = output / "embeddings"
        destination_embeddings.mkdir(exist_ok=True)
        for voice in source_embeddings.glob("*.safetensors"):
            shutil.copy2(voice, destination_embeddings / voice.name)
    return output
