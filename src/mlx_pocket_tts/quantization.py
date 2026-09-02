from __future__ import annotations

import json
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


def apply_quantization(model, config: dict, weights: dict[str, mx.array]) -> None:
    quantization = config.get("quantization")
    if not quantization:
        return
    group_size = int(quantization["group_size"])
    scope = str(quantization.get("scope", "flow_lm"))

    def predicate(path, module):
        return (
            path.startswith(scope)
            and hasattr(module, "to_quantized")
            and hasattr(module, "weight")
            and module.weight.shape[-1] % group_size == 0
            and f"{path}.scales" in weights
        )

    nn.quantize(
        model,
        group_size=group_size,
        bits=int(quantization["bits"]),
        mode=str(quantization.get("mode", "affine")),
        class_predicate=predicate,
    )


def quantize_artifact(
    source: Path,
    output: Path,
    *,
    bits: int = 8,
    group_size: int = 64,
) -> dict:
    from .loader import load

    model = load(source, lazy=True)

    def predicate(path, module):
        return (
            path.startswith("flow_lm")
            and hasattr(module, "to_quantized")
            and hasattr(module, "weight")
            and module.weight.shape[-1] % group_size == 0
        )

    nn.quantize(
        model,
        group_size=group_size,
        bits=bits,
        mode="affine",
        class_predicate=predicate,
    )
    weights = dict(tree_flatten(model.parameters()))
    output.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(output / "model.safetensors"), weights)
    config = json.loads((source / "config.json").read_text())
    config["quantization"] = {
        "bits": bits,
        "group_size": group_size,
        "mode": "affine",
        "scope": "flow_lm",
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    shutil.copy2(source / "tokenizer.model", output / "tokenizer.model")
    if (source / "embeddings").is_dir():
        shutil.copytree(source / "embeddings", output / "embeddings", dirs_exist_ok=True)
    report = {
        "pass": True,
        "source": str(source),
        "bits": bits,
        "group_size": group_size,
        "scope": "flow_lm",
        "parameter_count": len(weights),
    }
    (output / "quantization.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
