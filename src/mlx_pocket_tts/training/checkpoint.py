from __future__ import annotations

import json
import os
import re
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten


def _atomic_safetensors(path: Path, tensors: dict[str, mx.array]) -> None:
    temporary = path.with_name(f".{path.stem}.tmp.{os.getpid()}.safetensors")
    mx.save_safetensors(str(temporary), tensors)
    temporary.replace(path)


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class EMA:
    def __init__(self, model, decay: float):
        if not 0 < decay < 1:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = decay
        self.shadow = {
            name: mx.array(value.astype(mx.float32))
            for name, value in tree_flatten(model.trainable_parameters())
        }
        mx.eval(self.shadow)

    def update(self, model) -> None:
        parameters = dict(tree_flatten(model.trainable_parameters()))
        for name in list(self.shadow):
            if name in parameters:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * parameters[
                    name
                ].astype(mx.float32)
        mx.eval(self.shadow)

    def load_state_dict(self, state: dict[str, mx.array]) -> None:
        self.shadow = {name: mx.array(value.astype(mx.float32)) for name, value in state.items()}
        mx.eval(self.shadow)


def save_checkpoint(
    run_dir: str | Path,
    step: int,
    model,
    optimizer,
    ema: EMA | None,
    *,
    num_keep: int = 3,
) -> Path:
    if num_keep < 1:
        raise ValueError("num_keep must be positive")
    run_dir = Path(run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    stem = f"checkpoint_{step:08d}"
    model_path = run_dir / f"{stem}.safetensors"
    optimizer_path = run_dir / f"optim_{step:08d}.safetensors"
    ema_path = run_dir / f"ema_{step:08d}.safetensors"
    metadata_path = run_dir / f"{stem}.json"
    _atomic_safetensors(model_path, dict(tree_flatten(model.parameters())))
    _atomic_safetensors(optimizer_path, dict(tree_flatten(optimizer.state)))
    if ema is not None:
        _atomic_safetensors(ema_path, ema.shadow)
    _atomic_json(
        metadata_path,
        {
            "format": "mlx-pocket-tts-training-v1",
            "step": step,
            "model": model_path.name,
            "optimizer": optimizer_path.name,
            "ema": ema_path.name if ema is not None else None,
        },
    )
    checkpoints = sorted(run_dir.glob("checkpoint_*.json"))
    for old_metadata in checkpoints[:-num_keep]:
        old = json.loads(old_metadata.read_text())
        for key in ("model", "optimizer", "ema"):
            if old.get(key):
                (run_dir / old[key]).unlink(missing_ok=True)
        old_metadata.unlink()
    return metadata_path


def latest_checkpoint(run_dir: str | Path) -> Path | None:
    candidates = sorted(
        path
        for path in Path(run_dir).expanduser().glob("checkpoint_*.json")
        if re.fullmatch(r"checkpoint_\d{8}\.json", path.name)
    )
    return candidates[-1] if candidates else None


def load_checkpoint(path: str | Path, model, optimizer=None, ema: EMA | None = None) -> int:
    path = Path(path).expanduser().resolve()
    metadata = json.loads(path.read_text())
    if metadata.get("format") != "mlx-pocket-tts-training-v1":
        raise ValueError(f"Unsupported checkpoint format in {path}")
    model.load_weights(list(mx.load(str(path.parent / metadata["model"])).items()), strict=True)
    if optimizer is not None:
        optimizer_state = mx.load(str(path.parent / metadata["optimizer"]))
        optimizer.state = tree_unflatten(list(optimizer_state.items()))
    if ema is not None and metadata.get("ema"):
        ema.load_state_dict(mx.load(str(path.parent / metadata["ema"])))
    mx.eval(model.parameters(), optimizer.state if optimizer is not None else [])
    return int(metadata["step"])
