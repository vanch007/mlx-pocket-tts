from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from .utils import download_if_necessary, make_cache_directory


def import_legacy_checkpoint(checkpoint: str | Path) -> Path:
    source = download_if_necessary(str(checkpoint))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    output_dir = make_cache_directory() / "legacy-checkpoints" / digest[:20]
    output = output_dir / "weights.npz"
    if output.is_file():
        return output
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / ".weights.tmp.npz"
    command = [sys.executable, "-m", "mlx_pocket_tts.legacy_importer", str(source), str(temporary)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"Legacy checkpoint import failed for {source}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    temporary.replace(output)
    (output_dir / "import.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "sha256": digest,
                "importer_stdout": completed.stdout.strip(),
            },
            indent=2,
        )
        + "\n"
    )
    return output


def materialize_checkpoint_artifact(artifact: str | Path, checkpoint: str | Path) -> Path:
    from .loader import load, resolve_model_path

    source_artifact = resolve_model_path(artifact)
    imported = import_legacy_checkpoint(checkpoint)
    fingerprint = hashlib.sha256()
    fingerprint.update((source_artifact / "config.json").read_bytes())
    fingerprint.update(imported.read_bytes())
    output = make_cache_directory() / "checkpoint-artifacts" / fingerprint.hexdigest()[:20]
    weights_path = output / "model.safetensors"
    if weights_path.is_file() and (output / "config.json").is_file():
        return output

    model = load(source_artifact)
    weights = mx.load(str(imported))
    target_shapes = {name: tuple(value.shape) for name, value in tree_flatten(model.parameters())}
    unknown = sorted(set(weights) - set(target_shapes))
    mismatched = sorted(
        name for name, value in weights.items() if tuple(value.shape) != target_shapes.get(name)
    )
    if unknown or mismatched:
        raise ValueError(
            f"Legacy checkpoint is incompatible: unknown={unknown}, shape_mismatches={mismatched}"
        )
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / ".model.tmp.safetensors"
    mx.save_safetensors(str(temporary), dict(tree_flatten(model.parameters())))
    temporary.replace(weights_path)
    for name in ("config.json", "tokenizer.model"):
        shutil.copy2(source_artifact / name, output / name)
    if (source_artifact / "embeddings").is_dir():
        shutil.copytree(source_artifact / "embeddings", output / "embeddings", dirs_exist_ok=True)
    return output
