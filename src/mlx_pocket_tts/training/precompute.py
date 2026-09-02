from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf
from mlx.utils import tree_flatten
from scipy.signal import resample_poly

from .data import Entry, load_entries


def mimi_encode_fingerprint(mimi) -> str:
    digest = hashlib.sha256()
    parameters = dict(tree_flatten(mimi.parameters()))
    encode_names = ("encoder", "encoder_transformer", "downsample")
    selected = [
        (name, value) for name, value in parameters.items() if name.startswith(encode_names)
    ]
    if not selected:
        selected = sorted(parameters.items())
    for name, value in sorted(selected):
        mx.eval(value)
        dtype = str(value.dtype)
        try:
            array = np.asarray(value)
        except ValueError:
            array = np.asarray(value.astype(mx.float32))
        digest.update(f"{name}:{array.shape}:{dtype}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _load_window(entry: Entry, sample_rate: int) -> mx.array:
    info = sf.info(str(entry.path))
    start_frame = round(entry.start * info.samplerate)
    frame_count = round(entry.duration * info.samplerate)
    audio, source_rate = sf.read(
        str(entry.path),
        start=start_frame,
        frames=frame_count,
        dtype="float32",
        always_2d=True,
    )
    if audio.size == 0:
        raise ValueError(f"Decoded empty audio window from {entry.path}")
    mono = audio.mean(axis=1)
    if source_rate != sample_rate:
        mono = resample_poly(mono, sample_rate, source_rate).astype(np.float32)
    return mx.array(mono[None, None, :], dtype=mx.float32)


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text)
    temporary.replace(path)


def _atomic_safetensors(path: Path, tensors: dict[str, mx.array]) -> None:
    temporary = path.with_name(f".{path.stem}.tmp.{os.getpid()}.safetensors")
    mx.save_safetensors(str(temporary), tensors)
    temporary.replace(path)


def precompute_manifest(manifest: str | Path, model, output_dir: str | Path) -> dict:
    manifest = Path(manifest).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    entries = load_entries(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    latents_dir = output_dir / "latents"
    latents_dir.mkdir(exist_ok=True)
    encoder_hash = mimi_encode_fingerprint(model.mimi)
    encoded = reused = 0
    annotated = []
    for entry in entries:
        source_hash = entry.source_fingerprint()
        cache_key = hashlib.sha256(f"{source_hash}:{encoder_hash}".encode()).hexdigest()
        destination = latents_dir / f"{cache_key}.safetensors"
        if destination.is_file():
            reused += 1
        else:
            audio = _load_window(entry, model.sample_rate)
            latents = model.mimi.encode_to_latent(audio)
            mx.eval(latents)
            _atomic_safetensors(destination, {"latents": latents[0]})
            encoded += 1
        value = entry.to_dict()
        value["path"] = os.path.relpath(entry.path, output_dir)
        value["latents_file"] = str(destination.relative_to(output_dir))
        value["source_fingerprint"] = source_hash
        annotated.append(json.dumps(value, ensure_ascii=False))

    output_manifest = output_dir / f"{manifest.stem}_latents.jsonl"
    _atomic_text(output_manifest, "\n".join(annotated) + "\n")
    meta = {
        "source_manifest": str(manifest),
        "encoder_fingerprint": encoder_hash,
        "sample_rate": model.sample_rate,
        "frame_rate": model.mimi.frame_rate,
        "entries": len(entries),
    }
    meta_path = output_dir / f"{manifest.stem}_latents.meta.json"
    _atomic_text(meta_path, json.dumps(meta, indent=2) + "\n")
    return {
        "status": "pass",
        "encoded": encoded,
        "reused": reused,
        "entries": len(entries),
        "manifest": str(output_manifest),
        "meta": str(meta_path),
        "encoder_fingerprint": encoder_hash,
    }
