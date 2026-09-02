"""Measure reference/generated speaker similarity with SpeechBrain ECAPA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
BACKEND = "speechbrain_ecapa_voxceleb"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _p10(values: list[float]) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = 0.1 * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _segments(path: Path, *, sample_rate: int = 16_000):
    import librosa
    import numpy as np
    import soundfile as sf

    wave, source_rate = sf.read(str(path), always_2d=True)
    if not wave.size:
        raise ValueError(f"empty audio: {path}")
    mono = wave.mean(axis=1).astype("float32")
    if not np.isfinite(mono).all():
        raise ValueError(f"non-finite audio: {path}")
    if int(source_rate) != sample_rate:
        mono = librosa.resample(mono, orig_sr=int(source_rate), target_sr=sample_rate)
    mono, _ = librosa.effects.trim(mono - float(mono.mean()), top_db=35)
    if len(mono) / sample_rate < 0.75:
        raise ValueError(f"insufficient voiced audio: {path}")
    window, hop = 3 * sample_rate, int(1.5 * sample_rate)
    if len(mono) <= window:
        starts = [0]
    else:
        starts = list(range(0, len(mono) - window + 1, hop))
        if starts[-1] != len(mono) - window:
            starts.append(len(mono) - window)
    if len(starts) > 16:
        indexes = np.linspace(0, len(starts) - 1, 16)
        starts = sorted({starts[round(index)] for index in indexes})
    output = []
    for start in starts:
        segment = mono[start : start + window].astype("float32", copy=False)
        valid = len(segment)
        if valid < window:
            segment = np.pad(segment, (0, window - valid))
        output.append((segment, valid))
    return output


def _classifier(cache: Path, device: str):
    import huggingface_hub
    import torch
    import torchaudio

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: []
    from speechbrain.inference.speaker import EncoderClassifier

    original = huggingface_hub.hf_hub_download

    def compatible(*args, **kwargs):
        if "use_auth_token" in kwargs and "token" not in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        else:
            kwargs.pop("use_auth_token", None)
        return original(*args, **kwargs)

    huggingface_hub.hf_hub_download = compatible
    try:
        classifier = EncoderClassifier.from_hparams(source=str(cache), savedir=str(cache))
    finally:
        huggingface_hub.hf_hub_download = original
    classifier.device = device
    for module in classifier.mods.values():
        module.to(torch.device(device)).eval()
    return classifier


def _encode(classifier, items, device: str):
    import numpy as np
    import torch

    length = max(len(samples) for samples, _ in items)
    batch = np.zeros((len(items), length), dtype="float32")
    ratios = np.zeros(len(items), dtype="float32")
    for index, (samples, valid) in enumerate(items):
        batch[index, : len(samples)] = samples
        ratios[index] = valid / length
    with torch.inference_mode():
        embeddings = classifier.encode_batch(
            torch.from_numpy(batch).to(device),
            torch.from_numpy(ratios).to(device),
            normalize=False,
        )
    return embeddings.flatten(1).float()


def analyze(generated: Path, reference: Path, requested_device: str) -> dict:
    import torch

    cache = Path.home() / ".cache/speechbrain/spkrec-ecapa-voxceleb"
    weights = cache / "embedding_model.ckpt"
    if not (cache / "hyperparams.yaml").is_file() or not weights.is_file():
        raise FileNotFoundError(f"incomplete local ECAPA cache: {cache}")
    preferred = "mps" if requested_device != "cpu" and torch.backends.mps.is_available() else "cpu"
    attempts = [preferred] + (["cpu"] if preferred == "mps" and requested_device == "auto" else [])
    error = None
    for device in attempts:
        try:
            classifier = _classifier(cache, device)
            generated_embeddings = _encode(classifier, _segments(generated), device)
            reference_items = _segments(reference)
            reference_embeddings = _encode(classifier, reference_items, device)
            centroid = torch.nn.functional.normalize(reference_embeddings.mean(dim=0), dim=0)
            reference_scores = (
                torch.nn.functional.cosine_similarity(
                    reference_embeddings, centroid.unsqueeze(0), dim=1
                )
                .detach()
                .cpu()
                .tolist()
            )
            generated_scores = (
                torch.nn.functional.cosine_similarity(
                    generated_embeddings, centroid.unsqueeze(0), dim=1
                )
                .detach()
                .cpu()
                .tolist()
            )
            median = float(statistics.median(generated_scores))
            tail = float(_p10([float(value) for value in generated_scores]))
            return {
                "backend": BACKEND,
                "model_id": MODEL_ID,
                "model_artifact_sha256": _sha256(weights),
                "device": device,
                "cosine": 0.65 * median + 0.35 * tail,
                "median_cosine": median,
                "p10_cosine": tail,
                "generated_segments": len(generated_scores),
                "reference_segments": len(reference_scores),
                "reference_consistency": float(_p10([float(value) for value in reference_scores])),
            }
        except Exception as exc:
            error = exc
    raise RuntimeError(str(error))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    args = parser.parse_args()
    print(json.dumps(analyze(args.generated, args.reference, args.device)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
