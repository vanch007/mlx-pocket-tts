import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import soundfile as sf

from mlx_pocket_tts.training import LatentDataLoader, load_entries, precompute_manifest


class FakeMimi(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_weight = mx.array([1.0])
        self.frame_rate = 12.5

    def encode_to_latent(self, audio):
        return audio[..., ::80]


class FakeModel:
    sample_rate = 8000

    def __init__(self):
        self.mimi = FakeMimi()


def _write_fixture(root: Path, transcript: str = "hello") -> tuple[Path, Path]:
    audio = root / "audio.wav"
    sf.write(audio, np.linspace(-0.2, 0.2, 8000, dtype=np.float32), 8000)
    manifest = root / "train.jsonl"
    manifest.write_text(
        json.dumps({"path": "audio.wav", "duration": 1.0, "transcript": transcript}) + "\n"
    )
    return audio, manifest


def test_manifest_validation_reports_line_and_field(tmp_path):
    _, manifest = _write_fixture(tmp_path, transcript=" ")
    with pytest.raises(ValueError, match=r"train.jsonl:1: transcript must not be empty"):
        load_entries(manifest)


def test_latent_cache_reuse_and_audio_invalidation(tmp_path):
    audio, manifest = _write_fixture(tmp_path)
    output = tmp_path / "cache"
    first = precompute_manifest(manifest, FakeModel(), output)
    assert first["encoded"] == 1 and first["reused"] == 0
    second = precompute_manifest(manifest, FakeModel(), output)
    assert second["encoded"] == 0 and second["reused"] == 1

    sf.write(audio, np.linspace(0.2, -0.2, 8000, dtype=np.float32), 8000)
    third = precompute_manifest(manifest, FakeModel(), output)
    assert third["encoded"] == 1 and third["reused"] == 0
    assert len(list((output / "latents").glob("*.safetensors"))) == 2


class FakeTokenizer:
    def encode(self, text):
        return list(range(len(text.split())))


def test_latent_loader_builds_official_training_batch(tmp_path):
    _, manifest = _write_fixture(tmp_path)
    output = tmp_path / "cache"
    result = precompute_manifest(manifest, FakeModel(), output)
    annotated = load_entries(result["manifest"])
    assert annotated[0].path == (tmp_path / "audio.wav").resolve()
    loader = LatentDataLoader(
        result["manifest"], FakeTokenizer(), batch_size=1, frame_rate=12.5, shuffle=False
    )
    batch = next(iter(loader))
    assert batch.latents.ndim == 3
    assert batch.mask.shape == batch.latents.shape[:2]
    assert batch.voice_latents.ndim == 3
    assert batch.num_voice_prompt_frames.tolist() == [batch.voice_latents.shape[1]]
    assert batch.text_tokens[0].tolist() == [0]

    shuffled = LatentDataLoader(
        result["manifest"], FakeTokenizer(), batch_size=1, frame_rate=12.5, shuffle=True
    )
    assert next(iter(shuffled)).latents.shape == batch.latents.shape
