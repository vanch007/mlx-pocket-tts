from pathlib import Path

import pytest

from mlx_pocket_tts.defaults import (
    get_default_text_for_language,
    get_default_voice_for_language,
)
from mlx_pocket_tts.loader import ensure_quantized_artifact, resolve_language_artifact


def _artifact(root: Path, name: str) -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}\n")
    (path / "model.safetensors").touch()
    return path


def test_official_language_defaults():
    assert get_default_voice_for_language("french_24l") == "estelle"
    assert get_default_voice_for_language("spanish") == "lola"
    assert get_default_voice_for_language(None) == "alba"
    assert get_default_text_for_language("german_24l").startswith("Hallo Welt")


def test_language_artifact_root_and_quantization(monkeypatch, tmp_path):
    base = _artifact(tmp_path, "english-clone-mlx")
    quantized = _artifact(tmp_path, "english-clone-mlx-8bit")
    monkeypatch.setenv("MLX_POCKET_TTS_MODEL_ROOT", str(tmp_path))
    assert resolve_language_artifact("english") == base
    assert resolve_language_artifact("english", quantize=True) == quantized


def test_unknown_and_unavailable_french_languages():
    with pytest.raises(ValueError, match="french_24l"):
        resolve_language_artifact("french")
    with pytest.raises(ValueError, match="Unknown language"):
        resolve_language_artifact("klingon")


def test_quantized_artifact_prefers_existing_sibling(tmp_path):
    base = _artifact(tmp_path, "model")
    quantized = _artifact(tmp_path, "model-8bit")
    assert ensure_quantized_artifact(base) == quantized
