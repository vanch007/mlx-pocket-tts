from pathlib import Path

import pytest

from mlx_pocket_tts.defaults import (
    get_default_text_for_language,
    get_default_voice_for_language,
)
from mlx_pocket_tts.loader import ensure_quantized_artifact, resolve_language_artifact
from mlx_pocket_tts.pocket_tts import Model
from mlx_pocket_tts.utils import predefined_voice_source


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


def test_predefined_voice_state_is_scoped_to_model_language():
    source = predefined_voice_source("alba", language="spanish")
    assert "/languages/spanish/embeddings/alba.safetensors@" in source
    assert predefined_voice_source("juergen", language="german").startswith("hf://kyutai/")
    with pytest.raises(ValueError, match="language-specific model origin"):
        predefined_voice_source("lola")


def test_language_predefined_voice_restores_full_model_state(monkeypatch, tmp_path):
    model = Model.__new__(Model)
    model.model_path = None
    model.origin = "spanish"
    voice_file = tmp_path / "lola.safetensors"
    restored = {"flow_cache": ["restored"]}
    monkeypatch.setattr(
        "mlx_pocket_tts.pocket_tts.download_if_necessary",
        lambda source: voice_file,
    )
    monkeypatch.setattr(model, "_load_voice_file", lambda path: restored)

    assert model.get_state_for_audio_prompt("lola") is restored


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
