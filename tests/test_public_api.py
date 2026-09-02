import pytest

import mlx_pocket_tts


def test_public_api():
    assert mlx_pocket_tts.__all__ == ["TTSModel", "export_model_state"]
    assert mlx_pocket_tts.export_model_state is mlx_pocket_tts.export_voice_state


def test_official_api_methods_and_properties():
    for method_name in (
        "load_model",
        "generate_audio",
        "generate_audio_stream",
        "get_state_for_audio_prompt",
    ):
        assert callable(getattr(mlx_pocket_tts.TTSModel, method_name))
    for property_name in ("device", "sample_rate"):
        assert isinstance(getattr(mlx_pocket_tts.TTSModel, property_name), property)


def test_load_model_rejects_conflicting_language_and_config():
    with pytest.raises(ValueError, match="both config and language"):
        mlx_pocket_tts.TTSModel.load_model(language="english", config="config.yaml")
