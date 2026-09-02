import struct

import mlx.core as mx
from fastapi.testclient import TestClient

from mlx_pocket_tts.server import create_app, pcm_wav_stream


def test_streaming_wav_header_and_pcm_payload():
    parts = list(pcm_wav_stream([mx.zeros((240,))], 24000))
    assert parts[0][:4] == b"RIFF"
    assert parts[0][8:12] == b"WAVE"
    assert struct.unpack("<I", parts[0][40:44])[0] == 0xFFFFFFFF
    assert len(parts[1]) == 480


class FakeModel:
    sample_rate = 24000
    model_revision = "test-revision"
    origin = "french_24l"

    def __init__(self):
        self.voices = []
        self.states = []

    def get_state_for_audio_prompt(self, voice, truncate=False):
        self.voices.append((str(voice), truncate))
        return {"voice": str(voice)}

    def _cached_get_state_for_audio_prompt(self, voice, truncate=False):
        return self.get_state_for_audio_prompt(voice, truncate)

    def generate_audio_stream(self, state, text):
        self.states.append((state, text))
        yield mx.zeros((240,))


def test_official_web_root_and_default_voice():
    model = FakeModel()
    client = TestClient(create_app(model, "estelle"))
    root = client.get("/")
    assert root.status_code == 200
    assert "Bonjour le monde" in root.text
    response = client.post("/tts", data={"text": "Bonjour."})
    assert response.status_code == 200
    assert response.content[:4] == b"RIFF"
    assert model.states == [({"voice": "estelle"}, "Bonjour.")]


def test_official_voice_url_validation_and_upload():
    model = FakeModel()
    client = TestClient(create_app(model, "alba"))
    rejected = client.post("/tts", data={"text": "Hello", "voice_url": "./local.wav"})
    assert rejected.status_code == 400
    accepted = client.post("/tts", data={"text": "Hello", "voice_url": "marius"})
    assert accepted.status_code == 200
    uploaded = client.post(
        "/tts",
        data={"text": "Hello"},
        files={"voice_wav": ("voice.wav", b"fake audio", "audio/wav")},
    )
    assert uploaded.status_code == 200
    assert model.voices[-1][1] is True


def test_official_endpoint_rejects_two_voice_inputs():
    client = TestClient(create_app(FakeModel(), "alba"))
    response = client.post(
        "/tts",
        data={"text": "Hello", "voice_url": "marius"},
        files={"voice_wav": ("voice.wav", b"fake audio", "audio/wav")},
    )
    assert response.status_code == 400
