import os
import struct
import tempfile
import threading
from pathlib import Path
from typing import Annotated

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse

from .defaults import get_default_text_for_language, get_default_voice_for_language
from .loader import load, load_model
from .utils import OFFICIAL_VOICE_NAMES


def pcm_wav_stream(chunks, sample_rate: int):
    # Streaming responses do not know the final length up front. The all-ones
    # RIFF/data sizes are the conventional sentinel accepted by stream players.
    yield struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        0xFFFFFFFF,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        0xFFFFFFFF,
    )
    for chunk in chunks:
        pcm = np.clip(np.asarray(chunk), -1.0, 1.0)
        yield (pcm * 32767).astype("<i2").tobytes()


def create_app(model, default_voice: str = "alba") -> FastAPI:
    app = FastAPI(
        title="Kyutai Pocket TTS API",
        description="MLX-native text-to-speech generation API",
        version="1.0.0",
    )
    default_state = model.get_state_for_audio_prompt(default_voice)
    generation_lock = threading.Lock()

    @app.get("/", response_class=HTMLResponse)
    def root():
        static_path = Path(__file__).parent / "static" / "index.html"
        content = static_path.read_text()
        return content.replace("DEFAULT_TEXT_PROMPT", get_default_text_for_language(model.origin))

    @app.get("/health")
    def health():
        return {"status": "healthy", "model_revision": model.model_revision}

    @app.post("/tts")
    async def tts(
        text: Annotated[str, Form()],
        voice_url: Annotated[str | None, Form()] = None,
        voice_wav: Annotated[UploadFile | None, File()] = None,
        voice: Annotated[str | None, Form()] = None,
    ):
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        if voice_url is None:
            voice_url = voice
        if voice_url is not None and voice_wav is not None:
            raise HTTPException(
                status_code=400, detail="Cannot provide both voice_url and voice_wav"
            )
        try:
            if voice_wav is not None:
                suffix = Path(voice_wav.filename).suffix if voice_wav.filename else ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
                    temporary.write(await voice_wav.read())
                    temporary_path = Path(temporary.name)
                try:
                    state = model.get_state_for_audio_prompt(temporary_path, truncate=True)
                finally:
                    os.unlink(temporary_path)
            elif voice_url is not None:
                if not (
                    voice_url.startswith(("http://", "https://", "hf://"))
                    or voice_url in OFFICIAL_VOICE_NAMES
                ):
                    raise ValueError(
                        "voice_url must be a predefined voice or start with http://, https://, or hf://"
                    )
                state = model._cached_get_state_for_audio_prompt(voice_url)
            else:
                state = default_state
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        async def locked_payloads():
            with generation_lock:
                chunks = model.generate_audio_stream(state, text)
                for payload in pcm_wav_stream(chunks, model.sample_rate):
                    yield payload

        return StreamingResponse(
            locked_payloads(),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=generated_speech.wav"},
        )

    return app


def run_server(
    model_id: str | None,
    revision: str | None,
    default_voice: str | None,
    host: str,
    port: int,
    *,
    language: str | None = None,
    config: str | None = None,
    quantize: bool = False,
    reload: bool = False,
) -> None:
    if model_id:
        model = load(model_id, revision=revision)
    else:
        model = load_model(language=language, config=config, quantize=quantize)
    if default_voice is None:
        default_voice = get_default_voice_for_language(language, config)
    uvicorn.run(create_app(model, default_voice), host=host, port=port, reload=reload)
