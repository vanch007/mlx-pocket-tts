import hashlib
from pathlib import Path
from urllib.parse import urlsplit

import mlx.core as mx
import requests
from huggingface_hub import hf_hub_download

_voices_names = [
    "alba",
    "anna",
    "azelma",
    "bill_boerst",
    "caro_davy",
    "charles",
    "cosette",
    "eponine",
    "eve",
    "fantine",
    "george",
    "jane",
    "javert",
    "marius",
    "jean",
    "mary",
    "michael",
    "paul",
    "peter_yearsley",
    "stuart_bell",
    "vera",
]
OFFICIAL_VOICE_NAMES = set(_voices_names) | {
    "giovanni",
    "lola",
    "juergen",
    "rafael",
    "estelle",
}
PREDEFINED_VOICE_REVISION = "e81d79e8194ad4c7ce879c87a4258ef20cbf2487"
PREDEFINED_VOICES = {
    name: (
        "hf://kyutai/pocket-tts-without-voice-cloning/embeddings/"
        f"{name}.safetensors@d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3"
    )
    for name in _voices_names
}


def predefined_voice_source(voice_name: str, language: str | None = None) -> str:
    """Return the released voice state matching the model language.

    Preset states are model-specific. Reusing the English root embedding with a
    German, Spanish, or other language checkpoint can cause immediate EOS.
    """
    if voice_name not in OFFICIAL_VOICE_NAMES:
        raise ValueError(
            f"Predefined voice '{voice_name}' not found; "
            f"available voices are {sorted(OFFICIAL_VOICE_NAMES)}."
        )
    if language is not None:
        return (
            "hf://kyutai/pocket-tts-without-voice-cloning/languages/"
            f"{language}/embeddings/{voice_name}.safetensors@{PREDEFINED_VOICE_REVISION}"
        )
    if voice_name not in PREDEFINED_VOICES:
        raise ValueError(
            f"Predefined voice '{voice_name}' requires a language-specific model origin."
        )
    return PREDEFINED_VOICES[voice_name]


def make_cache_directory() -> Path:
    cache_dir = Path.home() / ".cache" / "mlx_pocket_tts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_if_necessary(path: str | Path) -> Path:
    path = str(path)
    if path.startswith("http://") or path.startswith("https://"):
        cache_dir = make_cache_directory()
        suffix = Path(urlsplit(path).path).suffix or ".bin"
        cached_file = cache_dir / f"{hashlib.sha256(path.encode()).hexdigest()}{suffix}"
        if not cached_file.exists():
            response = requests.get(path, timeout=30)
            response.raise_for_status()
            cached_file.write_bytes(response.content)
        return cached_file
    if path.startswith("hf://"):
        path = path.removeprefix("hf://")
        parts = path.split("/")
        repo_id = "/".join(parts[:2])
        filename = "/".join(parts[2:])
        if "@" in filename:
            filename, revision = filename.split("@", 1)
        else:
            revision = None
        cached_file = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
        return Path(cached_file)
    return Path(path)


def load_predefined_voice(voice_name: str, language: str | None = None) -> mx.array:
    voice_file = download_if_necessary(predefined_voice_source(voice_name, language))
    return mx.load(voice_file)["audio_prompt"]
