from __future__ import annotations

from pathlib import Path

DEFAULT_LANGUAGE = "english"
DEFAULT_TEXT_FOR_LANGUAGE = {
    "english": (
        "Hello world. I am Kyutai's Pocket TTS. I'm fast enough to run on small CPUs. "
        "I hope you'll like me."
    ),
    "french": (
        "Bonjour le monde. Je suis le TTS de poche de Kyutai. Je suis assez rapide pour "
        "fonctionner sur de petits CPU. J'espère que vous m'aimerez."
    ),
    "german": (
        "Hallo Welt. Ich bin Pocket TTS von Kyutai. Ich bin schnell genug, um auch auf "
        "kleinen CPUs zu laufen. Ich hoffe, ich gefalle dir."
    ),
    "portuguese": (
        "Olá mundo. Eu sou o Pocket TTS da Kyutai. Sou rápido o suficiente para rodar em "
        "CPUs pequenas. Espero que você goste de mim."
    ),
    "italian": (
        "Ciao mondo. Sono il Pocket TTS di Kyutai. Sono abbastanza veloce da funzionare su "
        "piccole CPU. Spero che ti piacerò."
    ),
    "spanish": (
        "Hola mundo. Soy el Pocket TTS de Kyutai. Soy lo suficientemente rápido para "
        "funcionar en pequeñas CPU. Espero que te guste."
    ),
}
DEFAULT_VOICE_FOR_LANGUAGE = {
    "french": "estelle",
    "german": "juergen",
    "italian": "giovanni",
    "portuguese": "rafael",
    "spanish": "lola",
}
DEFAULT_VOICE_FALLBACK = "alba"

# Names match the pinned official config stems. Values are local converted artifact names.
LANGUAGE_ARTIFACTS = {
    "english": "english-clone-mlx",
    "english_2026-01": "english_2026-01-public-mlx",
    "english_2026-04": "english-clone-mlx",
    "french_24l": "french_24l-public-mlx",
    "german": "german-public-mlx",
    "german_24l": "german_24l-public-mlx",
    "italian": "italian-public-mlx",
    "italian_24l": "italian_24l-public-mlx",
    "portuguese": "portuguese-public-mlx",
    "portuguese_24l": "portuguese_24l-public-mlx",
    "spanish": "spanish-public-mlx",
    "spanish_24l": "spanish_24l-public-mlx",
}


def get_default_text_for_language(language: str | None) -> str:
    for key, text in DEFAULT_TEXT_FOR_LANGUAGE.items():
        if language is not None and key in language:
            return text
    return DEFAULT_TEXT_FOR_LANGUAGE[DEFAULT_LANGUAGE]


def get_default_voice_for_language(
    language: str | None, config: str | Path | None = None, checkpoint: str | Path | None = None
) -> str:
    if config is not None or checkpoint is not None:
        return "hf://kyutai/tts-voices/alba-mackenna/casual.wav"
    for key, voice in DEFAULT_VOICE_FOR_LANGUAGE.items():
        if language is not None and key in language:
            return voice
    return DEFAULT_VOICE_FALLBACK
