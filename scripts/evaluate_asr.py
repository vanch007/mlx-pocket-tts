#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import mlx_whisper

CASES = {
    "english": ("en", "Hello from Pocket TTS on Apple silicon."),
    "english_2026-01": ("en", "This older English checkpoint still works with MLX."),
    "english_2026-04": ("en", "The latest English checkpoint runs locally with MLX."),
    "french_24l": ("fr", "Bonjour, ce modèle français fonctionne localement sur Apple silicon."),
    "spanish_24l": ("es", "Hola, este modelo español funciona localmente con Apple silicon."),
    "german_24l": ("de", "Hallo, dieses deutsche Modell läuft lokal auf Apple silicon."),
    "german": ("de", "Hallo, dieses kompakte deutsche Modell läuft lokal."),
    "portuguese_24l": ("pt", "Olá, este modelo português funciona localmente no Apple silicon."),
    "portuguese": ("pt", "Olá, este modelo português compacto funciona localmente."),
    "italian_24l": ("it", "Ciao, questo modello italiano funziona localmente su Apple silicon."),
    "italian": ("it", "Ciao, questo modello italiano compatto funziona localmente."),
    "spanish": ("es", "Hola, este modelo español compacto funciona localmente."),
    "community_clone": ("en", "The cloned reference voice works in the standalone MLX port."),
    "community_clone_reuse": ("en", "The exported cloned voice state can be reused."),
    "official_clone": (
        "en",
        "The newest official Pocket TTS voice cloning weights now run natively with MLX.",
    ),
    "official_clone_reuse": (
        "en",
        "Reusable voice states preserve the reference speaker across multiple generations.",
    ),
    "official_clone_8bit": (
        "en",
        "Eight bit voice cloning also runs locally with the newest official weights.",
    ),
}


def words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    normalized = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    return re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, actual in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected != actual),
                )
            )
        previous = current
    return previous[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-wer", type=float, default=0.5)
    args = parser.parse_args()
    results = []
    for name, (language, reference) in CASES.items():
        path = args.audio_dir / f"{name}.wav"
        response = mlx_whisper.transcribe(
            str(path),
            path_or_hf_repo=args.model,
            language=language,
            task="transcribe",
            temperature=0.0,
            verbose=False,
        )
        transcript = str(response.get("text", "")).strip()
        reference_words = words(reference)
        edits = edit_distance(reference_words, words(transcript))
        wer = edits / max(1, len(reference_words))
        results.append(
            {
                "name": name,
                "language": language,
                "reference": reference,
                "transcript": transcript,
                "wer": round(wer, 4),
                "status": "pass" if transcript and wer <= args.max_wer else "fail",
            }
        )
    report = {
        "status": "pass" if all(item["status"] == "pass" for item in results) else "fail",
        "asr_model": args.model,
        "max_wer": args.max_wer,
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
