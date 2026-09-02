from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Entry:
    path: Path
    duration: float
    transcript: str
    words: tuple[dict[str, Any], ...] = ()
    start: float = 0.0
    latents_file: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, manifest: Path, line_number: int) -> "Entry":
        location = f"{manifest}:{line_number}"
        for field in ("path", "duration", "transcript"):
            if field not in value:
                raise ValueError(f"{location}: missing required field {field!r}")
        path = Path(str(value["path"])).expanduser()
        if not path.is_absolute():
            path = (manifest.parent / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{location}: audio file not found: {path}")
        duration = float(value["duration"])
        start = float(value.get("start", 0.0))
        transcript = str(value["transcript"]).strip()
        if duration <= 0:
            raise ValueError(f"{location}: duration must be positive, got {duration}")
        if start < 0:
            raise ValueError(f"{location}: start must be non-negative, got {start}")
        if not transcript:
            raise ValueError(f"{location}: transcript must not be empty")
        words = tuple(value.get("words") or ())
        for index, word in enumerate(words):
            if not isinstance(word, dict) or not str(word.get("word", "")).strip():
                raise ValueError(f"{location}: invalid words[{index}]")
        return cls(path, duration, transcript, words, start, value.get("latents_file"))

    def source_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.path.read_bytes())
        digest.update(f"{self.start:.9f}:{self.duration:.9f}:{self.transcript}".encode())
        digest.update(json.dumps(self.words, sort_keys=True, ensure_ascii=False).encode())
        return digest.hexdigest()

    def to_dict(self, *, relative_to: Path | None = None) -> dict[str, Any]:
        path = self.path
        if relative_to is not None:
            try:
                path = path.relative_to(relative_to)
            except ValueError:
                pass
        value: dict[str, Any] = {
            "path": str(path),
            "duration": self.duration,
            "transcript": self.transcript,
        }
        if self.start:
            value["start"] = self.start
        if self.words:
            value["words"] = list(self.words)
        if self.latents_file:
            value["latents_file"] = self.latents_file
        return value


def load_entries(path: str | Path) -> list[Entry]:
    manifest = Path(path).expanduser().resolve()
    entries = []
    for line_number, line in enumerate(manifest.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{manifest}:{line_number}: invalid JSON: {error.msg}") from error
        if not isinstance(value, dict):
            raise ValueError(f"{manifest}:{line_number}: each line must be a JSON object")
        entries.append(Entry.from_dict(value, manifest=manifest, line_number=line_number))
    if not entries:
        raise ValueError(f"{manifest}: manifest contains no entries")
    return entries
