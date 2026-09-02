from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from .data import Entry, load_entries


@dataclass
class Batch:
    latents: mx.array
    mask: mx.array
    text_tokens: list[mx.array]
    voice_latents: mx.array
    num_voice_prompt_frames: mx.array


def _pad_time(value: mx.array, length: int) -> mx.array:
    if value.shape[0] >= length:
        return value[:length]
    return mx.pad(value, ((0, length - value.shape[0]), (0, 0)))


class LatentDataLoader:
    def __init__(
        self,
        manifest: str | Path,
        tokenizer,
        batch_size: int,
        *,
        max_duration_sec: float = 30.0,
        max_voice_prompt_sec: float = 5.0,
        frame_rate: float = 12.5,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.manifest = Path(manifest).expanduser().resolve()
        self.entries = load_entries(self.manifest)
        if any(entry.latents_file is None for entry in self.entries):
            raise ValueError(f"{self.manifest}: every entry must define latents_file")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_target_frames = max(1, round(max_duration_sec * frame_rate))
        self.max_prompt_frames = max(1, round(max_voice_prompt_sec * frame_rate))
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        indices = list(range(len(self.entries)))
        if self.shuffle:
            generator = mx.random.key(self.seed + self.epoch)
            indices = [int(i) for i in mx.random.permutation(len(indices), key=generator).tolist()]
        self.epoch += 1
        for start in range(0, len(indices), self.batch_size):
            yield self._collate(
                [self.entries[index] for index in indices[start : start + self.batch_size]]
            )

    def _load(self, entry: Entry) -> mx.array:
        path = self.manifest.parent / str(entry.latents_file)
        tensors = mx.load(str(path))
        if "latents" not in tensors:
            raise ValueError(f"{path}: missing 'latents' tensor")
        value = tensors["latents"]
        if value.ndim != 2:
            raise ValueError(f"{path}: expected [C,T] latents, got {tuple(value.shape)}")
        return value.T.astype(mx.float32)

    def _collate(self, entries: list[Entry]) -> Batch:
        targets = [self._load(entry)[: self.max_target_frames] for entry in entries]
        prompts = [target[: self.max_prompt_frames] for target in targets]
        target_width = max(target.shape[0] for target in targets)
        prompt_width = max(prompt.shape[0] for prompt in prompts)
        latents = mx.stack([_pad_time(target, target_width) for target in targets])
        mask = mx.stack([mx.arange(target_width) < target.shape[0] for target in targets])
        voice_latents = mx.stack([_pad_time(prompt, prompt_width) for prompt in prompts])
        text_tokens = [
            mx.array(self.tokenizer.encode(entry.transcript), dtype=mx.int32) for entry in entries
        ]
        lengths = mx.array([prompt.shape[0] for prompt in prompts], dtype=mx.int32)
        return Batch(latents, mask, text_tokens, voice_latents, lengths)
