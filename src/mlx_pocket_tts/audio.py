from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_audio(path: str | Path, sample_rate: int = 24_000) -> mx.array:
    samples, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
    samples = samples.mean(axis=1)
    if source_rate != sample_rate:
        samples = resample_poly(samples, sample_rate, source_rate).astype(np.float32)
    return mx.array(samples, dtype=mx.float32)


def write_audio(path: str | Path, audio: mx.array, sample_rate: int) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), np.asarray(audio, dtype=np.float32), sample_rate, subtype="PCM_16")
