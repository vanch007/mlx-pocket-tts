from pathlib import Path

import mlx.core as mx


def export_voice_state(model_state: dict, destination: str | Path) -> Path:
    """Save a reusable Pocket TTS voice cache in the official safetensors layout."""
    tensors = {}
    for index, cache in enumerate(model_state.get("flow_cache", [])):
        if cache.keys is None:
            continue
        prefix = f"transformer.layers.{index}.self_attn/"
        keys = cache.keys[..., : cache.offset, :].transpose(0, 2, 1, 3)
        values = cache.values[..., : cache.offset, :].transpose(0, 2, 1, 3)
        tensors[prefix + "cache"] = mx.stack([keys, values])
        tensors[prefix + "offset"] = mx.array([cache.offset], dtype=mx.int64)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(destination), tensors)
    return destination
