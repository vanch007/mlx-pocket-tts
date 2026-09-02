"""Isolated legacy PyTorch checkpoint reader.

This module is only launched as a subprocess. Production inference and native
training never import torch.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _normalize_name(name: str) -> str | None:
    name = name.removeprefix("module.")
    if name == "flow_lm.speaker_proj_weight":
        return "speaker_proj_weight"
    if name.startswith(("flow_lm.", "mimi.")) or name == "speaker_proj_weight":
        return name
    return None


def import_checkpoint(source: Path, output: Path, *, use_ema: bool = True) -> dict:
    try:
        import numpy as np
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Legacy .pt import requires PyTorch in the isolated importer environment."
        ) from error

    payload = torch.load(source, map_location="cpu", weights_only=True)
    state = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f"{source} holds no 'model' state dictionary")
    if use_ema and isinstance(payload.get("ema"), dict):
        state = {**state, **payload["ema"]}
    converted = {}
    ignored = []
    for name, value in state.items():
        normalized = _normalize_name(name)
        if normalized is None:
            ignored.append(name)
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} is not a tensor")
        tensor = value.detach().cpu().contiguous()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        converted[normalized] = tensor.numpy()
    if not any(name.startswith("flow_lm.") for name in converted):
        raise ValueError(f"No flow_lm.* tensors found in {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **converted)
    return {"tensors": len(converted), "ignored": ignored, "used_ema": use_ema}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--raw", action="store_true", help="Do not overlay EMA weights")
    args = parser.parse_args()
    report = import_checkpoint(args.source, args.output, use_ema=not args.raw)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
