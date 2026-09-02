import importlib.util
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_pocket_tts.legacy import import_legacy_checkpoint


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="isolated importer needs torch"
)
def test_legacy_importer_overlays_ema_without_runtime_torch_import(tmp_path):
    import torch

    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": {
                "flow_lm.out_eos.bias": torch.tensor([1.0]),
                "flow.w_s_t.0.weight": torch.ones((1, 1)),
            },
            "ema": {"flow_lm.out_eos.bias": torch.tensor([2.0])},
        },
        checkpoint,
    )
    imported = import_legacy_checkpoint(checkpoint)
    weights = mx.load(str(imported))
    assert sorted(weights) == ["flow_lm.out_eos.bias"]
    assert weights["flow_lm.out_eos.bias"].tolist() == [2.0]


def test_production_modules_do_not_import_torch():
    package = Path(__file__).parents[1] / "src" / "mlx_pocket_tts"
    offenders = []
    for path in package.rglob("*.py"):
        if path.name == "legacy_importer.py":
            continue
        text = path.read_text()
        if "import torch" in text or "from torch" in text:
            offenders.append(str(path.relative_to(package)))
    assert offenders == []
