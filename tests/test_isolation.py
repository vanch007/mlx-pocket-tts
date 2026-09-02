from pathlib import Path


def test_runtime_does_not_import_mlx_audio():
    root = Path(__file__).parents[1] / "src" / "mlx_pocket_tts"
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text()
        if "import mlx_audio" in text or "from mlx_audio" in text:
            offenders.append(str(path.relative_to(root)))
    assert offenders == []
