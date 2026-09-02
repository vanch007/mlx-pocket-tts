#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from fastapi.testclient import TestClient

from mlx_pocket_tts import load
from mlx_pocket_tts.server import create_app


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client = TestClient(create_app(load(args.model)))
    health = client.get("/health")
    audio = client.post(
        "/tts",
        data={"text": "The local HTTP streaming endpoint works.", "voice": "alba"},
    )
    empty = client.post("/tts", data={"text": ""})
    report = {
        "status": "pass"
        if health.status_code == 200
        and audio.status_code == 200
        and audio.content[:4] == b"RIFF"
        and empty.status_code in {400, 422}
        else "fail",
        "health_status": health.status_code,
        "health": health.json(),
        "tts_status": audio.status_code,
        "content_type": audio.headers.get("content-type"),
        "bytes": len(audio.content),
        "riff": audio.content[:4].decode("ascii", errors="replace"),
        "empty_text_status": empty.status_code,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
