#!/usr/bin/env python3
"""One-shot batch runner that bypasses shell escaping issues with S3 keys."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gemini_pilot.runner import run_cli

BUCKET = "s3://rezora-whisperx-us-east-1-864981718771/audio_m4a_aac/pretraining"

AUDIO_S3_URIS = [
    f"{BUCKET}/Qa-ppZFUp0g.m4a",
    f"{BUCKET}/D4uiHjHW4AU.m4a",
    f"{BUCKET}/DEt3IRqqUVs.m4a",
    f"{BUCKET}/Krnsw9WZtRA.m4a",
    f"{BUCKET}/FkgGv2iMjEo.m4a",
    f"{BUCKET}/UWLJ81ezcpU.m4a",
    f"{BUCKET}/RL6Y5qig0Wg.m4a",
    f"{BUCKET}/gDdoZC9Nhgk.m4a",
    f"{BUCKET}/ci-FdcWiJiA.m4a",
    f"{BUCKET}/CfMJ01KP_ns.m4a",
]

# Important: do NOT point this at any user-managed local dataset folder.
# This cache is only for temporary S3 downloads performed by the pilot.
CACHE_DIR = Path(".cache/gemini_pilot/audio_s3_eval_batch")
OUT_DIR = Path("artifacts/gemini_pilot") / f"eval_batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def main() -> int:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    argv: list[str] = []
    for uri in AUDIO_S3_URIS:
        argv += ["--audio-s3-uri", uri]
    argv += ["--cache-dir", str(CACHE_DIR), "--out-dir", str(OUT_DIR)]
    return run_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
