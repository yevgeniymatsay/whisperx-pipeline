#!/usr/bin/env python3
"""Run a 5-model Gemini sweep on a local audio directory, then emit eval JSON+MD.

This is intentionally a standalone "one command" runner that:
- Uses local audio files (fast; avoids S3 downloads for audio)
- Writes run artifacts under artifacts/gemini_pilot/run_<utc>/
- Writes eval artifacts under gemini_pilot/eval/<run_id>_eval.(json|md)

Prompt + schema are defined in gemini_pilot/ and are not modified by this script.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gemini_pilot.runner import run_cli as run_gemini_pilot_cli  # noqa: E402


DEFAULT_AUDIO_DIR = Path("/Volumes/Yevgeniy's Drive/tmp_gemini_audio")
DEFAULT_GT_S3_PREFIX = "s3://rezora-whisperx-us-east-1-864981718771/labeling/corrected_boundaries/v1/"
SUPPORTED_EXTS = {".m4a", ".mp3", ".wav", ".flac", ".aac", ".aiff", ".ogg"}


def _utc_compact() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


def _list_audio_files(audio_dir: Path) -> list[Path]:
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"--audio-dir is not a directory: {audio_dir}")
    files = [
        p
        for p in audio_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS and not p.name.startswith(".")
    ]
    files.sort(key=lambda p: p.name)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run local-dir Gemini sweep (5 models) and write artifacts for manual review."
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=DEFAULT_AUDIO_DIR,
        help=f"Directory containing exactly 10 audio files (default: {DEFAULT_AUDIO_DIR})",
    )
    parser.add_argument(
        "--gt-s3-prefix",
        type=str,
        default=DEFAULT_GT_S3_PREFIX,
        help="S3 prefix for GT label JSONs (default: corrected_boundaries v1).",
    )
    parser.add_argument(
        "--model-concurrency",
        type=int,
        default=5,
        help="Parallelism across models per audio input (default: 5).",
    )
    parser.add_argument(
        "--print-max-chars",
        type=int,
        default=0,
        help="Max chars to print per model output (0 = no truncation; default: 0).",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resolve inputs and write manifests without calling Gemini (default: false).",
    )
    args = parser.parse_args()

    audio_dir = args.audio_dir.expanduser().resolve()
    audio_files = _list_audio_files(audio_dir)
    if len(audio_files) != 10:
        listing = "\n".join(f"  - {p.name}" for p in audio_files[:25])
        raise SystemExit(
            f"Expected exactly 10 audio files in {audio_dir}, found {len(audio_files)}.\n"
            f"First files:\n{listing}"
        )

    out_dir = REPO_ROOT / "artifacts" / "gemini_pilot" / f"run_{_utc_compact()}"

    # 1) Run Gemini pilot sweep
    pilot_argv: list[str] = ["--out-dir", str(out_dir)]
    for p in audio_files:
        pilot_argv += ["--audio-path", str(p)]
    pilot_argv += [
        "--model-concurrency",
        str(int(args.model_concurrency)),
        "--print-max-chars",
        str(int(args.print_max_chars)),
    ]
    if args.dry_run:
        pilot_argv.append("--dry-run")

    pilot_rc = run_gemini_pilot_cli(pilot_argv)
    if pilot_rc != 0:
        print(f"[WARN] Gemini pilot exited with code={pilot_rc}.")

    if args.dry_run:
        print("")
        print("Dry-run complete (no Gemini calls).")
        print("")
        print("Planned run artifacts:")
        print(f"  run_dir: {out_dir}")
        print(f"  run_manifest.json: {out_dir / 'run_manifest.json'}")
        print(f"  run_summary.json: {out_dir / 'run_summary.json'}")
        return pilot_rc

    # 2) Run eval (JSON + Markdown under gemini_pilot/eval/)
    eval_script = REPO_ROOT / "scripts" / "eval_gemini_call_detector.py"
    eval_cmd = [
        sys.executable,
        str(eval_script),
        "--run-dir",
        str(out_dir),
        "--gt-s3-prefix",
        str(args.gt_s3_prefix),
        "--no-print-details",
    ]
    print("")
    print("Running eval:")
    print("  " + " ".join(eval_cmd))
    eval_proc = subprocess.run(eval_cmd, check=False)
    eval_rc = int(eval_proc.returncode or 0)

    print("")
    print("Run artifacts:")
    print(f"  run_dir: {out_dir}")
    print(f"  run_summary.json: {out_dir / 'run_summary.json'}")
    print(f"  eval.json: {REPO_ROOT / 'gemini_pilot' / 'eval' / f'{out_dir.name}_eval.json'}")
    print(f"  eval.md: {REPO_ROOT / 'gemini_pilot' / 'eval' / f'{out_dir.name}_eval.md'}")
    return pilot_rc if pilot_rc != 0 else eval_rc


if __name__ == "__main__":
    raise SystemExit(main())
