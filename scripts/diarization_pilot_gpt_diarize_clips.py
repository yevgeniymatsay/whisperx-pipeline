#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.call_extractor_wavlm.io import utc_now_compact

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AzureAudioConfig:
    endpoint: str
    api_key: str
    api_version: str
    deployment: str


def _maybe_load_dotenv() -> None:
    """Best-effort .env loader (no external dependency).

    This mirrors the repo's Azure GPT client behavior: read key=value pairs from `.env`
    and only set env vars that are not already defined.
    """
    path = Path(os.getcwd()) / ".env"
    if not path.is_file():
        return
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if not key:
                continue
            os.environ.setdefault(key, value)
    except OSError:
        return


def _load_azure_audio_config(*, deployment_override: Optional[str]) -> AzureAudioConfig:
    def _read() -> tuple[str, str, str, str]:
        endpoint = (os.environ.get("AZURE_OPENAI_ENDPOINT") or "").strip()
        api_key = (os.environ.get("AZURE_OPENAI_API_KEY") or "").strip()
        api_version = (os.environ.get("OPENAI_API_VERSION") or "").strip()
        deployment = (deployment_override or os.environ.get("AZURE_OPENAI_DEPLOYMENT_TRANSCRIBE_DIARIZE") or "").strip()
        return endpoint, api_key, api_version, deployment

    endpoint, api_key, api_version, deployment = _read()
    if not (endpoint and api_key and api_version and deployment):
        _maybe_load_dotenv()
        endpoint, api_key, api_version, deployment = _read()

    missing = [k for k, v in {
        "AZURE_OPENAI_ENDPOINT": endpoint,
        "AZURE_OPENAI_API_KEY": api_key,
        "OPENAI_API_VERSION": api_version,
        "AZURE_OPENAI_DEPLOYMENT_TRANSCRIBE_DIARIZE (or --deployment)": deployment,
    }.items() if not v]
    if missing:
        raise RuntimeError("Missing Azure OpenAI env vars: " + ", ".join(missing))
    return AzureAudioConfig(endpoint=endpoint, api_key=api_key, api_version=api_version, deployment=deployment)


def _make_azure_client(cfg: AzureAudioConfig):
    try:
        from openai import AzureOpenAI
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing `openai` SDK. Install it in your pilot venv.") from e

    return AzureOpenAI(api_key=cfg.api_key, azure_endpoint=cfg.endpoint, api_version=cfg.api_version)


def _read_metadata_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp_{utc_now_compact()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _transcribe_diarize(
    client,
    *,
    deployment: str,
    audio_path: Path,
    language: str,
    retries: int,
    sleep_base_s: float,
) -> Dict[str, Any]:
    last_err: Optional[BaseException] = None
    for attempt in range(retries + 1):
        try:
            with audio_path.open("rb") as f:
                resp = client.audio.transcriptions.create(
                    model=deployment,
                    file=f,
                    language=language,
                    response_format="diarized_json",
                    timestamp_granularities=["segment"],
                )
            if hasattr(resp, "model_dump"):
                return resp.model_dump()
            if isinstance(resp, dict):
                return resp
            # Fallback: best-effort JSON conversion
            return json.loads(json.dumps(resp))
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt >= retries:
                break
            sleep_s = float(sleep_base_s) * (2 ** attempt)
            logger.warning(f"Azure diarize failed (attempt {attempt+1}/{retries+1}): {type(e).__name__}: {e}; sleeping {sleep_s:.1f}s")
            time.sleep(sleep_s)
    assert last_err is not None
    raise RuntimeError(f"Azure diarize failed after {retries+1} attempts: {last_err}") from last_err


def main() -> int:
    parser = argparse.ArgumentParser(description="Run gpt-4o-transcribe-diarize over call clips (diarized_json).")
    parser.add_argument("--pilot-dir", type=Path, required=True, help="Pilot output dir created by diarization_pilot_build_clips.py")
    parser.add_argument("--deployment", type=str, default=None, help="Azure deployment name override")
    parser.add_argument("--language", type=str, default="en", help="Language code (default: en)")
    parser.add_argument("--retries", type=int, default=3, help="Retry count on transient failures")
    parser.add_argument("--sleep-base-s", type=float, default=2.0, help="Base retry backoff in seconds")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of clips to process")
    args = parser.parse_args()

    pilot_dir = Path(args.pilot_dir)
    meta_path = pilot_dir / "metadata.jsonl"
    if not meta_path.exists():
        raise SystemExit(f"Missing metadata: {meta_path}")

    cfg = _load_azure_audio_config(deployment_override=args.deployment)
    client = _make_azure_client(cfg)

    diarized_dir = pilot_dir / "diarized"
    rows = _read_metadata_jsonl(meta_path)
    if args.limit is not None:
        rows = rows[: int(args.limit)]
    logger.info(f"Processing {len(rows)} clips -> {diarized_dir}")

    done = 0
    skipped = 0
    for row in rows:
        clip_id = str(row["clip_id"])
        clip_path = Path(row["clip_path"])
        out_path = diarized_dir / f"{clip_id}.diarized_json.json"
        if out_path.exists() and out_path.stat().st_size > 0:
            skipped += 1
            continue

        if not clip_path.exists():
            raise SystemExit(f"Missing clip audio: {clip_path} (clip_id={clip_id})")

        data = _transcribe_diarize(
            client,
            deployment=cfg.deployment,
            audio_path=clip_path,
            language=str(args.language),
            retries=int(args.retries),
            sleep_base_s=float(args.sleep_base_s),
        )
        # Stamp minimal provenance (no secrets).
        data.setdefault("metadata", {})
        data["metadata"].update({
            "clip_id": clip_id,
            "source_clip_path": str(clip_path),
            "azure_deployment": cfg.deployment,
            "language": str(args.language),
        })
        _dump_json(out_path, data)
        done += 1
        if done % 10 == 0 or done == len(rows):
            logger.info(f"Diarized {done}/{len(rows)} clips (skipped={skipped})")

    logger.info(f"Done. diarized={done} skipped={skipped} out_dir={diarized_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
