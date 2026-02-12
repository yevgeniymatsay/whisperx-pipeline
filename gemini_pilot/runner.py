from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BACKOFF_S,
    DEFAULT_TIMEOUT_S,
    cached_audio_path_for_s3_uri,
    infer_audio_mime_type,
    parse_s3_uri,
    resolve_gemini_api_key,
    resolve_models,
    sanitize_token,
    utc_now_compact,
)
from .prompts import CALL_TIMESTAMP_PROMPT_V1
from .timestamps import parse_timestamp_response

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioInput:
    source_id: str
    source_kind: str  # local | s3
    source_ref: str
    local_path: Path
    mime_type: str
    size_bytes: int | None


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp_{utc_now_compact()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _download_s3_to_path(*, bucket: str, key: str, dst_path: Path) -> None:
    import boto3

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, str(dst_path))


def _resolve_audio_inputs(
    *,
    audio_paths: list[Path],
    audio_s3_uris: list[str],
    cache_dir: Path,
    dry_run: bool,
) -> list[AudioInput]:
    out: list[AudioInput] = []
    seen_refs: set[str] = set()

    for raw_path in audio_paths:
        p = raw_path.expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Local audio file not found: {p}")
        mime_type = infer_audio_mime_type(p)
        source_ref = str(p)
        if source_ref in seen_refs:
            continue
        seen_refs.add(source_ref)
        source_hash = hashlib.sha1(source_ref.encode("utf-8")).hexdigest()[:8]
        source_id = f"local_{sanitize_token(p.stem, fallback='audio')}_{source_hash}"
        out.append(
            AudioInput(
                source_id=source_id,
                source_kind="local",
                source_ref=source_ref,
                local_path=p,
                mime_type=mime_type,
                size_bytes=p.stat().st_size,
            )
        )

    for uri in audio_s3_uris:
        parsed = parse_s3_uri(uri)
        source_ref = f"s3://{parsed.bucket}/{parsed.key}"
        if source_ref in seen_refs:
            continue
        seen_refs.add(source_ref)
        cache_path = cached_audio_path_for_s3_uri(source_ref, cache_dir=cache_dir)

        if not dry_run:
            if not cache_path.exists():
                logger.info(f"Downloading {source_ref} -> {cache_path}")
                _download_s3_to_path(bucket=parsed.bucket, key=parsed.key, dst_path=cache_path)
            if not cache_path.is_file():
                raise FileNotFoundError(f"S3 cache target missing after download: {cache_path}")
            mime_type = infer_audio_mime_type(cache_path)
            size_bytes: int | None = cache_path.stat().st_size
        else:
            mime_type = infer_audio_mime_type(Path(parsed.key))
            size_bytes = cache_path.stat().st_size if cache_path.exists() else None

        source_hash = hashlib.sha1(source_ref.encode("utf-8")).hexdigest()[:8]
        source_id = f"s3_{sanitize_token(Path(parsed.key).stem, fallback='audio')}_{source_hash}"
        out.append(
            AudioInput(
                source_id=source_id,
                source_kind="s3",
                source_ref=source_ref,
                local_path=cache_path,
                mime_type=mime_type,
                size_bytes=size_bytes,
            )
        )

    if not out:
        raise ValueError("No audio inputs resolved. Provide --audio-path and/or --audio-s3-uri.")
    return out


def _make_gemini_client(*, api_key: str, timeout_s: float):
    from google import genai
    from google.genai import types

    timeout_ms = max(1000, int(round(float(timeout_s) * 1000.0)))
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_ms))


def _to_jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(exclude_none=True)  # type: ignore[no-any-return]
        except Exception:
            return {"repr": repr(obj)}
    if isinstance(obj, (dict, list, str, int, float, bool)):
        return obj
    return {"repr": repr(obj)}


def _extract_response_text(response: Any) -> str:
    text = ""
    try:
        maybe = response.text
        text = maybe.strip() if isinstance(maybe, str) else ""
    except Exception:
        text = ""
    if text:
        return text

    parts: list[str] = []
    for cand in getattr(response, "candidates", []) or []:
        content = getattr(cand, "content", None)
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                parts.append(part_text.strip())
    return "\n".join(parts).strip()


def _run_single_model(
    *,
    client: Any,
    audio_input: AudioInput,
    model: str,
    prompt: str,
    max_retries: int,
    retry_backoff_s: float,
) -> tuple[str, dict[str, Any]]:
    uploaded_file = None
    last_err: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            uploaded_file = client.files.upload(
                file=str(audio_input.local_path),
                config={"mimeType": audio_input.mime_type},
            )
            response = client.models.generate_content(
                model=model,
                contents=[prompt, uploaded_file],
            )
            response_text = _extract_response_text(response)
            response_meta = {
                "response_id": getattr(response, "response_id", None),
                "model_version": getattr(response, "model_version", None),
                "usage_metadata": _to_jsonable(getattr(response, "usage_metadata", None)),
                "prompt_feedback": _to_jsonable(getattr(response, "prompt_feedback", None)),
                "upload_file": {
                    "name": getattr(uploaded_file, "name", None),
                    "uri": getattr(uploaded_file, "uri", None),
                    "mime_type": getattr(uploaded_file, "mime_type", None),
                },
            }
            return response_text, response_meta
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt >= max_retries:
                break
            sleep_s = float(retry_backoff_s) * (2 ** attempt)
            logger.warning(
                "Gemini call failed (model=%s, input=%s, attempt=%d/%d): %s: %s; sleeping %.1fs",
                model,
                audio_input.source_id,
                attempt + 1,
                max_retries + 1,
                type(exc).__name__,
                exc,
                sleep_s,
            )
            time.sleep(sleep_s)
        finally:
            if uploaded_file is not None:
                try:
                    client.files.delete(name=uploaded_file.name)
                except Exception as delete_exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to delete uploaded Gemini file (model=%s, input=%s): %s",
                        model,
                        audio_input.source_id,
                        delete_exc,
                    )
                uploaded_file = None
    assert last_err is not None
    raise RuntimeError(
        f"Gemini request failed after {max_retries + 1} attempts "
        f"(model={model}, input={audio_input.source_id}): {last_err}"
    ) from last_err


def run_pilot(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    models = resolve_models(list(args.model or []))
    prompt = CALL_TIMESTAMP_PROMPT_V1
    run_id = f"run_{utc_now_compact()}"
    out_dir = Path(args.out_dir) if args.out_dir else (Path("artifacts") / "gemini_pilot" / run_id)
    cache_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    audio_inputs = _resolve_audio_inputs(
        audio_paths=list(args.audio_path or []),
        audio_s3_uris=list(args.audio_s3_uri or []),
        cache_dir=cache_dir,
        dry_run=bool(args.dry_run),
    )

    manifest = {
        "run_id": run_id,
        "created_at_utc": utc_now_compact(),
        "dry_run": bool(args.dry_run),
        "models": models,
        "settings": {
            "timeout_s": float(args.timeout_s),
            "max_retries": int(args.max_retries),
            "retry_backoff_s": float(args.retry_backoff_s),
            "continue_on_error": bool(args.continue_on_error),
            "cache_dir": str(cache_dir),
        },
        "prompt": {
            "name": "call_timestamp_prompt_v1",
            "text": prompt,
        },
        "audio_inputs": [
            {
                "source_id": a.source_id,
                "source_kind": a.source_kind,
                "source_ref": a.source_ref,
                "local_path": str(a.local_path),
                "mime_type": a.mime_type,
                "size_bytes": a.size_bytes,
            }
            for a in audio_inputs
        ],
    }
    _write_json(out_dir / "run_manifest.json", manifest)

    if args.dry_run:
        logger.info(
            "Dry-run complete: inputs=%d models=%d out_dir=%s",
            len(audio_inputs),
            len(models),
            out_dir,
        )
        summary = {
            "run_id": run_id,
            "dry_run": True,
            "inputs_count": len(audio_inputs),
            "models_count": len(models),
            "planned_requests": len(audio_inputs) * len(models),
            "out_dir": str(out_dir),
        }
        _write_json(out_dir / "run_summary.json", summary)
        return 0

    api_key = resolve_gemini_api_key(cwd=Path.cwd())
    client = _make_gemini_client(api_key=api_key, timeout_s=float(args.timeout_s))

    total = len(audio_inputs) * len(models)
    done = 0
    errors = 0
    results: list[dict[str, Any]] = []
    for audio_input in audio_inputs:
        input_dir = out_dir / "outputs" / audio_input.source_id
        input_dir.mkdir(parents=True, exist_ok=True)
        for model in models:
            done += 1
            model_slug = sanitize_token(model, fallback="model")
            raw_path = input_dir / f"{model_slug}.raw.txt"
            json_path = input_dir / f"{model_slug}.result.json"
            err_path = input_dir / f"{model_slug}.error.json"
            try:
                response_text, response_meta = _run_single_model(
                    client=client,
                    audio_input=audio_input,
                    model=model,
                    prompt=prompt,
                    max_retries=int(args.max_retries),
                    retry_backoff_s=float(args.retry_backoff_s),
                )
                raw_path.write_text((response_text or "") + "\n", encoding="utf-8")
                parse_payload: dict[str, Any]
                if args.parse_timestamps:
                    parsed = parse_timestamp_response(response_text)
                    parse_payload = {
                        "enabled": True,
                        "ambiguous": bool(parsed.ambiguous),
                        "no_call_segments": bool(parsed.no_call_segments),
                        "warnings": list(parsed.warnings),
                        "segments": [asdict(seg) for seg in parsed.segments],
                        "dropped_due_to_ambiguity": bool(parsed.ambiguous),
                    }
                else:
                    parse_payload = {
                        "enabled": False,
                        "reason": "raw_output_first_mode",
                    }
                payload = {
                    "status": "ok",
                    "source_id": audio_input.source_id,
                    "source_kind": audio_input.source_kind,
                    "source_ref": audio_input.source_ref,
                    "local_path": str(audio_input.local_path),
                    "mime_type": audio_input.mime_type,
                    "model": model,
                    "raw_text_path": str(raw_path),
                    "parse": parse_payload,
                    "response_meta": response_meta,
                }
                _write_json(json_path, payload)
                results.append(
                    {
                        "status": "ok",
                        "source_id": audio_input.source_id,
                        "model": model,
                        "result_json": str(json_path),
                    }
                )
                logger.info(
                    "OK %d/%d input=%s model=%s segments=%d ambiguous=%s",
                    done,
                    total,
                    audio_input.source_id,
                    model,
                    (len(parse_payload.get("segments", [])) if parse_payload.get("enabled") else -1),
                    (parse_payload.get("ambiguous") if parse_payload.get("enabled") else "n/a"),
                )
            except Exception as exc:  # noqa: BLE001
                errors += 1
                payload = {
                    "status": "error",
                    "source_id": audio_input.source_id,
                    "source_kind": audio_input.source_kind,
                    "source_ref": audio_input.source_ref,
                    "local_path": str(audio_input.local_path),
                    "mime_type": audio_input.mime_type,
                    "model": model,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                _write_json(err_path, payload)
                results.append(
                    {
                        "status": "error",
                        "source_id": audio_input.source_id,
                        "model": model,
                        "error_json": str(err_path),
                    }
                )
                logger.error(
                    "ERR %d/%d input=%s model=%s: %s: %s",
                    done,
                    total,
                    audio_input.source_id,
                    model,
                    type(exc).__name__,
                    exc,
                )
                if not args.continue_on_error:
                    summary = {
                        "run_id": run_id,
                        "dry_run": False,
                        "completed_requests": done,
                        "total_requests": total,
                        "errors": errors,
                        "results": results,
                        "out_dir": str(out_dir),
                    }
                    _write_json(out_dir / "run_summary.json", summary)
                    return 1

    summary = {
        "run_id": run_id,
        "dry_run": False,
        "completed_requests": done,
        "total_requests": total,
        "errors": errors,
        "results": results,
        "out_dir": str(out_dir),
    }
    _write_json(out_dir / "run_summary.json", summary)
    logger.info("Done requests=%d errors=%d out_dir=%s", done, errors, out_dir)
    return 0 if errors == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gemini audio pilot: extract candidate real-call timestamps from local files and S3 URIs."
    )
    parser.add_argument(
        "--audio-path",
        type=Path,
        action="append",
        default=[],
        help="Local audio file path (repeatable).",
    )
    parser.add_argument(
        "--audio-s3-uri",
        type=str,
        action="append",
        default=[],
        help="S3 audio URI in form s3://bucket/key (repeatable).",
    )
    parser.add_argument(
        "--model",
        type=str,
        action="append",
        default=[],
        help="Gemini model override (repeatable). Defaults to gemini-3-flash-preview and gemini-2.0-flash.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: artifacts/gemini_pilot/run_<utc>).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/gemini_pilot/audio"),
        help="Local cache dir for S3-downloaded audio (default: .cache/gemini_pilot/audio).",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"HTTP timeout in seconds (default: {DEFAULT_TIMEOUT_S}).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Retry count per request (default: {DEFAULT_MAX_RETRIES}).",
    )
    parser.add_argument(
        "--retry-backoff-s",
        type=float,
        default=DEFAULT_RETRY_BACKOFF_S,
        help=f"Base exponential backoff in seconds (default: {DEFAULT_RETRY_BACKOFF_S}).",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue processing remaining requests after an error (default: true).",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resolve inputs/models/output paths without calling Gemini (default: false).",
    )
    parser.add_argument(
        "--parse-timestamps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Parse CALL_SEGMENT lines from model text. Default false for raw-output-first evaluation.",
    )
    return parser


def run_cli(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.audio_path and not args.audio_s3_uri:
        parser.error("Provide at least one --audio-path or --audio-s3-uri input.")
    try:
        return run_pilot(args)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 2
