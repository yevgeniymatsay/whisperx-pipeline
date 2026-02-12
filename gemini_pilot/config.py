from __future__ import annotations

import hashlib
import mimetypes
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_MODELS: tuple[str, ...] = ("gemini-3-flash-preview", "gemini-2.0-flash")
DEFAULT_TIMEOUT_S: float = 300.0
DEFAULT_MAX_RETRIES: int = 2
DEFAULT_RETRY_BACKOFF_S: float = 2.0

_AUDIO_MIME_BY_EXT: dict[str, str] = {
    ".wav": "audio/wav",
    ".mp3": "audio/mp3",
    ".aiff": "audio/aiff",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
}
_SAFE_TOKEN_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class ParsedS3Uri:
    bucket: str
    key: str


def utc_now_compact() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


def maybe_load_dotenv(cwd: Path | None = None) -> None:
    """Best-effort dotenv loader that only sets missing env vars."""
    root = Path(cwd or os.getcwd())
    env_path = root / ".env"
    if not env_path.is_file():
        return
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    for raw_line in lines:
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
        if key:
            os.environ.setdefault(key, value)


def resolve_gemini_api_key(*, cwd: Path | None = None) -> str:
    def _read() -> str:
        return (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()

    key = _read()
    if not key:
        disable = (os.environ.get("GEMINI_PILOT_DISABLE_DOTENV") or "").strip().lower()
        if disable not in {"1", "true", "yes"}:
            maybe_load_dotenv(cwd=cwd)
        key = _read()
    if not key:
        raise RuntimeError(
            "Missing Gemini API key. Set GEMINI_API_KEY (preferred) or GOOGLE_API_KEY "
            "in shell env or .env before running the pilot."
        )
    return key


def resolve_models(model_overrides: list[str] | None) -> list[str]:
    raw = model_overrides or list(DEFAULT_MODELS)
    out: list[str] = []
    seen: set[str] = set()
    for model in raw:
        m = model.strip()
        if not m or m in seen:
            continue
        out.append(m)
        seen.add(m)
    if not out:
        raise ValueError("No models selected. Provide at least one --model.")
    return out


def parse_s3_uri(uri: str) -> ParsedS3Uri:
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Invalid S3 URI (expected s3://bucket/key): {uri}")
    bucket = (parsed.netloc or "").strip()
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI (missing bucket/key): {uri}")
    return ParsedS3Uri(bucket=bucket, key=key)


def infer_audio_mime_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in _AUDIO_MIME_BY_EXT:
        return _AUDIO_MIME_BY_EXT[ext]
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed and guessed.startswith("audio/"):
        if guessed == "audio/mpeg":
            return "audio/mp3"
        return guessed
    raise ValueError(f"Unsupported audio extension for {path}. Supported: {sorted(_AUDIO_MIME_BY_EXT)}")


def sanitize_token(value: str, *, fallback: str = "item", max_len: int = 80) -> str:
    safe = _SAFE_TOKEN_RE.sub("_", value).strip("._")
    if not safe:
        safe = fallback
    return safe[:max_len]


def cached_audio_path_for_s3_uri(uri: str, *, cache_dir: Path) -> Path:
    parsed = parse_s3_uri(uri)
    suffix = Path(parsed.key).suffix or ".bin"
    key_tail = Path(parsed.key).name
    stem = sanitize_token(Path(key_tail).stem, fallback="audio")
    digest = hashlib.sha1(uri.encode("utf-8")).hexdigest()[:12]
    filename = f"{stem}_{digest}{suffix}"
    return cache_dir / filename
