"""Upstream adapters for call segmenter dataset build / inference.

We support multiple upstream sources for diarization + transcript timing:
  - "whisperx": WhisperX (Whisper + pyannote) run artifacts in S3 (chunks).
  - "azure": GPT-4o transcribe-diarize merged JSON (local or S3).

Important constraints (project policy):
  - No lexical boundary heuristics (keywords like "hello/bye" etc.).
  - No diarization-only shortcut rules (e.g., "#speakers >= 2 => call").
  - Speaker identity is treated as anonymous; we preserve labels only as IDs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import S3_BUCKET
from ..transcriber import DiarizationSegment


@dataclass(frozen=True)
class UpstreamData:
    """Normalized upstream payload used by dataset build / inference."""

    upstream: str  # "whisperx" | "azure"
    run_id: str  # for traceability (may be a sentinel for non-WhisperX sources)
    segments: List[DiarizationSegment]
    words: List[dict]  # optional; can be empty to disable text features safely
    timeline_end_s: float
    mp3_key: str
    mp3_duration_s: Optional[float]
    # Coverage intervals where upstream produced artifacts; None means "no coverage filtering".
    chunk_coverage: Optional[List[Tuple[float, float]]]


def azure_run_id_for_source(
    *,
    merged_local_root: Optional[Path] = None,
    merged_s3_prefix: Optional[str] = None,
) -> str:
    """Build a stable run_id-like tag for Azure merged outputs.

    We intentionally avoid embedding full local paths or long S3 prefixes in the run_id.
    """
    if merged_s3_prefix:
        parts = [p for p in str(merged_s3_prefix).strip("/").split("/") if p]
        if len(parts) >= 2:
            tag = f"{parts[-2]}_{parts[-1]}"
        elif parts:
            tag = parts[-1]
        else:
            tag = "azure"
        return f"azure_{tag}"

    if merged_local_root is not None:
        p = Path(merged_local_root)
        # Common local layout: artifacts/.../<stamp>/videos/<video_id>/merged.diarized.json
        if p.name == "videos" and p.parent.name:
            tag = p.parent.name
        else:
            tag = p.name or "azure"
        return f"azure_{tag}"

    return "azure"


def parse_azure_merged_diarized(doc: Dict[str, Any]) -> List[DiarizationSegment]:
    """Parse Azure `merged.diarized.json` into DiarizationSegments.

    Expected schema (observed in artifacts):
      {
        "segments": [{"start": 0.0, "end": 1.23, "speaker": "spk_0", "text": "..."}],
        "text": "..."
      }
    """
    segs: List[DiarizationSegment] = []
    for s in doc.get("segments", []) or []:
        try:
            t0 = s.get("start", s.get("t0_abs"))
            t1 = s.get("end", s.get("t1_abs"))
            spk = s.get("speaker", s.get("spk"))
            if t0 is None or t1 is None or spk is None:
                continue
            t0_f = float(t0)
            t1_f = float(t1)
            if t1_f <= t0_f:
                continue
            segs.append(DiarizationSegment(t0_abs=t0_f, t1_abs=t1_f, spk=str(spk)))
        except Exception:
            continue
    segs.sort(key=lambda x: x.t0_abs)
    return segs


def load_azure_merged_local(merged_local_root: Path, video_id: str) -> Dict[str, Any]:
    """Load Azure merged diarization JSON from local filesystem.

    `merged_local_root` should contain per-video directories:
      <root>/<video_id>/merged.diarized.json
    """
    path = merged_local_root / video_id / "merged.diarized.json"
    return json.loads(path.read_text())


def load_azure_merged_s3(s3_client, merged_s3_prefix: str, video_id: str) -> Dict[str, Any]:
    """Load Azure merged diarization JSON from S3.

    Convention (planned): <prefix>/<video_id>.json
    """
    key = merged_s3_prefix.rstrip("/") + f"/{video_id}.json"
    resp = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
    return json.loads(resp["Body"].read())
