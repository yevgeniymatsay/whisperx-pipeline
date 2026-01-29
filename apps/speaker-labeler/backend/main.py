"""Speaker Labeler API."""
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Header, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import s3_client
import labels
import boundaries

app = FastAPI(title="Speaker Labeler")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Length", "Content-Range", "Accept-Ranges"],
)

# Cache calls list
_calls_cache: List[Dict[str, Any]] = []


def get_calls() -> List[Dict[str, Any]]:
    """Get cached calls list."""
    global _calls_cache
    if not _calls_cache:
        _calls_cache = s3_client.list_calls()
    return _calls_cache


class LabelRequest(BaseModel):
    speaker_roles: Dict[str, str]


@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/api/progress")
def get_progress():
    """Get labeling progress."""
    calls = get_calls()
    labeled = labels.get_labeled_call_ids()
    skipped = labels.get_skipped_call_ids()
    current_index = labels.get_current_index()

    return {
        "total": len(calls),
        "labeled": len(labeled),
        "skipped": len(skipped),
        "remaining": len(calls) - len(labeled) - len(skipped),
        "current_index": current_index,
    }


@app.get("/api/queue")
def get_queue():
    """Get all calls with their status."""
    calls = get_calls()
    labeled = labels.get_labeled_call_ids()
    skipped = labels.get_skipped_call_ids()

    result = []
    for call in calls:
        call_id = call["call_id"]
        status = "labeled" if call_id in labeled else ("skipped" if call_id in skipped else "pending")
        result.append({**call, "status": status})

    return result


@app.get("/api/queue/current")
def get_current_call():
    """Get current call to label."""
    calls = get_calls()
    current_index = labels.get_current_index()

    if current_index >= len(calls):
        return {"done": True, "message": "All calls labeled"}

    call = calls[current_index]
    return {
        "done": False,
        "index": current_index,
        "total": len(calls),
        **call,
    }


@app.get("/api/calls/{call_id}")
def get_call(call_id: str):
    """Get full call data."""
    calls = get_calls()

    # Find call in list
    call_info = None
    for c in calls:
        if c["call_id"] == call_id:
            call_info = c
            break

    if not call_info:
        raise HTTPException(status_code=404, detail="Call not found")

    # Load call data from S3
    data = s3_client.get_call_data(call_info["s3_key"])
    if not data:
        raise HTTPException(status_code=404, detail="Call data not found in S3")

    # Get audio URL
    audio_url = s3_client.get_audio_url(
        call_info["video_id"],
        call_info["call_id"],
        call_info["run_id"],
    )

    # Extract unique speakers
    turns = data.get("turns", [])
    speakers = list(dict.fromkeys(t.get("spk") for t in turns if t.get("spk")))

    # Parse start/end time from call_id (format: video_id_startMs_endMs)
    # e.g., "-POaWp9_UaM_143317_179145" -> start=143.317s, end=179.145s
    call_id_parts = call_id.rsplit("_", 2)
    start_time_s = 0.0
    end_time_s = 0.0
    if len(call_id_parts) == 3:
        try:
            start_time_s = int(call_id_parts[1]) / 1000.0
            end_time_s = int(call_id_parts[2]) / 1000.0
        except ValueError:
            pass

    # Calculate duration from clip bounds
    duration_s = end_time_s - start_time_s if end_time_s > start_time_s else 0

    return {
        "call_id": call_id,
        "video_id": call_info["video_id"],
        "audio_url": audio_url,
        "start_time_s": start_time_s,
        "duration_s": duration_s,
        "speakers": speakers,
        "turns": turns,
    }


@app.post("/api/calls/{call_id}/labels")
def save_call_labels(call_id: str, request: LabelRequest):
    """Save labels for a call."""
    calls = get_calls()

    # Find call to get video_id
    call_info = None
    for c in calls:
        if c["call_id"] == call_id:
            call_info = c
            break

    if not call_info:
        raise HTTPException(status_code=404, detail="Call not found")

    labels.save_labels(
        call_info["video_id"],
        call_id,
        request.speaker_roles,
    )

    return {"success": True}


@app.post("/api/calls/{call_id}/skip")
def skip_call(call_id: str):
    """Skip a call."""
    labels.skip_call(call_id)
    return {"success": True}


@app.post("/api/navigate/{direction}")
def navigate(direction: str):
    """Navigate prev/next."""
    calls = get_calls()
    current = labels.get_current_index()

    if direction == "next":
        new_index = min(current + 1, len(calls))  # Allow reaching len(calls) for "All done" state
    elif direction == "prev":
        new_index = max(current - 1, 0)
    else:
        raise HTTPException(status_code=400, detail="Invalid direction")

    labels.set_current_index(new_index)
    return {"index": new_index}


# =============================================================================
# Boundary Editor Endpoints
# =============================================================================


class BoundaryRequest(BaseModel):
    boundaries: List[Dict[str, float]]  # [{"start_s": 0.0, "end_s": 123.4}, ...]


@app.get("/api/videos")
def list_videos():
    """List all videos with their call counts and correction status."""
    videos = s3_client.list_videos()
    corrected_ids = boundaries.get_corrected_video_ids()

    result = []
    for v in videos:
        result.append({
            **v,
            "has_corrections": v["video_id"] in corrected_ids,
        })

    return result


@app.get("/api/videos/{video_id}/boundaries")
def get_video_boundaries(video_id: str):
    """Get both auto-detected and corrected boundaries for a video."""
    # Get auto-detected boundaries from existing calls
    auto_boundaries = s3_client.get_auto_detected_boundaries(video_id)

    # Get corrected boundaries if any
    corrected = boundaries.load_corrected_boundaries(video_id)
    corrected_list = boundaries.boundaries_to_dict(corrected)

    return {
        "video_id": video_id,
        "auto_boundaries": auto_boundaries,
        "corrected_boundaries": corrected_list,
        "has_corrections": len(corrected_list) > 0,
    }


@app.post("/api/videos/{video_id}/boundaries")
def save_video_boundaries(video_id: str, request: BoundaryRequest):
    """Save corrected boundaries for a video."""
    boundaries.save_corrected_boundaries(video_id, request.boundaries)
    return {"success": True, "count": len(request.boundaries)}


@app.get("/api/videos/{video_id}/audio-url")
def get_video_audio_url(video_id: str):
    """Get presigned URL for full video audio."""
    url = s3_client.get_full_video_audio_url(video_id)
    if not url:
        raise HTTPException(status_code=404, detail="Audio file not found")
    return {"url": url}


@app.get("/api/videos/{video_id}/audio")
def stream_video_audio(video_id: str, range: Optional[str] = Header(None)):
    """Stream full video audio from S3 with Range support."""
    audio_key = s3_client.get_full_video_audio_key(video_id)
    if not audio_key:
        raise HTTPException(status_code=404, detail="Audio file not found")

    s3 = s3_client.get_s3_client()

    # Get file size first
    head = s3.head_object(Bucket=s3_client.S3_BUCKET, Key=audio_key)
    file_size = head["ContentLength"]

    # Parse Range header
    start = 0
    end = file_size - 1

    if range:
        # Parse "bytes=start-end" or "bytes=start-"
        range_str = range.replace("bytes=", "")
        if "-" in range_str:
            parts = range_str.split("-")
            if parts[0]:
                start = int(parts[0])
            if parts[1]:
                end = int(parts[1])

    # Ensure valid range
    if start >= file_size:
        raise HTTPException(status_code=416, detail="Range not satisfiable")
    end = min(end, file_size - 1)
    content_length = end - start + 1

    def generate():
        # Use Range header for S3 request
        s3_range = f"bytes={start}-{end}"
        response = s3.get_object(
            Bucket=s3_client.S3_BUCKET,
            Key=audio_key,
            Range=s3_range
        )
        for chunk in response["Body"].iter_chunks(chunk_size=65536):
            yield chunk

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Cache-Control": "public, max-age=3600",
    }

    status_code = 206 if range else 200
    return StreamingResponse(
        generate(),
        status_code=status_code,
        media_type="audio/mpeg",
        headers=headers
    )
