"""S3 client for loading calls and audio."""
import json
from typing import List, Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError

from config import S3_BUCKET, AWS_REGION, S3_PREFIX


def get_s3_client():
    """Get boto3 S3 client."""
    return boto3.client("s3", region_name=AWS_REGION)


def list_calls(prefix: str = S3_PREFIX) -> List[Dict[str, Any]]:
    """List all calls under the given prefix."""
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")

    calls = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("spk_turns.json"):
                # Parse path: runs/{video_id}/{run_id}/calls/{call_id}/spk_turns.json
                parts = key.split("/")
                if len(parts) >= 5:
                    calls.append({
                        "s3_key": key,
                        "video_id": parts[1],
                        "run_id": parts[2],
                        "call_id": parts[4],
                    })
    return calls


def get_call_data(s3_key: str) -> Optional[Dict[str, Any]]:
    """Load call data from S3."""
    s3 = get_s3_client()
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
        data = json.loads(response["Body"].read())
        return data
    except ClientError:
        return None


def get_audio_url(video_id: str, call_id: str, run_id: str) -> Optional[str]:
    """Generate presigned URL for call audio."""
    s3 = get_s3_client()
    # Audio key: runs/{video_id}/{run_id}/calls/{call_id}/audio.mp3
    audio_key = f"runs/{video_id}/{run_id}/calls/{call_id}/audio.mp3"

    try:
        # Verify file exists first
        s3.head_object(Bucket=S3_BUCKET, Key=audio_key)

        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": audio_key},
            ExpiresIn=3600,
        )
        return url
    except ClientError as e:
        print(f"Audio file not found: {audio_key}")
        return None


def list_videos() -> List[Dict[str, Any]]:
    """List all videos that have been processed."""
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")

    # Get unique video IDs from runs/ prefix
    video_ids = set()
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix="runs/", Delimiter="/"):
        for prefix in page.get("CommonPrefixes", []):
            # prefix looks like "runs/VIDEO_ID/"
            parts = prefix["Prefix"].rstrip("/").split("/")
            if len(parts) == 2:
                video_ids.add(parts[1])

    # Build video info list
    videos = []
    for video_id in sorted(video_ids):
        # Count calls for this video
        call_count = 0
        for page in paginator.paginate(
            Bucket=S3_BUCKET,
            Prefix=f"runs/{video_id}/",
        ):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith("spk_turns.json"):
                    call_count += 1

        videos.append({
            "video_id": video_id,
            "call_count": call_count,
        })

    return videos


def get_full_video_audio_key(video_id: str) -> Optional[str]:
    """Find the S3 key for full video audio."""
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix="audio/pretraining/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(f" - {video_id}.mp3") or key.endswith(f"-{video_id}.mp3"):
                return key

    print(f"Full video audio not found for video_id: {video_id}")
    return None


def get_full_video_audio_url(video_id: str) -> Optional[str]:
    """Generate presigned URL for full video audio."""
    audio_key = get_full_video_audio_key(video_id)
    if not audio_key:
        return None

    s3 = get_s3_client()
    try:
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": audio_key},
            ExpiresIn=3600,
        )
        return url
    except ClientError as e:
        print(f"Error generating presigned URL: {e}")
        return None


def get_auto_detected_boundaries(video_id: str) -> List[Dict[str, Any]]:
    """Get auto-detected call boundaries for a video (from existing calls)."""
    calls = list_calls(f"runs/{video_id}/")

    boundaries = []
    for call in calls:
        # Parse start/end from call_id: {video_id}_{start_ms}_{end_ms}
        call_id = call["call_id"]
        parts = call_id.rsplit("_", 2)
        if len(parts) == 3:
            try:
                start_ms = int(parts[1])
                end_ms = int(parts[2])
                boundaries.append({
                    "call_id": call_id,
                    "start_s": start_ms / 1000.0,
                    "end_s": end_ms / 1000.0,
                    "type": "auto",
                })
            except ValueError:
                pass

    # Sort by start time
    boundaries.sort(key=lambda x: x["start_s"])
    return boundaries
