# scripts/whisperx_pipeline/generate_label_queue.py
"""Generate labeling queue from extracted calls."""
import json
import boto3
from pathlib import Path
from typing import List
import sys

from .config import S3_BUCKET, AWS_REGION


def generate_preview(turns: List[dict], max_turns: int = 12) -> str:
    """Generate text preview for labeling UI."""
    lines = []
    for t in turns[:max_turns]:
        text = t.get("text", "")[:60]
        lines.append(f"{t['spk']}: {text}")
    if len(turns) > max_turns:
        lines.append(f"... ({len(turns) - max_turns} more turns)")
    return "\n".join(lines)


def get_speakers_summary(turns: List[dict]) -> str:
    """Get summary of speakers in the call."""
    speakers: dict[str, int] = {}
    for t in turns:
        spk = t["spk"]
        speakers[spk] = speakers.get(spk, 0) + 1
    return ", ".join(f"{spk} ({count} turns)" for spk, count in sorted(speakers.items()))


def generate_queue(s3_prefix: str, output_path: str, limit: int = 100) -> None:
    """
    Generate labeling queue JSONL from S3 extracted calls.

    Args:
        s3_prefix: S3 prefix to scan (e.g., "runs/VIDEO_ID/RUN_ID/calls/")
        output_path: Local path for queue.jsonl
        limit: Max items to include
    """
    s3 = boto3.client("s3", region_name=AWS_REGION)

    # List all spk_turns.json files
    paginator = s3.get_paginator("list_objects_v2")
    items: List[str] = []

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("spk_turns.json"):
                items.append(key)

    print(f"Found {len(items)} calls under {s3_prefix}")

    queue: List[dict] = []
    for key in items[:limit]:
        try:
            response = s3.get_object(Bucket=S3_BUCKET, Key=key)
            data = json.loads(response["Body"].read())
            turns = data.get("turns", [])

            # Extract call_id and video_id from path
            # Format: runs/{video_id}/{run_id}/calls/{call_id}/spk_turns.json
            parts = key.split("/")
            video_id = parts[1] if len(parts) > 1 else "unknown"
            call_id = parts[4] if len(parts) > 4 else key

            item = {
                "call_id": call_id,
                "video_id": video_id,
                "s3_key": key,
                "preview_text": generate_preview(turns),
                "speakers_summary": get_speakers_summary(turns),
                "turns": turns,
                "call_start_abs": data.get("call_start_abs", 0.0),
                "num_turns": len(turns),
                "num_speakers": len(set(t["spk"] for t in turns)) if turns else 0
            }
            queue.append(item)

        except Exception as e:
            print(f"Error loading {key}: {e}")

    # Sort by number of speakers (prioritize 3+ speaker calls for labeling)
    queue.sort(key=lambda x: (-x["num_speakers"], x["call_id"]))

    with open(output_path, "w") as f:
        for item in queue:
            f.write(json.dumps(item) + "\n")

    print(f"Generated queue with {len(queue)} items: {output_path}")
    print(f"  3+ speakers: {sum(1 for q in queue if q['num_speakers'] >= 3)}")
    print(f"  2 speakers: {sum(1 for q in queue if q['num_speakers'] == 2)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_label_queue.py <s3_prefix> <output.jsonl>")
        print("Example: python generate_label_queue.py runs/abc123/ queue.jsonl")
        sys.exit(1)
    generate_queue(sys.argv[1], sys.argv[2])
