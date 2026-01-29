"""Extract audio segments for each call from full video audio."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path
import boto3

from .config import S3_BUCKET, AWS_REGION


def extract_audio_segment(
    input_path: str,
    output_path: str,
    start_s: float,
    end_s: float,
) -> bool:
    """Extract audio segment using ffmpeg."""
    duration = end_s - start_s
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ss", str(start_s),
        "-t", str(duration),
        "-acodec", "libmp3lame",
        "-q:a", "2",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Extract audio segments for calls from S3"
    )
    parser.add_argument(
        "s3_prefix",
        help="S3 prefix to scan (e.g., 'runs/' or 'runs/VIDEO_ID/')"
    )
    parser.add_argument(
        "--bucket",
        default=S3_BUCKET,
        help=f"S3 bucket (default: {S3_BUCKET})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be extracted without doing it"
    )
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=AWS_REGION)
    paginator = s3.get_paginator("list_objects_v2")

    # Find all spk_turns.json files
    calls_to_process = []
    for page in paginator.paginate(Bucket=args.bucket, Prefix=args.s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("spk_turns.json"):
                parts = key.split("/")
                if len(parts) >= 5:
                    calls_to_process.append({
                        "spk_turns_key": key,
                        "video_id": parts[1],
                        "run_id": parts[2],
                        "call_id": parts[4],
                    })

    print(f"Found {len(calls_to_process)} calls to process")

    # Group by video to avoid re-downloading audio
    by_video: dict[str, list] = {}
    for call in calls_to_process:
        vid = call["video_id"]
        if vid not in by_video:
            by_video[vid] = []
        by_video[vid].append(call)

    print(f"Across {len(by_video)} videos")

    # Build a lookup of video_id -> audio S3 key
    # Audio files are named like: "Title Here - VIDEO_ID.mp3"
    print("Scanning for audio files...")
    audio_lookup: dict[str, str] = {}
    for prefix in ["audio/pretraining/", "audio/expired_listing/", "audio/"]:
        for page in paginator.paginate(Bucket=args.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".mp3"):
                    # Extract video_id from filename (after last " - " before .mp3)
                    filename = key.split("/")[-1]
                    if " - " in filename:
                        vid = filename.rsplit(" - ", 1)[-1].replace(".mp3", "")
                        audio_lookup[vid] = key
                    else:
                        # Simple filename like video_id.mp3
                        vid = filename.replace(".mp3", "")
                        audio_lookup[vid] = key

    print(f"Found {len(audio_lookup)} audio files in S3")

    # Check which videos have audio
    missing_audio = [vid for vid in by_video if vid not in audio_lookup]
    if missing_audio:
        print(f"WARNING: {len(missing_audio)} videos missing audio:")
        for vid in missing_audio[:5]:
            print(f"  - {vid}")
        if len(missing_audio) > 5:
            print(f"  ... and {len(missing_audio) - 5} more")

    if args.dry_run:
        found_audio = [vid for vid in by_video if vid in audio_lookup]
        print(f"\nDry run summary:")
        print(f"  Videos with audio: {len(found_audio)}")
        print(f"  Videos missing audio: {len(missing_audio)}")
        print(f"  Calls ready to extract: {sum(len(by_video[v]) for v in found_audio)}")
        return

    extracted = 0
    errors = 0

    for video_id, video_calls in by_video.items():
        print(f"\nProcessing video: {video_id} ({len(video_calls)} calls)")

        # Find audio file for this video
        audio_key = audio_lookup.get(video_id)
        if not audio_key:
            print(f"  ERROR: No audio file found for video {video_id}")
            errors += len(video_calls)
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            local_audio = Path(tmpdir) / "full.mp3"

            try:
                s3.download_file(args.bucket, audio_key, str(local_audio))
            except Exception as e:
                print(f"  ERROR: Could not download {audio_key}: {e}")
                errors += len(video_calls)
                continue

            for call in video_calls:
                # Load spk_turns.json to get timestamps
                try:
                    response = s3.get_object(Bucket=args.bucket, Key=call["spk_turns_key"])
                    data = json.loads(response["Body"].read())
                except Exception as e:
                    print(f"  ERROR loading {call['call_id']}: {e}")
                    errors += 1
                    continue

                turns = data.get("turns", [])
                if not turns:
                    print(f"  SKIP {call['call_id']}: no turns")
                    continue

                # Calculate start/end from turns
                start_s = min(t.get("t0_abs", t.get("t0", 0)) for t in turns)
                end_s = max(t.get("t1_abs", t.get("t1", 0)) for t in turns)

                # Add small padding
                start_s = max(0, start_s - 0.5)
                end_s = end_s + 0.5

                # Extract segment
                local_segment = Path(tmpdir) / f"{call['call_id']}.mp3"
                if not extract_audio_segment(str(local_audio), str(local_segment), start_s, end_s):
                    print(f"  ERROR extracting {call['call_id']}")
                    errors += 1
                    continue

                # Upload to S3
                output_key = f"runs/{video_id}/{call['run_id']}/calls/{call['call_id']}/audio.mp3"
                try:
                    s3.upload_file(str(local_segment), args.bucket, output_key)
                    print(f"  Extracted: {call['call_id']} ({end_s - start_s:.1f}s)")
                    extracted += 1
                except Exception as e:
                    print(f"  ERROR uploading {call['call_id']}: {e}")
                    errors += 1

    print(f"\nDone! Extracted: {extracted}, Errors: {errors}")


if __name__ == "__main__":
    main()
