#!/usr/bin/env python3
"""
Run WhisperX pipeline on EC2 instance for selected videos.

Reads video IDs from data/selected_videos.txt, finds matching audio files in S3,
and runs the WhisperX pipeline on the EC2 instance via SSH.

Usage:
    python scripts/run_whisperx_on_ec2.py                    # Dry-run (preview)
    python scripts/run_whisperx_on_ec2.py --execute          # Run all videos
    python scripts/run_whisperx_on_ec2.py --execute --limit 5  # Run first 5 videos
    python scripts/run_whisperx_on_ec2.py --video-id ABC123  # Run single video
"""

import argparse
import signal
import subprocess
import sys
from pathlib import Path

import boto3

# Graceful stop handling - finish current video, then exit
_stop_requested = False


def _signal_handler(sig, frame):
    """Handle Ctrl+C by setting stop flag instead of immediate exit."""
    global _stop_requested
    if _stop_requested:
        # Second Ctrl+C - force exit
        print("\n\nForce exit requested. Killing process...")
        sys.exit(1)
    print("\n\n⚠ Stop requested. Will exit after current video finishes...")
    print("  (Press Ctrl+C again to force exit immediately)\n")
    _stop_requested = True


signal.signal(signal.SIGINT, _signal_handler)

# Paths
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SELECTED_PATH = DATA_DIR / "selected_videos.txt"

# S3 config
S3_BUCKET = "rezora-whisperx-us-east-1-864981718771"
AUDIO_PREFIX = "audio/pretraining/"

# EC2 config
EC2_HOST = "ubuntu@13.217.101.70"
SSH_KEY = Path.home() / ".ssh" / "whisperx-key-east1.pem"

# AWS Secrets Manager
HF_TOKEN_SECRET = "hf-token"
HF_TOKEN_REGION = "us-east-1"


def load_selected_videos() -> list[str]:
    """Load selected video IDs from file."""
    if not SELECTED_PATH.exists():
        print(f"Error: {SELECTED_PATH} not found")
        print("Run the TUI first: python scripts/select_whisperx_videos.py")
        sys.exit(1)

    with open(SELECTED_PATH) as f:
        return [line.strip() for line in f if line.strip()]


def find_audio_files(s3_client, video_ids: set[str]) -> dict[str, str]:
    """Find S3 audio files matching video IDs.

    Returns dict mapping video_id -> S3 key.
    """
    matches: dict[str, str] = {}
    paginator = s3_client.get_paginator("list_objects_v2")

    print(f"Scanning s3://{S3_BUCKET}/{AUDIO_PREFIX}...")

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=AUDIO_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = key.split("/")[-1]

            for vid in video_ids:
                if vid in filename and vid not in matches:
                    matches[vid] = key
                    break

    return matches


def get_hf_token() -> str:
    """Retrieve HuggingFace token from Secrets Manager."""
    sm = boto3.client("secretsmanager", region_name=HF_TOKEN_REGION)
    response = sm.get_secret_value(SecretId=HF_TOKEN_SECRET)
    return response["SecretString"]


def check_already_processed(s3_client, video_id: str) -> bool:
    """Check if video has already been processed (has runs/ output)."""
    prefix = f"runs/{video_id}/"
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix, MaxKeys=1)
    return response.get("KeyCount", 0) > 0


def run_pipeline_on_ec2(video_id: str, audio_key: str, hf_token: str, dry_run: bool = True) -> bool:
    """Run WhisperX pipeline on EC2 for a single video."""
    # Build the command to run on EC2
    # Escape single quotes in audio_key for shell safety
    audio_key_escaped = audio_key.replace("'", "'\\''")
    remote_cmd = f"""
cd ~ && \\
export HF_TOKEN='{hf_token}' && \\
export PYTHONPATH="$HOME:$PYTHONPATH" && \\
export TORCH_HOME="$HOME/.cache/torch" && \\
export HF_HOME="$HOME/.cache/huggingface" && \\
python3 -m pipeline.cli --video-id='{video_id}' --audio-key='{audio_key_escaped}' --no-db
"""

    ssh_cmd = [
        "ssh",
        "-i", str(SSH_KEY),
        "-o", "StrictHostKeyChecking=no",
        "-o", "ServerAliveInterval=60",
        "-o", "ServerAliveCountMax=10",
        EC2_HOST,
        remote_cmd.strip()
    ]

    if dry_run:
        print(f"  [DRY-RUN] Would run: {video_id}")
        print(f"    Audio: {audio_key}")
        return True

    print(f"  Running: {video_id}")
    print(f"    Audio: {audio_key}")

    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=False,  # Stream output live
            timeout=3600,  # 1 hour timeout per video
        )
        if result.returncode == 0:
            print(f"  ✓ Completed: {video_id}")
            return True
        else:
            print(f"  ✗ Failed: {video_id} (exit code {result.returncode})")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout: {video_id}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {video_id} - {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run WhisperX pipeline on EC2 for selected videos"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run the pipeline (default is dry-run)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit to first N videos",
    )
    parser.add_argument(
        "--video-id",
        help="Run only this specific video ID",
    )
    parser.add_argument(
        "--video-list",
        type=Path,
        help="Path to file with video IDs (one per line)",
    )
    parser.add_argument(
        "--skip-processed",
        action="store_true",
        help="Skip videos that already have output in runs/",
    )
    args = parser.parse_args()

    # Load video IDs
    if args.video_id:
        video_ids = [args.video_id]
    elif args.video_list:
        if not args.video_list.exists():
            print(f"Error: {args.video_list} not found")
            sys.exit(1)
        with open(args.video_list) as f:
            video_ids = [line.strip() for line in f if line.strip()]
    else:
        video_ids = load_selected_videos()

    print(f"Found {len(video_ids)} selected video(s)")

    if args.limit:
        video_ids = video_ids[:args.limit]
        print(f"Limited to first {args.limit}")

    # Find matching audio files
    s3 = boto3.client("s3")
    audio_files = find_audio_files(s3, set(video_ids))

    # Report missing
    missing = set(video_ids) - set(audio_files.keys())
    if missing:
        print(f"\n⚠ {len(missing)} video(s) not found in S3:")
        for vid in sorted(missing)[:5]:
            print(f"    - {vid}")
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more")

    found_ids = [vid for vid in video_ids if vid in audio_files]
    print(f"\nFound {len(found_ids)} matching audio file(s)")

    # Skip already processed if requested
    if args.skip_processed:
        print("\nChecking for already processed videos...")
        to_process = []
        for vid in found_ids:
            if check_already_processed(s3, vid):
                print(f"  Skipping {vid} (already has output)")
            else:
                to_process.append(vid)
        found_ids = to_process
        print(f"Will process {len(found_ids)} video(s)")

    if not found_ids:
        print("\nNo videos to process.")
        return

    # Get HF token
    print("\nRetrieving HuggingFace token...")
    hf_token = get_hf_token()
    print("✓ Got HF token")

    # Process videos
    print(f"\n{'=' * 60}")
    print(f"{'EXECUTING' if args.execute else 'DRY RUN - Preview only'}")
    print(f"{'=' * 60}\n")

    succeeded = 0
    failed = 0

    for i, vid in enumerate(found_ids, 1):
        # Check for graceful stop before starting next video
        if _stop_requested:
            print(f"\n⚠ Stopping as requested. Processed {i-1}/{len(found_ids)} videos.")
            print(f"  Remaining videos will be picked up on next run (--skip-processed).")
            break

        print(f"[{i}/{len(found_ids)}] {vid}")
        audio_key = audio_files[vid]

        if run_pipeline_on_ec2(vid, audio_key, hf_token, dry_run=not args.execute):
            succeeded += 1
        else:
            failed += 1
        print()

    # Summary
    print(f"{'=' * 60}")
    print(f"Summary: {succeeded} succeeded, {failed} failed")
    if not args.execute:
        print(f"\nRun with --execute to actually process videos")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
