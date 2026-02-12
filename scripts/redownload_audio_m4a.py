#!/usr/bin/env python3
"""
Re-download YouTube audio as M4A (AAC) and upload to S3.

Extracts YouTube video IDs from existing MP3 filenames in S3
(audio/pretraining/), re-downloads in native M4A (AAC) format
via yt-dlp, uploads to audio_m4a_aac/pretraining/, and deletes
the local file immediately after each upload.

Usage:
    python scripts/redownload_audio_m4a.py                         # Dry-run
    python scripts/redownload_audio_m4a.py --execute                # Download + upload
    python scripts/redownload_audio_m4a.py --execute --limit 5      # First 5 only
    python scripts/redownload_audio_m4a.py --execute --video-id ID  # Single video
    python scripts/redownload_audio_m4a.py --execute --concurrency 3
"""

import argparse
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import boto3

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "m4a_redownload_manifest.json"
DOWNLOAD_DIR = REPO_ROOT / "artifacts" / "m4a_downloads"

# ---------------------------------------------------------------------------
# S3 config
# ---------------------------------------------------------------------------
BUCKET = "rezora-whisperx-us-east-1-864981718771"
SOURCE_PREFIX = "audio/pretraining/"
DEST_PREFIX = "audio_m4a_aac/pretraining/"

# ---------------------------------------------------------------------------
# Video ID regex — same as pipeline/call_extractor_wavlm/io.py:20
# plus a fallback for the one oddly-named file.
# ---------------------------------------------------------------------------
_STANDARD_RE = re.compile(r" - ([A-Za-z0-9_-]{11})\.mp3$")
_PRETRAINING_PREFIX_RE = re.compile(r"^pretraining-([A-Za-z0-9_-]{11})-")

# ---------------------------------------------------------------------------
# Graceful stop (matches scripts/run_on_ec2.py pattern)
# ---------------------------------------------------------------------------
_stop_requested = False


def _signal_handler(_sig, _frame):
    global _stop_requested
    if _stop_requested:
        print("\n\nForce exit. Saving manifest...")
        sys.exit(1)
    print("\n\nStop requested. Finishing current download...")
    print("  (Ctrl+C again to force exit)\n")
    _stop_requested = True


signal.signal(signal.SIGINT, _signal_handler)

# Thread-safe manifest writes
_manifest_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_video_id(filename: str) -> str | None:
    """Extract 11-char YouTube video ID from an MP3 filename."""
    m = _STANDARD_RE.search(filename)
    if m:
        return m.group(1)
    m = _PRETRAINING_PREFIX_RE.search(filename)
    if m:
        logger.info("Used fallback regex for: %s", filename)
        return m.group(1)
    return None


def list_s3_mp3_files(s3_client) -> list[dict]:
    """List MP3 files from audio/pretraining/, filtering out NoCalls."""
    paginator = s3_client.get_paginator("list_objects_v2")
    files: list[dict] = []
    skipped_nocalls = 0
    skipped_no_id = 0

    for page in paginator.paginate(Bucket=BUCKET, Prefix=SOURCE_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".mp3"):
                continue
            filename = key.split("/")[-1]

            if filename.startswith("NoCalls"):
                skipped_nocalls += 1
                continue

            video_id = extract_video_id(filename)
            if not video_id:
                logger.warning("No video ID extracted from: %s", filename)
                skipped_no_id += 1
                continue

            files.append(
                {
                    "key": key,
                    "filename": filename,
                    "video_id": video_id,
                    "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
                }
            )

    logger.info(
        "Found %d eligible MP3s (skipped %d NoCalls, %d no-ID)",
        len(files),
        skipped_nocalls,
        skipped_no_id,
    )
    return files


def list_existing_m4a_files(s3_client) -> set[str]:
    """Get video IDs already uploaded to the M4A destination."""
    paginator = s3_client.get_paginator("list_objects_v2")
    existing: set[str] = set()
    m4a_id_re = re.compile(r"([A-Za-z0-9_-]{11})\.m4a$")

    for page in paginator.paginate(Bucket=BUCKET, Prefix=DEST_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = key.split("/")[-1]
            m = m4a_id_re.search(filename)
            if m:
                existing.add(m.group(1))

    logger.info("Found %d existing M4A files in S3", len(existing))
    return existing


def load_manifest() -> dict | None:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return None


def save_manifest(manifest: dict) -> None:
    with _manifest_lock:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = MANIFEST_PATH.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=2)
        tmp.replace(MANIFEST_PATH)


# ---------------------------------------------------------------------------
# Download + Upload
# ---------------------------------------------------------------------------


def download_m4a(video_id: str, download_dir: Path) -> Path | None:
    """Download native M4A audio from YouTube via yt-dlp.

    Prefers native AAC (m4a) stream. Falls back to best audio + transcode.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    output_template = str(download_dir / "%(id)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-f",
        "bestaudio[ext=m4a]/bestaudio",
        "-x",
        "--audio-format",
        "m4a",
        "--audio-quality",
        "0",
        "-o",
        output_template,
        "--no-overwrites",
        url,
    ]

    for attempt in range(2):
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0:
                break

            stderr = result.stderr
            if attempt == 0 and ("429" in stderr or "Too Many Requests" in stderr):
                logger.warning("Rate limited for %s, sleeping 60s...", video_id)
                time.sleep(60)
                continue

            logger.error(
                "Download failed for %s: %s", video_id, stderr.strip()[:300]
            )
            return None

        except subprocess.TimeoutExpired:
            logger.error("Download timed out for %s (attempt %d)", video_id, attempt + 1)
            if attempt == 0:
                continue
            return None

    # Find the downloaded M4A file — look for video_id in the filename
    candidates = [
        f
        for f in download_dir.iterdir()
        if video_id in f.name and f.suffix == ".m4a" and not f.name.endswith(".part")
    ]

    if not candidates:
        logger.error("Downloaded file not found for %s in %s", video_id, download_dir)
        return None

    return max(candidates, key=lambda f: f.stat().st_mtime)


def upload_to_s3(s3_client, local_path: Path) -> str:
    """Upload M4A file to S3. Returns the destination key."""
    dest_key = f"{DEST_PREFIX}{local_path.name}"
    s3_client.upload_file(str(local_path), BUCKET, dest_key)
    return dest_key


def _remove_local(path: Path) -> None:
    """Best-effort delete of a local file."""
    try:
        os.remove(path)
    except OSError:
        pass


def process_one_video(
    video_id: str,
    entry: dict,
    download_dir: Path,
    s3_client,
) -> str:
    """Download, upload, clean up one video. Returns status string."""
    local_path = download_m4a(video_id, download_dir)
    if local_path is None:
        entry["attempted_at"] = datetime.now(tz=timezone.utc).isoformat()
        return "failed"

    # Upload with one retry
    for attempt in range(2):
        try:
            dest_key = upload_to_s3(s3_client, local_path)
            entry["m4a_s3_key"] = dest_key
            entry["uploaded_at"] = datetime.now(tz=timezone.utc).isoformat()
            _remove_local(local_path)
            return "uploaded"
        except Exception as e:
            if attempt == 0:
                logger.warning("Upload retry for %s: %s", video_id, e)
                time.sleep(2)
            else:
                logger.error("Upload failed for %s: %s", video_id, e)

    # Delete local even on failure to preserve disk space
    _remove_local(local_path)
    entry["attempted_at"] = datetime.now(tz=timezone.utc).isoformat()
    return "upload_failed"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Re-download YouTube audio in M4A (AAC) and upload to S3"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually download and upload (default is dry-run)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max videos to process (0 = all)",
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default=None,
        help="Process a single video ID (for testing)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Parallel downloads (default: 3, max recommended: 5)",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=DOWNLOAD_DIR,
        help="Temp download directory",
    )
    args = parser.parse_args()

    if args.concurrency > 5:
        logger.warning("Concurrency > 5 may trigger YouTube rate limits")

    # Check yt-dlp is installed
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        logger.error("yt-dlp not found. Install with: brew install yt-dlp")
        sys.exit(1)

    s3 = boto3.client("s3")

    # --- Step 1-3: List, filter, extract ---
    logger.info("Listing MP3 files from s3://%s/%s ...", BUCKET, SOURCE_PREFIX)
    mp3_files = list_s3_mp3_files(s3)

    # Deduplicate by video_id (keep first seen)
    seen: dict[str, dict] = {}
    for f in mp3_files:
        vid = f["video_id"]
        if vid in seen:
            logger.warning(
                "Duplicate video ID %s: %s vs %s", vid, seen[vid]["key"], f["key"]
            )
        else:
            seen[vid] = f
    mp3_files = list(seen.values())

    # Filter to single video if requested
    if args.video_id:
        mp3_files = [f for f in mp3_files if f["video_id"] == args.video_id]
        if not mp3_files:
            logger.error("Video ID %s not found in S3", args.video_id)
            sys.exit(1)

    # --- Step 4: Check existing M4A files for resume ---
    logger.info("Checking existing M4A files in s3://%s/%s ...", BUCKET, DEST_PREFIX)
    existing = list_existing_m4a_files(s3)

    # --- Step 5: Build manifest ---
    manifest = {
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "bucket": BUCKET,
        "source_prefix": SOURCE_PREFIX,
        "dest_prefix": DEST_PREFIX,
        "total_eligible": len(mp3_files),
        "videos": {},
    }

    pending: list[dict] = []
    for f in mp3_files:
        vid = f["video_id"]
        status = "already_uploaded" if vid in existing else "pending"
        manifest["videos"][vid] = {
            "youtube_url": f["youtube_url"],
            "source_s3_key": f["key"],
            "status": status,
        }
        if status == "pending":
            pending.append(f)

    save_manifest(manifest)

    # --- Step 6: Summary / dry-run ---
    print(f"\n{'=' * 60}")
    print(f"  Total eligible MP3s:    {len(mp3_files)}")
    print(f"  Already uploaded (M4A): {len(mp3_files) - len(pending)}")
    print(f"  Pending downloads:      {len(pending)}")
    print(f"{'=' * 60}")

    if not args.execute:
        print("\nDRY RUN - YouTube URLs to download:\n")
        for f in pending[:20]:
            print(f"  {f['video_id']}  {f['youtube_url']}")
        if len(pending) > 20:
            print(f"  ... and {len(pending) - 20} more")
        print(f"\nManifest saved to: {MANIFEST_PATH}")
        print("Run with --execute to download and upload")
        return

    # --- Step 7-11: Download, upload, clean up ---
    if args.limit > 0:
        pending = pending[: args.limit]
        logger.info("Limited to %d videos", args.limit)

    if not pending:
        logger.info("Nothing to download — all videos already uploaded.")
        return

    args.download_dir.mkdir(parents=True, exist_ok=True)

    succeeded = 0
    failed = 0

    if args.concurrency <= 1:
        # Sequential
        for i, f in enumerate(pending, 1):
            if _stop_requested:
                logger.info("Stop requested. Saving manifest and exiting.")
                break

            vid = f["video_id"]
            logger.info("[%d/%d] Processing %s ...", i, len(pending), vid)
            status = process_one_video(
                vid, manifest["videos"][vid], args.download_dir, s3
            )
            manifest["videos"][vid]["status"] = status

            if status == "uploaded":
                succeeded += 1
                logger.info("[%d/%d] Uploaded %s", i, len(pending), vid)
            else:
                failed += 1
                logger.error("[%d/%d] Failed %s (%s)", i, len(pending), vid, status)

            save_manifest(manifest)
    else:
        # Parallel
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {}
            for f in pending:
                if _stop_requested:
                    break
                vid = f["video_id"]
                thread_s3 = boto3.client("s3")
                future = executor.submit(
                    process_one_video,
                    vid,
                    manifest["videos"][vid],
                    args.download_dir,
                    thread_s3,
                )
                futures[future] = vid

            for future in as_completed(futures):
                vid = futures[future]
                try:
                    status = future.result()
                except Exception as e:
                    logger.error("Unexpected error for %s: %s", vid, e)
                    status = "failed"
                    manifest["videos"][vid]["status"] = status

                manifest["videos"][vid]["status"] = status
                if status == "uploaded":
                    succeeded += 1
                else:
                    failed += 1

                save_manifest(manifest)

    # --- Final summary ---
    total_already = len(mp3_files) - len(pending)
    print(f"\n{'=' * 60}")
    print(f"  Succeeded:        {succeeded}")
    print(f"  Failed:           {failed}")
    print(f"  Already uploaded: {total_already}")
    print(f"{'=' * 60}")
    print(f"Manifest: {MANIFEST_PATH}")

    if failed:
        failed_ids = [
            vid
            for vid, v in manifest["videos"].items()
            if v["status"] in ("failed", "upload_failed")
        ]
        print(f"\nFailed video IDs ({len(failed_ids)}):")
        for vid in failed_ids:
            print(f"  {vid}  https://www.youtube.com/watch?v={vid}")


if __name__ == "__main__":
    main()
