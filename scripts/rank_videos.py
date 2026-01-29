#!/usr/bin/env python3
"""
Rank WhisperX videos by quality signals for speaker labeling selection.

Scoring factors:
- View count (35%): Log-scaled, higher = better
- Duration (25%): Sweet spot 10-45 min
- Title signals (25%): Keywords indicating real calls
- Channel diversity (15%): Boost underrepresented channels
"""

import json
import math
import re
import csv
import sys
from pathlib import Path
from collections import Counter

import boto3

# Config
WHISPERX_BUCKET = "rezora-whisperx-us-east-1-864981718771"
AUDIO_PREFIX = "audio/pretraining/"
CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "video_views_cache.json"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "whisperx_video_rankings.csv"

# Scoring weights
WEIGHT_VIEWS = 0.35
WEIGHT_DURATION = 0.25
WEIGHT_TITLE = 0.25
WEIGHT_DIVERSITY = 0.15

# Title keywords
STRONG_POSITIVE = ["live", "cold call", "sales call", "100", "50", "real"]
WEAK_POSITIVE = ["call", "dialing", "prospect", "meeting", "objection"]
NEGATIVE = ["roleplay", "role play", "script", "tutorial", "how to", "tips", "training"]


def extract_video_id(s3_key: str) -> str | None:
    """Extract 11-char YouTube video ID from S3 key."""
    match = re.search(r' - ([a-zA-Z0-9_-]{11})\.mp3$', s3_key)
    return match.group(1) if match else None


def score_views(view_count: int) -> float:
    """Log-scaled view score. 1000=0.5, 10000=0.75, 100000=1.0"""
    if view_count <= 0:
        return 0.0
    log_views = math.log10(view_count)
    # Scale: log10(1000)=3, log10(100000)=5 -> map to 0.5-1.0
    return min(1.0, max(0.0, (log_views - 2) / 3))


def score_duration(duration_sec: int) -> float:
    """Score duration with sweet spot at 10-45 minutes."""
    minutes = duration_sec / 60
    if 10 <= minutes <= 45:
        return 1.0
    elif 5 <= minutes < 10:
        return 0.7
    elif 45 < minutes <= 90:
        return 0.7
    elif 2 <= minutes < 5:
        return 0.4
    elif 90 < minutes <= 120:
        return 0.5
    else:
        return 0.2


def score_title(title: str) -> float:
    """Score title based on keyword signals."""
    title_lower = title.lower()
    score = 0.5  # Base score

    # Strong positives
    for kw in STRONG_POSITIVE:
        if kw in title_lower:
            score += 0.15

    # Weak positives
    for kw in WEAK_POSITIVE:
        if kw in title_lower:
            score += 0.05

    # Negatives
    for kw in NEGATIVE:
        if kw in title_lower:
            score -= 0.2

    # Bonus for numbers (suggests multiple calls)
    if re.search(r'\b\d{2,3}\b', title):  # 2-3 digit numbers
        score += 0.1

    return min(1.0, max(0.0, score))


def calculate_diversity_scores(videos: list[dict], channel_counts: Counter) -> dict[str, float]:
    """Calculate diversity bonus for underrepresented channels."""
    max_count = max(channel_counts.values()) if channel_counts else 1
    diversity_scores = {}

    for v in videos:
        channel = v.get("channel", "Unknown")
        count = channel_counts.get(channel, 1)
        # Inverse relationship: fewer videos from channel = higher diversity score
        diversity_scores[v["video_id"]] = 1.0 - (count / max_count) * 0.5

    return diversity_scores


def main():
    # 1. List videos from S3
    print("Fetching video list from S3...")
    s3 = boto3.client("s3", region_name="us-east-1")

    paginator = s3.get_paginator("list_objects_v2")
    video_ids = []
    for page in paginator.paginate(Bucket=WHISPERX_BUCKET, Prefix=AUDIO_PREFIX):
        for obj in page.get("Contents", []):
            vid = extract_video_id(obj["Key"])
            if vid:
                video_ids.append(vid)

    print(f"Found {len(video_ids)} videos in S3")

    # 2. Load metadata cache
    print("Loading metadata cache...")
    if not CACHE_PATH.exists():
        print(f"ERROR: Cache not found at {CACHE_PATH}")
        sys.exit(1)

    with open(CACHE_PATH) as f:
        cache = json.load(f)

    # 3. Score each video
    print("Scoring videos...")
    videos = []
    channel_counts = Counter()

    for vid in video_ids:
        if vid not in cache:
            print(f"  Warning: {vid} not in cache, skipping")
            continue

        meta = cache[vid]
        channel_counts[meta.get("channel", "Unknown")] += 1

        videos.append({
            "video_id": vid,
            "title": meta.get("title", ""),
            "view_count": meta.get("view_count", 0),
            "duration": meta.get("duration", 0),
            "channel": meta.get("channel", "Unknown"),
            "view_score": score_views(meta.get("view_count", 0)),
            "duration_score": score_duration(meta.get("duration", 0)),
            "title_score": score_title(meta.get("title", "")),
        })

    # Calculate diversity scores
    diversity_scores = calculate_diversity_scores(videos, channel_counts)

    # Calculate final scores
    for v in videos:
        v["diversity_score"] = diversity_scores[v["video_id"]]
        v["total_score"] = (
            WEIGHT_VIEWS * v["view_score"] +
            WEIGHT_DURATION * v["duration_score"] +
            WEIGHT_TITLE * v["title_score"] +
            WEIGHT_DIVERSITY * v["diversity_score"]
        )

    # Sort by total score descending
    videos.sort(key=lambda x: x["total_score"], reverse=True)

    # 4. Output to CSV
    print(f"Writing rankings to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "rank", "video_id", "title", "channel", "view_count", "duration_min",
            "total_score", "view_score", "duration_score", "title_score", "diversity_score"
        ])
        writer.writeheader()

        for i, v in enumerate(videos, 1):
            writer.writerow({
                "rank": i,
                "video_id": v["video_id"],
                "title": v["title"][:80],  # Truncate for readability
                "channel": v["channel"],
                "view_count": v["view_count"],
                "duration_min": round(v["duration"] / 60, 1),
                "total_score": round(v["total_score"], 3),
                "view_score": round(v["view_score"], 3),
                "duration_score": round(v["duration_score"], 3),
                "title_score": round(v["title_score"], 3),
                "diversity_score": round(v["diversity_score"], 3),
            })

    # 5. Print top 10
    print("\n" + "="*80)
    print("TOP 10 VIDEOS FOR SPEAKER LABELING")
    print("="*80)
    for i, v in enumerate(videos[:10], 1):
        dur_min = v["duration"] / 60
        print(f"\n{i}. {v['title'][:60]}...")
        print(f"   ID: {v['video_id']} | Views: {v['view_count']:,} | Duration: {dur_min:.1f}min")
        print(f"   Score: {v['total_score']:.3f} (view:{v['view_score']:.2f} dur:{v['duration_score']:.2f} title:{v['title_score']:.2f} div:{v['diversity_score']:.2f})")
        print(f"   Channel: {v['channel']}")

    print(f"\nFull rankings saved to: {OUTPUT_PATH}")
    print(f"Total videos ranked: {len(videos)}")


if __name__ == "__main__":
    main()
