# scripts/whisperx_pipeline/extract_training_features.py
"""Extract features from S3 calls for training the role classifier."""
import json
import boto3
import argparse
from typing import List
from pathlib import Path

from .config import S3_BUCKET, AWS_REGION
from .role_features import extract_all_speaker_features


def extract_training_features(
    s3_prefix: str,
    output_path: str,
    limit: int = 0,
    bucket: str | None = None,
    region: str | None = None
) -> None:
    """
    Load calls from S3, extract features, save as JSON for training.

    Args:
        s3_prefix: S3 prefix to scan (e.g., "runs/VIDEO_ID/")
        output_path: Local path for features.json
        limit: Max calls to process (0 = unlimited)
        bucket: S3 bucket (defaults to config)
        region: AWS region (defaults to config)
    """
    bucket = bucket or S3_BUCKET
    region = region or AWS_REGION
    s3 = boto3.client("s3", region_name=region)

    # List all spk_turns.json files
    paginator = s3.get_paginator("list_objects_v2")
    call_keys: List[str] = []

    for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("spk_turns.json"):
                call_keys.append(key)

    print(f"Found {len(call_keys)} calls under s3://{bucket}/{s3_prefix}")

    if limit > 0:
        call_keys = call_keys[:limit]
        print(f"Limiting to {limit} calls")

    features_data: List[dict] = []
    errors = 0

    for i, key in enumerate(call_keys):
        try:
            # Load spk_turns.json from S3
            response = s3.get_object(Bucket=bucket, Key=key)
            data = json.loads(response["Body"].read())
            turns = data.get("turns", [])

            if not turns:
                print(f"  Skipping {key} - no turns")
                continue

            # Extract call_id and video_id from path
            # Format: runs/{video_id}/{run_id}/calls/{call_id}/spk_turns.json
            parts = key.split("/")
            if len(parts) >= 5:
                video_id = parts[1]
                call_id = parts[4]
            else:
                # Fallback: use key as call_id
                video_id = "unknown"
                call_id = Path(key).stem

            # Extract features for all speakers
            call_start_abs = data.get("call_start_abs", 0.0)
            speaker_features = extract_all_speaker_features(turns, call_start_abs)

            features_data.append({
                "call_id": call_id,
                "video_id": video_id,
                "s3_key": key,
                "speakers": speaker_features,
                "num_speakers": len(speaker_features),
                "num_turns": len(turns)
            })

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(call_keys)} calls...")

        except Exception as e:
            print(f"  Error processing {key}: {e}")
            errors += 1

    # Save features
    with open(output_path, "w") as f:
        json.dump(features_data, f, indent=2)

    print(f"\nExtracted features for {len(features_data)} calls")
    print(f"  3+ speakers: {sum(1 for f in features_data if f['num_speakers'] >= 3)}")
    print(f"  2 speakers: {sum(1 for f in features_data if f['num_speakers'] == 2)}")
    print(f"  Errors: {errors}")
    print(f"Saved to: {output_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract features from S3 calls for role classifier training"
    )
    parser.add_argument(
        "s3_prefix",
        help="S3 prefix to scan (e.g., 'runs/VIDEO_ID/' or 'runs/')"
    )
    parser.add_argument(
        "output",
        help="Output path for features JSON file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max calls to process (0 = unlimited)"
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help=f"S3 bucket (default: {S3_BUCKET})"
    )
    parser.add_argument(
        "--region",
        default=None,
        help=f"AWS region (default: {AWS_REGION})"
    )

    args = parser.parse_args()
    extract_training_features(
        args.s3_prefix,
        args.output,
        limit=args.limit,
        bucket=args.bucket,
        region=args.region
    )


if __name__ == "__main__":
    main()
