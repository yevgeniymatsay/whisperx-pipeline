#!/usr/bin/env python3
"""
Enhanced pair detection - finds ALL before/after pairs from curated JSONL.
"""

import boto3
import json
import csv
from pathlib import Path
from collections import defaultdict

S3_BUCKET = "rezora-data-pipeline-864981718771"
AWS_REGION = "us-east-2"
OUTPUT_DIR = Path("artifacts/s3_audit")

# Curated files in order (v6 is latest with manual fixes)
CURATED_FILES = [
    "pretraining/curated/pretraining_v6_structure_validated.jsonl",
    "pretraining/curated/pretraining_v5_profanity_cleaned.jsonl",
    "pretraining/curated/pretraining_v4_quality_reviewed.jsonl",
    "pretraining/curated/pretraining_v3_final.jsonl",
    "pretraining/curated/pretraining_v2_curated.jsonl",
]

# Before prefixes to search
BEFORE_PREFIXES = [
    "pretraining/extracted/",
    "pretraining/accepted/",
]

s3 = boto3.client("s3", region_name=AWS_REGION)


def download_file(key: str) -> str:
    """Download full file content."""
    response = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return response["Body"].read().decode("utf-8")


def list_objects(prefix: str) -> dict:
    """List all objects under prefix, return dict keyed by filename."""
    objects = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = Path(key).name
            objects[filename] = {
                "key": key,
                "s3_uri": f"s3://{S3_BUCKET}/{key}",
                "size": obj["Size"],
            }
    return objects


def main():
    print("=" * 60)
    print("ENHANCED PAIR DETECTION")
    print("=" * 60)

    # Build index of "before" files
    print("\n[1/3] Indexing 'before' files (extracted + accepted)...")
    before_index = {}
    for prefix in BEFORE_PREFIXES:
        objs = list_objects(prefix)
        print(f"  {prefix}: {len(objs)} files")
        before_index.update(objs)

    print(f"  Total 'before' files: {len(before_index)}")

    # Parse curated files and find pairs
    print("\n[2/3] Parsing curated JSONL files...")
    all_pairs = []
    seen_sources = set()

    for curated_key in CURATED_FILES:
        try:
            content = download_file(curated_key)
            lines = content.strip().split("\n")
            print(f"  {curated_key}: {len(lines)} conversations")

            for line in lines:
                item = json.loads(line)
                source_file = item.get("_source_file", "")

                if source_file and source_file not in seen_sources:
                    seen_sources.add(source_file)

                    # Find matching before file
                    if source_file in before_index:
                        before_info = before_index[source_file]
                        all_pairs.append({
                            "source_file": source_file,
                            "before_s3_uri": before_info["s3_uri"],
                            "before_prefix": before_info["key"].split("/")[1],
                            "after_s3_uri": f"s3://{S3_BUCKET}/{curated_key}",
                            "curated_version": curated_key.split("/")[-1].replace(".jsonl", ""),
                            "scalar_score": item.get("scalar_score"),
                            "call_direction": item.get("_call_direction"),
                            "num_turns_after": len([m for m in item.get("messages", []) if m["role"] != "system"]),
                        })

        except Exception as e:
            print(f"    Error: {e}")

    print(f"\n[3/3] Found {len(all_pairs)} unique pairs")

    # Also check for items in extracted but NOT in any curated (potential rejects)
    curated_sources = set(p["source_file"] for p in all_pairs)
    extracted_only = []
    for filename, info in before_index.items():
        if "extracted" in info["key"] and filename not in curated_sources:
            extracted_only.append({
                "s3_uri": info["s3_uri"],
                "filename": filename,
                "reason": "in_extracted_not_curated",
            })

    print(f"  Items in extracted but not curated: {len(extracted_only)}")

    # Write enhanced pairs.csv
    pairs_path = OUTPUT_DIR / "pairs.csv"
    with open(pairs_path, "w", newline="") as f:
        fieldnames = [
            "pair_id", "source_file", "before_s3_uri", "before_prefix",
            "after_s3_uri", "curated_version", "scalar_score",
            "call_direction", "num_turns_after"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, pair in enumerate(all_pairs, 1):
            pair["pair_id"] = i
            writer.writerow(pair)
    print(f"\n  Updated {pairs_path} ({len(all_pairs)} pairs)")

    # Append extracted-only items to rejected.csv
    rejected_path = OUTPUT_DIR / "rejected.csv"

    # Read existing rejected items
    existing_rejected = []
    with open(rejected_path, "r") as f:
        reader = csv.DictReader(f)
        existing_rejected = list(reader)

    # Add extracted-only items
    for item in extracted_only:
        existing_rejected.append({
            "s3_uri": item["s3_uri"],
            "evidence_source": "in_extracted_not_curated",
            "notes": f"File {item['filename']} was extracted but never made it to curated"
        })

    with open(rejected_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["s3_uri", "evidence_source", "notes"])
        writer.writeheader()
        writer.writerows(existing_rejected)
    print(f"  Updated {rejected_path} ({len(existing_rejected)} total items)")

    # Update summary
    summary_path = OUTPUT_DIR / "dataset_summary.json"
    with open(summary_path, "r") as f:
        summary = json.load(f)

    summary["pair_count"] = len(all_pairs)
    summary["rejected_count"] = len(existing_rejected)
    summary["curated_sources_found"] = len(curated_sources)
    summary["extracted_not_curated"] = len(extracted_only)

    # Add version breakdown
    version_counts = defaultdict(int)
    for pair in all_pairs:
        version_counts[pair["curated_version"]] += 1
    summary["pairs_by_curated_version"] = dict(version_counts)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Updated {summary_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nPairs by curated version:")
    for version, count in sorted(version_counts.items()):
        print(f"  {version}: {count}")
    print(f"\nTotal pairs: {len(all_pairs)}")
    print(f"Rejected/dropped: {len(existing_rejected)}")


if __name__ == "__main__":
    main()
