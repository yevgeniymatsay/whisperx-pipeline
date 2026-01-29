#!/usr/bin/env python3
"""
S3 Audit Script for Judge LLM Training Data

This script audits the S3 bucket to find:
- Before/after pairs (extracted -> curated)
- Rejected items (for negative examples)
- Schema information

Outputs:
- artifacts/s3_audit/s3_objects.jsonl
- artifacts/s3_audit/pairs.csv
- artifacts/s3_audit/rejected.csv
- artifacts/s3_audit/dataset_summary.json
- artifacts/s3_audit/REPORT.md
"""

import boto3
import json
import csv
import hashlib
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Optional

# Configuration
S3_BUCKET = "rezora-data-pipeline-864981718771"
AWS_REGION = "us-east-2"
OUTPUT_DIR = Path("artifacts/s3_audit")

# Prefixes to scan
PREFIXES_TO_SCAN = [
    "",  # Root level
    "pretraining/",
    "curated/",
]

# Classification rules
PATH_TAGS = {
    "accepted": ["accepted", "approved", "clean", "final"],
    "rejected": ["rejected", "bad", "filtered", "trash", "quarantine"],
    "review": ["review", "pending", "needs-review"],
    "extracted": ["extracted", "raw"],
    "curated": ["curated", "cleaned", "fixed"],
    "training": ["training", "sft", "dataset"],
    "dropped": ["dropped"],
    "transcripts": ["transcript", "transcribed"],
    "audio": ["audio", "mp3", "wav"],
    "quality": ["quality-review", "quality"],
    "structure": ["structure-review", "structure"],
    "profanity": ["profanity-review", "profanity"],
    "similarity": ["similarity-review", "similarity", "dedupe"],
}

s3 = boto3.client("s3", region_name=AWS_REGION)


def classify_object(key: str, size: int) -> dict:
    """Classify an S3 object based on its path and extension."""
    ext = Path(key).suffix.lower()
    path_lower = key.lower()

    # Find matching path tags
    tags = []
    for tag, patterns in PATH_TAGS.items():
        for pattern in patterns:
            if pattern in path_lower:
                tags.append(tag)
                break

    # Guess type based on path and extension
    guessed_type = "unknown"
    confidence = 0.3

    if ext in [".json", ".jsonl"]:
        if "curated" in path_lower or "cleaned" in path_lower or "fixed" in path_lower:
            guessed_type = "clean_sft"
            confidence = 0.9
        elif "extracted" in path_lower:
            guessed_type = "raw_sft"
            confidence = 0.85
        elif "accepted" in path_lower:
            guessed_type = "accepted_sft"
            confidence = 0.85
        elif "rejected" in path_lower:
            guessed_type = "rejected"
            confidence = 0.9
        elif "dropped" in path_lower:
            guessed_type = "dropped"
            confidence = 0.85
        elif "review" in path_lower:
            guessed_type = "review_candidate"
            confidence = 0.8
        elif "training" in path_lower:
            guessed_type = "training_data"
            confidence = 0.85
        elif "transcript" in path_lower:
            guessed_type = "raw_transcript"
            confidence = 0.8
        else:
            guessed_type = "sft_json"
            confidence = 0.5
    elif ext in [".csv", ".tsv"]:
        guessed_type = "manifest_or_log"
        confidence = 0.6
    elif ext in [".mp3", ".wav", ".m4a"]:
        guessed_type = "audio"
        confidence = 0.95
    elif ext in [".txt"]:
        guessed_type = "text_file"
        confidence = 0.5
    elif ext in [".parquet"]:
        guessed_type = "parquet_dataset"
        confidence = 0.8

    return {
        "ext": ext,
        "path_tags": list(set(tags)),
        "guessed_type": guessed_type,
        "confidence": confidence,
    }


def list_all_objects(prefix: str = "") -> list:
    """List all objects under a prefix with pagination."""
    objects = []
    paginator = s3.get_paginator("list_objects_v2")

    try:
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                objects.append(obj)
    except Exception as e:
        print(f"  Error listing {prefix}: {e}")

    return objects


def get_object_metadata(key: str) -> dict:
    """Get HEAD metadata for an object."""
    try:
        response = s3.head_object(Bucket=S3_BUCKET, Key=key)
        return {
            "content_type": response.get("ContentType", ""),
            "user_metadata": response.get("Metadata", {}),
            "head_success": True,
        }
    except Exception as e:
        return {
            "content_type": "",
            "user_metadata": {},
            "head_success": False,
            "head_error": str(e),
        }


def get_object_tags(key: str) -> list:
    """Get tags for an object."""
    try:
        response = s3.get_object_tagging(Bucket=S3_BUCKET, Key=key)
        return response.get("TagSet", [])
    except Exception:
        return []


def sample_object_content(key: str, max_bytes: int = 50000) -> Optional[str]:
    """Download first N bytes of an object."""
    try:
        response = s3.get_object(
            Bucket=S3_BUCKET,
            Key=key,
            Range=f"bytes=0-{max_bytes}"
        )
        return response["Body"].read().decode("utf-8", errors="ignore")
    except Exception:
        return None


def infer_schema_from_content(content: str, key: str) -> dict:
    """Infer schema from JSON/JSONL content."""
    schema_info = {
        "format": "unknown",
        "has_messages": False,
        "has_turns": False,
        "has_conversation_id": False,
        "has_source_file": False,
        "sample_keys": [],
        "id_fields": [],
    }

    try:
        # Try JSONL first
        lines = content.strip().split("\n")
        if len(lines) > 1:
            first_obj = json.loads(lines[0])
            schema_info["format"] = "jsonl"
        else:
            first_obj = json.loads(content)
            schema_info["format"] = "json"

        # Analyze structure
        if isinstance(first_obj, dict):
            keys = list(first_obj.keys())
            schema_info["sample_keys"] = keys[:20]

            # Check for common patterns
            if "messages" in first_obj:
                schema_info["has_messages"] = True
                msgs = first_obj.get("messages", [])
                if msgs and isinstance(msgs[0], dict):
                    schema_info["message_keys"] = list(msgs[0].keys())
            if "turns" in first_obj:
                schema_info["has_turns"] = True
            if "conversation_id" in first_obj:
                schema_info["has_conversation_id"] = True
                schema_info["id_fields"].append("conversation_id")
            if "_source_file" in first_obj:
                schema_info["has_source_file"] = True
                schema_info["id_fields"].append("_source_file")
            if "source_file" in first_obj:
                schema_info["id_fields"].append("source_file")
            if "transcript_id" in first_obj:
                schema_info["id_fields"].append("transcript_id")

        if schema_info["format"] == "jsonl":
            schema_info["line_count_sample"] = len(lines)

    except json.JSONDecodeError:
        schema_info["format"] = "invalid_json"
    except Exception as e:
        schema_info["error"] = str(e)

    return schema_info


def find_pairs(objects_by_type: dict) -> list:
    """Find before/after pairs by matching source files."""
    pairs = []

    # Get extracted objects (before)
    extracted = {}
    for obj in objects_by_type.get("raw_sft", []) + objects_by_type.get("accepted_sft", []):
        filename = Path(obj["key"]).name
        extracted[filename] = obj

    # Get curated objects (after)
    curated = {}
    for obj in objects_by_type.get("clean_sft", []):
        key = obj["key"]
        # For JSONL files, we need to parse them to find source_file references
        if key.endswith(".jsonl"):
            content = sample_object_content(key, max_bytes=500000)
            if content:
                for line in content.strip().split("\n"):
                    try:
                        item = json.loads(line)
                        source_file = item.get("_source_file", "")
                        if source_file:
                            if source_file not in curated:
                                curated[source_file] = {
                                    "curated_key": key,
                                    "items": []
                                }
                            curated[source_file]["items"].append(item)
                    except json.JSONDecodeError:
                        continue

    # Match pairs
    pair_id = 0
    for filename, extracted_obj in extracted.items():
        if filename in curated:
            pair_id += 1
            pairs.append({
                "pair_id": pair_id,
                "raw_s3_uri": f"s3://{S3_BUCKET}/{extracted_obj['key']}",
                "clean_s3_uri": f"s3://{S3_BUCKET}/{curated[filename]['curated_key']}",
                "pairing_method": "filename_id",
                "notes": f"Matched by _source_file={filename}",
            })

    return pairs


def find_rejected_items(objects_by_type: dict) -> list:
    """Find rejected/dropped items."""
    rejected = []

    for obj in objects_by_type.get("rejected", []):
        rejected.append({
            "s3_uri": f"s3://{S3_BUCKET}/{obj['key']}",
            "evidence_source": "path_heuristic_rejected",
            "notes": f"Found in rejected prefix",
        })

    for obj in objects_by_type.get("dropped", []):
        rejected.append({
            "s3_uri": f"s3://{S3_BUCKET}/{obj['key']}",
            "evidence_source": "path_heuristic_dropped",
            "notes": f"Found in dropped prefix (never judged)",
        })

    return rejected


def generate_report(
    all_objects: list,
    objects_by_type: dict,
    pairs: list,
    rejected: list,
    schemas: dict,
    bucket_info: dict,
) -> str:
    """Generate the markdown report."""

    total_size = sum(o.get("size", 0) for o in all_objects)
    total_size_mb = total_size / (1024 * 1024)

    report = f"""# S3 Audit Report for Judge LLM Training Data

Generated: {datetime.now().isoformat()}

## Executive Summary

- **Bucket**: `{S3_BUCKET}`
- **Region**: `{AWS_REGION}`
- **Total Objects Scanned**: {len(all_objects):,}
- **Total Size**: {total_size_mb:,.2f} MB
- **Identified Pairs**: {len(pairs)}
- **Rejected/Dropped Items**: {len(rejected)}

## AWS Identity

```
Account: {bucket_info.get('account', 'N/A')}
User: {bucket_info.get('user', 'N/A')}
```

## Bucket Structure

### Top-Level Prefixes

| Prefix | Purpose | Object Count |
|--------|---------|--------------|
"""

    # Count objects by prefix
    prefix_counts = defaultdict(int)
    for obj in all_objects:
        key = obj.get("key", "")
        prefix = key.split("/")[0] + "/" if "/" in key else "(root)"
        prefix_counts[prefix] += 1

    for prefix, count in sorted(prefix_counts.items(), key=lambda x: -x[1]):
        purpose = "Unknown"
        if "pretraining" in prefix:
            purpose = "Pre-training pipeline data"
        elif "curated" in prefix:
            purpose = "Human-curated/fixed data"
        elif "accepted" in prefix:
            purpose = "Auto-accepted SFT data"
        elif "rejected" in prefix:
            purpose = "Rejected SFT data"
        elif "extracted" in prefix:
            purpose = "LLM-extracted calls"
        elif "training" in prefix:
            purpose = "Final training JSONL"
        elif "audio" in prefix:
            purpose = "Source audio files"
        elif "transcript" in prefix:
            purpose = "AWS Transcribe output"
        elif "review" in prefix:
            purpose = "Pending human review"
        elif "dropped" in prefix:
            purpose = "Dropped (eligibility failures)"
        report += f"| `{prefix}` | {purpose} | {count:,} |\n"

    report += f"""
### Pretraining Sub-Prefixes

| Prefix | Purpose | Est. Count |
|--------|---------|------------|
"""

    pretraining_prefixes = defaultdict(int)
    for obj in all_objects:
        key = obj.get("key", "")
        if key.startswith("pretraining/"):
            parts = key.split("/")
            if len(parts) > 1:
                sub = f"pretraining/{parts[1]}/"
                pretraining_prefixes[sub] += 1

    for prefix, count in sorted(pretraining_prefixes.items(), key=lambda x: -x[1]):
        purpose = prefix.replace("pretraining/", "").replace("/", "")
        report += f"| `{prefix}` | {purpose} | {count:,} |\n"

    report += f"""
## Object Classification

### By Guessed Type

| Type | Count | Total Size (MB) |
|------|-------|-----------------|
"""

    type_stats = defaultdict(lambda: {"count": 0, "size": 0})
    for obj in all_objects:
        t = obj.get("guessed_type", "unknown")
        type_stats[t]["count"] += 1
        type_stats[t]["size"] += obj.get("size", 0)

    for t, stats in sorted(type_stats.items(), key=lambda x: -x[1]["count"]):
        size_mb = stats["size"] / (1024 * 1024)
        report += f"| `{t}` | {stats['count']:,} | {size_mb:,.2f} |\n"

    report += f"""
### By Extension

| Extension | Count |
|-----------|-------|
"""

    ext_counts = defaultdict(int)
    for obj in all_objects:
        ext = obj.get("ext", "(none)")
        ext_counts[ext] += 1

    for ext, count in sorted(ext_counts.items(), key=lambda x: -x[1])[:15]:
        report += f"| `{ext}` | {count:,} |\n"

    report += """
## Schema Analysis

### SFT Data Formats Found

"""

    for prefix, schema in schemas.items():
        report += f"""
#### `{prefix}`

- **Format**: {schema.get('format', 'unknown')}
- **Has messages array**: {schema.get('has_messages', False)}
- **Has turns array**: {schema.get('has_turns', False)}
- **ID Fields**: {', '.join(schema.get('id_fields', [])) or 'None found'}
- **Sample Keys**: `{', '.join(schema.get('sample_keys', [])[:10])}`

"""

    report += f"""
## Before/After Pairs for Training

Found **{len(pairs)}** matched pairs where we have both:
- **Before**: Original extracted JSON (from LLM extraction)
- **After**: Curated/fixed JSON (after human review)

### Pairing Strategy

Pairs were identified by matching `_source_file` field in curated JSONL to filenames in extracted prefix.

### Sample Pairs

| Pair ID | Raw (Before) | Clean (After) | Method |
|---------|--------------|---------------|--------|
"""

    for pair in pairs[:10]:
        raw_short = pair["raw_s3_uri"].split("/")[-1][:40]
        clean_short = pair["clean_s3_uri"].split("/")[-1][:40]
        report += f"| {pair['pair_id']} | `{raw_short}...` | `{clean_short}...` | {pair['pairing_method']} |\n"

    if len(pairs) > 10:
        report += f"\n*... and {len(pairs) - 10} more pairs (see pairs.csv)*\n"

    report += f"""
## Rejected/Negative Examples

Found **{len(rejected)}** items that were rejected or dropped.

### By Evidence Source

| Source | Count |
|--------|-------|
"""

    evidence_counts = defaultdict(int)
    for r in rejected:
        evidence_counts[r["evidence_source"]] += 1

    for src, count in sorted(evidence_counts.items(), key=lambda x: -x[1]):
        report += f"| `{src}` | {count:,} |\n"

    report += """
## What to Train On

### Recommended Training Sources

#### Positive Examples (Good Quality → High Score)

1. **Curated JSONL files** - Human-verified and fixed
   - `s3://rezora-data-pipeline-864981718771/pretraining/curated/pretraining_v6_structure_validated.jsonl`
   - Contains 407 conversations with manual fixes

2. **Accepted prefix** - Auto-accepted high-quality calls
   - `s3://rezora-data-pipeline-864981718771/pretraining/accepted/`

#### Negative Examples (Bad Quality → Low Score)

1. **Rejected prefix** - Failed quality checks
   - `s3://rezora-data-pipeline-864981718771/pretraining/rejected/`

2. **Dropped prefix** - Failed eligibility (too short, no turns)
   - `s3://rezora-data-pipeline-864981718771/pretraining/dropped/`

3. **Before versions of pairs** - Pre-fix state for comparison
   - `s3://rezora-data-pipeline-864981718771/pretraining/extracted/`

#### Before/After Pairs (for learning what "good fixes" look like)

- Use `pairs.csv` manifest
- Match by `_source_file` field
- Contains structural fixes: role swaps, merged turns, text edits

### Proposed Train/Val/Test Split

Split by conversation ID hash to avoid leakage:

```python
import hashlib

def split_by_hash(conversation_id: str) -> str:
    h = int(hashlib.md5(conversation_id.encode()).hexdigest(), 16)
    bucket = h % 100
    if bucket < 80:
        return "train"
    elif bucket < 90:
        return "val"
    else:
        return "test"
```

Recommended ratios: **80/10/10** (train/val/test)

## If We Were to Consolidate This Mess Tomorrow

### Proposed Clean S3 Layout

```
s3://rezora-data-pipeline-864981718771/
└── datasets/
    └── judge_v1/
        ├── manifests/
        │   ├── pairs.jsonl          # All before/after pairs
        │   ├── rejected.jsonl       # All rejected items
        │   ├── splits.json          # train/val/test assignments
        │   └── schema.json          # Canonical schema definition
        │
        ├── raw/                     # Original extractions (before)
        │   └── {conversation_id}.json
        │
        ├── clean/                   # Fixed versions (after)
        │   └── {conversation_id}.json
        │
        ├── rejected/                # Negative examples
        │   └── {conversation_id}.json
        │
        └── training/
            ├── train.jsonl          # Final training file
            ├── val.jsonl            # Validation file
            └── test.jsonl           # Test file
```

### Canonical Schema for Training

```json
{{
  "conversation_id": "youtube_XXX_real_call_N",
  "version": "before" | "after",
  "quality_label": "good" | "bad" | "needs_fix",
  "issues": ["role_swap", "merged_turn", "truncated", ...],
  "turns": [
    {{"role": "assistant", "text": "..."}},
    {{"role": "user", "text": "..."}}
  ],
  "metadata": {{
    "source_video": "...",
    "call_direction": "outbound",
    "num_turns": 42
  }}
}}
```

### Migration Script Outline

1. Parse all curated JSONL → extract `_source_file` mappings
2. For each curated item, copy corresponding extracted file to `raw/`
3. Convert curated format to canonical schema → save to `clean/`
4. Copy rejected/dropped to `rejected/` with schema normalization
5. Generate manifests and split assignments
6. Validate schema consistency

## Appendix: Files Generated

| File | Description |
|------|-------------|
| `s3_objects.jsonl` | One line per S3 object with classification |
| `pairs.csv` | Before/after pair mappings |
| `rejected.csv` | Rejected items list |
| `dataset_summary.json` | Statistical summary |

---

*Generated by `tools/s3_audit/audit.py`*
"""

    return report


def main():
    print("=" * 60)
    print("S3 AUDIT FOR JUDGE LLM TRAINING DATA")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get AWS identity
    sts = boto3.client("sts", region_name=AWS_REGION)
    identity = sts.get_caller_identity()
    bucket_info = {
        "account": identity.get("Account"),
        "user": identity.get("Arn", "").split("/")[-1],
    }
    print(f"\nAWS Account: {bucket_info['account']}")
    print(f"User: {bucket_info['user']}")

    # List all objects
    print(f"\n[1/6] Listing all objects in s3://{S3_BUCKET}...")
    all_objects_raw = list_all_objects("")
    print(f"  Found {len(all_objects_raw):,} objects")

    # Classify objects
    print("\n[2/6] Classifying objects...")
    all_objects = []
    objects_by_type = defaultdict(list)

    for obj in all_objects_raw:
        key = obj.get("Key", "")
        size = obj.get("Size", 0)
        classification = classify_object(key, size)

        enriched = {
            "bucket": S3_BUCKET,
            "key": key,
            "s3_uri": f"s3://{S3_BUCKET}/{key}",
            "size": size,
            "last_modified": obj.get("LastModified", "").isoformat() if obj.get("LastModified") else "",
            "etag": obj.get("ETag", "").strip('"'),
            "storage_class": obj.get("StorageClass", "STANDARD"),
            **classification,
        }

        all_objects.append(enriched)
        objects_by_type[classification["guessed_type"]].append(enriched)

    print(f"  Classified into {len(objects_by_type)} types")

    # Sample schemas
    print("\n[3/6] Sampling schemas from key prefixes...")
    schemas = {}
    schema_sample_keys = [
        "pretraining/extracted/",
        "pretraining/curated/",
        "pretraining/accepted/",
        "pretraining/rejected/",
        "pretraining/training/",
    ]

    for prefix in schema_sample_keys:
        # Find first JSON/JSONL file in this prefix
        for obj in all_objects:
            if obj["key"].startswith(prefix) and obj["ext"] in [".json", ".jsonl"]:
                content = sample_object_content(obj["key"], max_bytes=100000)
                if content:
                    schemas[prefix] = infer_schema_from_content(content, obj["key"])
                    schemas[prefix]["sample_file"] = obj["key"]
                    print(f"  {prefix}: {schemas[prefix].get('format', 'unknown')} format")
                break

    # HEAD metadata for candidate files
    print("\n[4/6] Getting HEAD metadata for candidate files...")
    candidate_count = 0
    for obj in all_objects[:500]:  # Sample first 500
        if obj["guessed_type"] in ["clean_sft", "raw_sft", "accepted_sft", "rejected"]:
            metadata = get_object_metadata(obj["key"])
            obj.update(metadata)
            candidate_count += 1

    print(f"  Got metadata for {candidate_count} candidate files")

    # Find pairs
    print("\n[5/6] Finding before/after pairs...")
    pairs = find_pairs(objects_by_type)
    print(f"  Found {len(pairs)} pairs")

    # Find rejected items
    print("\n[6/6] Finding rejected/dropped items...")
    rejected = find_rejected_items(objects_by_type)
    print(f"  Found {len(rejected)} rejected/dropped items")

    # Write outputs
    print("\nWriting outputs...")

    # s3_objects.jsonl
    jsonl_path = OUTPUT_DIR / "s3_objects.jsonl"
    with open(jsonl_path, "w") as f:
        for obj in all_objects:
            f.write(json.dumps(obj, default=str) + "\n")
    print(f"  {jsonl_path}")

    # pairs.csv
    pairs_path = OUTPUT_DIR / "pairs.csv"
    with open(pairs_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "raw_s3_uri", "clean_s3_uri", "pairing_method", "notes"])
        writer.writeheader()
        writer.writerows(pairs)
    print(f"  {pairs_path}")

    # rejected.csv
    rejected_path = OUTPUT_DIR / "rejected.csv"
    with open(rejected_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["s3_uri", "evidence_source", "notes"])
        writer.writeheader()
        writer.writerows(rejected)
    print(f"  {rejected_path}")

    # dataset_summary.json
    summary = {
        "generated_at": datetime.now().isoformat(),
        "bucket": S3_BUCKET,
        "total_objects": len(all_objects),
        "total_bytes": sum(o.get("size", 0) for o in all_objects),
        "counts_by_type": {t: len(objs) for t, objs in objects_by_type.items()},
        "pair_count": len(pairs),
        "rejected_count": len(rejected),
        "schemas_found": list(schemas.keys()),
    }
    summary_path = OUTPUT_DIR / "dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  {summary_path}")

    # REPORT.md
    report = generate_report(all_objects, objects_by_type, pairs, rejected, schemas, bucket_info)
    report_path = OUTPUT_DIR / "REPORT.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  {report_path}")

    print("\n" + "=" * 60)
    print("AUDIT COMPLETE")
    print("=" * 60)
    print(f"\nDeliverables in {OUTPUT_DIR}/:")
    print(f"  - REPORT.md")
    print(f"  - s3_objects.jsonl ({len(all_objects):,} objects)")
    print(f"  - pairs.csv ({len(pairs)} pairs)")
    print(f"  - rejected.csv ({len(rejected)} items)")
    print(f"  - dataset_summary.json")


if __name__ == "__main__":
    main()
