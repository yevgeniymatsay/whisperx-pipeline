# S3 Audit Report for Judge LLM Training Data

Generated: 2026-01-27T00:15:00

## Executive Summary

| Metric | Value |
|--------|-------|
| **Bucket** | `rezora-data-pipeline-864981718771` |
| **Region** | `us-east-2` |
| **Total Objects Scanned** | 14,432 |
| **Total Size** | 19.5 GB |
| **Before/After Pairs Found** | **506** |
| **Rejected/Dropped Items** | **6,871** |
| **Curated Conversations (v6)** | 407 |

## AWS Identity

```
Account: 864981718771
User: claude-chat-app
```

---

## Part A: Repo S3 Touchpoints

| File | Bucket/Prefix | Purpose |
|------|---------------|---------|
| `config/settings.py` | `rezora-data-pipeline-864981718771` | Main bucket definition |
| `config/settings.py` | `pretraining/extracted/` | LLM-extracted calls |
| `config/settings.py` | `pretraining/accepted/` | Auto-accepted calls |
| `config/settings.py` | `pretraining/rejected/` | Failed quality checks |
| `config/settings.py` | `pretraining/dropped/` | Eligibility failures |
| `config/settings.py` | `pretraining/curated/` | Human-fixed data |
| `config/settings.py` | `pretraining/training/` | Final JSONL output |
| `scripts/judge_pretraining_calls.py` | `pretraining/extracted/` | Input for judging |
| `scripts/validate_structure.py` | `pretraining/curated/` | Structure validation source |
| `scripts/cleanlab_regression_audit.py` | `pretraining/quality-review/` | Quality audit output |

---

## Part B: Bucket Structure

### Top-Level Prefixes

| Prefix | Purpose | Object Count | Size (Est.) |
|--------|---------|--------------|-------------|
| `pretraining/` | Pre-training pipeline data | 13,787 | ~19 GB |
| `audio/` | Source audio (expired listing) | 55 | ~480 MB |
| `transcripts/` | AWS Transcribe output | 47 | ~15 MB |
| `extracted/` | Extracted calls (expired listing) | 155 | ~3 MB |
| `accepted/` | Accepted (expired listing) | 146 | ~3 MB |
| `rejected/` | Rejected (expired listing) | 24 | ~0.5 MB |
| `curated/` | Top-level curated (before/after pairs JSON) | 1 | ~164 KB |
| `training/` | Training JSONL (expired listing) | 8 | ~2 MB |

### Pretraining Sub-Prefixes (Primary Data Source)

| Prefix | Purpose | Count |
|--------|---------|-------|
| `pretraining/extracted/` | Raw LLM extractions (BEFORE) | 3,228 |
| `pretraining/dropped/` | Failed eligibility | 2,043 |
| `pretraining/rejected/` | Failed quality | 2,051 |
| `pretraining/accepted/` | Auto-accepted | 537 |
| `pretraining/audio/` | Source MP3 files | 552 |
| `pretraining/review/` | Pending human review | 441 |
| `pretraining/transcripts/` | AWS Transcribe JSON | 386 |
| `pretraining/curated/` | Human-fixed JSONL (AFTER) | 5 |
| `pretraining/training/` | Final training files | 182 |
| `pretraining/quality-review/` | Quality audit data | 11 |
| `pretraining/structure-review/` | Structure audit data | 116 |
| `pretraining/profanity-review/` | Profanity audit data | 21 |
| `pretraining/similarity-review/` | Dedup audit data | 10 |
| `pretraining/dedupe-review/` | Dedup review data | 10 |

---

## Part C: Object Classification

### By Guessed Type

| Type | Count | Size (MB) | Description |
|------|-------|-----------|-------------|
| `raw_sft` | 7,462 | ~150 | Extracted JSON (before fixes) |
| `rejected` | 2,106 | ~40 | Failed quality checks |
| `dropped` | 2,074 | ~25 | Eligibility failures |
| `accepted_sft` | 848 | ~20 | Auto-accepted calls |
| `review_candidate` | 703 | ~15 | Pending human review |
| `audio` | 607 | ~18,500 | Source MP3 files |
| `raw_transcript` | 433 | ~100 | AWS Transcribe output |
| `training_data` | 190 | ~40 | Final JSONL files |
| `clean_sft` | 6 | ~8 | Curated JSONL files |
| `unknown` | 3 | <1 | Unclassified |

### By Extension

| Extension | Count |
|-----------|-------|
| `.json` | 13,229 |
| `.mp3` | 607 |
| `.jsonl` | 193 |
| `.csv` | 8 |
| (other) | 5 |

---

## Part D: Schema Analysis

### Extracted JSON Schema (`pretraining/extracted/`)

```json
{
  "source_file": "pretraining/transcripts/pretraining-VIDEO_ID-TIMESTAMP.json",
  "transcript_id": "youtube_VIDEO_ID",
  "source_dataset": "youtube",
  "segment_id": 1,
  "content_type": "real_call|commentary|roleplay",
  "call_direction": "outbound|inbound|null",
  "call_type": "conversation|voicemail|null",
  "turns": [
    {"role": "assistant|user", "text": "..."}
  ],
  "flags": ["none", ...],
  "segmentation_confidence": 0.85,
  "judge": {
    "scores": {...},
    "turn_analysis": {...}
  }
}
```

### Curated JSONL Schema (`pretraining/curated/`)

```json
{
  "messages": [
    {"role": "system", "content": "[call_direction:outbound] [version:v1]"},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "scalar_score": 0.8875,
  "_source_file": "youtube_VIDEO_ID_real_call_N.json",
  "_call_direction": "outbound",
  "_source_dataset": "youtube"
}
```

### Key ID Fields for Pairing

| Field | Location | Example |
|-------|----------|---------|
| `_source_file` | curated JSONL | `youtube_oO60fSjdw1A_real_call_25.json` |
| Filename | extracted JSON | Same as `_source_file` |
| `transcript_id` | extracted JSON | `youtube_oO60fSjdw1A` |

---

## Part E: Before/After Pairs

### Summary

| Curated Version | Pairs Found | Notes |
|-----------------|-------------|-------|
| `pretraining_v6_structure_validated` | **407** | Latest, with manual structural fixes |
| `pretraining_v5_profanity_cleaned` | 9 | Profanity cleanup |
| `pretraining_v3_final` | 4 | Earlier final version |
| `pretraining_v2_curated` | 86 | First curated batch |
| **Total Unique Pairs** | **506** | |

### Pairing Strategy Used

1. Parse all 5 curated JSONL files
2. Extract `_source_file` field from each conversation
3. Match to files in `pretraining/extracted/` (3,228 files)
4. Deduplicate across versions (track first appearance)

### Sample Pairs (from v6 - highest quality fixes)

| Source File | Before (Extracted) | After (Curated) | Fix Type |
|-------------|-------------------|-----------------|----------|
| `youtube_oO60fSjdw1A_real_call_25.json` | 75 turns, roles inverted | 74 turns, roles fixed | 61 role swaps |
| `youtube_Uytq1t3zAz8_real_call_1.json` | 48 turns, roles inverted | 51 turns, splits added | 47 role swaps |
| `youtube_mxCrbMSfup4_real_call_1.json` | 51 turns, partial swap | 50 turns, roles fixed | 41 role swaps |
| `youtube_lS4QGaLxKbU_real_call_23.json` | 97 turns | 93 turns | 36 role swaps + 4 merges |
| `youtube_LKUqmMaJECY_real_call_4.json` | 50 turns | 47 turns | 34 role swaps + 3 merges |

### Common Fixes Applied (from detailed analysis)

| Fix Type | Count | % of Dataset | Description |
|----------|-------|--------------|-------------|
| Truncation | 107 | 26% | Opening/closing removed |
| Role Swap | 76 | 19% | Assistant↔User roles inverted |
| Merges/Splits | 81 | 20% | Turn boundary corrections |
| Text Edit | 15 | 4% | Content modifications |

---

## Part F: Rejected/Negative Examples

### Summary

| Source | Count | Notes |
|--------|-------|-------|
| `pretraining/rejected/` | 2,051 | Failed quality checks |
| `pretraining/dropped/` | 2,043 | Eligibility failures (too short, no turns) |
| In extracted, not curated | 2,691 | Never made it through pipeline |
| Other rejected paths | 86 | Expired listing pipeline |
| **Total** | **6,871** | |

### Categories of Rejection

| Evidence Source | Count | Meaning |
|-----------------|-------|---------|
| `path_heuristic_rejected` | 2,051 | File in `rejected/` prefix |
| `path_heuristic_dropped` | 2,043 | File in `dropped/` prefix |
| `in_extracted_not_curated` | 2,691 | Extracted but never curated |

---

## What to Train On

### Recommended Training Sources

#### ✅ Positive Examples (Good Quality → High Score)

| Source | S3 URI | Count | Use For |
|--------|--------|-------|---------|
| **Curated v6** | `s3://.../pretraining/curated/pretraining_v6_structure_validated.jsonl` | 407 | Gold standard (after fixes) |
| **Accepted** | `s3://.../pretraining/accepted/` | 537 | Auto-accepted high quality |

#### ❌ Negative Examples (Bad Quality → Low Score)

| Source | S3 URI | Count | Use For |
|--------|--------|-------|---------|
| **Rejected** | `s3://.../pretraining/rejected/` | 2,051 | Quality failures |
| **Dropped** | `s3://.../pretraining/dropped/` | 2,043 | Eligibility failures |
| **Extracted not curated** | See `rejected.csv` | 2,691 | Never passed review |

#### 🔄 Before/After Pairs (for preference learning)

| Type | S3 URI | Count |
|------|--------|-------|
| **Before (raw)** | `s3://.../pretraining/extracted/{source_file}` | 506 |
| **After (fixed)** | `s3://.../pretraining/curated/pretraining_v6_*.jsonl` | 407 (v6) |

Use `pairs.csv` to match before→after by `source_file` field.

### Proposed Train/Val/Test Split

```python
import hashlib

def get_split(conversation_id: str) -> str:
    """Deterministic split by conversation ID hash."""
    h = int(hashlib.md5(conversation_id.encode()).hexdigest(), 16)
    bucket = h % 100
    if bucket < 80:
        return "train"
    elif bucket < 90:
        return "val"
    else:
        return "test"

# Example:
# get_split("youtube_oO60fSjdw1A_real_call_25") → "train"
```

**Recommended split: 80/10/10 (train/val/test)**

### TRL Training Format Suggestion

For reward model / DPO training:

```json
{
  "conversation_id": "youtube_oO60fSjdw1A_real_call_25",
  "prompt": "<conversation for the judge to evaluate>",
  "chosen": "<high quality version (after fix)>",
  "rejected": "<low quality version (before fix)>",
  "labels": {
    "quality_score": 0.85,
    "issues_detected": ["role_swap", "merged_turn"],
    "issues_fixed": true
  }
}
```

---

## If We Were to Consolidate This Mess Tomorrow

### Proposed Clean S3 Layout

```
s3://rezora-data-pipeline-864981718771/
└── datasets/
    └── judge_v1/
        ├── manifests/
        │   ├── pairs.jsonl           # All 506 before/after pairs
        │   ├── positives.jsonl       # Good quality items (944)
        │   ├── negatives.jsonl       # Bad quality items (6,871)
        │   ├── splits.json           # train/val/test assignments
        │   └── schema.json           # Canonical schema
        │
        ├── raw/                      # Original extractions (before)
        │   └── {conversation_id}.json
        │
        ├── clean/                    # Fixed versions (after)
        │   └── {conversation_id}.json
        │
        ├── rejected/                 # Negative examples
        │   └── {conversation_id}.json
        │
        └── training/
            ├── train.jsonl
            ├── val.jsonl
            └── test.jsonl
```

### Canonical Schema

```json
{
  "conversation_id": "youtube_oO60fSjdw1A_real_call_25",
  "version": "before|after",
  "quality_label": "good|bad|needs_fix",
  "quality_score": 0.85,
  "issues": ["role_swap", "merged_turn", "truncated"],
  "issues_fixed": true,
  "turns": [
    {"role": "assistant", "text": "..."},
    {"role": "user", "text": "..."}
  ],
  "metadata": {
    "source_video": "youtube_oO60fSjdw1A",
    "call_direction": "outbound",
    "call_type": "conversation",
    "num_turns": 74,
    "curated_version": "v6"
  }
}
```

### Migration Checklist

- [ ] Parse `pretraining_v6_structure_validated.jsonl` → extract all 407 conversations
- [ ] For each, copy matching file from `pretraining/extracted/` → `datasets/judge_v1/raw/`
- [ ] Convert curated format to canonical schema → `datasets/judge_v1/clean/`
- [ ] Copy `pretraining/rejected/` + `pretraining/dropped/` → `datasets/judge_v1/rejected/`
- [ ] Generate `manifests/pairs.jsonl` with before→after mappings
- [ ] Generate `manifests/positives.jsonl` (curated + accepted)
- [ ] Generate `manifests/negatives.jsonl` (rejected + dropped + extracted-not-curated)
- [ ] Apply hash-based split → `manifests/splits.json`
- [ ] Build `training/{train,val,test}.jsonl` files
- [ ] Validate schema consistency across all files

---

## Appendix: Generated Files

| File | Description | Rows |
|------|-------------|------|
| `s3_objects.jsonl` | All S3 objects with classification | 14,432 |
| `pairs.csv` | Before/after pair mappings | 506 |
| `rejected.csv` | Rejected/dropped items | 6,871 |
| `dataset_summary.json` | Statistical summary | - |

### Quick Access Commands

```bash
# View pairs
head -20 artifacts/s3_audit/pairs.csv

# Count by curated version
cut -d',' -f6 artifacts/s3_audit/pairs.csv | sort | uniq -c

# Count rejected by source
cut -d',' -f2 artifacts/s3_audit/rejected.csv | sort | uniq -c

# Download sample before/after pair
aws s3 cp s3://rezora-data-pipeline-864981718771/pretraining/extracted/youtube_oO60fSjdw1A_real_call_25.json ./before.json
aws s3 cp s3://rezora-data-pipeline-864981718771/pretraining/curated/pretraining_v6_structure_validated.jsonl ./curated.jsonl
grep "youtube_oO60fSjdw1A_real_call_25" curated.jsonl > ./after.json
```

### Files Already Generated (from earlier analysis)

| File | Location | Description |
|------|----------|-------------|
| `top_5_before_after_pairs.json` | `s3://.../curated/` | Detailed diff of 5 most-changed conversations |
| `diverse_changes_analysis.json` | Local | Analysis of change types |

---

*Generated by `tools/s3_audit/audit.py` + `tools/s3_audit/enhance_pairs.py`*
