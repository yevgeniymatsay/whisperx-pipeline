# S3 Audit Tool

Audits the S3 bucket to produce a structured inventory of training data for the Judge LLM.

## Usage

```bash
# From project root
python tools/s3_audit/audit.py
```

## Outputs

All outputs are written to `artifacts/s3_audit/`:

| File | Description |
|------|-------------|
| `REPORT.md` | Human-readable audit report |
| `s3_objects.jsonl` | One JSON per S3 object with metadata |
| `pairs.csv` | Before/after pair mappings |
| `rejected.csv` | Rejected items list |
| `dataset_summary.json` | Statistical summary |

## Requirements

- AWS credentials configured (`aws configure`)
- boto3 installed (`pip install boto3`)
- Read access to `rezora-data-pipeline-864981718771` bucket

## What It Does

1. Lists all objects in the bucket
2. Classifies by path/extension heuristics
3. Samples schemas from key prefixes
4. Finds before/after pairs by matching `_source_file`
5. Identifies rejected/dropped items
6. Generates manifests and report

## Safe Operations

This tool is **READ-ONLY**:
- Uses `list-objects-v2`, `head-object`, `get-object` (range reads)
- Does NOT modify, tag, or delete any objects
- Does NOT dump full content (only schema sampling)
