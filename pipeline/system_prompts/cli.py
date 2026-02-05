#!/usr/bin/env python3
"""CLI for generating synthetic system prompts and exporting training files.

Usage:
  # Generate prompt objects from GenRM accepted calls
  python -m pipeline.system_prompts.cli generate \
    --input-prefix genrm/accepted/ \
    --output-prefix synthetic_system_prompts/v1/ \
    --limit 50

  # Export SFT JSONL
  python -m pipeline.system_prompts.cli export-jsonl \
    --input-prefix synthetic_system_prompts/v1/ \
    --output-path out/pretraining.jsonl

  # Export DPO pairs (provider-agnostic)
  python -m pipeline.system_prompts.cli export-dpo \
    --input-prefix synthetic_system_prompts/v1/ \
    --output-path out/prompt_dpo_pairs.jsonl
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import S3_BUCKET
from .export import export_dpo_jsonl, export_sft_jsonl
from .generator import generate_prompt_object
from .s3_io import list_keys, load_json, object_exists, write_json


def _call_id_from_key(key: str) -> str:
    base = key.split("/")[-1]
    return base[:-5] if base.endswith(".json") else base


def cmd_generate(args: argparse.Namespace) -> int:
    bucket = args.bucket
    input_prefix = args.input_prefix
    output_prefix = args.output_prefix
    limit = args.limit
    dry_run = args.dry_run
    skip_existing = args.skip_existing
    max_workers = args.max_workers
    topic_clarity_threshold = args.topic_clarity_threshold

    keys = list_keys(bucket, input_prefix, suffix=".json")
    keys.sort()
    if limit:
        keys = keys[:limit]

    print(f"Found {len(keys)} source objects under s3://{bucket}/{input_prefix}")

    stats = {"ok": 0, "skipped": 0, "error": 0, "no_turns": 0}

    def worker(src_key: str) -> Tuple[str, str]:
        call_id = _call_id_from_key(src_key)
        out_key = f"{output_prefix}{call_id}.json"

        if skip_existing and object_exists(bucket, out_key):
            return ("skipped", src_key)

        src = load_json(bucket, src_key)
        turns = src.get("turns") or []
        if not turns:
            return ("no_turns", src_key)

        result = generate_prompt_object(
            turns=turns,
            source_bucket=bucket,
            source_key=src_key,
            call_id=call_id,
            topic_clarity_threshold=topic_clarity_threshold,
        )
        write_json(bucket, out_key, result.data, dry_run=dry_run)
        return ("ok", src_key)

    if max_workers <= 1:
        for k in keys:
            try:
                status, _ = worker(k)
                stats[status] += 1
            except Exception as e:  # noqa: BLE001
                stats["error"] += 1
                print(f"ERROR: {k}: {e}")
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(worker, k): k for k in keys}
            for fut in as_completed(futs):
                k = futs[fut]
                try:
                    status, _ = fut.result()
                    stats[status] += 1
                except Exception as e:  # noqa: BLE001
                    stats["error"] += 1
                    print(f"ERROR: {k}: {e}")

    print("Done.")
    print(f"  ok: {stats['ok']}")
    print(f"  skipped: {stats['skipped']}")
    print(f"  no_turns: {stats['no_turns']}")
    print(f"  error: {stats['error']}")
    return 0 if stats["error"] == 0 else 2


def _load_prompt_objects(bucket: str, prefix: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    keys = list_keys(bucket, prefix, suffix=".json")
    keys.sort()
    if limit:
        keys = keys[:limit]
    objs: List[Dict[str, Any]] = []
    for k in keys:
        objs.append(load_json(bucket, k))
    return objs


def cmd_export_jsonl(args: argparse.Namespace) -> int:
    objs = _load_prompt_objects(args.bucket, args.input_prefix, args.limit)
    if not objs:
        print("No objects found.")
        return 1

    meta = export_sft_jsonl(
        objs,
        output_path=Path(args.output_path),
        topic_clarity_threshold=args.topic_clarity_threshold,
        detailed_cap=args.detailed_cap,
        write_manifest=not args.no_manifest,
    )
    print(f"Wrote {meta['written']} examples to {args.output_path}")
    return 0


def cmd_export_dpo(args: argparse.Namespace) -> int:
    objs = _load_prompt_objects(args.bucket, args.input_prefix, args.limit)
    if not objs:
        print("No objects found.")
        return 1
    meta = export_dpo_jsonl(objs, output_path=Path(args.output_path))
    print(f"Wrote {meta['written']} preference pairs to {args.output_path}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Synthetic system prompt generation + QA export tools",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate prompt objects from GenRM accepted calls")
    gen.add_argument("--bucket", default=S3_BUCKET)
    gen.add_argument("--input-prefix", default="genrm/accepted/")
    gen.add_argument("--output-prefix", default="synthetic_system_prompts/v1/")
    gen.add_argument("--limit", type=int)
    gen.add_argument("--dry-run", action="store_true")
    gen.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip calls that already have an output object (default: True)",
    )
    gen.add_argument("--max-workers", type=int, default=1)
    gen.add_argument("--topic-clarity-threshold", type=float, default=0.5)
    gen.set_defaults(func=cmd_generate)

    exp = sub.add_parser("export-jsonl", help="Export SFT-ready JSONL")
    exp.add_argument("--bucket", default=S3_BUCKET)
    exp.add_argument("--input-prefix", default="synthetic_system_prompts/v1/")
    exp.add_argument("--output-path", required=True)
    exp.add_argument("--limit", type=int)
    exp.add_argument("--topic-clarity-threshold", type=float, default=0.5)
    exp.add_argument("--detailed-cap", type=float, default=0.15)
    exp.add_argument("--no-manifest", action="store_true")
    exp.set_defaults(func=cmd_export_jsonl)

    dpo = sub.add_parser("export-dpo", help="Export preference pairs as JSONL")
    dpo.add_argument("--bucket", default=S3_BUCKET)
    dpo.add_argument("--input-prefix", default="synthetic_system_prompts/v1/")
    dpo.add_argument("--output-path", required=True)
    dpo.add_argument("--limit", type=int)
    dpo.set_defaults(func=cmd_export_dpo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
