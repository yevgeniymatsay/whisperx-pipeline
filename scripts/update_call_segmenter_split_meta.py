#!/usr/bin/env python3
"""Update a call-segmenter split meta JSON by appending new video_ids to TRAIN.

This keeps the EVAL holdout fixed so experiments stay comparable.

Usage:
  python scripts/update_call_segmenter_split_meta.py \\
      --in-split-meta data/call_segmenter/split_meta_v1_plus3_plus10nocall.json \\
      --new-video-ids pgYGE9jmNPA mxCrbMSfup4 \\
      --out-split-meta data/call_segmenter/split_meta_v1_plus3_plus10nocall_plus2.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List


def load_video_ids_from_file(path: Path) -> List[str]:
    ids: List[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line)
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Append new video_ids to train split (keep eval fixed)")
    parser.add_argument("--in-split-meta", type=Path, required=True, help="Existing split meta JSON")
    parser.add_argument("--out-split-meta", type=Path, required=True, help="Output split meta JSON")
    parser.add_argument("--new-video-ids", type=str, nargs="*", default=None,
                        help="New labeled video_ids to append to train")
    parser.add_argument("--new-video-list", type=Path, default=None,
                        help="File with new labeled video_ids (one per line)")
    args = parser.parse_args()

    if not args.in_split_meta.exists():
        raise FileNotFoundError(args.in_split_meta)

    new_ids: List[str] = []
    if args.new_video_list is not None:
        if not args.new_video_list.exists():
            raise FileNotFoundError(args.new_video_list)
        new_ids.extend(load_video_ids_from_file(args.new_video_list))
    if args.new_video_ids:
        new_ids.extend([str(v).strip() for v in args.new_video_ids if str(v).strip()])

    # De-dup while preserving order.
    seen = set()
    new_ids = [v for v in new_ids if not (v in seen or seen.add(v))]

    meta = json.loads(args.in_split_meta.read_text())
    train = [str(v) for v in meta.get("train_video_ids", [])]
    eval_ = [str(v) for v in meta.get("eval_video_ids", [])]

    if not train or not eval_:
        raise ValueError("Input split meta must contain train_video_ids and eval_video_ids")

    eval_set = set(eval_)
    train_set = set(train)

    added: List[str] = []
    skipped_in_eval: List[str] = []
    skipped_existing: List[str] = []

    for vid in new_ids:
        if vid in eval_set:
            skipped_in_eval.append(vid)
            continue
        if vid in train_set:
            skipped_existing.append(vid)
            continue
        train.append(vid)
        train_set.add(vid)
        added.append(vid)

    out = dict(meta)
    out["train_video_ids"] = train
    out["eval_video_ids"] = eval_
    note = out.get("notes", "")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    extra = f"{stamp}: appended {len(added)} train video_ids"
    out["notes"] = (note + " | " + extra) if note else extra

    args.out_split_meta.parent.mkdir(parents=True, exist_ok=True)
    args.out_split_meta.write_text(json.dumps(out, indent=2) + "\n")

    print(f"Wrote: {args.out_split_meta}")
    print(f"Train videos: {len(train)} (added {len(added)})")
    print(f"Eval videos (unchanged): {len(eval_)}")
    if added:
        print(f"Added: {added}")
    if skipped_existing:
        print(f"Skipped (already in train): {skipped_existing}")
    if skipped_in_eval:
        print(f"Skipped (in eval): {skipped_in_eval}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

