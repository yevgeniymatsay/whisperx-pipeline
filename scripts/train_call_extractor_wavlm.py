#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from transformers import (
    AutoFeatureExtractor,
    TrainingArguments,
    Trainer,
)

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import AWS_REGION, S3_BUCKET
from pipeline.call_extractor_wavlm.audio_cache import ensure_flac_cached
from pipeline.call_extractor_wavlm.io import (
    build_audio_index,
    ffprobe_duration_s,
    git_short_sha,
    s3_download_if_missing,
    s3_read_json,
    s3_upload_directory,
    utc_now_compact,
)
from pipeline.call_extractor_wavlm.labels import TargetConfig, parse_video_labels
from pipeline.call_extractor_wavlm.model import (
    WavLMFrameClassifier,
    WavLMFrameClassifierConfig,
)
from pipeline.call_extractor_wavlm.training import Collator, ChunkDataset, Example, make_examples_for_video, set_seed
from pipeline.call_extractor_wavlm.types import CallBoundary, ChunkingConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune WavLM-Large for audio-only call extraction (frame heads).")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/call_extractor/train_wavlm_large_v1.config.json"),
        help="Training config (.config.json)",
    )
    args = parser.parse_args()

    cfg = _load_json(args.config)
    split_cfg = _load_json(Path(cfg["split_config_path"]))
    out_cfg = _load_json(Path(cfg["output_config_path"]))

    sr_hz = int(cfg.get("sr_hz", 16000))
    seed = int(cfg.get("seed", 1337))
    set_seed(seed)

    repo_root = Path(__file__).parent.parent
    run_id = f"run_{utc_now_compact()}_{git_short_sha(repo_root)}"
    local_artifacts_dir = Path(out_cfg.get("local_artifacts_dir", "artifacts/call_extractor/wavlm_large_v1")) / run_id
    cache_dir = Path(out_cfg.get("local_cache_dir", ".cache/call_extractor_wavlm"))
    audio_cache_dir = cache_dir / "audio"
    flac_cache_dir = cache_dir / "audio_flac"
    audio_cache_dir.mkdir(parents=True, exist_ok=True)
    flac_cache_dir.mkdir(parents=True, exist_ok=True)
    local_artifacts_dir.mkdir(parents=True, exist_ok=True)

    label_prefix = str(split_cfg["label_prefix"])
    audio_prefixes = list(split_cfg["audio_prefixes"])

    audio_index = build_audio_index(S3_BUCKET, audio_prefixes, region=AWS_REGION)

    train_video_ids = list(split_cfg["train_video_ids"])
    eval_video_ids = list(split_cfg["eval_video_ids"])

    def _load_video(video_id: str) -> Optional[tuple[Path, list[CallBoundary], float]]:
        audio_key = audio_index.get(video_id)
        if not audio_key:
            logger.warning(f"Skip {video_id}: audio not found in prefixes")
            return None
        label_key = f"{label_prefix}{video_id}.json"
        label_data = s3_read_json(S3_BUCKET, label_key, region=AWS_REGION)
        labels = parse_video_labels(label_data)

        mp3_path = audio_cache_dir / f"{video_id}.mp3"
        s3_download_if_missing(S3_BUCKET, audio_key, mp3_path, region=AWS_REGION)

        flac_path = flac_cache_dir / f"{video_id}.flac"
        ensure_flac_cached(mp3_path=mp3_path, flac_path=flac_path, sr_hz=int(sr_hz))

        duration_s = ffprobe_duration_s(flac_path)
        return flac_path, labels.boundaries, float(duration_s)

    chunk_cfg = ChunkingConfig(
        chunk_total_s=float(cfg.get("chunk_total_s", 30.0)),
        core_s=float(cfg.get("core_s", 20.0)),
        margin_s=float(cfg.get("margin_s", 5.0)),
    )

    boundary_k = int(cfg.get("boundary_k_per_boundary", 3))
    boundary_jitter_s = float(cfg.get("boundary_jitter_s", 1.0))
    in_call_samples = int(cfg.get("in_call_samples_per_video", 4))
    out_call_samples = int(cfg.get("out_call_samples_per_video", 4))
    no_call_out_call_samples = int(cfg.get("no_call_out_call_samples_per_video", out_call_samples))

    # Keep eval sampling small: Trainer eval is for sanity checks / "best checkpoint by eval_loss",
    # not full-gate selection (which is done by predict+sweep on the fixed eval videos).
    eval_boundary_k = int(cfg.get("eval_boundary_k_per_boundary", min(2, boundary_k)))
    eval_boundary_jitter_s = float(cfg.get("eval_boundary_jitter_s", boundary_jitter_s))
    eval_in_call_samples = int(cfg.get("eval_in_call_samples_per_video", max(1, in_call_samples // 2)))
    eval_out_call_samples = int(cfg.get("eval_out_call_samples_per_video", max(1, out_call_samples // 2)))

    train_examples: list[Example] = []
    for vid in train_video_ids:
        loaded = _load_video(str(vid))
        if not loaded:
            continue
        audio_path, boundaries, duration_s = loaded
        is_no_call = len(boundaries) == 0
        train_examples.extend(
            make_examples_for_video(
                video_id=str(vid),
                audio_path=audio_path,
                boundaries=boundaries,
                duration_s=float(duration_s),
                chunk_cfg=chunk_cfg,
                seed=seed,
                boundary_k=boundary_k,
                boundary_jitter_s=boundary_jitter_s,
                in_call_samples=0 if is_no_call else in_call_samples,
                out_call_samples=no_call_out_call_samples if is_no_call else out_call_samples,
            )
        )
    logger.info(f"Train examples: {len(train_examples)} from {len(train_video_ids)} videos")

    eval_examples: list[Example] = []
    for vid in eval_video_ids:
        loaded = _load_video(str(vid))
        if not loaded:
            continue
        audio_path, boundaries, duration_s = loaded
        is_no_call = len(boundaries) == 0
        eval_examples.extend(
            make_examples_for_video(
                video_id=str(vid),
                audio_path=audio_path,
                boundaries=boundaries,
                duration_s=float(duration_s),
                chunk_cfg=chunk_cfg,
                seed=seed + 1,
                boundary_k=eval_boundary_k,
                boundary_jitter_s=eval_boundary_jitter_s,
                in_call_samples=0 if is_no_call else eval_in_call_samples,
                out_call_samples=no_call_out_call_samples if is_no_call else eval_out_call_samples,
            )
        )
    logger.info(f"Eval examples: {len(eval_examples)} from {len(eval_video_ids)} videos")

    feature_extractor = AutoFeatureExtractor.from_pretrained(str(cfg.get("base_model_name", "microsoft/wavlm-large")))

    model_cfg = WavLMFrameClassifierConfig(
        base_model_name=str(cfg.get("base_model_name", "microsoft/wavlm-large")),
        head_dropout=float(cfg.get("head_dropout", 0.10)),
        freeze_feature_encoder=bool(cfg.get("freeze_feature_encoder", True)),
        pos_weight_in_call=float(cfg.get("pos_weight_in_call", 1.0)),
        pos_weight_start=float(cfg.get("pos_weight_start", 10.0)),
        pos_weight_end=float(cfg.get("pos_weight_end", 10.0)),
    )
    model = WavLMFrameClassifier(model_cfg)

    target_cfg = TargetConfig(
        start_tolerance_s=float(cfg.get("start_tolerance_s", 0.20)),
        end_tolerance_s=float(cfg.get("end_tolerance_s", 0.20)),
        boundary_target_shape=str(cfg.get("boundary_target_shape", "binary")),
    )
    collator = Collator(
        feature_extractor=feature_extractor,
        sr_hz=sr_hz,
        target_cfg=target_cfg,
        model_config=model.wavlm.config,
    )

    train_ds = ChunkDataset(train_examples, sr_hz=sr_hz)
    eval_ds = ChunkDataset(eval_examples, sr_hz=sr_hz)

    training_args = TrainingArguments(
        output_dir=str(local_artifacts_dir / "trainer"),
        num_train_epochs=float(cfg.get("num_train_epochs", 3)),
        max_steps=int(cfg.get("max_steps", -1)),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 2)),
        per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", 2)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(cfg.get("learning_rate", 2e-5)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.05)),
        logging_steps=int(cfg.get("logging_steps", 25)),
        eval_strategy=str(cfg.get("eval_strategy", "epoch")),
        save_strategy=str(cfg.get("save_strategy", "epoch")),
        save_total_limit=int(cfg.get("save_total_limit", 3)),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=bool(cfg.get("fp16", True)),
        dataloader_num_workers=int(cfg.get("dataloader_num_workers", 2)),
        remove_unused_columns=False,
        report_to=[],
        seed=int(seed),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=feature_extractor,
    )

    trainer.train()

    best_dir = local_artifacts_dir / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    model.wavlm.save_pretrained(best_dir)
    feature_extractor.save_pretrained(best_dir)
    torch.save(model.classifier.state_dict(), best_dir / "frame_heads.pt")

    s3_prefix = str(out_cfg.get("s3_output_prefix", "call_extractor/wavlm_large_v1/")).rstrip("/")
    model_prefix = f"{s3_prefix}/models/{run_id}"
    logger.info(f"Upload model artifacts to s3://{S3_BUCKET}/{model_prefix}/")
    s3_upload_directory(bucket=S3_BUCKET, prefix=model_prefix, src_dir=best_dir, region=AWS_REGION)

    (local_artifacts_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "git_sha": git_short_sha(repo_root),
                "train_config": cfg,
                "split_config": split_cfg,
                "output_config": out_cfg,
                "s3_model_prefix": model_prefix,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    logger.info(f"Done: {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
