#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import transformers
from transformers import AutoFeatureExtractor, Trainer, TrainingArguments

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import AWS_REGION, S3_BUCKET
from pipeline.call_extractor_wavlm.audio_cache import ensure_flac_cached
from pipeline.call_extractor_wavlm.io import (
    build_audio_index,
    ffprobe_duration_s,
    s3_download_if_missing,
    s3_read_json,
)
from pipeline.call_extractor_wavlm.labels import TargetConfig, parse_video_labels
from pipeline.call_extractor_wavlm.model import WavLMFrameClassifier, WavLMFrameClassifierConfig
from pipeline.call_extractor_wavlm.training import ChunkDataset, Collator, make_examples_for_video, set_seed
from pipeline.call_extractor_wavlm.types import CallBoundary, ChunkingConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _nonzero_frac(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return 0.0
    return float((x != 0).float().mean().item())


@torch.inference_mode()
def _forward_report(*, model: torch.nn.Module, batch: dict, device: str) -> dict:
    input_values = batch["input_values"].to(device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    labels = batch["labels"].to(device)
    label_mask = batch["label_mask"].to(device)

    out = model(input_values=input_values, attention_mask=attention_mask, labels=labels, label_mask=label_mask)
    logits = out["logits"].detach().float().cpu()
    loss = float(out["loss"].detach().float().cpu().item())
    probs = torch.sigmoid(logits)

    mask = batch["label_mask"].bool()
    y = batch["labels"]
    p = probs
    report: dict = {
        "logits_shape": list(logits.shape),
        "labels_shape": list(y.shape),
        "label_mask_shape": list(batch["label_mask"].shape),
        "label_mask_nonzero_frac": float(mask.float().mean().item()) if mask.numel() else 0.0,
        "loss": float(loss),
    }
    for head_i, name in enumerate(["in_call", "start", "end"]):
        y_head = y[..., head_i]
        p_head = p[..., head_i]
        y_core = y_head[mask]
        p_core = p_head[mask]
        pos_frac = float((y_core >= 0.5).float().mean().item()) if y_core.numel() else 0.0
        report[f"{name}_pos_frac_core"] = pos_frac
        report[f"{name}_pos_sum_core"] = float(y_core.sum().item()) if y_core.numel() else 0.0
        if y_core.numel():
            pos = y_core >= 0.5
            neg = ~pos
            report[f"{name}_p_mean_pos_core"] = float(p_core[pos].mean().item()) if pos.any() else None
            report[f"{name}_p_mean_neg_core"] = float(p_core[neg].mean().item()) if neg.any() else None
            q = torch.quantile(p_core, torch.tensor([0.5, 0.9, 0.99]))
            report[f"{name}_p_quantiles_core"] = [float(q[0].item()), float(q[1].item()), float(q[2].item())]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="HF-docs-first preflight for WavLM call extractor training.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/call_extractor/train_wavlm_large_v4.config.json"),
        help="Training config (.config.json) to mirror for preflight.",
    )
    parser.add_argument("--video-id", type=str, default=None, help="Optional labeled video_id to use for preflight.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sr-hz", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--overfit-steps", type=int, default=50, help="0 disables the overfit test.")
    parser.add_argument("--overfit-lr", type=float, default=1e-4)
    args = parser.parse_args()

    cfg = _load_json(args.config)
    split_cfg = _load_json(Path(cfg["split_config_path"]))

    set_seed(int(args.seed))

    logger.info(f"torch={torch.__version__} transformers={transformers.__version__}")
    logger.info(f"device={args.device} cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"gpu={torch.cuda.get_device_name(0)}")

    cache_dir = Path(cfg.get("local_cache_dir", ".cache/call_extractor_wavlm"))
    audio_cache_dir = cache_dir / "audio"
    flac_cache_dir = cache_dir / "audio_flac"
    audio_cache_dir.mkdir(parents=True, exist_ok=True)
    flac_cache_dir.mkdir(parents=True, exist_ok=True)

    label_prefix = str(split_cfg["label_prefix"])
    audio_prefixes = list(split_cfg["audio_prefixes"])
    audio_index = build_audio_index(S3_BUCKET, audio_prefixes, region=AWS_REGION)

    eval_video_ids = [str(v) for v in split_cfg["eval_video_ids"]]
    train_video_ids = [str(v) for v in split_cfg["train_video_ids"]]

    video_id: Optional[str] = args.video_id
    if video_id is None:
        # Prefer a call-heavy eval video so boundary targets exist.
        candidates = eval_video_ids + train_video_ids
        for vid in candidates:
            label_key = f"{label_prefix}{vid}.json"
            labels = parse_video_labels(s3_read_json(S3_BUCKET, label_key, region=AWS_REGION))
            if labels.boundaries:
                video_id = vid
                break
    if video_id is None:
        raise RuntimeError("No labeled video with boundaries found in split config.")

    audio_key = audio_index.get(video_id)
    if not audio_key:
        raise RuntimeError(f"Audio not found in S3 prefixes for video_id={video_id}")

    label_key = f"{label_prefix}{video_id}.json"
    labels = parse_video_labels(s3_read_json(S3_BUCKET, label_key, region=AWS_REGION))
    boundaries: list[CallBoundary] = list(labels.boundaries)
    logger.info(f"video_id={video_id} boundaries={len(boundaries)} audio_key={audio_key}")

    mp3_path = audio_cache_dir / f"{video_id}.mp3"
    s3_download_if_missing(S3_BUCKET, audio_key, mp3_path, region=AWS_REGION)
    flac_path = flac_cache_dir / f"{video_id}.flac"
    ensure_flac_cached(mp3_path=mp3_path, flac_path=flac_path, sr_hz=int(args.sr_hz))
    duration_s = float(ffprobe_duration_s(flac_path))

    chunk_cfg = ChunkingConfig(
        chunk_total_s=float(cfg.get("chunk_total_s", 30.0)),
        core_s=float(cfg.get("core_s", 20.0)),
        margin_s=float(cfg.get("margin_s", 5.0)),
    )

    # Build a tiny boundary-centered dataset: enough to validate shapes + non-degenerate start/end.
    examples = make_examples_for_video(
        video_id=video_id,
        audio_path=flac_path,
        boundaries=boundaries,
        duration_s=duration_s,
        chunk_cfg=chunk_cfg,
        seed=int(args.seed),
        boundary_k=1,
        boundary_jitter_s=0.0,
        in_call_samples=0,
        out_call_samples=0,
    )
    if len(examples) == 0:
        raise RuntimeError("No training examples produced for preflight (unexpected).")
    examples = examples[:2]

    feature_extractor = AutoFeatureExtractor.from_pretrained(str(cfg.get("base_model_name", "microsoft/wavlm-large")))
    target_cfg = TargetConfig(
        start_tolerance_s=float(cfg.get("start_tolerance_s", 0.20)),
        end_tolerance_s=float(cfg.get("end_tolerance_s", 0.20)),
        boundary_target_shape=str(cfg.get("boundary_target_shape", "binary")),
    )

    model_cfg = WavLMFrameClassifierConfig(
        base_model_name=str(cfg.get("base_model_name", "microsoft/wavlm-large")),
        head_dropout=float(cfg.get("head_dropout", 0.10)),
        freeze_feature_encoder=bool(cfg.get("freeze_feature_encoder", True)),
        pos_weight_in_call=float(cfg.get("pos_weight_in_call", 1.0)),
        pos_weight_start=float(cfg.get("pos_weight_start", 10.0)),
        pos_weight_end=float(cfg.get("pos_weight_end", 10.0)),
    )
    model = WavLMFrameClassifier(model_cfg).to(args.device)
    model.train()

    collator = Collator(
        feature_extractor=feature_extractor,
        sr_hz=int(args.sr_hz),
        target_cfg=target_cfg,
        model_config=model.wavlm.config,
    )
    ds = ChunkDataset(examples, sr_hz=int(args.sr_hz))
    batch = collator([ds[0], ds[min(1, len(ds) - 1)]])

    # Basic tensor sanity.
    logger.info(f"input_values.shape={tuple(batch['input_values'].shape)}")
    logger.info(f"labels.shape={tuple(batch['labels'].shape)} label_mask.shape={tuple(batch['label_mask'].shape)}")
    logger.info(f"label_mask nonzero frac={_nonzero_frac(batch['label_mask']):.3f}")

    rep0 = _forward_report(model=model, batch=batch, device=str(args.device))
    logger.info(f"forward_report_before={json.dumps(rep0, sort_keys=True)}")

    if int(args.overfit_steps) <= 0:
        logger.info("Overfit test disabled (--overfit-steps 0).")
        return 0

    # HF best practice: overfit one batch to validate label wiring / learning dynamics.
    out_dir = Path("artifacts/call_extractor/wavlm_large_v1") / f"preflight_{video_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_args = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=int(args.overfit_steps),
        learning_rate=float(args.overfit_lr),
        weight_decay=0.0,
        warmup_ratio=0.0,
        logging_steps=max(1, int(args.overfit_steps // 5)),
        eval_strategy="no",
        save_strategy="no",
        fp16=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=[],
        seed=int(args.seed),
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=collator,
        train_dataset=ds,
        processing_class=feature_extractor,
    )
    trainer.train()

    model.eval()
    rep1 = _forward_report(model=model, batch=batch, device=str(args.device))
    logger.info(f"forward_report_after={json.dumps(rep1, sort_keys=True)}")

    # Minimal gate: loss should remain finite and some head should show separation.
    if not np.isfinite(rep1["loss"]):
        raise RuntimeError("Post-overfit loss is not finite (NaN/inf).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

