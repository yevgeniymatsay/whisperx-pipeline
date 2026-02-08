#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor, Trainer, TrainingArguments

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import AWS_REGION, S3_BUCKET
from pipeline.call_extractor_wavlm.io import (
    build_audio_index,
    ffprobe_duration_s,
    s3_download_if_missing,
    s3_read_json,
)
from pipeline.call_extractor_wavlm.audio_cache import ensure_flac_cached
from pipeline.call_extractor_wavlm.chunking import sample_boundary_chunks
from pipeline.call_extractor_wavlm.labels import TargetConfig, parse_video_labels
from pipeline.call_extractor_wavlm.model import WavLMFrameClassifier, WavLMFrameClassifierConfig
from pipeline.call_extractor_wavlm.training import Collator, ChunkDataset, Example, make_examples_for_video, set_seed
from pipeline.call_extractor_wavlm.types import CallBoundary, ChunkingConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


class _RepeatItemDataset(Dataset):
    def __init__(self, items: list[dict[str, Any]], *, n: int):
        if not items:
            raise ValueError("items must be non-empty")
        self._items = list(items)
        self._n = int(n)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._items[int(idx) % int(len(self._items))]


def _pick_video_id(
    *,
    video_ids: list[str],
    audio_index: dict[str, str],
    label_prefix: str,
) -> Optional[str]:
    for vid in video_ids:
        if vid not in audio_index:
            continue
        try:
            data = s3_read_json(S3_BUCKET, f"{label_prefix}{vid}.json", region=AWS_REGION)
            labels = parse_video_labels(data)
        except Exception:
            continue
        if labels.boundaries:
            return str(vid)
    return None


def _batch_stats(*, probs: np.ndarray, labels: np.ndarray, label_mask: np.ndarray) -> dict[str, Any]:
    mask = label_mask.astype(bool, copy=False)
    if mask.size == 0 or not mask.any():
        return {"mask_frac": 0.0}

    out: dict[str, Any] = {"mask_frac": float(mask.mean())}
    for name, col in [("in_call", 0), ("start", 1), ("end", 2)]:
        y = labels[..., col]
        p = probs[..., col]
        y_m = y[mask]
        p_m = p[mask]
        pos = y_m >= 0.5
        neg = ~pos
        out[name] = {
            "pos_frac": float(pos.mean()) if y_m.size else 0.0,
            "mean_p_pos": float(p_m[pos].mean()) if pos.any() else None,
            "mean_p_neg": float(p_m[neg].mean()) if neg.any() else None,
            "p50": float(np.quantile(p_m, 0.5)) if p_m.size else 0.0,
            "p90": float(np.quantile(p_m, 0.9)) if p_m.size else 0.0,
            "p99": float(np.quantile(p_m, 0.99)) if p_m.size else 0.0,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="HF-docs-first preflight checks for WavLM call extractor training.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/call_extractor/train_wavlm_large_v1.config.json"),
        help="Training config (.config.json)",
    )
    parser.add_argument("--video-id", type=str, default=None, help="Optional video_id to use for the preflight batch")
    parser.add_argument("--overfit-steps", type=int, default=50, help="Trainer max_steps for 'overfit one batch'")
    parser.add_argument("--overfit-lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("artifacts/call_extractor/wavlm_large_v1/preflight_report.json"),
    )
    args = parser.parse_args()

    import transformers as _tf

    logger.info(f"torch={torch.__version__} cuda={torch.cuda.is_available()} device={args.device}")
    if torch.cuda.is_available():
        logger.info(f"gpu_name={torch.cuda.get_device_name(0)}")
    logger.info(f"transformers={_tf.__version__}")

    cfg = _load_json(args.config)
    split_cfg = _load_json(Path(cfg["split_config_path"]))
    out_cfg = _load_json(Path(cfg["output_config_path"]))

    sr_hz = int(cfg.get("sr_hz", 16000))
    seed = int(cfg.get("seed", 1337))
    set_seed(seed)

    cache_dir = Path(out_cfg.get("local_cache_dir", ".cache/call_extractor_wavlm"))
    audio_cache_dir = cache_dir / "audio"
    audio_cache_dir.mkdir(parents=True, exist_ok=True)

    label_prefix = str(split_cfg["label_prefix"])
    audio_prefixes = list(split_cfg["audio_prefixes"])
    audio_index = build_audio_index(S3_BUCKET, audio_prefixes, region=AWS_REGION)

    video_id = args.video_id
    if not video_id:
        video_id = _pick_video_id(
            video_ids=list(split_cfg["train_video_ids"]),
            audio_index=audio_index,
            label_prefix=label_prefix,
        )
    if not video_id:
        raise RuntimeError("Could not find a train video_id with audio + at least one boundary label")

    audio_key = audio_index.get(video_id)
    if not audio_key:
        raise RuntimeError(f"Audio key not found for video_id={video_id}")

    label_data = s3_read_json(S3_BUCKET, f"{label_prefix}{video_id}.json", region=AWS_REGION)
    labels = parse_video_labels(label_data)
    boundaries: list[CallBoundary] = list(labels.boundaries)
    if not boundaries:
        raise RuntimeError(f"Selected preflight video has no boundaries: {video_id}")

    audio_path = audio_cache_dir / f"{video_id}.mp3"
    s3_download_if_missing(S3_BUCKET, audio_key, audio_path, region=AWS_REGION)
    flac_path = (cache_dir / "audio_flac") / f"{video_id}.flac"
    flac_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_flac_cached(mp3_path=audio_path, flac_path=flac_path, sr_hz=int(sr_hz))
    duration_s = float(ffprobe_duration_s(flac_path))

    chunk_cfg = ChunkingConfig(
        chunk_total_s=float(cfg.get("chunk_total_s", 30.0)),
        core_s=float(cfg.get("core_s", 20.0)),
        margin_s=float(cfg.get("margin_s", 5.0)),
    )
    examples: list[Example] = make_examples_for_video(
        video_id=str(video_id),
        audio_path=flac_path,
        boundaries=boundaries,
        duration_s=float(duration_s),
        chunk_cfg=chunk_cfg,
        seed=seed,
        boundary_k=int(cfg.get("boundary_k_per_boundary", 3)),
        boundary_jitter_s=float(cfg.get("boundary_jitter_s", 1.0)),
        in_call_samples=0,
        out_call_samples=0,
    )
    if not examples:
        raise RuntimeError("Failed to generate boundary-centered training examples for preflight")

    # Build one example centered on a start boundary, and one centered on an end boundary, so we can sanity-check
    # both sparse target heads are non-degenerate inside the core region.
    rng = np.random.default_rng(int(seed) ^ (hash(video_id) & 0xFFFF_FFFF))
    start_times = [float(b.start_s) for b in boundaries]
    end_times = [float(b.end_s) for b in boundaries]

    start_starts = sample_boundary_chunks(
        boundary_times_s=start_times,
        duration_s=float(duration_s),
        cfg=chunk_cfg,
        k_per_boundary=1,
        jitter_s=0.0,
        rng=rng,
    )
    end_starts = sample_boundary_chunks(
        boundary_times_s=end_times,
        duration_s=float(duration_s),
        cfg=chunk_cfg,
        k_per_boundary=1,
        jitter_s=0.0,
        rng=rng,
    )

    def _mk_example(start_s: float) -> Example:
        core_start = float(start_s) + float(chunk_cfg.margin_s)
        core_end = core_start + float(chunk_cfg.core_s)
        return Example(
            video_id=str(video_id),
            audio_path=flac_path,
            boundaries=boundaries,
            chunk_start_s=float(start_s),
            chunk_total_s=float(chunk_cfg.chunk_total_s),
            core_start_abs_s=float(core_start),
            core_end_abs_s=float(core_end),
        )

    ex_start = _mk_example(float(start_starts[0])) if start_starts else examples[0]
    ex_end = _mk_example(float(end_starts[0])) if end_starts else examples[0]

    feature_extractor = AutoFeatureExtractor.from_pretrained(str(cfg.get("base_model_name", "microsoft/wavlm-large")))
    model_cfg = WavLMFrameClassifierConfig(
        base_model_name=str(cfg.get("base_model_name", "microsoft/wavlm-large")),
        head_dropout=float(cfg.get("head_dropout", 0.10)),
        freeze_feature_encoder=bool(cfg.get("freeze_feature_encoder", True)),
        pos_weight_in_call=float(cfg.get("pos_weight_in_call", 1.0)),
        pos_weight_start=float(cfg.get("pos_weight_start", 10.0)),
        pos_weight_end=float(cfg.get("pos_weight_end", 10.0)),
    )
    model = WavLMFrameClassifier(model_cfg).to(args.device)
    trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    total = sum(int(p.numel()) for p in model.parameters())
    logger.info(f"params_total={total:,} trainable={trainable:,} freeze_feature_encoder={model_cfg.freeze_feature_encoder}")

    target_cfg = TargetConfig(
        start_tolerance_s=float(cfg.get("start_tolerance_s", 0.20)),
        end_tolerance_s=float(cfg.get("end_tolerance_s", 0.20)),
        boundary_target_shape=str(cfg.get("boundary_target_shape", "binary")),
    )
    collator = Collator(feature_extractor=feature_extractor, sr_hz=sr_hz, target_cfg=target_cfg, model_config=model.wavlm.config)

    ds = ChunkDataset([ex_start, ex_end], sr_hz=sr_hz)
    item_start = ds[0]  # decode once
    item_end = ds[1]  # decode once

    batch = collator([item_start])
    batch = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    with torch.no_grad():
        out = model(
            input_values=batch["input_values"],
            attention_mask=batch.get("attention_mask"),
            labels=batch["labels"],
            label_mask=batch["label_mask"],
        )
        logits = out["logits"]
        loss0 = float(out["loss"].detach().float().cpu().item())
    logger.info(f"one_batch: logits_shape={tuple(logits.shape)} loss={loss0:.6f}")

    probs0 = torch.sigmoid(logits).detach().float().cpu().numpy()
    labels0 = batch["labels"].detach().float().cpu().numpy()
    mask0 = batch["label_mask"].detach().float().cpu().numpy()
    stats0 = _batch_stats(probs=probs0, labels=labels0, label_mask=mask0)
    logger.info(f"one_batch stats: {json.dumps(stats0, sort_keys=True)}")

    # Target density sanity: ensure *both* sparse heads appear inside the core region for some boundary-centered chunk.
    core = mask0.astype(bool, copy=False)
    start_pos = int((labels0[..., 1][core] >= 0.5).sum())
    end_pos = int((labels0[..., 2][core] >= 0.5).sum())
    ok_start = start_pos > 0
    ok_end = end_pos > 0

    if not ok_end:
        batch_e = collator([item_end])
        batch_e = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in batch_e.items()}
        with torch.no_grad():
            out_e = model(
                input_values=batch_e["input_values"],
                attention_mask=batch_e.get("attention_mask"),
                labels=batch_e["labels"],
                label_mask=batch_e["label_mask"],
            )
        logits_e = out_e["logits"]
        loss_e = float(out_e["loss"].detach().float().cpu().item())
        probs_e = torch.sigmoid(logits_e).detach().float().cpu().numpy()
        labels_e = batch_e["labels"].detach().float().cpu().numpy()
        mask_e = batch_e["label_mask"].detach().float().cpu().numpy()
        stats_e = _batch_stats(probs=probs_e, labels=labels_e, label_mask=mask_e)
        logger.info(f"end_batch: logits_shape={tuple(logits_e.shape)} loss={loss_e:.6f}")
        logger.info(f"end_batch stats: {json.dumps(stats_e, sort_keys=True)}")

        core_e = mask_e.astype(bool, copy=False)
        end_pos = int((labels_e[..., 2][core_e] >= 0.5).sum())
        ok_end = end_pos > 0

    if not ok_start or not ok_end:
        raise RuntimeError(
            f"Preflight target density failed (core positives): start_ok={ok_start} end_ok={ok_end}. "
            "Check timing alignment / tolerance bands / sampling."
        )

    # Overfit one batch (HF debugging best practice): ensure loss decreases on the same batch.
    overfit_steps = int(args.overfit_steps)
    if overfit_steps > 0:
        rep_ds = _RepeatItemDataset([item_start, item_end], n=2)
        training_args = TrainingArguments(
            output_dir=str(Path(out_cfg.get('local_artifacts_dir', 'artifacts/call_extractor/wavlm_large_v1')) / "preflight_overfit"),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=float(args.overfit_lr),
            max_steps=int(overfit_steps),
            logging_steps=max(1, int(overfit_steps // 10)),
            eval_strategy="no",
            save_strategy="no",
            fp16=False,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to=[],
            seed=int(seed),
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=collator,
            train_dataset=rep_ds,
            processing_class=feature_extractor,
        )
        trainer.train()

        with torch.no_grad():
            out2 = model(
                input_values=batch["input_values"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["labels"],
                label_mask=batch["label_mask"],
            )
            loss1 = float(out2["loss"].detach().float().cpu().item())
            logits1 = out2["logits"]
        logger.info(f"overfit: loss_start={loss0:.6f} loss_end={loss1:.6f}")

        probs1 = torch.sigmoid(logits1).detach().float().cpu().numpy()
        stats1 = _batch_stats(probs=probs1, labels=labels0, label_mask=mask0)
        logger.info(f"overfit stats: {json.dumps(stats1, sort_keys=True)}")

        if not np.isfinite(loss1) or loss1 >= loss0 * 0.98:
            raise RuntimeError(
                f"Overfit sanity failed: loss did not decrease (start={loss0:.6f} end={loss1:.6f}). "
                "Potential wiring issue (labels/mask misaligned, columns dropped, LR too low, frozen params)."
            )

    report = {
        "video_id": str(video_id),
        "duration_s": float(duration_s),
        "seed": int(seed),
        "chunk_cfg": {"chunk_total_s": chunk_cfg.chunk_total_s, "core_s": chunk_cfg.core_s, "margin_s": chunk_cfg.margin_s},
        "loss_one_batch": float(loss0),
        "stats_one_batch": stats0,
        "overfit_steps": int(overfit_steps),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    logger.info(f"Wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
