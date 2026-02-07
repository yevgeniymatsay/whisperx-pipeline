#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoFeatureExtractor,
    TrainingArguments,
    Trainer,
)

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import AWS_REGION, S3_BUCKET
from pipeline.call_extractor_wavlm.chunking import (
    sample_boundary_chunks,
    sample_in_call_chunks,
    sample_out_of_call_chunks,
)
from pipeline.call_extractor_wavlm.io import (
    build_audio_index,
    ffprobe_duration_s,
    get_s3,
    git_short_sha,
    s3_download_if_missing,
    s3_read_json,
    s3_upload_directory,
    utc_now_compact,
    decode_audio_segment_to_float32,
)
from pipeline.call_extractor_wavlm.labels import TargetConfig, make_core_mask, make_targets_for_frames, parse_video_labels
from pipeline.call_extractor_wavlm.model import (
    WavLMFrameClassifier,
    WavLMFrameClassifierConfig,
    feat_extract_output_length,
    feat_extract_timing_from_config,
)
from pipeline.call_extractor_wavlm.types import CallBoundary, ChunkingConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


@dataclass(frozen=True)
class Example:
    video_id: str
    audio_path: Path
    boundaries: list[CallBoundary]
    chunk_start_s: float
    chunk_total_s: float
    core_start_abs_s: float
    core_end_abs_s: float


class ChunkDataset(Dataset):
    def __init__(self, examples: Sequence[Example], *, sr_hz: int):
        self.examples = list(examples)
        self.sr_hz = int(sr_hz)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[int(idx)]
        audio = decode_audio_segment_to_float32(
            ex.audio_path,
            start_s=float(ex.chunk_start_s),
            duration_s=float(ex.chunk_total_s),
            sr_hz=int(self.sr_hz),
        )
        return {
            "video_id": ex.video_id,
            "audio": audio,
            "boundaries": ex.boundaries,
            "chunk_start_s": float(ex.chunk_start_s),
            "core_start_abs_s": float(ex.core_start_abs_s),
            "core_end_abs_s": float(ex.core_end_abs_s),
        }


class Collator:
    def __init__(
        self,
        *,
        feature_extractor,
        sr_hz: int,
        target_cfg: TargetConfig,
        model_config,
    ):
        self.feature_extractor = feature_extractor
        self.sr_hz = int(sr_hz)
        self.target_cfg = target_cfg
        self.model_config = model_config

        self.timing = feat_extract_timing_from_config(config=self.model_config, sr_hz=self.sr_hz)

    def __call__(self, batch: list[dict]) -> dict:
        audios = [b["audio"] for b in batch]
        inputs = self.feature_extractor(
            audios,
            sampling_rate=int(self.sr_hz),
            return_tensors="pt",
            padding=True,
        )

        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            input_lengths = attention_mask.sum(dim=-1).to(torch.int64).tolist()
        else:
            input_lengths = [int(v.shape[-1]) for v in inputs["input_values"]]

        out_lens = [feat_extract_output_length(input_length_samples=int(L), config=self.model_config) for L in input_lengths]
        max_out_len = int(max(out_lens)) if out_lens else 0

        labels = torch.zeros((len(batch), max_out_len, 3), dtype=torch.float32)
        label_mask = torch.zeros((len(batch), max_out_len), dtype=torch.float32)

        stride_s = float(self.timing.stride_s)
        offset_s = float(self.timing.offset_s)

        for bi, (b, out_len) in enumerate(zip(batch, out_lens)):
            chunk_start = float(b["chunk_start_s"])
            frame_times_abs = chunk_start + (np.arange(int(out_len), dtype=np.float32) * float(stride_s) + float(offset_s))

            in_call, start, end = make_targets_for_frames(
                frame_times_abs,
                boundaries=b["boundaries"],
                cfg=self.target_cfg,
            )
            core_mask = make_core_mask(
                frame_times_abs,
                core_start_abs_s=float(b["core_start_abs_s"]),
                core_end_abs_s=float(b["core_end_abs_s"]),
            ).astype(np.float32)

            lab = np.stack([in_call, start, end], axis=-1).astype(np.float32)
            labels[bi, : int(out_len), :] = torch.from_numpy(lab)
            label_mask[bi, : int(out_len)] = torch.from_numpy(core_mask)

        return {
            "input_values": inputs["input_values"],
            "attention_mask": attention_mask,
            "labels": labels,
            "label_mask": label_mask,
        }


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _make_examples_for_video(
    *,
    video_id: str,
    audio_path: Path,
    boundaries: Sequence[CallBoundary],
    duration_s: float,
    chunk_cfg: ChunkingConfig,
    seed: int,
    boundary_k: int,
    boundary_jitter_s: float,
    in_call_samples: int,
    out_call_samples: int,
) -> list[Example]:
    rng = np.random.default_rng(int(seed) ^ (hash(video_id) & 0xFFFF_FFFF))
    boundary_times: list[float] = []
    for b in boundaries:
        boundary_times.extend([float(b.start_s), float(b.end_s)])

    starts: list[float] = []
    starts.extend(
        sample_boundary_chunks(
            boundary_times_s=boundary_times,
            duration_s=float(duration_s),
            cfg=chunk_cfg,
            k_per_boundary=int(boundary_k),
            jitter_s=float(boundary_jitter_s),
            rng=rng,
        )
    )
    starts.extend(
        sample_in_call_chunks(
            boundaries=boundaries,
            duration_s=float(duration_s),
            cfg=chunk_cfg,
            n_samples=int(in_call_samples),
            rng=rng,
        )
    )
    starts.extend(
        sample_out_of_call_chunks(
            boundaries=boundaries,
            duration_s=float(duration_s),
            cfg=chunk_cfg,
            n_samples=int(out_call_samples),
            rng=rng,
        )
    )

    examples: list[Example] = []
    for s in starts:
        core_start = float(s) + float(chunk_cfg.margin_s)
        core_end = core_start + float(chunk_cfg.core_s)
        examples.append(
            Example(
                video_id=str(video_id),
                audio_path=audio_path,
                boundaries=list(boundaries),
                chunk_start_s=float(s),
                chunk_total_s=float(chunk_cfg.chunk_total_s),
                core_start_abs_s=float(core_start),
                core_end_abs_s=float(core_end),
            )
        )
    return examples


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
    _set_seed(seed)

    repo_root = Path(__file__).parent.parent
    run_id = f"run_{utc_now_compact()}_{git_short_sha(repo_root)}"
    local_artifacts_dir = Path(out_cfg.get("local_artifacts_dir", "artifacts/call_extractor/wavlm_large_v1")) / run_id
    cache_dir = Path(out_cfg.get("local_cache_dir", ".cache/call_extractor_wavlm"))
    audio_cache_dir = cache_dir / "audio"
    audio_cache_dir.mkdir(parents=True, exist_ok=True)
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

        audio_path = audio_cache_dir / f"{video_id}.mp3"
        s3_download_if_missing(S3_BUCKET, audio_key, audio_path, region=AWS_REGION)
        duration_s = ffprobe_duration_s(audio_path)
        return audio_path, labels.boundaries, float(duration_s)

    chunk_cfg = ChunkingConfig(
        chunk_total_s=float(cfg.get("chunk_total_s", 30.0)),
        core_s=float(cfg.get("core_s", 20.0)),
        margin_s=float(cfg.get("margin_s", 5.0)),
    )

    boundary_k = int(cfg.get("boundary_k_per_boundary", 3))
    boundary_jitter_s = float(cfg.get("boundary_jitter_s", 1.0))
    in_call_samples = int(cfg.get("in_call_samples_per_video", 4))
    out_call_samples = int(cfg.get("out_call_samples_per_video", 4))

    train_examples: list[Example] = []
    for vid in train_video_ids:
        loaded = _load_video(str(vid))
        if not loaded:
            continue
        audio_path, boundaries, duration_s = loaded
        train_examples.extend(
            _make_examples_for_video(
                video_id=str(vid),
                audio_path=audio_path,
                boundaries=boundaries,
                duration_s=float(duration_s),
                chunk_cfg=chunk_cfg,
                seed=seed,
                boundary_k=boundary_k,
                boundary_jitter_s=boundary_jitter_s,
                in_call_samples=in_call_samples,
                out_call_samples=out_call_samples,
            )
        )
    logger.info(f"Train examples: {len(train_examples)} from {len(train_video_ids)} videos")

    eval_examples: list[Example] = []
    for vid in eval_video_ids:
        loaded = _load_video(str(vid))
        if not loaded:
            continue
        audio_path, boundaries, duration_s = loaded
        eval_examples.extend(
            _make_examples_for_video(
                video_id=str(vid),
                audio_path=audio_path,
                boundaries=boundaries,
                duration_s=float(duration_s),
                chunk_cfg=chunk_cfg,
                seed=seed + 1,
                boundary_k=boundary_k,
                boundary_jitter_s=boundary_jitter_s,
                in_call_samples=max(1, in_call_samples // 2),
                out_call_samples=max(1, out_call_samples // 2),
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
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 2)),
        per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", 2)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(cfg.get("learning_rate", 2e-5)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.05)),
        logging_steps=int(cfg.get("logging_steps", 25)),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=int(cfg.get("save_total_limit", 3)),
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
