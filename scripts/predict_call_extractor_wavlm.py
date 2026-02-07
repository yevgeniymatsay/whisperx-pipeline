#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoProcessor

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import AWS_REGION, S3_BUCKET
from pipeline.call_extractor_wavlm.chunking import iter_inference_chunk_specs
from pipeline.call_extractor_wavlm.decode import DecodeConfig, probabilities_to_segments
from pipeline.call_extractor_wavlm.io import (
    build_audio_index,
    decode_audio_segment_to_float32,
    ffprobe_duration_s,
    s3_download_if_missing,
    s3_upload_file,
)
from pipeline.call_extractor_wavlm.model import WavLMFrameClassifier, WavLMFrameClassifierConfig
from pipeline.call_extractor_wavlm.types import ChunkingConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_decode_cfg(path: Path) -> DecodeConfig:
    data = _load_json(path)
    return DecodeConfig(**data)


def _load_model(model_dir: Path, *, device: str) -> tuple[Any, torch.nn.Module]:
    processor = AutoProcessor.from_pretrained(str(model_dir))
    model_cfg = WavLMFrameClassifierConfig(base_model_name=str(model_dir), freeze_feature_encoder=False)
    model = WavLMFrameClassifier(model_cfg)
    state = torch.load(model_dir / "frame_heads.pt", map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)
    return processor, model


@torch.inference_mode()
def _predict_video(
    *,
    video_id: str,
    audio_path: Path,
    model,
    processor,
    sr_hz: int,
    chunk_cfg: ChunkingConfig,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    duration_s = ffprobe_duration_s(audio_path)
    timing = model.feat_timing(sr_hz=sr_hz)
    stride_s = float(timing.stride_s)
    offset_s = float(timing.offset_s)

    all_times: list[np.ndarray] = []
    all_in_call: list[np.ndarray] = []
    all_start: list[np.ndarray] = []
    all_end: list[np.ndarray] = []

    for spec in iter_inference_chunk_specs(duration_s, chunk_cfg):
        audio = decode_audio_segment_to_float32(
            audio_path,
            start_s=float(spec.chunk_start_s),
            duration_s=float(spec.chunk_total_s),
            sr_hz=int(sr_hz),
        )
        inputs = processor(audio, sampling_rate=int(sr_hz), return_tensors="pt", padding=True)
        input_values = inputs["input_values"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        out = model(input_values=input_values, attention_mask=attention_mask)
        logits = out["logits"][0].detach().float().cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))

        out_len = probs.shape[0]
        times_abs = float(spec.chunk_start_s) + (
            (np.arange(out_len, dtype=np.float32) * float(stride_s)) + float(offset_s)
        )

        core_mask = (times_abs >= float(spec.core_start_s)) & (times_abs < float(spec.core_end_s))
        if not core_mask.any():
            continue

        all_times.append(times_abs[core_mask])
        all_in_call.append(probs[core_mask, 0].astype(np.float32))
        all_start.append(probs[core_mask, 1].astype(np.float32))
        all_end.append(probs[core_mask, 2].astype(np.float32))

    if not all_times:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    t = np.concatenate(all_times)
    in_call = np.concatenate(all_in_call)
    start = np.concatenate(all_start)
    end = np.concatenate(all_end)

    order = np.argsort(t)
    return t[order], in_call[order], start[order], end[order]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run WavLM call extractor inference and decode segments.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Local model dir containing frame_heads.pt")
    parser.add_argument("--split-config", type=Path, default=Path("configs/call_extractor/split_v1.config.json"))
    parser.add_argument("--output-config", type=Path, default=Path("configs/call_extractor/output_wavlm_large_v1.config.json"))
    parser.add_argument("--decode-config", type=Path, default=Path("configs/call_extractor/decode_wavlm_large_v1.config.json"))
    parser.add_argument(
        "--subset",
        type=str,
        choices=["eval", "train", "all"],
        default="eval",
        help="Which split videos to process (default: eval)",
    )
    parser.add_argument("--sr-hz", type=int, default=16000)
    parser.add_argument("--chunk-total-s", type=float, default=30.0)
    parser.add_argument("--core-s", type=float, default=20.0)
    parser.add_argument("--margin-s", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--upload", action=argparse.BooleanOptionalAction, default=True, help="Upload segments to S3")
    args = parser.parse_args()

    split_cfg = _load_json(args.split_config)
    out_cfg = _load_json(args.output_config)
    decode_cfg = _load_decode_cfg(args.decode_config)

    video_ids: list[str] = []
    if args.subset == "eval":
        video_ids = list(split_cfg["eval_video_ids"])
    elif args.subset == "train":
        video_ids = list(split_cfg["train_video_ids"])
    else:
        video_ids = list(split_cfg["eval_video_ids"]) + list(split_cfg["train_video_ids"])

    audio_prefixes = list(split_cfg["audio_prefixes"])
    audio_index = build_audio_index(S3_BUCKET, audio_prefixes, region=AWS_REGION)

    cache_dir = Path(out_cfg.get("local_cache_dir", ".cache/call_extractor_wavlm"))
    audio_cache_dir = cache_dir / "audio"
    audio_cache_dir.mkdir(parents=True, exist_ok=True)

    processor, model = _load_model(args.model_dir, device=args.device)
    chunk_cfg = ChunkingConfig(
        chunk_total_s=float(args.chunk_total_s),
        core_s=float(args.core_s),
        margin_s=float(args.margin_s),
    )

    s3_prefix = str(out_cfg.get("s3_output_prefix", "call_extractor/wavlm_large_v1/")).rstrip("/")
    local_pred_dir = Path(out_cfg.get("local_artifacts_dir", "artifacts/call_extractor/wavlm_large_v1")) / "predictions"
    local_pred_dir.mkdir(parents=True, exist_ok=True)

    for idx, vid in enumerate(video_ids):
        audio_key = audio_index.get(str(vid))
        if not audio_key:
            logger.warning(f"[{idx+1}/{len(video_ids)}] Skip {vid}: audio not found")
            continue
        audio_path = audio_cache_dir / f"{vid}.mp3"
        s3_download_if_missing(S3_BUCKET, audio_key, audio_path, region=AWS_REGION)

        t, in_call, start, end = _predict_video(
            video_id=str(vid),
            audio_path=audio_path,
            model=model,
            processor=processor,
            sr_hz=int(args.sr_hz),
            chunk_cfg=chunk_cfg,
            device=str(args.device),
        )
        segments = probabilities_to_segments(
            times_s=t,
            in_call_p=in_call,
            start_p=start,
            end_p=end,
            cfg=decode_cfg,
        )

        out_json = {
            "video_id": str(vid),
            "audio_s3_key": str(audio_key),
            "model_dir": str(args.model_dir),
            "segments": segments,
        }

        local_path = local_pred_dir / f"{vid}.json"
        local_path.write_text(json.dumps(out_json, indent=2, sort_keys=True) + "\n")

        if args.upload:
            dst_key = f"{s3_prefix}/segments/{vid}.json"
            s3_upload_file(bucket=S3_BUCKET, key=dst_key, src_path=local_path, region=AWS_REGION)

        npz_path = local_pred_dir / f"{vid}.npz"
        np.savez_compressed(npz_path, times_s=t, in_call=in_call, start=start, end=end)
        if args.upload:
            dst_key = f"{s3_prefix}/probs/{vid}.npz"
            s3_upload_file(bucket=S3_BUCKET, key=dst_key, src_path=npz_path, region=AWS_REGION)

        logger.info(f"[{idx+1}/{len(video_ids)}] {vid}: segments={len(segments)} frames={t.shape[0]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

