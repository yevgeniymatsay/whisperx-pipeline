#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from transformers import AutoFeatureExtractor

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.call_extractor_wavlm.model import WavLMFrameClassifier, WavLMFrameClassifierConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a Trainer checkpoint into a predictable model_dir (wavlm backbone + frame_heads.pt)."
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Trainer checkpoint dir (contains pytorch_model.bin)")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output model dir")
    parser.add_argument("--base-model-name", type=str, default="microsoft/wavlm-large", help="Base HF checkpoint id")
    args = parser.parse_args()

    ckpt_bin = args.checkpoint_dir / "pytorch_model.bin"
    ckpt_safe = args.checkpoint_dir / "model.safetensors"
    if not ckpt_safe.exists() and not ckpt_bin.exists():
        raise FileNotFoundError(f"Missing checkpoint weights (expected {ckpt_safe} or {ckpt_bin})")

    if ckpt_safe.exists():
        logger.info(f"Load checkpoint (safetensors): {ckpt_safe}")
        try:
            from safetensors.torch import load_file
        except Exception as e:  # pragma: no cover
            raise RuntimeError("safetensors is required to load model.safetensors: pip install safetensors") from e
        state = load_file(str(ckpt_safe))
    else:
        logger.info(f"Load checkpoint: {ckpt_bin}")
        state = torch.load(ckpt_bin, map_location="cpu")

    feature_extractor = AutoFeatureExtractor.from_pretrained(str(args.base_model_name))
    model_cfg = WavLMFrameClassifierConfig(base_model_name=str(args.base_model_name), freeze_feature_encoder=False)
    model = WavLMFrameClassifier(model_cfg)
    model.load_state_dict(state, strict=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.wavlm.save_pretrained(args.out_dir)
    feature_extractor.save_pretrained(args.out_dir)
    torch.save(model.classifier.state_dict(), args.out_dir / "frame_heads.pt")

    logger.info(f"Wrote model_dir: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
