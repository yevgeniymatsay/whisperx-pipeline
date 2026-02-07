from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch
from torch import nn
from transformers import WavLMModel

from .types import FeatExtractTiming


def _as_int_list(values: Iterable[int]) -> list[int]:
    return [int(v) for v in values]


def feat_extract_timing_from_config(*, config, sr_hz: int) -> FeatExtractTiming:
    conv_kernel = getattr(config, "conv_kernel", None)
    conv_stride = getattr(config, "conv_stride", None)

    if conv_kernel is None or conv_stride is None:
        # Reasonable defaults for WavLM/Wav2Vec2 style feature extractor.
        stride_samples = 320
        receptive_field_samples = 703
        offset_samples = 351
        return FeatExtractTiming(
            sr_hz=int(sr_hz),
            stride_samples=int(stride_samples),
            offset_samples=int(offset_samples),
            receptive_field_samples=int(receptive_field_samples),
        )

    kernels = _as_int_list(conv_kernel)
    strides = _as_int_list(conv_stride)
    if len(kernels) != len(strides) or len(kernels) == 0:
        raise ValueError("Invalid conv_kernel/conv_stride in model config")

    receptive = 1
    for k, s in zip(kernels, strides):
        receptive = (receptive - 1) * int(s) + int(k)
    total_stride = int(math.prod(strides))
    offset = int(round((receptive - 1) / 2.0))

    return FeatExtractTiming(
        sr_hz=int(sr_hz),
        stride_samples=int(total_stride),
        offset_samples=int(offset),
        receptive_field_samples=int(receptive),
    )


def feat_extract_output_length(*, input_length_samples: int, config) -> int:
    """Compute feature extractor output length matching HF conv stack semantics."""
    conv_kernel = getattr(config, "conv_kernel", None)
    conv_stride = getattr(config, "conv_stride", None)
    if conv_kernel is None or conv_stride is None:
        stride = 320
        return int(max(0, (int(input_length_samples) // stride)))

    length = int(input_length_samples)
    for k, s in zip(conv_kernel, conv_stride):
        k_i = int(k)
        s_i = int(s)
        length = int(math.floor((length - k_i) / s_i) + 1)
    return max(0, int(length))


@dataclass(frozen=True)
class WavLMFrameClassifierConfig:
    base_model_name: str = "microsoft/wavlm-large"
    head_dropout: float = 0.10
    freeze_feature_encoder: bool = True

    pos_weight_in_call: float = 1.0
    pos_weight_start: float = 10.0
    pos_weight_end: float = 10.0


class WavLMFrameClassifier(nn.Module):
    def __init__(self, cfg: WavLMFrameClassifierConfig):
        super().__init__()
        self.cfg = cfg

        self.wavlm = WavLMModel.from_pretrained(cfg.base_model_name)
        if cfg.freeze_feature_encoder and hasattr(self.wavlm, "freeze_feature_encoder"):
            self.wavlm.freeze_feature_encoder()

        self.dropout = nn.Dropout(float(cfg.head_dropout))
        self.classifier = nn.Linear(int(self.wavlm.config.hidden_size), 3)

        pos_weight = torch.tensor(
            [float(cfg.pos_weight_in_call), float(cfg.pos_weight_start), float(cfg.pos_weight_end)],
            dtype=torch.float32,
        )
        self.register_buffer("_pos_weight", pos_weight, persistent=False)
        self._loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=self._pos_weight)

    def feat_timing(self, *, sr_hz: int) -> FeatExtractTiming:
        return feat_extract_timing_from_config(config=self.wavlm.config, sr_hz=int(sr_hz))

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        label_mask: Optional[torch.Tensor] = None,
    ):
        out = self.wavlm(input_values=input_values, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        logits = self.classifier(self.dropout(hidden))

        result = {"logits": logits}
        if labels is None:
            return result

        if label_mask is None:
            raise ValueError("labels provided without label_mask")

        if labels.shape != logits.shape:
            raise ValueError(f"labels shape {labels.shape} must match logits shape {logits.shape}")

        mask = label_mask.to(dtype=logits.dtype)
        loss_raw = self._loss_fn(logits, labels)
        loss = (loss_raw * mask.unsqueeze(-1)).sum()
        denom = mask.sum().clamp(min=1.0) * float(logits.shape[-1])
        result["loss"] = loss / denom
        return result

