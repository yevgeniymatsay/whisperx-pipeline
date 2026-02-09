from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .audio_cache import read_flac_segment_float32
from .chunking import sample_boundary_chunks, sample_in_call_chunks, sample_out_of_call_chunks
from .labels import TargetConfig, make_core_mask, make_targets_for_frames
from .model import feat_extract_output_length, feat_extract_timing_from_config
from .types import CallBoundary, ChunkingConfig


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
        audio = read_flac_segment_float32(
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
        audio_lengths = [int(a.shape[0]) for a in audios]
        inputs = self.feature_extractor(
            audios,
            sampling_rate=int(self.sr_hz),
            return_tensors="pt",
            padding=True,
        )

        attention_mask = inputs.get("attention_mask", None)
        feat_norm = getattr(self.model_config, "feat_extract_norm", None)
        use_attention_mask = (attention_mask is not None) and (str(feat_norm).lower() != "group")
        if not use_attention_mask:
            attention_mask = None
        else:
            # transformers attention mask is conceptually boolean; keeping it bool avoids downstream
            # dtype mismatches in torch attention implementations.
            attention_mask = attention_mask.to(dtype=torch.bool)
        input_lengths = audio_lengths

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


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def make_examples_for_video(
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
