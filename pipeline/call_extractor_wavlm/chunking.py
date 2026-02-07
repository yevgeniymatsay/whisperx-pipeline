from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np

from .types import CallBoundary, ChunkingConfig, InferenceChunkSpec


def iter_inference_chunk_specs(duration_s: float, cfg: ChunkingConfig) -> Iterator[InferenceChunkSpec]:
    """Yield chunk specs whose core windows tile [0, duration_s]."""
    if duration_s <= 0:
        raise ValueError(f"duration_s must be > 0 (got {duration_s})")

    if duration_s <= cfg.chunk_total_s:
        yield InferenceChunkSpec(
            chunk_start_s=0.0,
            chunk_total_s=cfg.chunk_total_s,
            core_start_s=0.0,
            core_end_s=float(duration_s),
        )
        return

    n_cores = int(math.ceil(float(duration_s) / float(cfg.core_s)))
    for i in range(n_cores):
        core_start = float(i) * float(cfg.core_s)
        if core_start >= duration_s:
            break
        core_end = min(float(duration_s), core_start + float(cfg.core_s))
        chunk_start = core_start - float(cfg.margin_s)
        chunk_start = max(0.0, min(chunk_start, float(duration_s) - float(cfg.chunk_total_s)))

        yield InferenceChunkSpec(
            chunk_start_s=chunk_start,
            chunk_total_s=float(cfg.chunk_total_s),
            core_start_s=core_start,
            core_end_s=core_end,
        )

        if core_end >= duration_s:
            break


def _core_interval_for_training_chunk(chunk_start_s: float, cfg: ChunkingConfig) -> tuple[float, float]:
    core_start = float(chunk_start_s) + float(cfg.margin_s)
    core_end = core_start + float(cfg.core_s)
    return core_start, core_end


def _interval_overlap_s(a0: float, a1: float, b0: float, b1: float) -> float:
    s = max(float(a0), float(b0))
    e = min(float(a1), float(b1))
    return max(0.0, e - s)


def core_overlaps_any_call(core_start_s: float, core_end_s: float, boundaries: Sequence[CallBoundary]) -> bool:
    for b in boundaries:
        if _interval_overlap_s(core_start_s, core_end_s, b.start_s, b.end_s) > 0.0:
            return True
    return False


def sample_boundary_chunks(
    *,
    boundary_times_s: Sequence[float],
    duration_s: float,
    cfg: ChunkingConfig,
    k_per_boundary: int,
    jitter_s: float,
    rng: np.random.Generator,
) -> list[float]:
    """Sample chunk starts such that each boundary time lands inside the core region."""
    if duration_s <= cfg.chunk_total_s:
        return [0.0]
    max_start = float(duration_s) - float(cfg.chunk_total_s)

    starts: list[float] = []
    for t in boundary_times_s:
        for _ in range(int(k_per_boundary)):
            pos_in_core = rng.uniform(0.0, float(cfg.core_s))
            pos_in_chunk = float(cfg.margin_s) + float(pos_in_core)
            start = float(t) - float(pos_in_chunk) + float(rng.uniform(-float(jitter_s), float(jitter_s)))
            start = max(0.0, min(start, max_start))

            core_start, core_end = _core_interval_for_training_chunk(start, cfg)
            if core_start <= float(t) <= core_end:
                starts.append(float(start))
    return starts


def sample_in_call_chunks(
    *,
    boundaries: Sequence[CallBoundary],
    duration_s: float,
    cfg: ChunkingConfig,
    n_samples: int,
    rng: np.random.Generator,
) -> list[float]:
    if not boundaries:
        return []
    if duration_s <= cfg.chunk_total_s:
        return [0.0]
    max_start = float(duration_s) - float(cfg.chunk_total_s)

    starts: list[float] = []
    for _ in range(int(n_samples)):
        b = boundaries[int(rng.integers(0, len(boundaries)))]
        if b.end_s <= b.start_s:
            continue
        t = float(rng.uniform(float(b.start_s), float(b.end_s)))
        core_center = float(cfg.margin_s) + float(cfg.core_s) / 2.0
        start = float(t) - core_center
        start = max(0.0, min(start, max_start))
        starts.append(float(start))
    return starts


def sample_out_of_call_chunks(
    *,
    boundaries: Sequence[CallBoundary],
    duration_s: float,
    cfg: ChunkingConfig,
    n_samples: int,
    rng: np.random.Generator,
    max_tries_per_sample: int = 200,
) -> list[float]:
    if duration_s <= cfg.chunk_total_s:
        return [0.0]
    max_start = float(duration_s) - float(cfg.chunk_total_s)

    starts: list[float] = []
    for _ in range(int(n_samples)):
        ok = False
        for _try in range(int(max_tries_per_sample)):
            start = float(rng.uniform(0.0, max_start))
            core_start, core_end = _core_interval_for_training_chunk(start, cfg)
            if not core_overlaps_any_call(core_start, core_end, boundaries):
                starts.append(float(start))
                ok = True
                break
        if not ok:
            # If the video is mostly calls, we may not find a pure negative core. Skip.
            continue
    return starts

