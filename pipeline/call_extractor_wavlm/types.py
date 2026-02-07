from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CallBoundary:
    start_s: float
    end_s: float


@dataclass(frozen=True)
class VideoLabels:
    video_id: str
    boundaries: list[CallBoundary]


@dataclass(frozen=True)
class FeatExtractTiming:
    """Timing for mapping WavLM feature frames to seconds."""

    sr_hz: int
    stride_samples: int
    offset_samples: int
    receptive_field_samples: int

    @property
    def stride_s(self) -> float:
        return float(self.stride_samples) / float(self.sr_hz)

    @property
    def offset_s(self) -> float:
        return float(self.offset_samples) / float(self.sr_hz)


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_total_s: float = 30.0
    core_s: float = 20.0
    margin_s: float = 5.0

    def __post_init__(self) -> None:
        expected = self.core_s + 2.0 * self.margin_s
        if abs(expected - self.chunk_total_s) > 1e-6:
            raise ValueError(
                f"Invalid chunking: chunk_total_s must equal core_s + 2*margin_s "
                f"(got chunk_total_s={self.chunk_total_s}, core_s={self.core_s}, margin_s={self.margin_s})"
            )


@dataclass(frozen=True)
class InferenceChunkSpec:
    chunk_start_s: float
    chunk_total_s: float
    core_start_s: float
    core_end_s: float

