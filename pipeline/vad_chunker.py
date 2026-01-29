# scripts/whisperx_pipeline/vad_chunker.py
"""VAD-based audio chunking for WhisperX processing."""
import os
from dataclasses import dataclass
from typing import List
import numpy as np
import torch
from .config import VADConfig


@dataclass
class Chunk:
    chunk_id: str
    start_ms: int
    end_ms: int
    start_s: float
    end_s: float
    duration_s: float


class VADChunker:
    """Split audio into chunks based on voice activity detection."""

    def __init__(self, config: VADConfig):
        self.config = config
        self._model = None
        self._utils = None

    def _load_vad_model(self):
        """Load Silero VAD model from cache (pre-downloaded in Docker build)."""
        if self._model is None:
            # Point torch.hub to the cached model from Docker build
            torch.hub.set_dir(os.environ.get("TORCH_HOME", "/app/.cache/torch"))
            self._model, self._utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True
            )
        return self._model, self._utils

    def get_speech_segments(self, audio: np.ndarray, sr: int = 16000) -> List[dict]:
        """Get speech timestamps from audio using Silero VAD."""
        model, utils = self._load_vad_model()
        get_speech_timestamps = utils[0]

        audio_tensor = torch.from_numpy(audio)
        segments = get_speech_timestamps(audio_tensor, model, sampling_rate=sr)

        return [
            {"start": s["start"] / sr, "end": s["end"] / sr}
            for s in segments
        ]

    def compute_chunks(
        self,
        speech_segments: List[dict],
        total_duration_s: float,
        video_id: str
    ) -> List[Chunk]:
        """Compute chunk boundaries from speech segments."""
        if not speech_segments:
            # Single chunk for entire audio
            return [self._make_chunk(video_id, 0.0, total_duration_s)]

        chunks = []
        current_start = max(0, speech_segments[0]["start"] - self.config.padding_s)

        for i, seg in enumerate(speech_segments):
            is_last = i == len(speech_segments) - 1

            if is_last:
                # End of audio
                end = min(total_duration_s, seg["end"] + self.config.padding_s)
                chunks.extend(self._split_if_needed(video_id, current_start, end))
            else:
                gap = speech_segments[i + 1]["start"] - seg["end"]

                if gap >= self.config.gap_threshold_s:
                    # Split here
                    end = min(total_duration_s, seg["end"] + self.config.padding_s)
                    chunks.extend(self._split_if_needed(video_id, current_start, end))
                    current_start = max(0, speech_segments[i + 1]["start"] - self.config.padding_s)

        # Merge tiny chunks
        return self._merge_small_chunks(chunks, video_id)

    def _split_if_needed(self, video_id: str, start: float, end: float) -> List[Chunk]:
        """Split chunk if it exceeds max duration."""
        duration = end - start
        if duration <= self.config.max_chunk_s:
            return [self._make_chunk(video_id, start, end)]

        # Split into max_chunk_s pieces
        chunks = []
        current = start
        while current < end:
            chunk_end = min(current + self.config.max_chunk_s, end)
            chunks.append(self._make_chunk(video_id, current, chunk_end))
            current = chunk_end
        return chunks

    def _merge_small_chunks(self, chunks: List[Chunk], video_id: str) -> List[Chunk]:
        """Merge chunks smaller than min duration."""
        if not chunks:
            return chunks

        merged = []
        current_start = chunks[0].start_s
        current_end = chunks[0].end_s

        for chunk in chunks[1:]:
            if current_end - current_start < self.config.min_chunk_s:
                # Extend current chunk
                current_end = chunk.end_s
            else:
                merged.append(self._make_chunk(video_id, current_start, current_end))
                current_start = chunk.start_s
                current_end = chunk.end_s

        # Handle last accumulated segment
        last_duration = current_end - current_start
        if last_duration < self.config.min_chunk_s and merged:
            # Merge trailing small chunk into previous
            prev = merged.pop()
            merged.append(self._make_chunk(video_id, prev.start_s, current_end))
        else:
            merged.append(self._make_chunk(video_id, current_start, current_end))

        return merged

    def _make_chunk(self, video_id: str, start_s: float, end_s: float) -> Chunk:
        """Create a Chunk with computed fields."""
        start_ms = int(start_s * 1000)
        end_ms = int(end_s * 1000)
        return Chunk(
            chunk_id=f"{video_id}_{start_ms}_{end_ms}",
            start_ms=start_ms,
            end_ms=end_ms,
            start_s=start_s,
            end_s=end_s,
            duration_s=end_s - start_s
        )
