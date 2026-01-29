# scripts/whisperx_pipeline/chunk_writer.py
"""Write chunk and call outputs to local filesystem or S3."""
import json
from pathlib import Path
from typing import List, Any
from dataclasses import asdict

from .config import PipelineConfig
from .transcriber import WordOutput, DiarizationSegment
from .quality_metrics import QualityMetrics, QualityReason


class ChunkWriter:
    """Write chunk artifacts to filesystem."""

    def __init__(self, base_path: str, config: PipelineConfig):
        self.base_path = Path(base_path)
        self.config = config

    def write_chunk(
        self,
        video_id: str,
        chunk_id: str,
        chunk_time_offset_s: float,
        words: List[WordOutput],
        diarization_segments: List[DiarizationSegment],
        quality_metrics: QualityMetrics,
        quality_reasons: List[QualityReason],
        content_type: str,
        duration_s: float
    ) -> Path:
        """Write all chunk artifacts to chunks/<chunk_id>/."""
        chunk_dir = self.base_path / "chunks" / chunk_id
        chunk_dir.mkdir(parents=True, exist_ok=True)

        # words.json
        words_data = {
            "video_id": video_id,
            "chunk_id": chunk_id,
            "chunk_time_offset_s": chunk_time_offset_s,
            "pipeline_version": self.config.pipeline_version,
            "asr_model": f"whisper-{self.config.whisperx.model}",
            "diarizer_model": "pyannote-3.1",
            "words": [asdict(w) for w in words]
        }
        self._write_json(chunk_dir / "words.json", words_data)

        # diarization_segments.json
        dia_data = {
            "segments": [asdict(s) for s in diarization_segments]
        }
        self._write_json(chunk_dir / "diarization_segments.json", dia_data)

        # chunk_metadata.json
        meta_data = {
            "video_id": video_id,
            "chunk_id": chunk_id,
            "chunk_time_offset_s": chunk_time_offset_s,
            "duration_s": duration_s,
            "speaker_count": len(set(w.spk for w in words)),
            "quality_score": self._compute_quality_score(quality_metrics),
            "quality_metrics": asdict(quality_metrics),
            "quality_reasons": [asdict(r) for r in quality_reasons],
            "content_type": content_type
        }
        self._write_json(chunk_dir / "chunk_metadata.json", meta_data)

        # RTTM format for compatibility
        self._write_rttm(chunk_dir / "diarization.rttm", video_id, diarization_segments)

        return chunk_dir

    def _compute_quality_score(self, metrics: QualityMetrics) -> float:
        """Compute overall quality score from metrics."""
        score = 1.0
        score -= metrics.overlap_ratio * 0.5
        score -= metrics.micro_turn_ratio * 0.3
        score -= metrics.narrator_ratio * 0.2
        if metrics.speaker_flip_suspected:
            score -= 0.3
        return max(0.0, min(1.0, score))

    def _write_json(self, path: Path, data: dict):
        """Write JSON with consistent formatting."""
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _write_rttm(self, path: Path, video_id: str, segments: List[DiarizationSegment]):
        """Write RTTM format for compatibility."""
        with open(path, "w") as f:
            for seg in segments:
                duration = seg.t1_abs - seg.t0_abs
                f.write(f"SPEAKER {video_id} 1 {seg.t0_abs:.3f} {duration:.3f} <NA> <NA> {seg.spk} <NA> <NA>\n")


class CallWriter:
    """Write call artifacts to filesystem."""

    def __init__(self, base_path: str, config: PipelineConfig):
        self.base_path = Path(base_path)
        self.config = config

    def write_call(
        self,
        video_id: str,
        call_id: str,
        call_start_abs: float,
        call_end_abs: float,
        words: List[dict],
        spk_turns: List[Any],  # List[Turn]
        boundary_confidence: float,
        start_evidence: dict,
        end_evidence: dict
    ) -> Path:
        """Write call artifacts to calls/<call_id>/."""
        call_dir = self.base_path / "calls" / call_id
        call_dir.mkdir(parents=True, exist_ok=True)

        # call_words.json - words for this call
        words_data = {
            "video_id": video_id,
            "call_id": call_id,
            "call_start_abs": call_start_abs,
            "call_end_abs": call_end_abs,
            "word_count": len(words),
            "words": words
        }
        self._write_json(call_dir / "call_words.json", words_data)

        # spk_turns.json - turns with original speaker labels
        turns_data = {
            "video_id": video_id,
            "call_id": call_id,
            "call_start_abs": call_start_abs,
            "turn_count": len(spk_turns),
            "turns": [
                {
                    "turn_id": t.turn_id,
                    "spk": t.spk,
                    "text": t.text,
                    "t0_abs": t.t0_abs,
                    "t1_abs": t.t1_abs,
                    "word_span_start": t.word_span_start,
                    "word_span_end": t.word_span_end
                }
                for t in spk_turns
            ]
        }
        self._write_json(call_dir / "spk_turns.json", turns_data)

        # call_metadata.json
        meta_data = {
            "video_id": video_id,
            "call_id": call_id,
            "call_start_abs": call_start_abs,
            "call_end_abs": call_end_abs,
            "duration_s": call_end_abs - call_start_abs,
            "turn_count": len(spk_turns),
            "word_count": len(words),
            "boundary_confidence": boundary_confidence,
            "start_evidence": start_evidence,
            "end_evidence": end_evidence,
            "role_assignment": None  # Filled by role predictor later
        }
        self._write_json(call_dir / "call_metadata.json", meta_data)

        return call_dir

    def _write_json(self, path: Path, data: dict):
        """Write JSON with consistent formatting."""
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
