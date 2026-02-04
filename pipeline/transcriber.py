# scripts/whisperx_pipeline/transcriber.py
"""WhisperX transcription with diarization."""
import os
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from .config import WhisperXConfig
from .lexical_matcher import tokenize


@dataclass
class WordOutput:
    i: int
    t0: float
    t1: float
    t0_abs: float
    t1_abs: float
    text: str
    text_norm: str
    spk: str


@dataclass
class DiarizationSegment:
    t0_abs: float
    t1_abs: float
    spk: str


class WhisperXTranscriber:
    """Transcribe audio using WhisperX with pyannote diarization."""

    def __init__(self, config: WhisperXConfig):
        self.config = config
        self._model = None
        self._align_model = None
        self._align_metadata = None
        self._diarize_pipeline = None

    def _load_models(self):
        """Lazy load WhisperX models."""
        if self._model is not None:
            return

        import whisperx

        self._model = whisperx.load_model(
            self.config.model,
            self.config.device,
            compute_type=self.config.compute_type
        )

    def _load_align_model(self, language: str):
        """Load alignment model for language."""
        import whisperx

        if self._align_model is None:
            self._align_model, self._align_metadata = whisperx.load_align_model(
                language_code=language,
                device=self.config.device
            )
        return self._align_model, self._align_metadata

    def _load_diarize_pipeline(self):
        """Load pyannote diarization pipeline."""
        if self._diarize_pipeline is not None:
            return self._diarize_pipeline

        from whisperx.diarize import DiarizationPipeline

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable required for diarization")

        self._diarize_pipeline = DiarizationPipeline(
            use_auth_token=hf_token,
            device=self.config.device
        )
        return self._diarize_pipeline

    def transcribe(
        self,
        audio: np.ndarray,
        chunk_time_offset_s: float = 0.0,
        min_speakers: int = 1,
        max_speakers: int = 5
    ) -> Tuple[List[WordOutput], List[DiarizationSegment]]:
        """
        Transcribe audio chunk with diarization.

        Args:
            audio: Audio as numpy array (16kHz mono)
            chunk_time_offset_s: Offset to add for absolute timestamps
            min_speakers: Minimum expected speakers
            max_speakers: Maximum expected speakers

        Returns:
            Tuple of (words, diarization_segments)
        """
        import whisperx

        self._load_models()

        # 1. ASR (language forced via config to skip detection)
        result = self._model.transcribe(
            audio,
            batch_size=self.config.batch_size,
            language=self.config.language
        )
        language = self.config.language

        # 2. Align
        align_model, align_metadata = self._load_align_model(language)
        result = whisperx.align(
            result["segments"],
            align_model,
            align_metadata,
            audio,
            self.config.device,
            return_char_alignments=False
        )

        # 3. Diarize
        diarize_pipeline = self._load_diarize_pipeline()
        diarize_segments = diarize_pipeline(
            audio,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )

        # 4. Assign speakers to words
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # 5. Convert to output format
        words = []
        for seg in result.get("segments", []):
            for word_data in seg.get("words", []):
                t0 = word_data.get("start", 0)
                t1 = word_data.get("end", t0)
                text = word_data.get("word", "")
                spk = word_data.get("speaker", "SPEAKER_00")

                tokens = tokenize(text)
                text_norm = tokens[0] if tokens else ""

                words.append(WordOutput(
                    i=len(words),
                    t0=t0,
                    t1=t1,
                    t0_abs=t0 + chunk_time_offset_s,
                    t1_abs=t1 + chunk_time_offset_s,
                    text=text.strip(),
                    text_norm=text_norm,
                    spk=spk
                ))

        # Extract diarization segments (handle both DataFrame and pyannote Annotation)
        dia_segments = []
        if hasattr(diarize_segments, 'itertracks'):
            # pyannote Annotation format
            for seg in diarize_segments.itertracks(yield_label=True):
                turn, _, speaker = seg
                dia_segments.append(DiarizationSegment(
                    t0_abs=turn.start + chunk_time_offset_s,
                    t1_abs=turn.end + chunk_time_offset_s,
                    spk=speaker
                ))
        else:
            # DataFrame format (newer whisperx versions)
            for _, row in diarize_segments.iterrows():
                dia_segments.append(DiarizationSegment(
                    t0_abs=row['start'] + chunk_time_offset_s,
                    t1_abs=row['end'] + chunk_time_offset_s,
                    spk=row['speaker']
                ))

        return words, dia_segments
