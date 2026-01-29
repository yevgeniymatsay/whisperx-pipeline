# scripts/whisperx_pipeline/pipeline.py
"""Main WhisperX pipeline orchestrator."""
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import boto3

from .config import PipelineConfig, S3_BUCKET
from .audio_preprocess import convert_to_wav, load_audio
from .vad_chunker import VADChunker, Chunk
from .transcriber import WhisperXTranscriber
from .quality_metrics import compute_quality_metrics, compute_narrator_ratio
from .call_splitter import CallSplitter
from .turn_builder import TurnBuilder
from .chunk_writer import ChunkWriter, CallWriter
from . import db


class WhisperXPipeline:
    """Orchestrate the full WhisperX transcription pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.s3 = boto3.client("s3")

    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def process_video(
        self,
        video_id: str,
        audio_s3_key: str,
        run_id: Optional[str] = None,
        conversation_type: Optional[str] = None,
        track_in_db: bool = True,
    ) -> dict:
        """
        Process a single video through the full pipeline.

        Args:
            video_id: Unique video identifier
            audio_s3_key: S3 key for audio file
            run_id: Optional run ID (generated if not provided)
            conversation_type: Type of conversation (pretraining, expired_listing).
                             If not provided, tries to infer from audio_s3_key.
            track_in_db: Whether to track progress in DynamoDB (default True)

        Returns:
            Video manifest dict
        """
        run_id = run_id or self._generate_run_id()
        run_prefix = f"runs/{video_id}/{run_id}"

        # Infer conversation_type from S3 key if not provided
        if conversation_type is None:
            if "pretraining" in audio_s3_key:
                conversation_type = "pretraining"
            elif "expired_listing" in audio_s3_key:
                conversation_type = "expired_listing"
            else:
                conversation_type = "pretraining"  # Default

        # Update video status in DynamoDB
        if track_in_db:
            try:
                db.update_video_status(video_id, "processing", run_id=run_id)
            except Exception as e:
                print(f"WARNING: Failed to update video status in DB: {e}")

        manifest = {
            "video_id": video_id,
            "run_id": run_id,
            "conversation_type": conversation_type,
            "pipeline_version": self.config.pipeline_version,
            "job_status": "running",
            "params": self._config_to_dict(),
            "chunks": {"total": 0, "ok": 0, "failed": 0},
            "calls": {"total": 0, "ok": 0, "failed": 0},
            "started_at": datetime.now().isoformat()
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Download and convert audio
            audio_path = tmpdir / "audio.wav"
            self._download_and_convert(audio_s3_key, audio_path)

            # Load audio
            audio = load_audio(str(audio_path))
            duration_s = len(audio) / 16000

            # VAD chunking
            vad = VADChunker(self.config.vad)
            speech_segments = vad.get_speech_segments(audio)
            chunks = vad.compute_chunks(speech_segments, duration_s, video_id)
            manifest["chunks"]["total"] = len(chunks)

            # Process each chunk
            transcriber = WhisperXTranscriber(self.config.whisperx)
            writer = ChunkWriter(str(tmpdir / "output"), self.config)

            all_words = []
            for chunk in chunks:
                try:
                    chunk_audio = audio[int(chunk.start_s * 16000):int(chunk.end_s * 16000)]
                    words, dia_segs = transcriber.transcribe(
                        chunk_audio,
                        chunk_time_offset_s=chunk.start_s
                    )

                    # Quality metrics
                    metrics, reasons = compute_quality_metrics(
                        [{"spk": w.spk, "t0_abs": w.t0_abs, "t1_abs": w.t1_abs, "text": w.text} for w in words],
                        [{"t0_abs": s.t0_abs, "t1_abs": s.t1_abs, "spk": s.spk} for s in dia_segs],
                        chunk.duration_s
                    )

                    # Determine content type
                    content_type = "narration_only" if metrics.narrator_ratio > self.config.quality.max_narrator_ratio else "call_like"

                    # Write chunk
                    writer.write_chunk(
                        video_id=video_id,
                        chunk_id=chunk.chunk_id,
                        chunk_time_offset_s=chunk.start_s,
                        words=words,
                        diarization_segments=dia_segs,
                        quality_metrics=metrics,
                        quality_reasons=reasons,
                        content_type=content_type,
                        duration_s=chunk.duration_s
                    )

                    # Collect words for call splitting
                    if content_type == "call_like":
                        all_words.extend([
                            {"spk": w.spk, "t0_abs": w.t0_abs, "t1_abs": w.t1_abs, "text": w.text, "text_norm": w.text_norm}
                            for w in words
                        ])

                    manifest["chunks"]["ok"] += 1

                except Exception as e:
                    manifest["chunks"]["failed"] += 1
                    import traceback
                    print(f"ERROR processing chunk {chunk.chunk_id}: {e}")
                    print(traceback.format_exc())

            # CRITICAL: Sort merged words by absolute time before splitting
            # Chunks may overlap or be processed out of order
            all_words.sort(key=lambda w: (w["t0_abs"], w["t1_abs"]))

            # Call splitting
            splitter = CallSplitter(self.config.split_calls, video_id)
            boundaries = splitter.find_boundaries(all_words)
            call_word_lists = splitter.split_words(all_words, boundaries)

            # Build turns and write call artifacts
            turn_builder = TurnBuilder()
            call_writer = CallWriter(str(tmpdir / "output"), self.config)
            manifest["calls"]["total"] = len(boundaries)

            for boundary, call_words in zip(boundaries, call_word_lists):
                turns = turn_builder.build_turns(call_words)

                # Write call artifacts to calls/<call_id>/
                call_writer.write_call(
                    video_id=video_id,
                    call_id=boundary.call_id,
                    call_start_abs=boundary.start_abs,
                    call_end_abs=boundary.end_abs,
                    words=call_words,
                    spk_turns=turns,
                    boundary_confidence=boundary.boundary_confidence,
                    start_evidence=boundary.start_evidence,
                    end_evidence=boundary.end_evidence
                )
                manifest["calls"]["ok"] += 1

                # Register call in DynamoDB
                if track_in_db:
                    try:
                        # Count unique speakers in turns
                        speakers = set(t.spk for t in turns)
                        s3_key = f"{run_prefix}/calls/{boundary.call_id}/spk_turns.json"

                        db.register_call(
                            call_id=boundary.call_id,
                            video_id=video_id,
                            run_id=run_id,
                            s3_key=s3_key,
                            turn_count=len(turns),
                            speaker_count=len(speakers),
                            conversation_type=conversation_type,
                        )
                    except Exception as e:
                        print(f"WARNING: Failed to register call {boundary.call_id} in DB: {e}")

            # Upload to S3
            self._upload_outputs(tmpdir / "output", run_prefix)

            manifest["job_status"] = "completed_ok" if manifest["chunks"]["failed"] == 0 else "completed_with_errors"
            manifest["completed_at"] = datetime.now().isoformat()

        # Update final video status in DynamoDB
        if track_in_db:
            try:
                final_status = "completed" if manifest["job_status"] == "completed_ok" else "failed"
                db.update_video_status(
                    video_id,
                    final_status,
                    run_id=run_id,
                    call_count=manifest["calls"]["ok"],
                )
            except Exception as e:
                print(f"WARNING: Failed to update final video status in DB: {e}")

        return manifest

    def _download_and_convert(self, s3_key: str, output_path: Path):
        """Download audio from S3 and convert to 16kHz mono WAV."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            self.s3.download_file(S3_BUCKET, s3_key, f.name)
            convert_to_wav(f.name, str(output_path))

    def _config_to_dict(self) -> dict:
        """Convert config to dict for manifest."""
        return {
            "vad": {
                "gap_threshold_s": self.config.vad.gap_threshold_s,
                "min_chunk_s": self.config.vad.min_chunk_s,
                "max_chunk_s": self.config.vad.max_chunk_s,
                "padding_s": self.config.vad.padding_s
            },
            "whisperx": {
                "model": self.config.whisperx.model,
                "batch_size": self.config.whisperx.batch_size,
                "compute_type": self.config.whisperx.compute_type
            },
            "split_calls": {
                "silence_threshold_s": self.config.split_calls.silence_threshold_s,
                "fuzzy_match_threshold": self.config.split_calls.fuzzy_match_threshold
            }
        }

    def _upload_outputs(self, local_dir: Path, s3_prefix: str):
        """Upload all outputs to S3."""
        for file_path in local_dir.rglob("*"):
            if file_path.is_file():
                relative = file_path.relative_to(local_dir)
                s3_key = f"{s3_prefix}/{relative}"
                self.s3.upload_file(str(file_path), S3_BUCKET, s3_key)
