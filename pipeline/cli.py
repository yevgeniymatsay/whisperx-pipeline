# scripts/whisperx_pipeline/cli.py
"""CLI entry point for WhisperX pipeline."""
import argparse
import json
import os
import sys
import numpy as np
import boto3

# Fix for PyTorch 2.6+ weights_only=True default breaking pyannote model loading
# Must be done before importing whisperx/pyannote
import torch

# Monkey-patch torch.load to FORCE weights_only=False for pyannote compatibility
# Required because pyannote checkpoints contain omegaconf objects
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False  # Force False, ignore any passed value
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Enable TF32 for 25-30% speedup on Ampere+ GPUs (A10G, A100)
# No effect on older GPUs (T4), safe to always enable
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from .config import PipelineConfig, S3_BUCKET
from .pipeline import WhisperXPipeline


def run_smoke_test():
    """Quick validation that all components are working. Catches 90% of GPU/config issues."""
    print("=== WhisperX Pipeline Smoke Test ===\n")

    # 1. Check torch + CUDA
    print("1. Checking PyTorch + CUDA...")
    import torch
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("   WARNING: CUDA not available, will run on CPU (slow)")

    # 2. Check HuggingFace token
    print("\n2. Checking HF_TOKEN...")
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print(f"   HF_TOKEN: set ({len(hf_token)} chars)")
    else:
        print("   ERROR: HF_TOKEN not set (required for diarization)")
        sys.exit(1)

    # 3. Load Whisper model
    print("\n3. Loading Whisper model...")
    import whisperx
    config = PipelineConfig()
    model = whisperx.load_model(config.whisperx.model, config.whisperx.device)
    print(f"   Whisper model: {config.whisperx.model} loaded")

    # 4. Load diarization pipeline
    print("\n4. Loading diarization pipeline...")
    from whisperx.diarize import DiarizationPipeline
    diarize = DiarizationPipeline(use_auth_token=hf_token, device=config.whisperx.device)
    print("   Diarization pipeline: loaded")

    # 5. Run transcription on dummy audio
    print("\n5. Running transcription on dummy audio...")
    dummy_audio = np.zeros(32000, dtype=np.float32)  # 2 seconds at 16kHz
    result = model.transcribe(dummy_audio, batch_size=1, language=config.whisperx.language)
    print(f"   Transcription: OK (language: {config.whisperx.language})")

    # 6. Run diarization on dummy audio (catches pyannote auth/model issues)
    print("\n6. Running diarization on dummy audio...")
    diarize_result = diarize(dummy_audio, min_speakers=1, max_speakers=2)
    # Handle both DataFrame (whisperx 3.x) and Annotation (older) return types
    if hasattr(diarize_result, 'itertracks'):
        segment_count = len(list(diarize_result.itertracks()))
    else:
        segment_count = len(diarize_result) if diarize_result is not None else 0
    print(f"   Diarization: OK ({segment_count} segments, 0 expected for silence)")

    print("\n=== Smoke Test PASSED ===")
    print("Pipeline is ready to process videos.")


def main():
    parser = argparse.ArgumentParser(description="WhisperX Transcription Pipeline")
    parser.add_argument("--video-id", help="Video ID to process")
    parser.add_argument("--run-id", help="Run ID (auto-generated if not provided)")
    parser.add_argument("--audio-key", help="S3 key for audio (default: audio/{video_id}.mp3)")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick validation of GPU/models")
    parser.add_argument("--no-db", action="store_true", help="Skip DynamoDB tracking")

    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()
        return

    if not args.video_id:
        parser.error("--video-id is required (unless using --smoke-test)")

    audio_key = args.audio_key or f"audio/{args.video_id}.mp3"

    config = PipelineConfig()
    pipeline = WhisperXPipeline(config)

    print(f"Processing video: {args.video_id}")
    manifest = pipeline.process_video(
        video_id=args.video_id,
        audio_s3_key=audio_key,
        run_id=args.run_id,
        track_in_db=not args.no_db,
    )

    print(f"Completed: {manifest['job_status']}")
    print(f"Chunks: {manifest['chunks']}")
    print(f"Calls: {manifest['calls']}")

    # Upload manifest
    s3 = boto3.client("s3")
    run_id = manifest['run_id']
    manifest_key = f"runs/{args.video_id}/{run_id}/video_manifest.json"
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=manifest_key,
        Body=json.dumps(manifest, indent=2)
    )

    # Two-step latest pointer update (immutable history + pointer)
    if manifest["job_status"] in ["completed_ok", "completed_with_errors"]:
        # Step 1: Write immutable history entry (never overwritten)
        history_key = f"latest/{args.video_id}/{run_id}.json"
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=history_key,
            Body=json.dumps({
                "run_id": run_id,
                "status": manifest["job_status"],
                "manifest_key": manifest_key
            })
        )

        # Step 2: Update pointer to latest run
        pointer_key = f"latest/{args.video_id}.json"
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=pointer_key,
            Body=json.dumps({"run_id": run_id, "status": manifest["job_status"]})
        )


if __name__ == "__main__":
    main()
