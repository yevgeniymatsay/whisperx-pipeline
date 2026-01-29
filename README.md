# WhisperX Pipeline

High-quality audio transcription pipeline using WhisperX with speaker diarization, call splitting, role classification, and GenRM-based quality judging.

## Components

- **Speaker Labeler**: Web app for manual speaker role labeling + boundary correction
- **WhisperX Pipeline**: Transcription, VAD chunking, call splitting, role prediction
- **GenRM Judge**: Qwen3-Nemotron-32B for quality evaluation (requires g5.24xlarge)

## Setup

1. Copy `.env.example` to `.env` and fill in credentials
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## S3 Bucket

`s3://rezora-whisperx-us-east-1-864981718771/`

| Prefix | Description |
|--------|-------------|
| `audio/pretraining/` | Source MP3 files |
| `runs/{video_id}/` | Pipeline output per video |
| `models/qwen3-nemotron-32b/` | GenRM model backup |

## GenRM Judge (EC2 g5.24xlarge)

The GenRM judge runs on a g5.24xlarge (4x A10G GPUs) with vLLM:
- Model: nvidia/Qwen3-Nemotron-32B-GenRM-Principle
- Server: vLLM with tensor-parallel-size=4

See `setup/` for EC2 deployment scripts.

## Architecture

```
EC2 g5.24xlarge (GenRM Judge)
├── 4x NVIDIA A10G GPUs
├── vLLM Server (Qwen3-Nemotron-32B)
└── API: http://localhost:8000/v1

EC2 g4dn.xlarge (WhisperX Transcription)
├── 1x NVIDIA T4 GPU
└── Docker container with WhisperX

S3: rezora-whisperx-us-east-1-864981718771
├── audio/pretraining/ (source MP3s)
├── runs/{video_id}/ (pipeline output)
├── models/qwen3-nemotron-32b/ (model backup)
└── genrm/{accepted,review,rejected}/ (judged calls)

Local: whisperx-pipeline/
├── apps/speaker-labeler/ (React + FastAPI)
├── pipeline/ (WhisperX + GenRM)
└── scripts/ (CLI tools)
```

## Quick Start

### Run Speaker Labeler
```bash
cd apps/speaker-labeler
./start.sh
```

### Run GenRM Workbench
```bash
streamlit run scripts/genrm_workbench.py
```

### Run WhisperX Pipeline
```bash
python -m pipeline.cli --video-id VIDEO_ID --run-id RUN_ID --audio-key audio/pretraining/file.mp3
```
