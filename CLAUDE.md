# CLAUDE.md

## Project Overview
WhisperX audio transcription pipeline that converts YouTube call recordings into multi-turn SFT training data for fine-tuning LLMs. Pipeline: Audio → VAD chunking → WhisperX ASR + diarization → call splitting → role classification → GenRM quality judging.

**Goal:** Generate high-quality agent/user conversation pairs from cold call recordings for supervised fine-tuning.

## AWS Access
- **Account:** 864981718771
- **IAM User:** claude-chat-app
- **Region:** us-east-1
- **CLI:** Configured via shared credentials file (~/.aws/credentials)

```bash
aws sts get-caller-identity  # Verify access
aws s3 ls s3://rezora-whisperx-us-east-1-864981718771/  # List bucket
```

## Quick Commands
```bash
# Speaker Labeler UI (localhost:5173)
cd apps/speaker-labeler && ./start.sh

# Run WhisperX pipeline
python -m pipeline.cli --video-id VIDEO_ID --audio-key s3://path/to.mp3

# Smoke test (validates GPU, models, HF_TOKEN)
python -m pipeline.cli --smoke-test

# GenRM workbench (Streamlit UI)
streamlit run scripts/genrm_workbench.py

# Run tests
pytest tests/
```

## Directory Structure
```
pipeline/           # Core Python pipeline
  genrm/            # GenRM quality judge (vLLM client, router, workbench)
apps/speaker-labeler/  # React + FastAPI labeling UI
scripts/            # Infrastructure and operational scripts
docker/whisperx/    # WhisperX Docker image
infra/              # AWS Batch configs, IAM policies
setup/              # EC2 setup scripts (GenRM, model download)
tests/              # pytest tests
```

## S3 Structure
**Bucket:** `rezora-whisperx-us-east-1-864981718771`
```
audio/pretraining/              # Source MP3s
runs/{video_id}/{run_id}/       # Pipeline output
  chunks/{chunk_id}/            # words.json, diarization_segments.json
  calls/{call_id}/              # spk_turns.json, call_metadata.json
latest/{video_id}.json          # Pointer to latest run
genrm/{accepted,review,rejected,role_fallback}/  # Routed SFT data
```

**Model Bucket:** `rezora-data-pipeline-864981718771`
```
models/qwen3-nemotron-32b/      # GenRM model backup
```

## GenRM Judge (Qwen3-Nemotron-32B)

### Overview
Quality judge that evaluates WhisperX output and routes to accepted/review/rejected buckets. Uses nvidia/Qwen3-Nemotron-32B-GenRM-Principle model on vLLM.

### EC2 Setup
- **Instance:** g5.24xlarge (4x A10G GPUs, 96GB VRAM)
- **Setup script:** `setup/setup_genrm_inference.sh`
- **Model download:** `setup/ec2-model-download-userdata.sh`

```bash
# On EC2: Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model ~/models/qwen3-nemotron-32b \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --port 8000
```

### Three Evaluation Principles
| Principle | Weight | Evaluates |
|-----------|--------|-----------|
| Integrity | 0.40 | Data authenticity, correct speaker attribution |
| SFT Value | 0.35 | Training value for cold call agent |
| Competence | 0.25 | Professional agent behavior |

### Routing Thresholds
| Score | Decision | S3 Path |
|-------|----------|---------|
| ≥0.80 | ACCEPT | `genrm/accepted/` |
| 0.60-0.79 | REVIEW | `genrm/review/` |
| <0.60 | REJECT | `genrm/rejected/` |
| Low confidence | ROLE_FALLBACK | `genrm/role_fallback/` |

### Current State
- ✅ EC2 setup scripts working
- ✅ vLLM serving with tensor parallelism
- ✅ 3-principle evaluation pipeline
- ✅ Routing to S3 buckets
- ⚠️ LLM fallback for role assignment not fully implemented
- ❌ Fine-tuning pipeline not yet built

## WhisperX Infrastructure

### EC2 Worker (whisperx-worker-1)
- **Instance:** g5.2xlarge (1x A10G GPU, 24GB VRAM)
- **IP:** 18.209.171.46
- **SSH Key:** whisperx-key

```bash
# Run pipeline on worker (from local machine)
python scripts/run_on_ec2.py --video-id VIDEO_ID

# Run batch of videos from selected_videos.txt
python scripts/run_on_ec2.py --execute --limit 5

# SSH directly to worker
ssh -i ~/.ssh/whisperx-key.pem ubuntu@18.209.171.46
```

### Docker Image
- **ECR:** `864981718771.dkr.ecr.us-east-1.amazonaws.com/whisperx-pipeline:latest`
- **Build:** `docker build -t whisperx-pipeline docker/whisperx/`

## SFT Data Format

**Input (WhisperX spk_turns.json):**
```json
{"turns": [{"spk": "SPEAKER_00", "text": "Hello...", "t0_abs": 0.0}]}
```

**Output (GenRM routed):**
```json
{
  "turns": [
    {"role": "assistant", "content": "Hello, I'm calling about..."},
    {"role": "user", "content": "Hi, who is this?"}
  ],
  "genrm_judgment": {"decision": "ACCEPT", "aggregate_score": 0.82}
}
```

## Speaker Labeler App
Two UIs at http://localhost:5173:
- `/boundary-editor` - Edit call start/end, export labels.json for ML training
- `/speaker-labeler` - Assign agent/user roles to speakers

**Keyboard shortcuts (BoundaryEditor):** Space=play, B=add boundary, D=delete, []=skip 5s, +-=zoom, arrows=nudge

## Code Conventions
- Files/functions: snake_case (`vad_chunker.py`, `process_video()`)
- Classes: PascalCase (`WhisperXPipeline`, `RolePredictorV2`)
- v2 suffix for newer implementations (`role_predictor_v2.py`)
- Dataclasses for typed data (`Chunk`, `WordOutput`, `CallBoundary`)
- Configuration in `pipeline/config.py` dataclasses

## Key Gotchas
- **PyTorch 2.6+ breaks pyannote:** `cli.py` monkey-patches `torch.load` - must happen BEFORE imports
- **HF_TOKEN required:** Diarization fails without HuggingFace token
- **Immutable chunks:** Never regenerate chunks within same run_id
- **60s feature window:** Role classifier only uses first 60s of call
- **GenRM principle order:** Messages must have "principle" role FIRST
- **v1 vs v2:** Always use v2 modules (`role_predictor_v2.py`)

## Environment Variables
```bash
# AWS
AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
AWS_REGION=us-east-1
S3_BUCKET=rezora-whisperx-us-east-1-864981718771

# HuggingFace (required for diarization)
HF_TOKEN=<huggingface_token>

# GenRM vLLM server
VLLM_BASE_URL=http://localhost:8000/v1
```

## Data Flow
```
MP3 → ffmpeg 16kHz WAV → Silero VAD chunks (30-600s)
    → WhisperX + pyannote → words.json + diarization
    → Call splitting (2.5s silence + greeting patterns)
    → Turn building → Role classification (3-class)
    → GenRM judge → Route to accepted/review/rejected
    → Multi-turn SFT data for fine-tuning
```

## Testing
```bash
pytest tests/                           # All tests
pytest tests/whisperx_pipeline/ -v      # Verbose
pytest --cov=pipeline --cov-report=html # Coverage
```

## Development Workflow (Claude Code)

### Committing Changes
After completing code changes:
- `/commit` - Create commits with auto-generated messages matching repo style
- `/commit-push-pr` - Commit, push, and create PR in one step
- `/clean_gone` - Clean up stale local branches (deleted from remote)

### Feature Development
For new features or substantial changes:
- `/feature-dev [description]` - Launch guided 7-phase workflow
- Phases: Discovery → Exploration → Questions → Design → Implementation → Review → Summary

Individual agents can be invoked directly:
- **code-explorer** - "Launch code-explorer to trace how [feature] works"
- **code-architect** - "Launch code-architect to design [component]"
- **code-reviewer** - "Launch code-reviewer to check my recent changes"

### Code Quality
- **Pyright LSP** - Automatic type checking on Python files (errors appear after edits)
- Fix type errors and unused variable warnings before committing

### Code Navigation (LSP vs Search)

**Use LSP for semantic code navigation:**
- `goToDefinition` - Find where a class/function is defined (avoids false positives in docs)
- `findReferences` - Find all usages of a symbol across the codebase
- `documentSymbol` - List all classes, methods, variables in a file with hierarchy
- `incomingCalls` - Find what functions call a given function
- `outgoingCalls` - Trace what a function calls (e.g., 37 calls from `process_video`)
- `hover` - Get type info and docstrings

**Use Grep/Glob for text search:**
- Search for text patterns, error messages, strings, comments
- Search non-Python files (JSON, YAML, Markdown, configs)
- Fuzzy/partial name search when exact symbol unknown
- Find files by naming pattern

**IMPORTANT: For Python code exploration, prefer LSP over Bash/Grep:**

| Task | Don't Use | Use Instead |
|------|-----------|-------------|
| Find where `CallBoundary` is defined | `grep -r "class CallBoundary"` | LSP `goToDefinition` |
| Find all usages of `load_video_data()` | `grep -r "load_video_data"` | LSP `findReferences` |
| List functions in a file | `grep "def "` or `cat` | LSP `documentSymbol` |
| Trace what calls `process_video()` | `grep "process_video("` | LSP `incomingCalls` |
| Trace what `process_video()` calls | Read file manually | LSP `outgoingCalls` |
| Get function signature/types | Read file | LSP `hover` |

LSP is faster, more accurate (ignores comments/strings), and understands Python semantics (inheritance, imports).

### AWS Operations
- **AWS CLI** - Configured for scripts and automation
- **AWS MCP** - Natural language AWS queries for interactive exploration
- Use CLI for scripts/CI, MCP for debugging and exploration
