# WhisperX Transcription Pipeline Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:writing-plans to create the implementation plan after this design is approved.

**Goal:** Replace AWS Transcribe with a WhisperX-based pipeline that produces better speaker diarization through VAD pre-splitting and pyannote, with a trained role classifier to auto-assign agent/user roles.

**Architecture:** Audio → VAD chunks → WhisperX + pyannote → canonical word stream → deterministic call splitting → role classifier → judge/fixer/router pipeline.

**Tech Stack:** WhisperX, pyannote 3.1, Silero VAD, AWS Batch (GPU), logistic regression role classifier.

---

## 1. Architecture Overview

The pipeline replaces AWS Transcribe with a WhisperX-based system that produces better speaker diarization through VAD pre-splitting and pyannote. It runs on an on-demand EC2 GPU instance via AWS Batch, triggered by job submission.

### Data Flow

```
S3/audio/*.mp3
    ↓ (submit_jobs.py → AWS Batch)
AWS Batch GPU Instance:
    ↓ ffmpeg → 16kHz mono WAV
    ↓ Silero VAD → chunks (≥2s silence, min 30s, max 10min, ±0.5s padding)
    ↓ WhisperX per chunk (ASR + alignment + pyannote)
    ↓ Output: canonical words.json (never regenerate)
S3/runs/<video_id>/<run_id>/chunks/
    ↓ split_calls.py (deterministic call boundaries)
S3/runs/<video_id>/<run_id>/calls/
    ↓ role_classifier (trained from bootstrap labels)
    ↓ High confidence → auto-assign roles
    ↓ Low confidence → LLM fallback
S3/runs/<video_id>/<run_id>/calls/ (with assistant/user roles)
    ↓ latest/<video_id>.json pointer updated
    ↓ existing judge/fixer/router pipeline
```

### Key Changes from Current Pipeline

| Component | Current (AWS Transcribe) | New (WhisperX) |
|-----------|-------------------------|----------------|
| Transcription | AWS Transcribe | Whisper large-v2 |
| Diarization | AWS internal (MaxSpeakerLabels) | pyannote 3.1 |
| Long audio handling | Full file (drift) | VAD pre-split (2-10 min chunks) |
| Call splitting | Extractor LLM | Deterministic `split_calls.py` |
| Role mapping | LLM inference | Trained classifier + LLM fallback |

---

## 2. Pipeline Stages

### Stage 1: Audio Preprocessing
- Input: `S3/audio/<video_id>.mp3`
- ffmpeg → 16kHz mono WAV (temp file on instance)
- Output: WAV ready for VAD

### Stage 2: VAD Chunking (Compute Optimization Only)
- Silero VAD splits on ≥2s silence gaps
- Min chunk size: 30 seconds
- Max chunk size: 10 minutes
- Padding: ±0.5s around cuts (preserves plosives/word starts)
- Purpose: Prevent WhisperX/pyannote drift on long audio
- **These are NOT calls** - just compute-friendly segments

### Stage 3: WhisperX + pyannote (Per Chunk)
- Whisper large-v2 ASR → segment-level transcript
- Forced word alignment → word-level timestamps
- pyannote diarization → speaker segments
- Word ∩ diarization → speaker label per word
- Output per chunk:
  - `words.json` (canonical - never regenerate)
  - `diarization_segments.json`
  - `diarization.rttm` (compatibility)
  - `chunk_metadata.json`

### Stage 4: Quality Gate + Narration Filter
- Quality gate: numeric score + reason codes (not boolean)
- Narration filter: mark regions as `call_like` vs `narration_like`
- Flags stored in `chunk_metadata.json`
- Low quality → route to expensive path / manual review

### Stage 5: Call Splitting (`split_calls.py`)
- Operates on merged word streams across VAD boundaries
- **MVP boundaries:** Silence gaps + lexical resets (greeting patterns)
- Speaker-embedding change points: Phase 2 hardening (optional)
- Output per call:
  - `call_words.json` (flattened with backrefs)
  - `boundary_confidence` + `boundary_evidence`

### Stage 5.5: Turn Building
- Build `spk_turns.json` from call words (speaker-based)
- Happens before role assignment
- Can regenerate when turn rules change

### Stage 6: Role Assignment
- Role classifier uses first 60s features
- Predicts which speaker is agent via argmax
- Confidence = |p_spk0 - p_spk1| (probability gap)
- High confidence (gap ≥ threshold) → auto-assign `assistant`/`user`
- Low confidence → LLM fallback
- Output: `role_turns.json` with `role_confidence` + `decision_features`

---

## 3. Data Schemas

### Canonical Layer (Never Regenerate)

**words.json (per chunk)**
```json
{
  "video_id": "abc123",
  "chunk_id": "abc123_0_185000",
  "chunk_time_offset_s": 0.0,
  "pipeline_version": "1.0.0",
  "asr_model": "whisper-large-v2",
  "diarizer_model": "pyannote-3.1",
  "words": [
    {
      "i": 0,
      "t0": 0.52,
      "t1": 0.78,
      "t0_abs": 0.52,
      "t1_abs": 0.78,
      "text": "Hello",
      "text_norm": "hello",
      "spk": "SPEAKER_00"
    }
  ]
}
```

**diarization_segments.json (per chunk)**
```json
{
  "segments": [
    {"t0_abs": 0.5, "t1_abs": 12.4, "spk": "SPEAKER_00"},
    {"t0_abs": 12.8, "t1_abs": 18.1, "spk": "SPEAKER_01"}
  ]
}
```

**chunk_metadata.json (per chunk)**
```json
{
  "video_id": "abc123",
  "chunk_id": "abc123_0_185000",
  "chunk_time_offset_s": 0.0,
  "duration_s": 185.0,
  "speaker_count": 2,
  "quality_score": 0.82,
  "quality_metrics": {
    "overlap_ratio": 0.09,
    "speaker_switches_per_min": 38,
    "median_turn_s": 0.8,
    "micro_turn_ratio": 0.12
  },
  "quality_reasons": [
    {"type": "high_overlap", "t0_abs": 45.0, "t1_abs": 52.0, "severity": 0.7}
  ],
  "content_type": "call_like"
}
```

### Derived Layer (Regeneratable)

**call_words.json (per call, flattened with backrefs)**
```json
{
  "video_id": "abc123",
  "call_id": "abc123_125020_298440",
  "pipeline_version": "1.0.0",
  "split_calls_version": "1.0.0",
  "call_start_abs": 125.02,
  "call_end_abs": 298.44,
  "source_chunks": ["abc123_0_185000", "abc123_185000_370000"],
  "has_missing_audio": false,
  "missing_regions": [],
  "boundary_confidence": 0.91,
  "boundary_evidence": {
    "start": {"type": "greeting_reset", "phrase_debug": "hi is this", "t_abs": 125.02},
    "end": {"type": "silence_gap", "gap_s": 3.2, "t_abs": 298.44}
  },
  "words": [
    {"call_i": 0, "chunk_id": "abc123_0_185000", "chunk_i": 0, "t0_abs": 125.02, "t1_abs": 125.28, "text": "Hi", "text_norm": "hi", "spk": "SPEAKER_00"}
  ]
}
```

**spk_turns.json (speaker-based, before role mapping)**
```json
{
  "video_id": "abc123",
  "call_id": "abc123_125020_298440",
  "pipeline_version": "1.0.0",
  "turns": [
    {
      "turn_id": 0,
      "spk": "SPEAKER_00",
      "text": "Hi is this John?",
      "t0_abs": 125.02,
      "t1_abs": 127.1,
      "word_span": {"call_i_start": 0, "call_i_end": 4}
    }
  ]
}
```

**role_turns.json (final output)**
```json
{
  "video_id": "abc123",
  "call_id": "abc123_125020_298440",
  "pipeline_version": "1.0.0",
  "role_model_version": "1.0.0",
  "role_mapping": {"SPEAKER_00": "assistant", "SPEAKER_01": "user"},
  "role_confidence": 0.96,
  "decision_features": {
    "diff_talk_time": 12.5,
    "diff_turn_count": 1,
    "diff_question_turn_rate": 0.15
  },
  "spk_turns_ref": "s3://bucket/runs/abc123/.../spk_turns.json",
  "turns": [
    {"role": "assistant", "text": "Hi is this John?", "t0_abs": 125.02, "t1_abs": 127.1, "word_span": {"call_i_start": 0, "call_i_end": 4}}
  ]
}
```

---

## 4. Error Handling & Routing

### Idempotency Keys (Collision-Proof)

```python
chunk_id = f"{video_id}_{chunk_start_ms}_{chunk_end_ms}"
call_id = f"{video_id}_{call_start_ms}_{call_end_ms}"
```

### Run Isolation (No In-Place Overwrites)

Each pipeline run writes to a unique prefix:
```
s3://bucket/runs/<video_id>/<run_id>/chunks/...
s3://bucket/runs/<video_id>/<run_id>/calls/...
s3://bucket/runs/<video_id>/<run_id>/video_manifest.json
```

Latest pointer (updated only on `completed_*` status):
```
s3://bucket/latest/<video_id>/<run_id>.json  # immutable
s3://bucket/latest/<video_id>.json           # pointer
→ {"run_id": "20260126_143052", "status": "completed_ok"}
```

### Parameter Persistence

**video_manifest.json:**
```json
{
  "video_id": "abc123",
  "run_id": "20260126_143052",
  "pipeline_version": "1.0.0",
  "job_status": "completed_ok",
  "params": {
    "vad": {"gap_threshold_s": 2.0, "min_chunk_s": 30, "max_chunk_s": 600, "padding_s": 0.5},
    "whisperx": {"model": "large-v2", "batch_size": 16, "compute_type": "float16"},
    "split_calls": {"silence_threshold_s": 2.5, "greeting_patterns": ["hi is this", "calling from"]},
    "role_classifier": {"confidence_threshold": 0.25, "model_version": "1.0.0"}
  },
  "chunks": {"total": 12, "ok": 11, "failed": 1},
  "calls": {"total": 8, "auto_complete": 6, "llm_fallback": 1, "incomplete": 1},
  "missing_regions": [{"t0_abs": 185.5, "t1_abs": 370.0, "reason": "chunk_failed"}]
}
```

### Chunk Status (Append-Only Log)

**chunk_events.jsonl:**
```jsonl
{"chunk_id": "abc123_0_185000", "event": "started", "ts": "..."}
{"chunk_id": "abc123_0_185000", "event": "completed", "ts": "..."}
{"chunk_id": "abc123_185000_370000", "event": "retry", "attempt": 2, "error": "OOM", "ts": "..."}
{"chunk_id": "abc123_185000_370000", "event": "failed", "error": "OOM after 2 retries", "ts": "..."}
```

**chunk_status.json (derived):**
```json
{
  "abc123_0_185000": {"status": "ok"},
  "abc123_185000_370000": {"status": "failed", "error": "OOM"}
}
```

### Retry Policy

| Error Type | Retry Strategy |
|------------|----------------|
| GPU OOM | Retry 1: lower batch_size. Retry 2: split chunk in half + lower batch. Then fail. |
| WhisperX transient | Retry 3x with backoff |
| S3 IO | Retry 5x with exponential backoff |
| Bad audio decode | **Fail fast** |
| pyannote load failure | **Fail job**, status = `MISCONFIGURED` |

### Quality Routing

**After WhisperX (per chunk):**
```python
if quality_score < 0.4:
    chunk_status = "needs_review"
elif narrator_ratio > 0.7:
    chunk_status = "narration_only"
else:
    chunk_status = "ok"
```

**After role assignment (per call):**
```python
if role_confidence_gap < 0.25:
    route_to = "llm_fallback"
elif quality_score < 0.6:
    route_to = "needs_review"
else:
    route_to = "auto_complete"
```

---

## 5. AWS Batch Orchestration

### Compute Environment

**GPU Instance:** `g5.xlarge` (1x A10G, 24GB VRAM)
- Cost: ~$1/hr on-demand, ~$0.35/hr spot

**Scaling:**
```
min vCPUs: 0
max vCPUs: 16 (4 concurrent jobs)
```

**Spot Strategy:** Use Spot with controller-based fallback to On-Demand.

### Job Definition

```yaml
containerProperties:
  image: {account}.dkr.ecr.us-east-2.amazonaws.com/whisperx-pipeline:latest

  resourceRequirements:
    - type: GPU
      value: "1"
    - type: VCPU
      value: "4"
    - type: MEMORY
      value: "16384"

  secrets:
    - name: HF_TOKEN
      valueFrom: arn:aws:secretsmanager:us-east-2:{account}:secret:hf-token

  environment:
    - name: S3_BUCKET
      value: rezora-data-pipeline-864981718771

  logConfiguration:
    logDriver: awslogs
    options:
      awslogs-group: /aws/batch/whisperx-pipeline
      awslogs-region: us-east-2
      awslogs-stream-prefix: whisperx

  command:
    - python
    - /app/transcribe_video.py
    - --video-id
    - Ref::video_id
    - --run-id
    - Ref::run_id
```

### Job Queue + Retry

```yaml
Queue: whisperx-transcription-queue
  Compute Environments:
    - whisperx-gpu-spot (priority 1)
    - whisperx-gpu-ondemand (priority 2)

retryStrategy:
  attempts: 2
  evaluateOnExit:
    - onExitCode: 137  # OOM/SIGKILL
      action: RETRY
    - onExitCode: 1    # Application error - internal retry handles it
      action: EXIT
```

### Spot → On-Demand Fallback Controller

Batch doesn't auto-fallback after N minutes. Use a Lambda controller:
```
Every 5 min:
  - Check jobs RUNNABLE > 10 min
  - If no Spot capacity: cancel, resubmit to On-Demand queue
  - Write status = resubmitted_to_ondemand
```

### Job Submission

**submit_jobs.py:**
```python
def submit_transcription_job(video_id: str) -> str:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    response = batch.submit_job(
        jobName=f"transcribe-{video_id}-{run_id}",
        jobQueue="whisperx-transcription-queue",
        jobDefinition="whisperx-pipeline",
        parameters={"video_id": video_id, "run_id": run_id},
        timeout={"attemptDurationSeconds": 7200}
    )
    return response["jobId"]
```

### Cost Controls

| Control | Setting |
|---------|---------|
| Per-job timeout | 2 hours |
| Max concurrent jobs | 4 |
| Alarm: stuck jobs | > 3 hours |
| Alarm: daily spend | > $50/day |

### CloudWatch Metrics (Cardinality-Safe)

Aggregate metrics only (no per-video dimensions):
```python
cloudwatch.put_metric_data(
    Namespace="WhisperXPipeline",
    MetricData=[
        {"MetricName": "VideosProcessed", "Value": 1},
        {"MetricName": "ChunksProcessed", "Value": 12},
        {"MetricName": "ChunksFailed", "Value": 1},
        {"MetricName": "TotalProcessingSeconds", "Value": 842}
    ]
)
```

Per-video details: CloudWatch Logs + S3 `video_manifest.json`.

---

## 6. Bootstrap Role Classifier

### Goal

Predict which diarized speaker (`SPEAKER_00` vs `SPEAKER_01`) is the agent/caller for each extracted call.

**Targets:**
- ≥97% accuracy at high-confidence threshold
- ≥85% of calls auto-assigned (≤15% to LLM fallback)

### Training Data Source

- Input: `spk_turns.json` from pipeline output
- Use only **first 60 seconds** of each call
- Split train/val by **video_id** (no leakage)

### Auto-Sampling Rules (`make_role_label_set.py`)

```python
def is_clean_candidate(call_metadata, spk_turns, call_start_abs) -> bool:
    first_60s_turns = [t for t in spk_turns if t["t0_abs"] < call_start_abs + 60]
    speakers = set(t["spk"] for t in first_60s_turns)
    total_words = sum(len(t["text"].split()) for t in first_60s_turns)

    return (
        len(speakers) == 2 and
        total_words >= 80 and
        len(first_60s_turns) >= 6 and
        call_metadata.get("overlap_ratio", 0) < 0.15 and
        not call_metadata.get("has_missing_audio", False) and
        call_metadata.get("content_type") != "narration_only"
    )
```

**Output:** `role_label_queue.jsonl` (200-500 candidates from ~50 videos)

### Labeling UX (`label_roles.py`)

```
─────────────────────────────────────────────
Call: abc123_125020_298440 (1/247)
─────────────────────────────────────────────
SPEAKER_00: Hi is this John?
SPEAKER_01: Yeah speaking.
SPEAKER_00: Cool, this is Mike calling from ABC Insurance...
─────────────────────────────────────────────
Who is the AGENT/CALLER?
  [0] SPEAKER_00    [1] SPEAKER_01    [s] Skip    [b] Back
>
```

**Output:** `role_labels.csv`

### Feature Extraction (`extract_role_features.py`)

**Per-speaker features (first 60s, filtered explicitly):**
```python
def extract_speaker_features(all_turns: list, speaker: str, call_start_abs: float) -> dict:
    # Explicit first-60s filtering
    first_60s_turns = [t for t in all_turns if t["t0_abs"] < call_start_abs + 60]
    spk_turns = [t for t in first_60s_turns if t["spk"] == speaker]

    # Fixed speaker ordering
    first_turn_spk = first_60s_turns[0]["spk"] if first_60s_turns else None

    durations = [t["t1_abs"] - t["t0_abs"] for t in spk_turns]
    word_counts = [len(t["text"].split()) for t in spk_turns]
    turn_count = len(spk_turns)
    question_count = sum(1 for t in spk_turns if is_question(t["text"]))

    return {
        "talk_time": sum(durations) if durations else 0,
        "turn_count": turn_count,
        "avg_turn_duration": safe_mean(durations),
        "median_turn_duration": safe_median(durations),
        "max_turn_duration": safe_max(durations),
        "question_turn_count": question_count,
        "question_turn_rate": question_count / turn_count if turn_count > 0 else 0,
        "long_turn_count": sum(1 for d in durations if d > 5),
        "short_turn_count": sum(1 for d in durations if d < 0.6),
        "word_count": sum(word_counts) if word_counts else 0,
        "avg_words_per_turn": safe_mean(word_counts),
        "first_speaker": 1 if speaker == first_turn_spk else 0,
    }

def is_question(text: str) -> bool:
    """Robust question detection."""
    text_lower = text.strip().lower()
    if text.rstrip().endswith("?"):
        return True
    words = text_lower.split()[:3]
    interrogatives = {"is", "are", "do", "did", "can", "could", "would", "what", "why", "how", "when", "where", "who"}
    return any(w in interrogatives for w in words)
```

**Call-level difference features:**
```python
def extract_call_features(call_id: str, spk_turns: list, call_start_abs: float) -> dict:
    # Hardcoded speaker ordering
    spk0, spk1 = "SPEAKER_00", "SPEAKER_01"

    feat0 = extract_speaker_features(spk_turns, spk0, call_start_abs)
    feat1 = extract_speaker_features(spk_turns, spk1, call_start_abs)

    diff = {f"diff_{k}": feat0[k] - feat1[k] for k in feat0}
    return {"call_id": call_id, "spk0": spk0, "spk1": spk1, **diff}
```

### Model Training (`train_role_model.py`)

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Video-level split (no leakage)
train_videos, val_videos = train_test_split(video_ids, test_size=0.2)
train_mask = features["video_id"].isin(train_videos)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression())
])

X = features[["diff_talk_time", "diff_turn_count", "diff_avg_turn_duration",
              "diff_question_turn_rate", "diff_avg_words_per_turn", ...]]
y = features["agent_is_spk0"]

model.fit(X[train_mask], y[train_mask])

# Calibration
probas = model.predict_proba(X[~train_mask])[:, 1]
confidence_gaps = np.abs(probas - 0.5) * 2
pred_is_spk0 = (probas >= 0.5).astype(int)

for gap_threshold in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
    high_conf_mask = confidence_gaps >= gap_threshold
    accuracy = (pred_is_spk0[high_conf_mask] == y_val[high_conf_mask]).mean()
    coverage = high_conf_mask.mean()
    print(f"Gap {gap_threshold}: accuracy={accuracy:.3f}, coverage={coverage:.3f}")
```

**Select `gap_threshold` where:** accuracy ≥ 97%, coverage maximized (≥85%).

### Deployment

**Artifacts:**
```
s3://bucket/models/role_classifier/
  role_model.pkl
  feature_schema.json
  training_metadata.json
```

**training_metadata.json:**
```json
{
  "role_model_version": "1.0.0",
  "trained_at": "2026-01-26T16:00:00",
  "training_samples": 412,
  "gap_threshold": 0.25,
  "val_accuracy": 0.973,
  "val_coverage": 0.86,
  "features": ["diff_talk_time", "diff_turn_count", ...]
}
```

---

## 7. Success Criteria

### Structural Transcript Quality

Measure on 100-call evaluation set (40 worst + 60 random):

| Metric | Target |
|--------|--------|
| Merged-turn contamination rate | <2% of calls |
| Speaker swap/drift rate | <3% of calls (<1% on clean audio) |
| Micro-turn ping-pong rate | 50%+ reduction vs AWS |

### Role Mapping Accuracy

| Metric | Target |
|--------|--------|
| High-confidence accuracy | ≥97% |
| High-confidence coverage | ≥85% |

### Business Outcomes

| Metric | Target |
|--------|--------|
| Manual review hours per 100 calls | 5× reduction |
| LLM fixer usage rate | <15% |
| Manual review rate | <5% |

### Ship Criteria

Ship WhisperX pipeline if:
- Mixed-speaker/swap errors are at least 3× lower than AWS on eval set
- LLM routing drops to ≤15%
- Manual review time drops ≥5× on 500-call batch

---

## 8. S3 Structure

```
s3://rezora-data-pipeline-864981718771/
├── audio/
│   └── {title} - {video_id}.mp3
├── runs/
│   └── {video_id}/
│       └── {run_id}/
│           ├── chunks/
│           │   └── {chunk_id}/
│           │       ├── words.json
│           │       ├── diarization_segments.json
│           │       ├── diarization.rttm
│           │       └── chunk_metadata.json
│           ├── calls/
│           │   └── {call_id}/
│           │       ├── call_words.json
│           │       ├── spk_turns.json
│           │       └── role_turns.json
│           ├── chunk_events.jsonl
│           ├── chunk_status.json
│           └── video_manifest.json
├── latest/
│   ├── {video_id}/
│   │   └── {run_id}.json
│   └── {video_id}.json
└── models/
    └── role_classifier/
        ├── role_model.pkl
        ├── feature_schema.json
        └── training_metadata.json
```

---

## 9. Implementation Phases

### Phase 1: MVP Pipeline (Weeks 1-2)
- Docker image with WhisperX + pyannote
- VAD splitting + WhisperX transcription
- Canonical word stream output
- Basic `split_calls.py` (silence + greeting resets)
- Manual role labeling (no classifier yet)
- AWS Batch job definition

### Phase 2: Role Classifier (Week 3)
- Run pipeline on 50 videos
- Auto-sample 200-500 clean candidates
- Label roles via CLI tool
- Train + calibrate classifier
- Integrate into pipeline

### Phase 3: Production Hardening (Week 4)
- Quality gate + narration filter
- Spot fallback controller
- CloudWatch metrics + alarms
- Evaluation on 100-call set
- Ship decision

### Phase 4: Future Improvements (Post-Ship)
- Speaker-embedding change points for call splitting
- NeMo MSDD if pyannote insufficient
- Active learning for role classifier
