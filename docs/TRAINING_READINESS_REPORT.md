# Training Readiness Report: Call-Boundary Classifier

**Repository:** whisperx-pipeline
**Date:** 2026-01-29
**Labeled Videos:** 51
**Target Scale:** 1000+ videos

---

## 1) Executive Summary

### What's Ready

| Component | Status | Evidence |
|-----------|--------|----------|
| Manual boundary labeling UI | ✅ Ready | S3-backed persistence, validation, export endpoint |
| Dataset builder script | ✅ Ready | Generates windowed parquet with 9 features |
| Export endpoint | ✅ Ready | `GET /api/export/labels` produces ML-ready format |
| Timestamps | ✅ Verified ABSOLUTE | Code evidence in transcriber.py:147-156 |
| Diarization features | ✅ Ready | 9 features computed per window |
| Host detection | ✅ Ready | Time-bin coverage algorithm |

### What's Missing / At Risk

| Issue | Severity | Location | Impact |
|-------|----------|----------|--------|
| Run ID not pinned | HIGH | build_call_segmenter_dataset.py:467-471 | Non-reproducible datasets |
| MP3 duration requires download | CRITICAL | build_call_segmenter_dataset.py:289-306 | 10-30 hours for 1000 videos |
| min/max speakers hardcoded | LOW | transcriber.py:86-87 | Always 2-3, not configurable |
| Leaky feature included | LOW | chunk_content_type in parquet | Must exclude from training |
| No drift threshold enforced | MEDIUM | Must pass `--exclude-drift-above` manually |

### Top 5 Risks

| # | Risk | Severity | Mitigation |
|---|------|----------|------------|
| 1 | **MP3 download bottleneck** | CRITICAL | Store duration in S3 metadata during upload |
| 2 | **Run ID drift** | HIGH | Add `--pin-run-ids` flag or save index.jsonl |
| 3 | **N+1 S3 listing in list_videos()** | MEDIUM | Refactor to single-pass aggregation |
| 4 | **Leaky chunk_content_type feature** | LOW | Already flagged in code comments, remove before training |
| 5 | **Hardcoded speaker limits** | LOW | Add to WhisperXConfig |

### Training Readiness Verdict

**✅ Ready to Train on 51 Videos: YES**

The infrastructure is complete for training on the current 51 labeled videos. All timestamps are verified absolute, the dataset builder generates proper windowed features, and the export endpoint produces ML-ready labels.

**For 1000+ Videos: Requires MP3 Duration Fix First**

The MP3 duration bottleneck makes large-scale training impractical without modification.

---

## 2) System Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           WHISPERX PIPELINE                                  │
└─────────────────────────────────────────────────────────────────────────────┘

                              SOURCE DATA
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  S3: audio/pretraining/{title} - {video_id}.mp3                             │
│       └─→ Full MP3 files (50-200MB each)                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: Audio Preprocessing                                                │
│  ─────────────────────────────                                              │
│  File: pipeline/audio_preprocess.py                                         │
│  Action: ffmpeg → 16kHz mono WAV                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: VAD Chunking                                                       │
│  ─────────────────────                                                       │
│  File: pipeline/vad_chunker.py                                              │
│  Model: Silero VAD (torch.hub)                                              │
│  Config: gap=2.0s, min=30s, max=600s, padding=0.5s                          │
│  Output: List[Chunk] with chunk_id = {video_id}_{start_ms}_{end_ms}         │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: Transcription + Diarization (per chunk)                            │
│  ────────────────────────────────────────────────                            │
│  File: pipeline/transcriber.py                                              │
│  ASR: whisper-large-v3 (batch=32, float16, en)                              │
│  Alignment: whisperx.align() → word timestamps                              │
│  Diarization: pyannote 3.1 (min=2, max=3 speakers)                          │
│  Output: words.json, diarization_segments.json                              │
│          └─→ t0_abs/t1_abs = ABSOLUTE timestamps                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: Quality Classification                                             │
│  ──────────────────────────────                                              │
│  File: pipeline/quality_metrics.py                                          │
│  Metrics: narrator_ratio, overlap_ratio, speaker_flip_suspected             │
│  Decision: narrator_ratio > 0.7 → "narration_only" (excluded from splits)   │
│  Output: chunk_metadata.json with content_type                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 5: Call Splitting (LEGACY - TO BE REPLACED BY ML)                     │
│  ───────────────────────────────────────────────────────                     │
│  File: pipeline/call_splitter.py                                            │
│  Heuristics: silence ≥2.5s OR greeting patterns (fuzzy)                     │
│  Output: CallBoundary with start_abs, end_abs, evidence                     │
│  Artifacts: spk_turns.json, call_metadata.json                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 6: S3 Upload                                                          │
│  ──────────────────                                                          │
│  File: pipeline/pipeline.py, pipeline/cli.py                                │
│  Paths:                                                                      │
│    runs/{video_id}/{run_id}/video_manifest.json                             │
│    runs/{video_id}/{run_id}/chunks/{chunk_id}/words.json                    │
│    runs/{video_id}/{run_id}/chunks/{chunk_id}/diarization_segments.json     │
│    runs/{video_id}/{run_id}/calls/{call_id}/spk_turns.json                  │
│    latest/{video_id}.json → pointer to latest run_id                        │
└─────────────────────────────────────────────────────────────────────────────┘

                          HUMAN LABELING LAYER
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  BoundaryEditor UI (localhost:5173/boundary-editor)                          │
│  ─────────────────────────────────────────────────                           │
│  Files: apps/speaker-labeler/frontend/src/pages/BoundaryEditor.tsx          │
│         apps/speaker-labeler/backend/main.py                                │
│         apps/speaker-labeler/backend/boundary_store.py                      │
│  Colors: RED = auto-detected, GREEN = human-corrected                       │
│  Storage: S3 labeling/corrected_boundaries/v1/{video_id}.json               │
│  Export: GET /api/export/labels → labels.json                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ML TRAINING DATA GENERATION                                                 │
│  ───────────────────────────────                                             │
│  File: scripts/build_call_segmenter_dataset.py                              │
│  Input: labels from S3 + diarization_segments.json                          │
│  Process:                                                                    │
│    1. Load labels (--labels-s3)                                             │
│    2. Resolve run_id via latest/{video_id}.json                             │
│    3. Load ALL chunks (narration + call_like)                               │
│    4. Merge diarization segments (absolute timestamps)                      │
│    5. Detect host speaker (30s bin coverage)                                │
│    6. Generate sliding windows (1s win, 0.5s hop)                           │
│    7. Compute 9 features per window                                         │
│    8. Assign y=IN_CALL/OUT_OF_CALL, ignore zone                             │
│  Output: windows_all.parquet, dataset_meta.json, report.json                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3) Detailed Findings

### SECTION A — Manual Boundary Labeling: What's stored and where

#### A.1 Data Flow

```
Frontend (BoundaryEditor.tsx:436-481)
    │ handleSave() transforms Boundary[] → {start_s, end_s}[]
    ▼
API Client (api.ts:90-91)
    │ POST /api/videos/{video_id}/boundaries
    ▼
FastAPI Endpoint (main.py:247-257)
    │ Validates via boundaries.validate_boundaries()
    ▼
Validation (boundaries.py:40-72)
    │ Checks: start < end, duration >= 1s, no overlaps
    ▼
S3 Persistence (boundary_store.py:63-100)
    │ 1. Sorts by start_s
    │ 2. Assigns call_index sequentially
    │ 3. Adds corrected_at timestamp
    ▼
S3: labeling/corrected_boundaries/v1/{video_id}.json
```

#### A.2 S3 Storage Details

- **Bucket:** `rezora-whisperx-us-east-1-864981718771`
- **Prefix:** `labeling/corrected_boundaries/v1/`
- **Key Pattern:** `{video_id}.json`

#### A.3 Stored JSON Schema

```json
{
    "video_id": "abc123",
    "updated_at": "2026-01-29T15:12:01.123456+00:00",
    "boundaries": [
        {
            "call_index": 0,
            "start_s": 12.5,
            "end_s": 145.8,
            "corrected_at": "2026-01-29T15:12:01.123456+00:00"
        }
    ]
}
```

#### A.4 Export Endpoint

- **Endpoint:** `GET /api/export/labels`
- **Implementation:** `boundary_store.py:161-185`
- **Output Format:**

```json
[
  {"video_id": "abc", "calls": [{"start": 12.5, "end": 145.8}]}
]
```

#### A.5 Timestamp Verification: ABSOLUTE

**Evidence from BoundaryEditor.tsx:318-356:**
```typescript
const time = wavesurferRef.current.getCurrentTime();  // ABSOLUTE from WaveSurfer
const newBoundary = {
  start_s: time,  // Direct WaveSurfer position = absolute MP3 time
  end_s: endTime,
};
```

WaveSurfer loads the FULL MP3 (`getVideoAudioUrl()` → full audio stream), so positions are absolute.

#### A.6 Sorting Behavior

**Boundaries ARE sorted before saving** (`boundary_store.py:83`):
```python
sorted_boundaries = sorted(boundaries, key=lambda x: x["start_s"])
```

**call_index is assigned sequentially** (`boundary_store.py:84-87`):
```python
for i, b in enumerate(sorted_boundaries):
    b["call_index"] = i
```

#### A.7 Risks

| Risk | Severity | Details |
|------|----------|---------|
| Race condition | MEDIUM | No optimistic locking for concurrent edits |
| "Copy from Auto" trap | LOW | Warning dialog at lines 436-447 mitigates |
| Empty boundaries | N/A | Correctly deletes S3 object (boundary_store.py:76-78) |

---

### SECTION B — Dataset Builder: correctness + reproducibility

#### B.1 Script Location

**Path:** `/Users/yevgeniymatsay/whisperx-pipeline/scripts/build_call_segmenter_dataset.py`

**CLI Usage:**
```bash
python scripts/build_call_segmenter_dataset.py \
    --labels-s3 \
    --output-dir data/call_segmenter/v1 \
    --win-s 1.0 --hop-s 0.5 --ignore-s 0.75
```

#### B.2 Key Defaults

| Parameter | Default | Location |
|-----------|---------|----------|
| `--win-s` | 1.0 | Line 833 |
| `--hop-s` | 0.5 | Line 834 |
| `--ignore-s` | 0.75 | Line 835 |
| `context_10s` | 10.0 | DatasetConfig:95 (hardcoded) |
| `context_30s` | 30.0 | DatasetConfig:96 (hardcoded) |

#### B.3 Chunk Loading: ALL Chunks

The builder loads ALL chunks regardless of content_type (`build_call_segmenter_dataset.py:483-496`):
```python
chunk_ids = list_chunks(s3_client, video_id, run_id)
for chunk_id in chunk_ids:
    meta = load_chunk_metadata(s3_client, video_id, run_id, chunk_id)
    chunks.append(meta)
```

#### B.4 Run ID Resolution: NOT PINNED (Risk)

**Critical finding** (`build_call_segmenter_dataset.py:467-471`):
```python
run_id = load_latest_run_id(s3_client, video_id)
```

Uses `latest/{video_id}.json` pointer which can change if pipeline reruns.

#### B.5 Diarization Timestamps: ABSOLUTE

**Proof from transcriber.py:165-166:**
```python
dia_segments.append(DiarizationSegment(
    t0_abs=turn.start + chunk_time_offset_s,  # ABSOLUTE
    t1_abs=turn.end + chunk_time_offset_s,    # ABSOLUTE
))
```

#### B.6 MP3 Duration: REQUIRES DOWNLOAD

**`build_call_segmenter_dataset.py:289-306`** downloads entire MP3 if metadata missing:
```python
s3_client.download_file(S3_BUCKET, mp3_key, tmp.name)
result = subprocess.run(["ffprobe", ...])
```

#### B.7 Output Schemas

**Parquet Columns (21 total):**

| Feature Columns (9) | Non-Feature Columns (12) |
|---------------------|--------------------------|
| speech_frac | video_id |
| host_speech_frac | run_id |
| nonhost_speech_frac | creator_id |
| num_active_speakers | mp3_key |
| nonhost_active | t_start |
| speaker_switches_10s | t_end |
| unique_nonhost_speakers_30s | t_mid |
| nonhost_turns_30s | chunk_content_type (LEAKY) |
| avg_nonhost_turn_len_30s | y |
| | ignore |
| | dist_to_boundary_s |

#### B.8 Smoke Test Procedure

```bash
# 1. Dry run on 2-3 videos
python scripts/build_call_segmenter_dataset.py \
    --labels-s3 \
    --video-ids VIDEO1 VIDEO2 VIDEO3 \
    --output-dir /tmp/smoke_test \
    --dry-run

# 2. Full build
python scripts/build_call_segmenter_dataset.py \
    --labels-s3 \
    --video-ids VIDEO1 VIDEO2 VIDEO3 \
    --output-dir /tmp/smoke_test

# 3. Verify outputs
cat /tmp/smoke_test/report.json | python -m json.tool
python -c "import pandas as pd; print(pd.read_parquet('/tmp/smoke_test/windows_all.parquet').head())"
```

---

### SECTION C — WhisperX Configuration (ASR + alignment)

#### C.1 Model Settings

**File:** `pipeline/config.py:14-20`

```python
@dataclass
class WhisperXConfig:
    model: str = "large-v3"
    batch_size: int = 32
    compute_type: str = "float16"
    device: str = "cuda"
    language: str = "en"
```

#### C.2 VAD Settings

**File:** `pipeline/config.py:6-11`

```python
@dataclass
class VADConfig:
    gap_threshold_s: float = 2.0
    min_chunk_s: float = 30.0
    max_chunk_s: float = 600.0
    padding_s: float = 0.5
```

#### C.3 words.json Schema

```json
{
  "video_id": "string",
  "chunk_id": "string",
  "chunk_time_offset_s": "float",
  "pipeline_version": "string",
  "asr_model": "whisper-large-v3",
  "diarizer_model": "pyannote-3.1",
  "words": [
    {
      "i": "int",
      "t0": "float (relative)",
      "t1": "float (relative)",
      "t0_abs": "float (ABSOLUTE)",
      "t1_abs": "float (ABSOLUTE)",
      "text": "string",
      "text_norm": "string",
      "spk": "SPEAKER_XX"
    }
  ]
}
```

#### C.4 Absolute Timestamp Proof

**`transcriber.py:147-156`:**
```python
words.append(WordOutput(
    t0_abs=t0 + chunk_time_offset_s,   # ABSOLUTE = relative + offset
    t1_abs=t1 + chunk_time_offset_s,   # ABSOLUTE = relative + offset
))
```

---

### SECTION D — Diarization Configuration (pyannote)

#### D.1 Model

- **Pipeline:** pyannote/speaker-diarization-3.1 (via WhisperX wrapper)
- **HF_TOKEN:** Required (`transcriber.py:72-74`)

#### D.2 Speaker Settings

**`transcriber.py:86-87` (HARDCODED):**
```python
min_speakers: int = 2,
max_speakers: int = 3
```

Not configurable via `PipelineConfig`.

#### D.3 diarization_segments.json Schema

```json
{
  "segments": [
    {
      "t0_abs": "float (ABSOLUTE)",
      "t1_abs": "float (ABSOLUTE)",
      "spk": "SPEAKER_XX"
    }
  ]
}
```

#### D.4 RTTM Generation

**`chunk_writer.py:88-93`:**
```python
f.write(f"SPEAKER {video_id} 1 {seg.t0_abs:.3f} {duration:.3f} <NA> <NA> {seg.spk} <NA> <NA>\n")
```

#### D.5 Runs on ALL Chunks

Diarization runs BEFORE content_type classification (`pipeline.py:110-123`).

---

### SECTION E — Content Type Classification Logic

#### E.1 Classification Logic

**`pipeline.py:123`:**
```python
content_type = "narration_only" if metrics.narrator_ratio > self.config.quality.max_narrator_ratio else "call_like"
```

**Threshold:** `max_narrator_ratio = 0.7` (`config.py:26`)

#### E.2 narrator_ratio Computation

**`quality_metrics.py:27-97`:**

| Pattern | Condition | Result |
|---------|-----------|--------|
| Strong narration | talk_ratio > 0.7 AND avg_turn > 8s | `talk_ratio * (avg_turn / 10)` |
| Moderate narration | talk_ratio > 0.6 AND avg_turn > 5s | `talk_ratio * 0.7` |
| Dialogue | else | `talk_ratio * 0.3` |

#### E.3 ML Classifier: NONE

**Purely heuristic** - no ML model, embeddings, or inference.

#### E.4 Downstream Dependencies

- Only `call_like` chunks contribute to call splitting
- ALL chunks get transcription, diarization, S3 upload

---

### SECTION F — Call Splitting Logic (Legacy)

#### F.1 Splitting Heuristics

**`call_splitter.py:39-74`:**

1. **Silence gaps:** `gap >= 2.5s` triggers split
2. **Greeting resets:** Fuzzy match of patterns after 30s

**Greeting Patterns (`config.py:34-40`):**
```python
["hi", "is", "this"],
["hello", "is", "this"],
["hi", "my", "name", "is"],
["this", "is"],
["calling", "from"],
["calling", "about"],
```

#### F.2 CallBoundary Schema

**`call_splitter.py:10-17`:**
```python
@dataclass
class CallBoundary:
    call_id: str                    # {video_id}_{start_ms}_{end_ms}
    start_abs: float
    end_abs: float
    boundary_confidence: float
    start_evidence: Dict[str, Any]
    end_evidence: Dict[str, Any]
```

#### F.3 UI Overlays

| Type | Color | Draggable |
|------|-------|-----------|
| Auto (heuristic) | Red | No |
| Corrected (manual) | Green | Yes |

#### F.4 ML Replacement Point

Replace `CallSplitter.find_boundaries()` at `pipeline.py:158-159`:
```python
splitter = CallSplitter(self.config.split_calls, video_id)
boundaries = splitter.find_boundaries(all_words)  # ← ML model goes here
```

---

### SECTION G — S3 Artifacts Inventory

#### Complete S3 Path Tree

```
rezora-whisperx-us-east-1-864981718771/
├── audio/
│   └── pretraining/{title} - {video_id}.mp3
│
├── runs/{video_id}/{run_id}/
│   ├── video_manifest.json
│   ├── chunks/{chunk_id}/
│   │   ├── words.json
│   │   ├── diarization_segments.json
│   │   ├── chunk_metadata.json
│   │   └── diarization.rttm
│   └── calls/{call_id}/
│       ├── call_words.json
│       ├── spk_turns.json
│       └── call_metadata.json
│
├── latest/
│   ├── {video_id}.json              # Pointer to latest run
│   └── {video_id}/{run_id}.json     # Immutable history
│
├── labeling/
│   └── corrected_boundaries/v1/{video_id}.json
│
└── genrm/
    ├── accepted/{call_id}.json      # Score >= 0.80
    ├── review/{call_id}.json        # 0.60 <= Score < 0.80
    └── rejected/{call_id}.json      # Score < 0.60
```

---

### SECTION H — Scale Readiness for 1000+ Videos

#### H.1 Top Bottlenecks

| Rank | Bottleneck | Location | Impact at 1000 Videos |
|------|------------|----------|----------------------|
| 1 | MP3 download for duration | build_call_segmenter_dataset.py:289-306 | 50-200GB transfer, 10-30 hours |
| 2 | list_videos() O(V²) | s3_client.py:69-101 | 5000+ LIST requests |
| 3 | get_mp3_key_for_video() | build_call_segmenter_dataset.py:256-267 | 1M LIST requests |
| 4 | Window features O(W*S) | features.py:58-67 | 1B comparisons |
| 5 | No checkpointing | build_call_segmenter_dataset.py:903-949 | Full recompute every run |

#### H.2 Mitigations

1. **Store duration in S3 metadata** - Eliminates 50-200GB transfer
2. **Add mp3_key to latest/{video_id}.json** - Eliminates O(V*N) listing
3. **Cache video index** - Fixes list_videos() O(V²)
4. **Binary search for segments** - Reduces O(W*S) to O(W*log(S))
5. **Add checkpointing** - Skip unchanged videos

#### H.3 Idempotency

- **Run ID:** Timestamp-based (`YYYYMMDD_HHMMSS_ffffff`)
- **Reprocessing:** Creates new directory, preserves old data
- **Collision risk:** Low for serial, risky for >10 concurrent workers

---

## 4) Checklist to Train Robustly on 1000+ Videos

### Pre-Training

- [ ] **Verify labeled count:** `aws s3 ls s3://rezora-whisperx-us-east-1-864981718771/labeling/corrected_boundaries/v1/ | wc -l`
- [ ] **Export labels:** `curl http://localhost:8000/api/export/labels > labels.json`
- [ ] **Pin run_ids:** Save current `index.jsonl` after first dataset build
- [ ] **Populate MP3 metadata:** Run script to add `x-amz-meta-duration-s` to all MP3 objects
- [ ] **Remove leaky feature:** Exclude `chunk_content_type` from training

### Dataset Build

- [ ] Run: `python scripts/build_call_segmenter_dataset.py --labels-s3 --output-dir data/call_segmenter/v2`
- [ ] Inspect `report.json` for anomalies:
  - `anomalies.timebase_drift` should be empty or known videos
  - `anomalies.ambiguous_host` should be < 5% of videos
- [ ] Verify class balance: `class_balance.in_call` should be 20-50%
- [ ] Save `index.jsonl` for reproducibility

### Training

- [ ] Split by `creator_id` (not random) to prevent leakage
- [ ] Use only 9 feature columns from `dataset_meta.json`
- [ ] Hold out 10-20% of videos for validation

### Post-Training

- [ ] Integration test: Replace `CallSplitter.find_boundaries()` with ML model
- [ ] Compare ML boundaries to human labels (precision/recall)
- [ ] Monitor boundary_confidence distribution

---

## 5) Appendix: Key Code Excerpts

### A. Boundary Save Function (`boundary_store.py:63-100`)

```python
def save(self, video_id: str, boundaries: List[Dict[str, Any]]) -> None:
    if not boundaries:
        self.delete(video_id)
        return

    now = datetime.now(timezone.utc).isoformat()
    sorted_boundaries = sorted(boundaries, key=lambda x: x["start_s"])
    for i, b in enumerate(sorted_boundaries):
        b["call_index"] = i
        if "corrected_at" not in b:
            b["corrected_at"] = now

    doc = {
        "video_id": video_id,
        "updated_at": now,
        "boundaries": sorted_boundaries,
    }

    self.s3.put_object(
        Bucket=self.bucket,
        Key=self._key(video_id),
        Body=json.dumps(doc, indent=2),
        ContentType="application/json",
    )
```

### B. Absolute Timestamp Proof (`transcriber.py:147-156`)

```python
words.append(WordOutput(
    i=len(words),
    t0=t0,
    t1=t1,
    t0_abs=t0 + chunk_time_offset_s,   # ABSOLUTE
    t1_abs=t1 + chunk_time_offset_s,   # ABSOLUTE
    text=text.strip(),
    text_norm=text_norm,
    spk=spk
))
```

### C. Label Assignment (`window_generator.py:89-124`)

```python
def assign_labels(
    t_mid: float,
    call_boundaries: List[CallBoundary],
    ignore_s: float,
) -> Tuple[int, int, float]:
    if not call_boundaries:
        return 0, 0, float("inf")

    y = 0
    for boundary in call_boundaries:
        if boundary.start <= t_mid <= boundary.end:
            y = 1
            break

    min_dist = float("inf")
    for boundary in call_boundaries:
        dist_to_start = abs(t_mid - boundary.start)
        dist_to_end = abs(t_mid - boundary.end)
        min_dist = min(min_dist, dist_to_start, dist_to_end)

    ignore = 1 if min_dist <= ignore_s else 0
    return y, ignore, min_dist
```

### D. Feature Computation (`features.py:70-100`)

```python
def compute_window_features(
    segments: List[DiarizationSegment],
    host_speaker: str,
    win_start: float,
    win_end: float,
    context_10s_start: float,
    context_30s_start: float,
) -> WindowFeatures:
    win_duration = win_end - win_start
    window_segs = _segments_in_range(segments, win_start, win_end)

    total_speech = 0.0
    host_speech = 0.0
    nonhost_speech = 0.0
    active_speakers: Set[str] = set()

    for seg in window_segs:
        duration = _intersect_duration(seg.t0_abs, seg.t1_abs, win_start, win_end)
        total_speech += duration
        active_speakers.add(seg.spk)
        if seg.spk == host_speaker:
            host_speech += duration
        else:
            nonhost_speech += duration

    speech_frac = min(1.0, total_speech / win_duration)
    # ... compute remaining features
```

### E. MP3 Duration Bottleneck (`build_call_segmenter_dataset.py:270-306`)

```python
def get_mp3_duration_s(s3_client, mp3_key: str) -> Optional[float]:
    # Fast path - check S3 metadata first
    try:
        head = s3_client.head_object(Bucket=S3_BUCKET, Key=mp3_key)
        if "x-amz-meta-duration-s" in head.get("Metadata", {}):
            return float(head["Metadata"]["x-amz-meta-duration-s"])
    except Exception:
        pass

    # Slow path - download entire MP3 (BOTTLENECK)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
        s3_client.download_file(S3_BUCKET, mp3_key, tmp.name)
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", ...],
            capture_output=True,
        )
        return float(result.stdout.strip())
```

---

## Essential Files Reference

| Purpose | File Path |
|---------|-----------|
| Boundary labeling UI | `apps/speaker-labeler/frontend/src/pages/BoundaryEditor.tsx` |
| Boundary API | `apps/speaker-labeler/backend/main.py` |
| Boundary S3 store | `apps/speaker-labeler/backend/boundary_store.py` |
| Boundary validation | `apps/speaker-labeler/backend/boundaries.py` |
| Dataset builder | `scripts/build_call_segmenter_dataset.py` |
| Window generation | `pipeline/call_segmenter/window_generator.py` |
| Feature computation | `pipeline/call_segmenter/features.py` |
| Host detection | `pipeline/call_segmenter/host_detector.py` |
| Pipeline config | `pipeline/config.py` |
| Transcriber | `pipeline/transcriber.py` |
| Call splitter | `pipeline/call_splitter.py` |
| Chunk writer | `pipeline/chunk_writer.py` |
| Pipeline orchestrator | `pipeline/pipeline.py` |
| CLI entry point | `pipeline/cli.py` |

---

**Report Generated:** 2026-01-29
**Total Videos Analyzed:** 51 labeled
**Confidence Level:** High (all critical paths verified with code evidence)
