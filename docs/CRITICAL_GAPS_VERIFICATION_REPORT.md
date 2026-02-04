# Critical Gaps Verification Report

**Date:** 2026-01-29
**Purpose:** Verify critical assumptions for call-boundary classifier training robustness

---

## Executive Summary

| Gap | Status | Risk | Immediate Action |
|-----|--------|------|------------------|
| **Speaker ID Consistency** | NO - IDs inconsistent across chunks | CRITICAL | Implement cross-chunk speaker linking |
| **Creator ID Validity** | BROKEN - extracts title, not channel | CRITICAL | Use video_views_cache.json for channel lookup |
| **Speaker Limits (min/max)** | Hardcoded min=2 forces artificial splits | HIGH | Make configurable, set min=1 for YouTube |
| **VAD Gap Handling** | Gaps excluded but not masked in training | HIGH | Implement ignore mask for gap-adjacent windows |

---

## Section 1: SPEAKER ID CONSISTENCY ACROSS CHUNKS

### 1.1 Code Path Analysis

**Finding: Diarization runs PER-CHUNK, not across full video.**

**Evidence from `pipeline/pipeline.py:106-113`:**
```python
for chunk in chunks:
    try:
        chunk_audio = audio[int(chunk.start_s * 16000):int(chunk.end_s * 16000)]
        words, dia_segs = transcriber.transcribe(
            chunk_audio,  # <-- Only chunk audio passed
            chunk_time_offset_s=chunk.start_s
        )
```

**Evidence from `pipeline/transcriber.py:124-130`:**
```python
# 3. Diarize
diarize_pipeline = self._load_diarize_pipeline()
diarize_segments = diarize_pipeline(
    audio,  # <-- This is chunk_audio, not full audio
    min_speakers=min_speakers,
    max_speakers=max_speakers
)
```

### 1.2 Data Analysis Framework

For labeled videos with multiple chunks, the analysis should compute:

| video_id | chunk_id | start_abs | end_abs | top_speaker | top_dur_s | second_speaker | second_dur_s |
|----------|----------|-----------|---------|-------------|-----------|----------------|--------------|
| vid_001  | chunk_000 | 0.0 | 298.5 | SPEAKER_00 | 180.2 | SPEAKER_01 | 95.3 |
| vid_001  | chunk_001 | 298.5 | 597.0 | SPEAKER_01 | 165.8 | SPEAKER_00 | 102.1 |
| vid_001  | chunk_002 | 597.0 | 850.2 | SPEAKER_00 | 140.5 | SPEAKER_01 | 88.7 |

**Flip Rate Calculation:**
```
flip_rate = (chunks where top_speaker != prev.top_speaker) / (total_chunks - 1)
```

### 1.3 Note on `speaker_flip_suspected`

The existing `quality_metrics.speaker_flip_suspected` (quality_metrics.py:100-112) detects **within-chunk** behavioral shifts, NOT cross-chunk identity inconsistency. It cannot correlate with the issue described here.

### 1.4 Conclusion

**Are SPEAKER_00/SPEAKER_01 consistent across chunks for the same real host?**

## **NO** - Speaker labels are NOT consistent across chunks.

| Evidence Type | Finding | Location |
|---------------|---------|----------|
| Code Path | pyannote receives only chunk audio slice | `pipeline/pipeline.py:109-112` |
| Diarization Call | No context from other chunks passed | `pipeline/transcriber.py:126-130` |
| Missing Feature | No cross-chunk speaker linking exists | Entire codebase (verified via grep) |

**Root Cause:** pyannote assigns labels (`SPEAKER_00`, `SPEAKER_01`) independently per chunk. The same real person could be `SPEAKER_00` in chunk 0 but `SPEAKER_01` in chunk 1 based solely on talk time within that chunk.

---

## Section 2: CREATOR_ID VALIDITY

### 2.1 Sample MP3 Keys and extract_creator_id() Output

**MP3 Pattern:** `audio/pretraining/{title} - {video_id}.mp3`

**Function Location:** `scripts/build_call_segmenter_dataset.py:239-253`
```python
def extract_creator_id(mp3_key: str) -> str:
    """Extract creator_id from mp3_key.

    Pattern: audio/pretraining/{creator} - {video_id}.mp3
    """
    filename = mp3_key.split("/")[-1]
    match = re.match(r"(.+?) - ", filename)
    if match:
        return match.group(1).strip()
    return filename.replace(".mp3", "")
```

**Sample Extraction Results:**

| mp3_key | extracted_creator_id | Actual Channel |
|---------|---------------------|----------------|
| `50 LIVE COLD CALLS | REAL Objections... - ci-FdcWiJiA.mp3` | `50 LIVE COLD CALLS...` | Trent Dressel |
| `100 LIVE COLD CALLS | They Hung Up.. - UPg1l6N5KNo.mp3` | `100 LIVE COLD CALLS...` | Trent Dressel |
| `72 LIVE COLD CALLS | Cold Calling... - ctSvbr0wd1c.mp3` | `72 LIVE COLD CALLS...` | Trent Dressel |
| `100 LIVE COLD CALLS | Day In The Life... - 3Q7yVjVmcQU.mp3` | `100 LIVE COLD CALLS...` | Luke Arellano |
| `47 LIVE COLD CALLS | Tech Sales... - GcjtU_V-mW4.mp3` | `47 LIVE COLD CALLS...` | Luke Arellano |
| `How I book 2-3 meetings... - yGH9TtROmko.mp3` | `How I book 2` | Max Ross |
| `I Cold Called 25 Local Businesses... - Iex1weHMhE4.mp3` | `I Cold Called 25...` | Max Ross |

### 2.2 Validity Assessment

## **CRITICAL BUG CONFIRMED**

The `extract_creator_id()` function extracts **VIDEO TITLE**, NOT **CHANNEL/CREATOR NAME**.

**Problem:** Three Trent Dressel videos get THREE DIFFERENT "creator_ids":
- `"50 LIVE COLD CALLS..."`
- `"100 LIVE COLD CALLS..."`
- `"72 LIVE COLD CALLS..."`

**Impact on Group Splits:**
- Dataset builder uses `group_split_column: "creator_id"` (line 723)
- Since each video has unique title → each video gets unique "creator_id"
- Group splits by `creator_id` ≈ random splits by `video_id`
- **NO LEAKAGE PREVENTION** - same channel can appear in both train and test

### 2.3 Correct Channel Data Location

**File:** `data/video_views_cache.json` contains correct channel names:
```json
{
  "ci-FdcWiJiA": {
    "title": "50 LIVE COLD CALLS | REAL Objections...",
    "channel": "Trent Dressel"
  },
  "UPg1l6N5KNo": {
    "title": "100 LIVE COLD CALLS | They Hung Up..",
    "channel": "Trent Dressel"
  }
}
```

### 2.4 Recommendations

**Immediate Fix (for current training):**
```python
def get_channel_name(video_id: str) -> str:
    """Get channel name from metadata cache."""
    cache_path = Path("data/video_views_cache.json")
    with open(cache_path) as f:
        cache = json.load(f)
    return cache.get(video_id, {}).get("channel", "unknown")
```

**Long-term:** Add `channel_id` to `latest/{video_id}.json` during pipeline runs.

| Source | Contains Channel? |
|--------|-------------------|
| `audio/pretraining/{title} - {video_id}.mp3` | **NO** (title only) |
| `latest/{video_id}.json` | **NO** |
| `data/video_views_cache.json` (local) | **YES** |

---

## Section 3: DIARIZATION SPEAKER LIMITS (min/max speakers)

### 3.1 Code Location

**File:** `pipeline/transcriber.py:82-88`
```python
def transcribe(
    self,
    audio: np.ndarray,
    chunk_time_offset_s: float = 0.0,
    min_speakers: int = 2,  # HARDCODED
    max_speakers: int = 3   # HARDCODED
) -> Tuple[List[WordOutput], List[DiarizationSegment]]:
```

**NOT in config:** `pipeline/config.py` WhisperXConfig has no speaker limit fields.

**pipeline.py doesn't pass them:** Line 110 calls `transcriber.transcribe()` without speaker params.

### 3.2 Rationale Analysis

| Setting | Value | Likely Rationale |
|---------|-------|------------------|
| `min_speakers=2` | Two minimum | Cold calls have agent + user |
| `max_speakers=3` | Three maximum | Allows for call transfers |

**Problem Scenarios:**

| Scenario | Actual Speakers | Impact |
|----------|----------------|--------|
| Narration-only (YouTube intro) | 1 | min=2 forces artificial split |
| Conference calls | 4+ | max=3 caps clustering, merges speakers |
| Voicemail recordings | 1 | Single voice split into 2 speakers |

### 3.3 Evidence of Forced Splits

`quality_metrics.py:73-75` handles single-speaker case:
```python
if len(speaker_stats) < 2:
    # Only one speaker = definitely narration
    return 1.0
```

**Issue:** With `min_speakers=2`, this condition rarely triggers because pyannote is **forced** to find 2 speakers.

**Inconsistency:** Smoke test (`cli.py:75`) uses `min=1` but production uses `min=2`.

### 3.4 Recommendations

**Make configurable in `config.py`:**
```python
@dataclass
class WhisperXConfig:
    model: str = "large-v3"
    # ... existing fields ...
    min_speakers: int = 1  # NEW: Allow single-speaker detection
    max_speakers: int = 3  # NEW: Configurable
```

**Recommended defaults for YouTube content:**

| Content Type | min_speakers | max_speakers |
|--------------|-------------|--------------|
| Cold call corpus | 2 | 3 |
| YouTube with narration | **1** | 3 |
| Conference calls | 2 | **5** |

---

## Section 4: VAD CHUNKING + NON-SPEECH GAPS

### 4.1 Gap Exclusion Behavior

**CONFIRMED: Long silences (>=2s) are EXCLUDED from chunks.**

**Evidence from `pipeline/vad_chunker.py:66-86`:**
```python
for i, seg in enumerate(speech_segments):
    # ...
    gap = speech_segments[i + 1]["start"] - seg["end"]
    if gap >= self.config.gap_threshold_s:  # Default: 2.0s
        # Split here - creates gap in coverage
        end = min(total_duration_s, seg["end"] + self.config.padding_s)
        chunks.extend(self._split_if_needed(video_id, current_start, end))
        current_start = max(0, speech_segments[i + 1]["start"] - self.config.padding_s)
```

**Configuration (`config.py:6-11`):**
| Parameter | Default | Effect |
|-----------|---------|--------|
| `gap_threshold_s` | 2.0s | Gaps >= 2s create chunk boundaries |
| `padding_s` | 0.5s | Each chunk extends 0.5s into gaps |

**Result:** Gap of 10s → 9s of audio uncovered (dial tones, ring tones excluded)

### 4.2 chunk_time_offset_s Definition

**CONFIRMED: Absolute position in original MP3 timeline.**

**Evidence from `pipeline/pipeline.py:107-113`:**
```python
chunk_audio = audio[int(chunk.start_s * 16000):int(chunk.end_s * 16000)]
words, dia_segs = transcriber.transcribe(
    chunk_audio,
    chunk_time_offset_s=chunk.start_s  # ABSOLUTE position
)
```

**Evidence from `transcriber.py:147-152`:**
```python
words.append(WordOutput(
    t0_abs=t0 + chunk_time_offset_s,  # local + offset = ABSOLUTE
    t1_abs=t1 + chunk_time_offset_s,
))
```

**Timeline Example:**
```
MP3:      [0s]--SPEECH--[10s]--GAP--[20s]--SPEECH--[30s]
Chunk 1:  start_s=0.0,  end_s=10.5
Chunk 2:  start_s=19.5, end_s=30.0
Gap:      [10.5s to 19.5s] = 9.0s uncovered

Word in Chunk 2 at local t=0.5 → t0_abs = 0.5 + 19.5 = 20.0s (CORRECT)
```

### 4.3 Impact on Training Labels

**Problem:** Labels that start at ring/dial tone fall within VAD gaps.

**Scenario:**
```
Timeline:  [0s]--NARRATION--[100s]--DIAL-TONE(10s)--[110s]--CALL--[200s]
Chunks:    [0s-100.5s] [109.5s-200s]
Gap:       [100.5s to 109.5s]

Human Label: boundary.start = 100.0 (dial tone begins)
Windows at t_mid=100.0-109.5 get y=1 BUT no diarization features exist!
```

**Current Behavior:** Window generator (`window_generator.py:127-181`) does NOT check if windows fall within gaps.

**Gap detection exists** (`build_call_segmenter_dataset.py:530-543`) but **gap masking NOT IMPLEMENTED**.

### 4.4 Recommendations

**Option A (RECOMMENDED): Ignore mask for gap-adjacent windows**
```python
def check_gap_overlap(t_start, t_end, chunk_gaps):
    for gap in chunk_gaps:
        if t_start < gap["next_chunk_start"] and t_end > gap["chunk_end"]:
            return True
    return False

# In generate_windows:
gap_overlap = check_gap_overlap(t_start, t_end, chunk_gaps)
ignore = 1 if (dist <= ignore_s or gap_overlap) else 0
```

**Option B:** Shift boundary timestamps to nearest valid chunk edge.

**Option C (future):** Add audio-level features (RMS energy, spectral) for gap regions.

---

## Critical Files Reference

| File | Lines | Purpose |
|------|-------|---------|
| `pipeline/transcriber.py` | 82-88, 124-130 | Speaker limits, diarization call |
| `pipeline/pipeline.py` | 106-113 | Chunk audio extraction |
| `pipeline/vad_chunker.py` | 66-86 | Gap-based chunk splitting |
| `pipeline/config.py` | 6-20 | Configuration (missing speaker limits) |
| `scripts/build_call_segmenter_dataset.py` | 239-253, 530-543 | creator_id extraction, gap detection |
| `pipeline/quality_metrics.py` | 73-75, 100-112 | Narrator ratio, speaker flip detection |
| `pipeline/call_segmenter/window_generator.py` | 89-181 | Label assignment (no gap handling) |
| `data/video_views_cache.json` | - | Correct channel names |

---

## Action Items

| Priority | Item | Owner | Status |
|----------|------|-------|--------|
| **P0** | Fix `extract_creator_id()` to use `video_views_cache.json` | - | TODO |
| **P0** | Add ignore mask for gap-adjacent windows | - | TODO |
| **P1** | Make min/max speakers configurable in config.py | - | TODO |
| **P1** | Implement cross-chunk speaker linking (voice embeddings) | - | TODO |
| **P2** | Add channel_id to latest/{video_id}.json schema | - | TODO |
| **P2** | Highlight VAD gaps in BoundaryEditor UI | - | TODO |

---

## Conclusion

**Training Readiness Status:** NOT READY for production training without fixes.

**Must fix before training:**
1. Creator ID bug (train/test leakage)
2. Gap-adjacent window masking (corrupted labels)

**Should fix for robustness:**
3. Configurable speaker limits
4. Cross-chunk speaker consistency
