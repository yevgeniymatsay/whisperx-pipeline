# Critical Gaps Quantification Report

**Date:** 2026-01-29
**Videos Analyzed:** 4 (with S3 boundary labels)
**Candidate Videos:** 51 (in selected_videos.txt)

---

## Executive Summary

| Metric | Value | Severity |
|--------|-------|----------|
| **Avg Speaker ID Flip Rate** | 71.4% | CRITICAL |
| **Calls with Start in VAD Gap** | 42.9% (3/7) | HIGH |
| **y=1 Windows with Zero Speech** | 32 windows | HIGH |
| **Channel Coverage** | 100% (51/51) | ✅ OK |

**Key Finding:** Only **4 videos** have actual boundary labels in S3. The "51 labeled videos" refers to the selection list, not completed labeling work.

---

## 1. Speaker ID Instability Metrics

**Output File:** `data/call_segmenter/analysis/speaker_id_instability.csv`

### Summary Statistics

| Metric | Value |
|--------|-------|
| Videos Analyzed | 4 |
| Avg Flip Rate | 71.38% |
| Avg Boundary Switches | 4.2 |
| Avg Unique Speakers | 2.8 |

### Per-Video Results

| video_id | chunks | flip_rate | boundary_switches | unique_speakers |
|----------|--------|-----------|-------------------|-----------------|
| -UzXRV8nHn4 | 2 | **100%** | 1 | 3 |
| 6L-8f1eYOOY | 2 | **100%** | 1 | 2 |
| 4ipwbOJRMck | 18 | 47.1% | 8 | 3 |
| -POaWp9_UaM | 14 | 38.5% | 7 | 3 |

### Interpretation

**flip_rate = 100%** for 2-chunk videos means the dominant speaker label (by talk time) changed between chunk 0 and chunk 1. This is almost certainly the same physical speaker being assigned different IDs.

**Evidence of problem:** Video `4ipwbOJRMck` with 18 chunks shows 8 dominant speaker changes:
```
chunk_0: SPEAKER_02 -> SPEAKER_00
chunk_1: SPEAKER_00 -> SPEAKER_01
chunk_3: SPEAKER_01 -> SPEAKER_00
...
```

### Impact on 9 Features

| Feature | Impact Level | Explanation |
|---------|-------------|-------------|
| `host_speech_frac` | **HIGH** | Host gets different label in new chunk → undercounted |
| `nonhost_speech_frac` | **HIGH** | Host speech miscounted as non-host |
| `num_active_speakers` | **MEDIUM** | Same person counted with multiple IDs |
| `speaker_switches_10s` | **HIGH** | Spurious switches at chunk boundaries |
| `unique_nonhost_speakers_30s` | **CRITICAL** | Same person counted multiple times |
| `nonhost_turns_30s` | **HIGH** | Host turns misclassified |
| `avg_nonhost_turn_len_30s` | **MEDIUM** | Distorted by incorrect attribution |
| `nonhost_active` | **MEDIUM** | May show false non-host activity |
| `speech_frac` | **NONE** | Speaker-agnostic, unaffected |

---

## 2. VAD Gap vs Labels Mismatch Metrics

**Output File:** `data/call_segmenter/analysis/vad_gap_label_mismatch.json`

### Summary Statistics

| Metric | Value |
|--------|-------|
| Total Labeled Calls | 7 |
| Calls with Start in VAD Gap | 3 (42.9%) |
| Calls with End in VAD Gap | 2 (28.6%) |
| Avg Covered Fraction | 95.21% |
| Avg Delta to First Speech | 4.65s |
| **y=1 Windows with Zero Speech** | **32** |
| **y=1 Windows with Zero Speakers** | **32** |

### Interpretation

**32 windows labeled as "in-call" (y=1) have zero speech features.** These are windows where:
1. The human labeler marked a call boundary (e.g., at dial tone)
2. But the VAD chunker excluded that region (no speech detected)
3. Result: Window gets `y=1` label but `speech_frac=0`

**This corrupts training data.** The model will learn that "zero speech = in-call" for some samples.

### Worst Calls by Coverage

| video_id | call_index | covered_fraction | delta_to_first_speech | start_in_gap |
|----------|------------|------------------|----------------------|--------------|
| -POaWp9_UaM | 1 | 84.5% | 12.84s | YES |
| -POaWp9_UaM | 0 | 91.3% | 6.71s | YES |
| -POaWp9_UaM | 3 | 92.2% | 10.13s | YES |

**Pattern:** Human labelers mark call starts at ring/dial tones, but VAD chunks don't cover these non-speech regions.

---

## 3. Channel/Creator ID Coverage

**Output File:** `data/call_segmenter/analysis/channel_coverage.json`

### Summary

| Category | Count | Coverage |
|----------|-------|----------|
| Labeled Videos | 51 | - |
| In Cache | 51 | 100% |
| With Channel | 51 | **100%** |
| Missing from Cache | 0 | - |

### Full Corpus

| Metric | Value |
|--------|-------|
| Total in Cache | 174 |
| With Channel | 174 (100%) |
| Unique Channels | 75 |

### Top Channels in Labeled Set

| Channel | Videos |
|---------|--------|
| Cody Askins | 6 |
| Pace Morby | 4 |
| Trent Dressel | 3 |
| Peter Roberts | 2 |
| Matt Macnamara | 2 |

### Conclusion

**Channel coverage is complete.** All 51 labeled videos have valid channel metadata in `video_views_cache.json`. The `extract_creator_id()` bug (extracts title instead of channel) can be fixed by reading from the cache.

**Recommended fix:**
```python
def get_channel_name(video_id: str) -> str:
    cache = json.load(open("data/video_views_cache.json"))
    return cache.get(video_id, {}).get("channel", "unknown")
```

---

## 4. Rerun Cost Plan

### Current Pipeline Cost Breakdown

| Stage | GPU Time | % of Total |
|-------|----------|------------|
| ASR (Whisper large-v3) | ~10-12 min | **~80%** |
| Alignment (wav2vec2) | ~1 min | ~5% |
| Diarization (pyannote) | ~1.5-2 min | ~10-15% |
| Post-processing | ~0.5 min | ~5% |

### Diarization-Only Rerun Savings

| Metric | Full Pipeline | Diarization Only | Savings |
|--------|---------------|------------------|---------|
| GPU Time per Video | ~15 min | ~2-3 min | **~85%** |
| Cost per Video | ~$0.25 | ~$0.04 | **~85%** |

### Proposed Script

**Location:** `scripts/rerun_diarization.py`

**CLI Interface:**
```bash
python scripts/rerun_diarization.py \
    --video-id VIDEO_ID \
    --min-speakers 1 \
    --max-speakers 4 \
    --dry-run
```

**Key Insight:** Word timestamps from alignment are **independent of diarization**. We can:
1. Reuse existing `words.json` timestamps
2. Re-run only pyannote diarization with new min/max settings
3. Re-assign speaker labels using `whisperx.assign_word_speakers()`

### Implementation Requirements

| File | Change |
|------|--------|
| `scripts/rerun_diarization.py` | NEW - CLI script for diarization-only reruns |
| `pipeline/transcriber.py` | Add `diarize_only()` method |

### Files to Reuse (No Regeneration)

- VAD chunk boundaries (deterministic)
- Word timestamps from alignment
- Normalized text

### Files to Regenerate

- `diarization_segments.json` (new speaker segments)
- `spk` field in `words.json` (re-assigned)
- `chunk_metadata.json` (speaker count changes)
- `spk_turns.json` (downstream)

---

## Action Items

### P0: Critical (Before Training)

1. **Fix VAD gap masking** - Set `ignore=1` for windows overlapping VAD gaps
   - File: `pipeline/call_segmenter/window_generator.py`
   - Impact: Prevents 32+ corrupted training samples

2. **Fix creator_id extraction** - Use `video_views_cache.json` instead of filename parsing
   - File: `scripts/build_call_segmenter_dataset.py:239-253`
   - Impact: Enables proper train/test group splits

### P1: High Priority

3. **Implement cross-chunk speaker linking** - Use voice embeddings to maintain consistent IDs
   - New module: `pipeline/speaker_linker.py`
   - Impact: Fixes 71% average flip rate

4. **Create diarization-only rerun script**
   - File: `scripts/rerun_diarization.py`
   - Impact: 85% cost savings for parameter experiments

### P2: Medium Priority

5. **Make min/max speakers configurable**
   - File: `pipeline/config.py` - Add to `WhisperXConfig`
   - Current: Hardcoded `min=2, max=3` in `transcriber.py:86-87`

6. **Add labeling progress tracking**
   - Only 4/51 videos have S3 boundary labels
   - Need clear status dashboard

---

## Data Files Generated

| File | Description |
|------|-------------|
| `data/call_segmenter/analysis/speaker_id_instability.csv` | Per-video speaker flip metrics |
| `data/call_segmenter/analysis/speaker_id_instability_analysis.md` | Feature impact analysis |
| `data/call_segmenter/analysis/vad_gap_label_mismatch.json` | Gap coverage metrics + worst calls |
| `data/call_segmenter/analysis/channel_coverage.json` | Channel metadata coverage |

---

## Key Takeaways

1. **Only 4 videos actually have boundary labels** - Not 51 as assumed
2. **71% average flip rate** - Speaker IDs are highly inconsistent across chunks
3. **32 corrupted training windows** - y=1 with zero speech features
4. **Channel coverage is 100%** - This is the one metric that's fine
5. **85% cost savings possible** - Diarization-only reruns are feasible
