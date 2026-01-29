# WhisperX Transcription Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a WhisperX-based transcription pipeline with VAD pre-splitting, pyannote diarization, deterministic call splitting, and a trained role classifier.

**Architecture:** Docker container runs on AWS Batch GPU. Audio → VAD chunks → Whisper large-v3 + pyannote → canonical words → split_calls → role classifier → existing judge/fixer pipeline.

**Tech Stack:** WhisperX, Whisper large-v3, pyannote 3.1, Silero VAD, AWS Batch, scikit-learn, boto3

**Design Reference:** `docs/plans/2026-01-26-whisperx-pipeline-design.md`

---

## ⚠️ Recommended Execution Order

**Do Task 15 (Docker) early - after Task 3 or 4.** The biggest time sink will be "CUDA doesn't work in Batch." Fail fast on container + GPU before writing 10 modules.

**Suggested order:**
1. Task 1-4 (Config, Audio, VAD, Quality metrics)
2. **Task 15 (Docker)** ← Build and verify GPU works
3. Task 5-14 (Continue pipeline modules)
4. Task 16-18 (CLI, Batch config)

**Notes on canonical data:**
- "Canonical" means per-run immutable (never regenerate within a run)
- It's OK to rerun the pipeline with a different ASR model (new run_id)
- Run isolation handles this: each run gets unique prefix

---

## Phase 1: Core Pipeline Infrastructure

### Task 1: Create Pipeline Config Module

**Files:**
- Create: `scripts/whisperx_pipeline/config.py`
- Create: `scripts/whisperx_pipeline/__init__.py`

**Step 1: Create the package directory**

```bash
mkdir -p scripts/whisperx_pipeline
```

**Step 2: Create config.py with all pipeline parameters**

```python
"""WhisperX pipeline configuration."""
from dataclasses import dataclass, field

@dataclass
class VADConfig:
    gap_threshold_s: float = 2.0
    min_chunk_s: float = 30.0
    max_chunk_s: float = 600.0
    padding_s: float = 0.5

@dataclass
class WhisperXConfig:
    model: str = "large-v3"
    batch_size: int = 16
    compute_type: str = "float16"
    device: str = "cuda"

@dataclass
class QualityConfig:
    min_quality_score: float = 0.4
    max_narrator_ratio: float = 0.7
    max_overlap_ratio: float = 0.3
    flip_detection_threshold: float = 0.4

@dataclass
class SplitCallsConfig:
    silence_threshold_s: float = 2.5
    greeting_tokens: list = field(default_factory=lambda: [
        ["hi", "is", "this"],
        ["hello", "is", "this"],
        ["hi", "my", "name", "is"],
        ["this", "is"],
        ["calling", "from"],
        ["calling", "about"],
    ])
    fuzzy_match_threshold: float = 0.8

@dataclass
class RoleClassifierConfig:
    confidence_threshold: float = 0.25
    model_version: str = "1.0.0"

@dataclass
class PipelineConfig:
    vad: VADConfig = field(default_factory=VADConfig)
    whisperx: WhisperXConfig = field(default_factory=WhisperXConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    split_calls: SplitCallsConfig = field(default_factory=SplitCallsConfig)
    role_classifier: RoleClassifierConfig = field(default_factory=RoleClassifierConfig)
    pipeline_version: str = "1.0.0"

S3_BUCKET = "rezora-data-pipeline-864981718771"
```

**Step 3: Create __init__.py**

```python
"""WhisperX transcription pipeline."""
from .config import PipelineConfig, S3_BUCKET
```

**Step 4: Commit**

```bash
git add scripts/whisperx_pipeline/
git commit -m "feat(whisperx): add pipeline config module"
```

---

### Task 2: Create Audio Preprocessing Module

**Files:**
- Create: `scripts/whisperx_pipeline/audio_preprocess.py`
- Create: `tests/whisperx_pipeline/__init__.py`
- Create: `tests/whisperx_pipeline/test_audio_preprocess.py`

**Step 1: Write the failing test**

```python
# tests/whisperx_pipeline/test_audio_preprocess.py
import pytest
import tempfile
import numpy as np
import soundfile as sf
from scripts.whisperx_pipeline.audio_preprocess import convert_to_wav, load_audio

def test_convert_to_wav_creates_16k_mono():
    """Convert MP3 to 16kHz mono WAV."""
    # Create a test WAV (we'll test with WAV since MP3 needs encoding)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        # 1 second of silence at 44.1kHz stereo
        sr = 44100
        audio = np.zeros((sr, 2), dtype=np.float32)
        sf.write(f.name, audio, sr)
        input_path = f.name

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        output_path = f.name

    convert_to_wav(input_path, output_path)

    # Verify output is 16kHz mono
    data, sr = sf.read(output_path)
    assert sr == 16000
    assert len(data.shape) == 1  # mono

def test_load_audio_returns_numpy_array():
    """Load audio returns numpy array at 16kHz."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sr = 16000
        audio = np.random.randn(sr).astype(np.float32)
        sf.write(f.name, audio, sr)
        path = f.name

    result = load_audio(path)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
```

**Step 2: Run test to verify it fails**

```bash
mkdir -p tests/whisperx_pipeline && touch tests/whisperx_pipeline/__init__.py
pytest tests/whisperx_pipeline/test_audio_preprocess.py -v
```
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# scripts/whisperx_pipeline/audio_preprocess.py
"""Audio preprocessing: convert to 16kHz mono WAV."""
import subprocess
import numpy as np
import soundfile as sf
from pathlib import Path

def convert_to_wav(input_path: str, output_path: str) -> None:
    """Convert audio file to 16kHz mono WAV using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)

def load_audio(path: str) -> np.ndarray:
    """Load audio file as numpy array."""
    data, sr = sf.read(path, dtype="float32")
    if sr != 16000:
        raise ValueError(f"Expected 16kHz, got {sr}Hz")
    return data
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/whisperx_pipeline/test_audio_preprocess.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/whisperx_pipeline/audio_preprocess.py tests/whisperx_pipeline/
git commit -m "feat(whisperx): add audio preprocessing module"
```

---

### Task 3: Create VAD Chunking Module

**Files:**
- Create: `scripts/whisperx_pipeline/vad_chunker.py`
- Create: `tests/whisperx_pipeline/test_vad_chunker.py`

**Step 1: Write the failing test**

```python
# tests/whisperx_pipeline/test_vad_chunker.py
import pytest
from scripts.whisperx_pipeline.vad_chunker import VADChunker, Chunk
from scripts.whisperx_pipeline.config import VADConfig

def test_chunk_dataclass():
    """Chunk has required fields."""
    chunk = Chunk(
        chunk_id="video_0_30000",
        start_ms=0,
        end_ms=30000,
        start_s=0.0,
        end_s=30.0,
        duration_s=30.0
    )
    assert chunk.chunk_id == "video_0_30000"
    assert chunk.duration_s == 30.0

def test_chunker_respects_min_max():
    """Chunks respect min/max duration."""
    config = VADConfig(min_chunk_s=30.0, max_chunk_s=60.0)
    chunker = VADChunker(config)

    # Mock: 3 speech segments with gaps
    speech_segments = [
        {"start": 0.0, "end": 25.0},   # First speech
        {"start": 28.0, "end": 55.0},  # Gap of 3s (>2s threshold)
        {"start": 58.0, "end": 120.0}, # Long segment
    ]

    chunks = chunker.compute_chunks(
        speech_segments=speech_segments,
        total_duration_s=120.0,
        video_id="test"
    )

    # Should merge first two (gap < min_chunk), split long one
    for chunk in chunks:
        assert chunk.duration_s >= 20.0  # Allow some slack for padding
        assert chunk.duration_s <= 65.0  # max + padding

def test_chunker_adds_padding():
    """Chunks have padding around boundaries."""
    config = VADConfig(padding_s=0.5, gap_threshold_s=2.0, min_chunk_s=10.0, max_chunk_s=600.0)
    chunker = VADChunker(config)

    speech_segments = [
        {"start": 1.0, "end": 10.0},
        {"start": 15.0, "end": 25.0},  # Gap of 5s
    ]

    chunks = chunker.compute_chunks(speech_segments, 30.0, "test")

    # First chunk should start at 0.5 (1.0 - 0.5 padding)
    assert chunks[0].start_s == 0.5
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/whisperx_pipeline/test_vad_chunker.py -v
```
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# scripts/whisperx_pipeline/vad_chunker.py
"""VAD-based audio chunking for WhisperX processing."""
import os
from dataclasses import dataclass
from typing import List
import numpy as np
import torch
from .config import VADConfig

@dataclass
class Chunk:
    chunk_id: str
    start_ms: int
    end_ms: int
    start_s: float
    end_s: float
    duration_s: float

class VADChunker:
    """Split audio into chunks based on voice activity detection."""

    def __init__(self, config: VADConfig):
        self.config = config
        self._model = None
        self._utils = None

    def _load_vad_model(self):
        """Load Silero VAD model from cache (pre-downloaded in Docker build)."""
        if self._model is None:
            # Point torch.hub to the cached model from Docker build
            torch.hub.set_dir(os.environ.get("TORCH_HOME", "/app/.cache/torch"))
            self._model, self._utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True
            )
        return self._model, self._utils

    def get_speech_segments(self, audio: np.ndarray, sr: int = 16000) -> List[dict]:
        """Get speech timestamps from audio using Silero VAD."""
        model, utils = self._load_vad_model()
        get_speech_timestamps = utils[0]

        audio_tensor = torch.from_numpy(audio)
        segments = get_speech_timestamps(audio_tensor, model, sampling_rate=sr)

        return [
            {"start": s["start"] / sr, "end": s["end"] / sr}
            for s in segments
        ]

    def compute_chunks(
        self,
        speech_segments: List[dict],
        total_duration_s: float,
        video_id: str
    ) -> List[Chunk]:
        """Compute chunk boundaries from speech segments."""
        if not speech_segments:
            # Single chunk for entire audio
            return [self._make_chunk(video_id, 0.0, total_duration_s)]

        chunks = []
        current_start = max(0, speech_segments[0]["start"] - self.config.padding_s)

        for i, seg in enumerate(speech_segments):
            is_last = i == len(speech_segments) - 1

            if is_last:
                # End of audio
                end = min(total_duration_s, seg["end"] + self.config.padding_s)
                chunks.extend(self._split_if_needed(video_id, current_start, end))
            else:
                gap = speech_segments[i + 1]["start"] - seg["end"]

                if gap >= self.config.gap_threshold_s:
                    # Split here
                    end = min(total_duration_s, seg["end"] + self.config.padding_s)
                    chunks.extend(self._split_if_needed(video_id, current_start, end))
                    current_start = max(0, speech_segments[i + 1]["start"] - self.config.padding_s)

        # Merge tiny chunks
        return self._merge_small_chunks(chunks, video_id)

    def _split_if_needed(self, video_id: str, start: float, end: float) -> List[Chunk]:
        """Split chunk if it exceeds max duration."""
        duration = end - start
        if duration <= self.config.max_chunk_s:
            return [self._make_chunk(video_id, start, end)]

        # Split into max_chunk_s pieces
        chunks = []
        current = start
        while current < end:
            chunk_end = min(current + self.config.max_chunk_s, end)
            chunks.append(self._make_chunk(video_id, current, chunk_end))
            current = chunk_end
        return chunks

    def _merge_small_chunks(self, chunks: List[Chunk], video_id: str) -> List[Chunk]:
        """Merge chunks smaller than min duration."""
        if not chunks:
            return chunks

        merged = []
        current_start = chunks[0].start_s
        current_end = chunks[0].end_s

        for chunk in chunks[1:]:
            if current_end - current_start < self.config.min_chunk_s:
                # Extend current chunk
                current_end = chunk.end_s
            else:
                merged.append(self._make_chunk(video_id, current_start, current_end))
                current_start = chunk.start_s
                current_end = chunk.end_s

        # Add last chunk
        merged.append(self._make_chunk(video_id, current_start, current_end))
        return merged

    def _make_chunk(self, video_id: str, start_s: float, end_s: float) -> Chunk:
        """Create a Chunk with computed fields."""
        start_ms = int(start_s * 1000)
        end_ms = int(end_s * 1000)
        return Chunk(
            chunk_id=f"{video_id}_{start_ms}_{end_ms}",
            start_ms=start_ms,
            end_ms=end_ms,
            start_s=start_s,
            end_s=end_s,
            duration_s=end_s - start_s
        )
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/whisperx_pipeline/test_vad_chunker.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/whisperx_pipeline/vad_chunker.py tests/whisperx_pipeline/test_vad_chunker.py
git commit -m "feat(whisperx): add VAD chunking module"
```

---

### Task 4: Create Quality Metrics Module

**Files:**
- Create: `scripts/whisperx_pipeline/quality_metrics.py`
- Create: `tests/whisperx_pipeline/test_quality_metrics.py`

**Step 1: Write the failing test**

```python
# tests/whisperx_pipeline/test_quality_metrics.py
import pytest
from scripts.whisperx_pipeline.quality_metrics import (
    compute_quality_metrics,
    compute_narrator_ratio,
    detect_speaker_flip,
    QualityMetrics,
    QualityReason
)

def test_narrator_ratio_high_for_monologue():
    """Narrator ratio is high when one speaker dominates with long turns."""
    # Speaker 0 talks for 50s in one turn, Speaker 1 says 3 words
    words = [
        {"spk": "SPEAKER_00", "t0_abs": 0.0, "t1_abs": 50.0, "text": "long monologue " * 100},
        {"spk": "SPEAKER_01", "t0_abs": 50.5, "t1_abs": 52.0, "text": "okay thanks bye"},
    ]

    ratio = compute_narrator_ratio(words, duration_s=55.0)
    assert ratio > 0.7  # High narrator ratio

def test_narrator_ratio_low_for_dialogue():
    """Narrator ratio is low for back-and-forth conversation."""
    words = []
    for i in range(20):
        spk = "SPEAKER_00" if i % 2 == 0 else "SPEAKER_01"
        words.append({
            "spk": spk,
            "t0_abs": i * 3.0,
            "t1_abs": i * 3.0 + 2.5,
            "text": f"turn {i} content here"
        })

    ratio = compute_narrator_ratio(words, duration_s=60.0)
    assert ratio < 0.3  # Low narrator ratio

def test_detect_speaker_flip_finds_swap():
    """Detect when speakers swap roles mid-call."""
    # First half: SPEAKER_00 leads (agent pattern)
    # Second half: SPEAKER_01 leads (agent pattern)
    words = [
        # First 30s - SPEAKER_00 asks questions, SPEAKER_01 short answers
        {"spk": "SPEAKER_00", "t0_abs": 0, "t1_abs": 5, "text": "Hi is this John calling about your property?"},
        {"spk": "SPEAKER_01", "t0_abs": 5, "t1_abs": 6, "text": "Yes"},
        {"spk": "SPEAKER_00", "t0_abs": 6, "t1_abs": 12, "text": "Great I wanted to ask about the listing"},
        {"spk": "SPEAKER_01", "t0_abs": 12, "t1_abs": 13, "text": "Okay"},
        # Second 30s - roles flip
        {"spk": "SPEAKER_01", "t0_abs": 30, "t1_abs": 38, "text": "So let me tell you about our services and pricing"},
        {"spk": "SPEAKER_00", "t0_abs": 38, "t1_abs": 39, "text": "Uh huh"},
        {"spk": "SPEAKER_01", "t0_abs": 39, "t1_abs": 48, "text": "We offer comprehensive coverage for your needs"},
        {"spk": "SPEAKER_00", "t0_abs": 48, "t1_abs": 49, "text": "I see"},
    ]

    flip_detected, reason = detect_speaker_flip(words)
    assert flip_detected
    assert reason.type == "speaker_flip_suspected"

def test_quality_metrics_struct():
    """QualityMetrics has all required fields."""
    metrics = QualityMetrics(
        overlap_ratio=0.05,
        speaker_switches_per_min=25.0,
        median_turn_s=2.5,
        micro_turn_ratio=0.1,
        narrator_ratio=0.2,
        speaker_flip_suspected=False
    )
    assert metrics.narrator_ratio == 0.2

def test_overlap_ratio_no_overcount():
    """Overlap ratio uses sweep line to avoid overcounting >2 overlapping segments.

    If 3 segments overlap at t=5-7:
    - Pairwise would count: (A,B) + (A,C) + (B,C) = 6s of overlap (WRONG)
    - Sweep line counts: 2s where active_speakers >= 2 (CORRECT)
    """
    from scripts.whisperx_pipeline.quality_metrics import _compute_overlap_ratio

    # Three segments all overlapping at t=5-7
    segments = [
        {"t0_abs": 0.0, "t1_abs": 7.0, "spk": "A"},   # 0-7
        {"t0_abs": 5.0, "t1_abs": 10.0, "spk": "B"},  # 5-10
        {"t0_abs": 5.0, "t1_abs": 8.0, "spk": "C"},   # 5-8
    ]

    ratio = _compute_overlap_ratio(segments)
    # Total duration: 0-10 = 10s
    # Overlap time: 5-8 = 3s (where 2+ speakers active)
    # Expected ratio: 3/10 = 0.3
    assert 0.25 < ratio < 0.35, f"Expected ~0.3, got {ratio}"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/whisperx_pipeline/test_quality_metrics.py -v
```
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# scripts/whisperx_pipeline/quality_metrics.py
"""Quality metrics for diarization output."""
from dataclasses import dataclass
from typing import List, Tuple, Optional
from statistics import median

@dataclass
class QualityReason:
    type: str
    t0_abs: float
    t1_abs: float
    severity: float
    details: Optional[str] = None

@dataclass
class QualityMetrics:
    overlap_ratio: float
    speaker_switches_per_min: float
    median_turn_s: float
    micro_turn_ratio: float
    narrator_ratio: float
    speaker_flip_suspected: bool

def compute_narrator_ratio(words: List[dict], duration_s: float) -> float:
    """
    Compute narrator ratio: how much the chunk resembles narration vs dialogue.

    High ratio (>0.7) = likely narration/monologue (one speaker dominates with long turns)
    Low ratio (<0.3) = likely dialogue (back-and-forth conversation)

    Method:
    1. Compute talk time per speaker
    2. Compute average turn duration per speaker
    3. If dominant speaker has >80% talk time AND avg turn >10s, it's narration
    """
    if not words or duration_s <= 0:
        return 0.0

    # Group consecutive words by speaker into turns
    turns = []
    current_spk = None
    current_start = None
    current_end = None

    for w in words:
        if w["spk"] != current_spk:
            if current_spk is not None:
                turns.append({"spk": current_spk, "start": current_start, "end": current_end})
            current_spk = w["spk"]
            current_start = w["t0_abs"]
        current_end = w["t1_abs"]

    if current_spk is not None:
        turns.append({"spk": current_spk, "start": current_start, "end": current_end})

    if not turns:
        return 0.0

    # Compute per-speaker stats
    speaker_stats = {}
    for turn in turns:
        spk = turn["spk"]
        dur = turn["end"] - turn["start"]
        if spk not in speaker_stats:
            speaker_stats[spk] = {"talk_time": 0.0, "turn_count": 0, "turn_durations": []}
        speaker_stats[spk]["talk_time"] += dur
        speaker_stats[spk]["turn_count"] += 1
        speaker_stats[spk]["turn_durations"].append(dur)

    if len(speaker_stats) < 2:
        # Only one speaker = definitely narration
        return 1.0

    # Find dominant speaker
    total_talk = sum(s["talk_time"] for s in speaker_stats.values())
    if total_talk <= 0:
        return 0.0

    dominant_spk = max(speaker_stats.keys(), key=lambda s: speaker_stats[s]["talk_time"])
    dominant = speaker_stats[dominant_spk]

    talk_ratio = dominant["talk_time"] / total_talk
    avg_turn = dominant["talk_time"] / dominant["turn_count"] if dominant["turn_count"] > 0 else 0

    # Narrator pattern: >70% talk time AND avg turn >8s
    if talk_ratio > 0.7 and avg_turn > 8.0:
        return min(1.0, talk_ratio * (avg_turn / 10.0))

    # Moderate narration: >60% talk time AND avg turn >5s
    if talk_ratio > 0.6 and avg_turn > 5.0:
        return talk_ratio * 0.7

    # Dialogue pattern
    return talk_ratio * 0.3

def detect_speaker_flip(words: List[dict], window_s: float = 30.0) -> Tuple[bool, Optional[QualityReason]]:
    """
    Detect if speakers appear to swap roles mid-call.

    This is different from high switch rate (which can be normal rapid dialogue).
    Speaker flip = the "agent pattern" (longer turns, questions) moves from one speaker to another.

    Method:
    1. Split into first half and second half
    2. Compute "agent score" per speaker in each half
    3. If dominant agent flips, flag it
    """
    if not words:
        return False, None

    # Get time range
    min_t = min(w["t0_abs"] for w in words)
    max_t = max(w["t1_abs"] for w in words)
    mid_t = (min_t + max_t) / 2

    if max_t - min_t < 20:  # Too short to detect flip
        return False, None

    first_half = [w for w in words if w["t0_abs"] < mid_t]
    second_half = [w for w in words if w["t0_abs"] >= mid_t]

    def agent_score(word_list: List[dict]) -> dict:
        """Compute agent-like score per speaker (longer turns, more questions)."""
        turns = _group_into_turns(word_list)
        scores = {}
        for spk, spk_turns in _group_turns_by_speaker(turns).items():
            if not spk_turns:
                scores[spk] = 0
                continue
            avg_dur = sum(t["end"] - t["start"] for t in spk_turns) / len(spk_turns)
            question_ratio = sum(1 for t in spk_turns if "?" in t.get("text", "")) / len(spk_turns)
            scores[spk] = avg_dur * (1 + question_ratio)
        return scores

    first_scores = agent_score(first_half)
    second_scores = agent_score(second_half)

    if not first_scores or not second_scores:
        return False, None

    first_agent = max(first_scores.keys(), key=lambda s: first_scores.get(s, 0))
    second_agent = max(second_scores.keys(), key=lambda s: second_scores.get(s, 0))

    # Check if agent flipped AND the scores are meaningfully different
    if first_agent != second_agent:
        first_gap = first_scores.get(first_agent, 0) - first_scores.get(second_agent, 0)
        second_gap = second_scores.get(second_agent, 0) - second_scores.get(first_agent, 0)

        if first_gap > 1.0 and second_gap > 1.0:
            return True, QualityReason(
                type="speaker_flip_suspected",
                t0_abs=mid_t - 5,
                t1_abs=mid_t + 5,
                severity=min(1.0, (first_gap + second_gap) / 10),
                details=f"Agent pattern moved from {first_agent} to {second_agent}"
            )

    return False, None

def _group_into_turns(words: List[dict]) -> List[dict]:
    """Group consecutive words by speaker into turns."""
    if not words:
        return []
    turns = []
    current = {"spk": words[0]["spk"], "start": words[0]["t0_abs"], "end": words[0]["t1_abs"], "text": words[0].get("text", "")}
    for w in words[1:]:
        if w["spk"] == current["spk"]:
            current["end"] = w["t1_abs"]
            current["text"] += " " + w.get("text", "")
        else:
            turns.append(current)
            current = {"spk": w["spk"], "start": w["t0_abs"], "end": w["t1_abs"], "text": w.get("text", "")}
    turns.append(current)
    return turns

def _group_turns_by_speaker(turns: List[dict]) -> dict:
    """Group turns by speaker."""
    result = {}
    for t in turns:
        spk = t["spk"]
        if spk not in result:
            result[spk] = []
        result[spk].append(t)
    return result

def compute_quality_metrics(
    words: List[dict],
    diarization_segments: List[dict],
    duration_s: float
) -> Tuple[QualityMetrics, List[QualityReason]]:
    """Compute all quality metrics for a chunk."""
    reasons = []

    # Overlap ratio (from diarization segments)
    overlap_ratio = _compute_overlap_ratio(diarization_segments)

    # Speaker switches per minute
    turns = _group_into_turns(words)
    switches = len(turns) - 1 if turns else 0
    switches_per_min = (switches / duration_s) * 60 if duration_s > 0 else 0

    # Median turn duration
    turn_durations = [t["end"] - t["start"] for t in turns]
    median_turn = median(turn_durations) if turn_durations else 0

    # Micro-turn ratio (turns < 0.7s)
    micro_turns = sum(1 for d in turn_durations if d < 0.7)
    micro_ratio = micro_turns / len(turns) if turns else 0

    # Narrator ratio
    narrator_ratio = compute_narrator_ratio(words, duration_s)

    # Speaker flip detection
    flip_detected, flip_reason = detect_speaker_flip(words)
    if flip_reason:
        reasons.append(flip_reason)

    # Add high overlap reason if needed
    if overlap_ratio > 0.15:
        reasons.append(QualityReason(
            type="high_overlap",
            t0_abs=0,
            t1_abs=duration_s,
            severity=overlap_ratio
        ))

    return QualityMetrics(
        overlap_ratio=overlap_ratio,
        speaker_switches_per_min=switches_per_min,
        median_turn_s=median_turn,
        micro_turn_ratio=micro_ratio,
        narrator_ratio=narrator_ratio,
        speaker_flip_suspected=flip_detected
    ), reasons

def _compute_overlap_ratio(segments: List[dict]) -> float:
    """
    Compute ratio of overlapping speech from diarization segments.

    Uses sweep line algorithm to avoid overcounting when >2 segments overlap.
    """
    if len(segments) < 2:
        return 0.0

    total_duration = max(s["t1_abs"] for s in segments) - min(s["t0_abs"] for s in segments)
    if total_duration <= 0:
        return 0.0

    # Sweep line: create events (+1 at start, -1 at end)
    events = []
    for seg in segments:
        events.append((seg["t0_abs"], +1))  # segment starts
        events.append((seg["t1_abs"], -1))  # segment ends

    events.sort(key=lambda e: (e[0], -e[1]))  # Sort by time, starts before ends at same time

    overlap_time = 0.0
    active_speakers = 0
    prev_time = None

    for time, delta in events:
        if prev_time is not None and active_speakers >= 2:
            overlap_time += time - prev_time
        active_speakers += delta
        prev_time = time

    return overlap_time / total_duration
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/whisperx_pipeline/test_quality_metrics.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/whisperx_pipeline/quality_metrics.py tests/whisperx_pipeline/test_quality_metrics.py
git commit -m "feat(whisperx): add quality metrics with narrator ratio and speaker flip detection"
```

---

### Task 5: Create Fuzzy Lexical Matcher for Call Splitting

**Files:**
- Create: `scripts/whisperx_pipeline/lexical_matcher.py`
- Create: `tests/whisperx_pipeline/test_lexical_matcher.py`

**Step 1: Write the failing test**

```python
# tests/whisperx_pipeline/test_lexical_matcher.py
import pytest
from scripts.whisperx_pipeline.lexical_matcher import (
    FuzzyLexicalMatcher,
    tokenize,
    token_similarity
)

def test_tokenize_normalizes():
    """Tokenize lowercases and strips punctuation."""
    assert tokenize("Hi, is this John?") == ["hi", "is", "this", "john"]
    assert tokenize("Hello! My name is") == ["hello", "my", "name", "is"]

def test_token_similarity_exact():
    """Exact token match has similarity 1.0."""
    assert token_similarity(["hi", "is", "this"], ["hi", "is", "this"]) == 1.0

def test_token_similarity_partial():
    """Partial overlap has proportional similarity."""
    # 2 out of 3 tokens match
    sim = token_similarity(["hi", "is", "this"], ["hi", "is", "that"])
    assert 0.6 < sim < 0.8

def test_fuzzy_matcher_finds_greeting():
    """Fuzzy matcher finds greeting despite ASR variance."""
    matcher = FuzzyLexicalMatcher(
        patterns=[["hi", "is", "this"], ["hello", "is", "this"]],
        threshold=0.7
    )

    # ASR might transcribe "Hi, is this John" as "Hi is this John" or "Hi, this is John"
    words = [
        {"text": "Hi", "t0_abs": 0.0},
        {"text": "is", "t0_abs": 0.2},
        {"text": "this", "t0_abs": 0.4},
        {"text": "John", "t0_abs": 0.6},
    ]

    matches = matcher.find_matches(words)
    assert len(matches) == 1
    assert matches[0]["pattern"] == ["hi", "is", "this"]
    assert matches[0]["t_abs"] == 0.0

def test_fuzzy_matcher_handles_asr_variance():
    """Matcher tolerates ASR errors like 'high' instead of 'hi'."""
    matcher = FuzzyLexicalMatcher(
        patterns=[["hi", "is", "this"]],
        threshold=0.6  # Lower threshold for fuzzy
    )

    # ASR transcribed "hi" as "high"
    words = [
        {"text": "High", "t0_abs": 0.0},
        {"text": "is", "t0_abs": 0.2},
        {"text": "this", "t0_abs": 0.4},
    ]

    matches = matcher.find_matches(words)
    # Should still find it because 2/3 tokens match exactly
    assert len(matches) >= 1
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/whisperx_pipeline/test_lexical_matcher.py -v
```
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# scripts/whisperx_pipeline/lexical_matcher.py
"""Fuzzy lexical matching for call boundary detection."""
import re
from typing import List, Optional
from difflib import SequenceMatcher

def tokenize(text: str) -> List[str]:
    """Normalize text to lowercase tokens, strip punctuation."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()

def token_similarity(tokens_a: List[str], tokens_b: List[str]) -> float:
    """
    Compute similarity between two token sequences.
    Uses both exact match ratio and fuzzy character matching.
    """
    if not tokens_a or not tokens_b:
        return 0.0

    # Exact token overlap
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    overlap = len(set_a & set_b)
    union = len(set_a | set_b)
    jaccard = overlap / union if union > 0 else 0

    # Sequential match (order matters for patterns)
    seq_matches = 0
    for i, tok_a in enumerate(tokens_a):
        if i < len(tokens_b):
            tok_b = tokens_b[i]
            if tok_a == tok_b:
                seq_matches += 1
            elif SequenceMatcher(None, tok_a, tok_b).ratio() > 0.8:
                seq_matches += 0.8  # Partial credit for similar tokens

    seq_ratio = seq_matches / max(len(tokens_a), len(tokens_b))

    # Combine both metrics
    return (jaccard + seq_ratio) / 2

class FuzzyLexicalMatcher:
    """Match greeting/reset patterns with fuzzy tolerance for ASR errors."""

    def __init__(self, patterns: List[List[str]], threshold: float = 0.7):
        self.patterns = patterns
        self.threshold = threshold

    def find_matches(self, words: List[dict]) -> List[dict]:
        """
        Find all pattern matches in word sequence.
        Returns list of matches with pattern, timestamp, and confidence.
        """
        matches = []

        for i in range(len(words)):
            for pattern in self.patterns:
                # Extract window of tokens
                window_size = len(pattern) + 1  # Allow one extra for flexibility
                window_words = words[i:i + window_size]

                if len(window_words) < len(pattern):
                    continue

                window_tokens = [tokenize(w.get("text", ""))[0] if tokenize(w.get("text", "")) else ""
                                for w in window_words]

                # Try different alignments within window
                best_sim = 0
                for offset in range(min(2, len(window_tokens) - len(pattern) + 1)):
                    candidate = window_tokens[offset:offset + len(pattern)]
                    sim = token_similarity(pattern, candidate)
                    best_sim = max(best_sim, sim)

                if best_sim >= self.threshold:
                    matches.append({
                        "pattern": pattern,
                        "t_abs": words[i]["t0_abs"],
                        "confidence": best_sim,
                        "word_index": i
                    })
                    break  # Only match one pattern per position

        return matches

    def find_greeting_resets(
        self,
        words: List[dict],
        min_gap_from_start_s: float = 30.0
    ) -> List[dict]:
        """
        Find greeting patterns that indicate a new call started.
        Only considers matches after min_gap_from_start_s.
        """
        if not words:
            return []

        start_t = words[0]["t0_abs"]
        all_matches = self.find_matches(words)

        # Filter to matches after the initial greeting window
        resets = [
            m for m in all_matches
            if m["t_abs"] > start_t + min_gap_from_start_s
        ]

        return resets
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/whisperx_pipeline/test_lexical_matcher.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/whisperx_pipeline/lexical_matcher.py tests/whisperx_pipeline/test_lexical_matcher.py
git commit -m "feat(whisperx): add fuzzy lexical matcher for call boundary detection"
```

---

*Continued in Part 2...*

---

## Phase 1 Continued: Core Pipeline

### Task 6: Create WhisperX Transcription Module

**Files:**
- Create: `scripts/whisperx_pipeline/transcriber.py`
- Create: `tests/whisperx_pipeline/test_transcriber.py`

**Step 1: Write the failing test**

```python
# tests/whisperx_pipeline/test_transcriber.py
import pytest
from scripts.whisperx_pipeline.transcriber import WhisperXTranscriber, WordOutput
from scripts.whisperx_pipeline.config import WhisperXConfig

def test_word_output_dataclass():
    """WordOutput has required fields."""
    word = WordOutput(
        i=0,
        t0=0.5,
        t1=0.8,
        t0_abs=100.5,
        t1_abs=100.8,
        text="Hello",
        text_norm="hello",
        spk="SPEAKER_00"
    )
    assert word.text_norm == "hello"

def test_transcriber_config():
    """Transcriber accepts config."""
    config = WhisperXConfig(model="large-v3", batch_size=8)
    transcriber = WhisperXTranscriber(config)
    assert transcriber.config.model == "large-v3"

# Integration test (requires GPU, skip in CI)
@pytest.mark.skip(reason="Requires GPU and model download")
def test_transcriber_produces_words():
    """Transcriber produces word list with timestamps."""
    config = WhisperXConfig(model="large-v3")
    transcriber = WhisperXTranscriber(config)
    # Would test with actual audio
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/whisperx_pipeline/test_transcriber.py -v -k "not skip"
```
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# scripts/whisperx_pipeline/transcriber.py
"""WhisperX transcription with diarization."""
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
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
        min_speakers: int = 2,
        max_speakers: int = 3
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

        # 1. ASR
        result = self._model.transcribe(audio, batch_size=self.config.batch_size)
        language = result.get("language", "en")

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
        for i, seg in enumerate(result.get("segments", [])):
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

        # Extract diarization segments
        dia_segments = []
        for seg in diarize_segments.itertracks(yield_label=True):
            turn, _, speaker = seg
            dia_segments.append(DiarizationSegment(
                t0_abs=turn.start + chunk_time_offset_s,
                t1_abs=turn.end + chunk_time_offset_s,
                spk=speaker
            ))

        return words, dia_segments
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/whisperx_pipeline/test_transcriber.py -v -k "not skip"
```
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/whisperx_pipeline/transcriber.py tests/whisperx_pipeline/test_transcriber.py
git commit -m "feat(whisperx): add WhisperX transcription module with large-v3"
```

---

### Task 7: Create Chunk Output Writer

**Files:**
- Create: `scripts/whisperx_pipeline/chunk_writer.py`
- Create: `tests/whisperx_pipeline/test_chunk_writer.py`

**Step 1: Write the failing test**

```python
# tests/whisperx_pipeline/test_chunk_writer.py
import pytest
import json
import tempfile
from pathlib import Path
from scripts.whisperx_pipeline.chunk_writer import ChunkWriter
from scripts.whisperx_pipeline.transcriber import WordOutput, DiarizationSegment
from scripts.whisperx_pipeline.quality_metrics import QualityMetrics, QualityReason
from scripts.whisperx_pipeline.config import PipelineConfig

def test_chunk_writer_creates_words_json():
    """ChunkWriter creates words.json with correct schema."""
    config = PipelineConfig()
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = ChunkWriter(base_path=tmpdir, config=config)

        words = [
            WordOutput(i=0, t0=0.5, t1=0.8, t0_abs=0.5, t1_abs=0.8,
                      text="Hello", text_norm="hello", spk="SPEAKER_00")
        ]
        segments = [
            DiarizationSegment(t0_abs=0.5, t1_abs=2.0, spk="SPEAKER_00")
        ]
        metrics = QualityMetrics(
            overlap_ratio=0.05, speaker_switches_per_min=20,
            median_turn_s=2.0, micro_turn_ratio=0.1,
            narrator_ratio=0.2, speaker_flip_suspected=False
        )

        writer.write_chunk(
            video_id="test_video",
            chunk_id="test_video_0_30000",
            chunk_time_offset_s=0.0,
            words=words,
            diarization_segments=segments,
            quality_metrics=metrics,
            quality_reasons=[],
            content_type="call_like",
            duration_s=30.0
        )

        # Verify files created (chunks/<chunk_id>/ per design)
        chunk_dir = Path(tmpdir) / "chunks" / "test_video_0_30000"
        assert (chunk_dir / "words.json").exists()
        assert (chunk_dir / "diarization_segments.json").exists()
        assert (chunk_dir / "chunk_metadata.json").exists()

        # Verify words.json schema
        with open(chunk_dir / "words.json") as f:
            data = json.load(f)
        assert data["video_id"] == "test_video"
        assert data["asr_model"] == "whisper-large-v3"
        assert len(data["words"]) == 1
        assert data["words"][0]["text_norm"] == "hello"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/whisperx_pipeline/test_chunk_writer.py -v
```
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# scripts/whisperx_pipeline/chunk_writer.py
"""Write chunk outputs to local filesystem or S3."""
import json
from pathlib import Path
from typing import List, Optional
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
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/whisperx_pipeline/test_chunk_writer.py -v
```
Expected: PASS

**Step 5: Add CallWriter class (same file)**

```python
# Add to scripts/whisperx_pipeline/chunk_writer.py

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
        spk_turns: List,  # List[Turn]
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
```

**Step 6: Commit**

```bash
git add scripts/whisperx_pipeline/chunk_writer.py tests/whisperx_pipeline/test_chunk_writer.py
git commit -m "feat(whisperx): add chunk and call output writers"
```

---

### Task 8: Create Call Splitter Module

**Files:**
- Create: `scripts/whisperx_pipeline/call_splitter.py`
- Create: `tests/whisperx_pipeline/test_call_splitter.py`

**Step 1: Write the failing test**

```python
# tests/whisperx_pipeline/test_call_splitter.py
import pytest
from scripts.whisperx_pipeline.call_splitter import CallSplitter, CallBoundary
from scripts.whisperx_pipeline.config import SplitCallsConfig

def test_call_boundary_dataclass():
    """CallBoundary has required fields."""
    boundary = CallBoundary(
        call_id="video_0_30000",
        start_abs=0.0,
        end_abs=30.0,
        boundary_confidence=0.9,
        start_evidence={"type": "start_of_audio"},
        end_evidence={"type": "silence_gap", "gap_s": 3.5}
    )
    assert boundary.boundary_confidence == 0.9

def test_splitter_finds_silence_boundaries():
    """Splitter detects call boundaries at silence gaps."""
    config = SplitCallsConfig(silence_threshold_s=2.5)
    splitter = CallSplitter(config, video_id="test")

    # Two calls with 3s silence gap between them
    words = [
        # Call 1: 0-25s
        {"t0_abs": 0.0, "t1_abs": 0.5, "text": "Hi", "spk": "SPEAKER_00"},
        {"t0_abs": 24.5, "t1_abs": 25.0, "text": "bye", "spk": "SPEAKER_00"},
        # Gap: 25s to 28s (3s silence)
        # Call 2: 28-50s
        {"t0_abs": 28.0, "t1_abs": 28.5, "text": "Hello", "spk": "SPEAKER_00"},
        {"t0_abs": 49.5, "t1_abs": 50.0, "text": "thanks", "spk": "SPEAKER_01"},
    ]

    boundaries = splitter.find_boundaries(words)
    assert len(boundaries) == 2

def test_splitter_finds_greeting_resets():
    """Splitter detects new calls at greeting patterns."""
    config = SplitCallsConfig(
        silence_threshold_s=2.5,
        greeting_tokens=[["hi", "is", "this"], ["hello", "is", "this"]]
    )
    splitter = CallSplitter(config, video_id="test")

    # One audio with greeting reset mid-stream (no long silence)
    words = [
        {"t0_abs": 0.0, "t1_abs": 0.3, "text": "Hi", "spk": "SPEAKER_00"},
        {"t0_abs": 0.3, "t1_abs": 0.5, "text": "is", "spk": "SPEAKER_00"},
        {"t0_abs": 0.5, "t1_abs": 0.8, "text": "this", "spk": "SPEAKER_00"},
        {"t0_abs": 0.8, "t1_abs": 1.2, "text": "John", "spk": "SPEAKER_00"},
        {"t0_abs": 30.0, "t1_abs": 30.3, "text": "okay", "spk": "SPEAKER_01"},
        {"t0_abs": 30.5, "t1_abs": 30.8, "text": "bye", "spk": "SPEAKER_00"},
        # Short gap (1.2s) but greeting reset
        {"t0_abs": 32.0, "t1_abs": 32.3, "text": "Hi", "spk": "SPEAKER_00"},
        {"t0_abs": 32.3, "t1_abs": 32.5, "text": "is", "spk": "SPEAKER_00"},
        {"t0_abs": 32.5, "t1_abs": 32.8, "text": "this", "spk": "SPEAKER_00"},
        {"t0_abs": 32.8, "t1_abs": 33.2, "text": "Mary", "spk": "SPEAKER_00"},
    ]

    boundaries = splitter.find_boundaries(words)
    assert len(boundaries) == 2
    assert boundaries[1].start_evidence["type"] == "greeting_reset"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/whisperx_pipeline/test_call_splitter.py -v
```
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# scripts/whisperx_pipeline/call_splitter.py
"""Split merged word stream into individual calls."""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from .config import SplitCallsConfig
from .lexical_matcher import FuzzyLexicalMatcher

@dataclass
class CallBoundary:
    call_id: str
    start_abs: float
    end_abs: float
    boundary_confidence: float
    start_evidence: Dict[str, Any]
    end_evidence: Dict[str, Any]

class CallSplitter:
    """Split word stream into calls using silence + lexical resets."""

    def __init__(self, config: SplitCallsConfig, video_id: str):
        self.config = config
        self.video_id = video_id
        self.matcher = FuzzyLexicalMatcher(
            patterns=config.greeting_tokens,
            threshold=config.fuzzy_match_threshold
        )

    def find_boundaries(self, words: List[dict]) -> List[CallBoundary]:
        """Find call boundaries in word stream."""
        if not words:
            return []

        # Find potential split points
        split_points = []

        # 1. Silence gaps
        for i in range(1, len(words)):
            gap = words[i]["t0_abs"] - words[i-1]["t1_abs"]
            if gap >= self.config.silence_threshold_s:
                split_points.append({
                    "t_abs": words[i]["t0_abs"],
                    "type": "silence_gap",
                    "gap_s": gap,
                    "confidence": min(1.0, gap / 5.0),
                    "word_index": i
                })

        # 2. Greeting resets (after initial 30s)
        greeting_matches = self.matcher.find_greeting_resets(
            words,
            min_gap_from_start_s=30.0
        )
        for match in greeting_matches:
            # Check if there isn't already a silence split nearby
            is_near_silence = any(
                abs(sp["t_abs"] - match["t_abs"]) < 5.0
                for sp in split_points if sp["type"] == "silence_gap"
            )
            if not is_near_silence:
                split_points.append({
                    "t_abs": match["t_abs"],
                    "type": "greeting_reset",
                    "phrase_debug": " ".join(match["pattern"]),
                    "confidence": match["confidence"],
                    "word_index": match["word_index"]
                })

        # Sort split points by time
        split_points.sort(key=lambda x: x["t_abs"])

        # Build call boundaries
        boundaries = []
        current_start = words[0]["t0_abs"]
        current_start_evidence = {"type": "start_of_audio"}

        for sp in split_points:
            # End current call
            end_t = words[sp["word_index"] - 1]["t1_abs"] if sp["word_index"] > 0 else sp["t_abs"]
            boundaries.append(self._make_boundary(
                start=current_start,
                end=end_t,
                start_evidence=current_start_evidence,
                end_evidence=sp,
                confidence=sp["confidence"]
            ))

            # Start new call
            current_start = sp["t_abs"]
            current_start_evidence = sp

        # Add final call
        if words:
            boundaries.append(self._make_boundary(
                start=current_start,
                end=words[-1]["t1_abs"],
                start_evidence=current_start_evidence,
                end_evidence={"type": "end_of_audio"},
                confidence=1.0
            ))

        return boundaries

    def _make_boundary(
        self,
        start: float,
        end: float,
        start_evidence: dict,
        end_evidence: dict,
        confidence: float
    ) -> CallBoundary:
        """Create CallBoundary with computed ID."""
        start_ms = int(start * 1000)
        end_ms = int(end * 1000)
        return CallBoundary(
            call_id=f"{self.video_id}_{start_ms}_{end_ms}",
            start_abs=start,
            end_abs=end,
            boundary_confidence=confidence,
            start_evidence=start_evidence,
            end_evidence=end_evidence
        )

    def split_words(
        self,
        words: List[dict],
        boundaries: List[CallBoundary]
    ) -> List[List[dict]]:
        """Split word list according to boundaries."""
        calls = []
        for boundary in boundaries:
            call_words = [
                w for w in words
                if boundary.start_abs <= w["t0_abs"] < boundary.end_abs
            ]
            calls.append(call_words)
        return calls
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/whisperx_pipeline/test_call_splitter.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/whisperx_pipeline/call_splitter.py tests/whisperx_pipeline/test_call_splitter.py
git commit -m "feat(whisperx): add call splitter with fuzzy greeting detection"
```

---

### Task 9: Create Turn Builder Module

**Files:**
- Create: `scripts/whisperx_pipeline/turn_builder.py`
- Create: `tests/whisperx_pipeline/test_turn_builder.py`

**Step 1: Write the failing test**

```python
# tests/whisperx_pipeline/test_turn_builder.py
import pytest
from scripts.whisperx_pipeline.turn_builder import TurnBuilder, Turn

def test_turn_dataclass():
    """Turn has required fields."""
    turn = Turn(
        turn_id=0,
        spk="SPEAKER_00",
        text="Hello there",
        t0_abs=0.0,
        t1_abs=1.5,
        word_span_start=0,
        word_span_end=2
    )
    assert turn.text == "Hello there"

def test_builder_groups_consecutive_words():
    """Builder groups consecutive same-speaker words into turns."""
    builder = TurnBuilder()

    words = [
        {"spk": "SPEAKER_00", "text": "Hi", "t0_abs": 0.0, "t1_abs": 0.3},
        {"spk": "SPEAKER_00", "text": "there", "t0_abs": 0.3, "t1_abs": 0.6},
        {"spk": "SPEAKER_01", "text": "Hello", "t0_abs": 0.8, "t1_abs": 1.1},
        {"spk": "SPEAKER_00", "text": "How", "t0_abs": 1.3, "t1_abs": 1.5},
        {"spk": "SPEAKER_00", "text": "are", "t0_abs": 1.5, "t1_abs": 1.7},
        {"spk": "SPEAKER_00", "text": "you", "t0_abs": 1.7, "t1_abs": 2.0},
    ]

    turns = builder.build_turns(words)

    assert len(turns) == 3
    assert turns[0].text == "Hi there"
    assert turns[0].spk == "SPEAKER_00"
    assert turns[1].text == "Hello"
    assert turns[1].spk == "SPEAKER_01"
    assert turns[2].text == "How are you"

def test_builder_tracks_word_spans():
    """Builder correctly tracks word span indices."""
    builder = TurnBuilder()

    words = [
        {"spk": "SPEAKER_00", "text": "One", "t0_abs": 0.0, "t1_abs": 0.3},
        {"spk": "SPEAKER_00", "text": "two", "t0_abs": 0.3, "t1_abs": 0.6},
        {"spk": "SPEAKER_01", "text": "Three", "t0_abs": 0.8, "t1_abs": 1.1},
    ]

    turns = builder.build_turns(words)

    assert turns[0].word_span_start == 0
    assert turns[0].word_span_end == 2  # exclusive
    assert turns[1].word_span_start == 2
    assert turns[1].word_span_end == 3

def test_builder_merges_small_gaps():
    """Builder merges same-speaker words with small gaps."""
    builder = TurnBuilder(max_gap_s=0.3)

    # Same speaker with small gap (0.2s) - should merge
    words = [
        {"spk": "SPEAKER_00", "text": "Hi", "t0_abs": 0.0, "t1_abs": 0.3},
        {"spk": "SPEAKER_00", "text": "there", "t0_abs": 0.5, "t1_abs": 0.8},  # 0.2s gap
    ]

    turns = builder.build_turns(words)
    assert len(turns) == 1
    assert turns[0].text == "Hi there"

def test_builder_splits_large_gaps():
    """Builder splits same-speaker words with large gaps (diarization artifacts)."""
    builder = TurnBuilder(max_gap_s=0.3)

    # Same speaker but large gap (2s) - should split
    words = [
        {"spk": "SPEAKER_00", "text": "First sentence", "t0_abs": 0.0, "t1_abs": 1.0},
        {"spk": "SPEAKER_00", "text": "Second sentence", "t0_abs": 3.0, "t1_abs": 4.0},  # 2s gap
    ]

    turns = builder.build_turns(words)
    assert len(turns) == 2
    assert turns[0].text == "First sentence"
    assert turns[1].text == "Second sentence"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/whisperx_pipeline/test_turn_builder.py -v
```
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# scripts/whisperx_pipeline/turn_builder.py
"""Build speaker turns from word stream."""
from dataclasses import dataclass
from typing import List

@dataclass
class Turn:
    turn_id: int
    spk: str
    text: str
    t0_abs: float
    t1_abs: float
    word_span_start: int  # inclusive
    word_span_end: int    # exclusive (half-open)

class TurnBuilder:
    """Build turns by grouping consecutive same-speaker words."""

    def __init__(self, max_gap_s: float = 0.3):
        """
        Args:
            max_gap_s: If same speaker resumes within this gap, merge into same turn.
                       Prevents overly-aggressive turn splitting from ASR segmentation weirdness.
        """
        self.max_gap_s = max_gap_s

    def build_turns(self, words: List[dict]) -> List[Turn]:
        """Build turns from word list with gap-based merging."""
        if not words:
            return []

        turns = []
        current_spk = words[0]["spk"]
        current_texts = [words[0]["text"]]
        current_start = words[0]["t0_abs"]
        current_end = words[0]["t1_abs"]
        span_start = 0

        for i, word in enumerate(words[1:], start=1):
            same_speaker = word["spk"] == current_spk
            gap = word["t0_abs"] - current_end

            # Merge if same speaker AND gap is small enough
            # Large gaps (even same speaker) indicate diarization artifacts or long pauses
            if same_speaker and gap <= self.max_gap_s:
                # Continue current turn
                current_texts.append(word["text"])
                current_end = word["t1_abs"]
            else:
                # Save current turn
                turns.append(Turn(
                    turn_id=len(turns),
                    spk=current_spk,
                    text=" ".join(current_texts),
                    t0_abs=current_start,
                    t1_abs=current_end,
                    word_span_start=span_start,
                    word_span_end=i
                ))

                # Start new turn
                current_spk = word["spk"]
                current_texts = [word["text"]]
                current_start = word["t0_abs"]
                current_end = word["t1_abs"]
                span_start = i

        # Add final turn
        turns.append(Turn(
            turn_id=len(turns),
            spk=current_spk,
            text=" ".join(current_texts),
            t0_abs=current_start,
            t1_abs=current_end,
            word_span_start=span_start,
            word_span_end=len(words)
        ))

        return turns
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/whisperx_pipeline/test_turn_builder.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/whisperx_pipeline/turn_builder.py tests/whisperx_pipeline/test_turn_builder.py
git commit -m "feat(whisperx): add turn builder module"
```

---

### Task 10: Create Main Pipeline Orchestrator

**Files:**
- Create: `scripts/whisperx_pipeline/pipeline.py`
- Create: `tests/whisperx_pipeline/test_pipeline.py`

**Step 1: Write the failing test**

```python
# tests/whisperx_pipeline/test_pipeline.py
import pytest
from scripts.whisperx_pipeline.pipeline import WhisperXPipeline
from scripts.whisperx_pipeline.config import PipelineConfig

def test_pipeline_config():
    """Pipeline accepts config."""
    config = PipelineConfig()
    pipeline = WhisperXPipeline(config)
    assert pipeline.config.whisperx.model == "large-v3"

def test_pipeline_generates_run_id():
    """Pipeline generates unique run IDs."""
    config = PipelineConfig()
    pipeline = WhisperXPipeline(config)

    run_id_1 = pipeline._generate_run_id()
    run_id_2 = pipeline._generate_run_id()

    assert run_id_1 != run_id_2
    assert len(run_id_1) > 10  # Timestamp format
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/whisperx_pipeline/test_pipeline.py -v
```
Expected: FAIL

**Step 3: Write minimal implementation**

```python
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
        run_id: Optional[str] = None
    ) -> dict:
        """
        Process a single video through the full pipeline.

        Args:
            video_id: Unique video identifier
            audio_s3_key: S3 key for audio file
            run_id: Optional run ID (generated if not provided)

        Returns:
            Video manifest dict
        """
        run_id = run_id or self._generate_run_id()
        run_prefix = f"runs/{video_id}/{run_id}"

        manifest = {
            "video_id": video_id,
            "run_id": run_id,
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
                    # Log error, continue with next chunk

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

            # Upload to S3
            self._upload_outputs(tmpdir / "output", run_prefix)

            manifest["job_status"] = "completed_ok" if manifest["chunks"]["failed"] == 0 else "completed_with_errors"
            manifest["completed_at"] = datetime.now().isoformat()

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
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/whisperx_pipeline/test_pipeline.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/whisperx_pipeline/pipeline.py tests/whisperx_pipeline/test_pipeline.py
git commit -m "feat(whisperx): add main pipeline orchestrator"
```

---

*Phase 1 Complete. Continue to Phase 2 for Role Classifier Bootstrap.*

---

## Phase 2: Role Classifier Bootstrap

### Task 11: Create Role Feature Extractor

**Files:**
- Create: `scripts/whisperx_pipeline/role_features.py`
- Create: `tests/whisperx_pipeline/test_role_features.py`

**Step 1: Write the failing test**

```python
# tests/whisperx_pipeline/test_role_features.py
import pytest
from scripts.whisperx_pipeline.role_features import (
    extract_speaker_features,
    extract_call_features,
    is_question
)

def test_is_question_punctuation():
    """Question mark detected."""
    assert is_question("How are you?")
    assert not is_question("I am fine.")

def test_is_question_interrogative():
    """Interrogative words detected."""
    assert is_question("Do you have time")
    assert is_question("What is your name")
    assert is_question("Can I help you")

def test_extract_speaker_features():
    """Extract features for one speaker."""
    turns = [
        {"spk": "SPEAKER_00", "t0_abs": 0, "t1_abs": 5, "text": "Hi is this John?"},
        {"spk": "SPEAKER_01", "t0_abs": 5, "t1_abs": 6, "text": "Yes"},
        {"spk": "SPEAKER_00", "t0_abs": 6, "t1_abs": 12, "text": "Great how are you doing today?"},
    ]

    features = extract_speaker_features(turns, "SPEAKER_00", call_start_abs=0)

    assert features["turn_count"] == 2
    assert features["first_speaker"] == 1
    assert features["question_turn_rate"] == 1.0  # Both turns have questions

def test_extract_call_features_difference():
    """Call features are differences between speakers."""
    turns = [
        {"spk": "SPEAKER_00", "t0_abs": 0, "t1_abs": 5, "text": "Hello there"},
        {"spk": "SPEAKER_01", "t0_abs": 5, "t1_abs": 6, "text": "Hi"},
    ]

    features = extract_call_features("call_1", turns, call_start_abs=0)

    assert "diff_talk_time" in features
    assert features["diff_talk_time"] == 4.0  # 5s - 1s
```

**Step 2-5: Implement, test, commit**

```python
# scripts/whisperx_pipeline/role_features.py
"""Feature extraction for role classification."""
from typing import List, Dict
from statistics import mean, median

def safe_mean(values: List[float]) -> float:
    return mean(values) if values else 0.0

def safe_median(values: List[float]) -> float:
    return median(values) if values else 0.0

def safe_max(values: List[float]) -> float:
    return max(values) if values else 0.0

def is_question(text: str) -> bool:
    """Robust question detection."""
    text_lower = text.strip().lower()
    if text.rstrip().endswith("?"):
        return True
    words = text_lower.split()[:3]
    interrogatives = {"is", "are", "do", "did", "can", "could", "would",
                     "what", "why", "how", "when", "where", "who", "whose", "which"}
    return any(w in interrogatives for w in words)

def extract_speaker_features(
    all_turns: List[dict],
    speaker: str,
    call_start_abs: float,
    window_s: float = 60.0
) -> Dict[str, float]:
    """Extract features for one speaker from first 60s."""
    # Filter to first 60s and this speaker
    first_60s = [t for t in all_turns if t["t0_abs"] < call_start_abs + window_s]
    spk_turns = [t for t in first_60s if t["spk"] == speaker]

    if not spk_turns:
        return {k: 0.0 for k in ["talk_time", "turn_count", "avg_turn_duration",
                                  "median_turn_duration", "max_turn_duration",
                                  "question_turn_count", "question_turn_rate",
                                  "long_turn_count", "short_turn_count",
                                  "word_count", "avg_words_per_turn", "first_speaker"]}

    first_turn_spk = first_60s[0]["spk"] if first_60s else None
    durations = [t["t1_abs"] - t["t0_abs"] for t in spk_turns]
    word_counts = [len(t["text"].split()) for t in spk_turns]
    turn_count = len(spk_turns)
    question_count = sum(1 for t in spk_turns if is_question(t["text"]))

    return {
        "talk_time": sum(durations),
        "turn_count": turn_count,
        "avg_turn_duration": safe_mean(durations),
        "median_turn_duration": safe_median(durations),
        "max_turn_duration": safe_max(durations),
        "question_turn_count": question_count,
        "question_turn_rate": question_count / turn_count if turn_count > 0 else 0,
        "long_turn_count": sum(1 for d in durations if d > 5),
        "short_turn_count": sum(1 for d in durations if d < 0.6),
        "word_count": sum(word_counts),
        "avg_words_per_turn": safe_mean(word_counts),
        "first_speaker": 1 if speaker == first_turn_spk else 0,
    }

def extract_call_features(
    call_id: str,
    spk_turns: List[dict],
    call_start_abs: float
) -> Dict[str, float]:
    """Extract call-level difference features."""
    # Hardcoded speaker ordering
    spk0, spk1 = "SPEAKER_00", "SPEAKER_01"

    feat0 = extract_speaker_features(spk_turns, spk0, call_start_abs)
    feat1 = extract_speaker_features(spk_turns, spk1, call_start_abs)

    diff = {f"diff_{k}": feat0[k] - feat1[k] for k in feat0}
    return {"call_id": call_id, "spk0": spk0, "spk1": spk1, **diff}
```

```bash
git add scripts/whisperx_pipeline/role_features.py tests/whisperx_pipeline/test_role_features.py
git commit -m "feat(whisperx): add role feature extraction"
```

---

### Task 12: Create Role Labeling CLI

**Files:**
- Create: `scripts/whisperx_pipeline/label_roles.py`

**Implementation:**

```python
# scripts/whisperx_pipeline/label_roles.py
"""CLI for labeling speaker roles."""
import json
import csv
from pathlib import Path
from datetime import datetime
import sys

def load_queue(queue_path: str) -> list:
    """Load labeling queue."""
    items = []
    with open(queue_path) as f:
        for line in f:
            items.append(json.loads(line))
    return items

def load_existing_labels(labels_path: str) -> set:
    """Load already-labeled call IDs."""
    labeled = set()
    if Path(labels_path).exists():
        with open(labels_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                labeled.add(row["call_id"])
    return labeled

def save_label(labels_path: str, video_id: str, call_id: str, agent_spk: str):
    """Append label to CSV."""
    file_exists = Path(labels_path).exists()
    with open(labels_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["video_id", "call_id", "agent_spk", "labeled_at"])
        writer.writerow([video_id, call_id, agent_spk, datetime.now().isoformat()])

def display_call(item: dict, index: int, total: int):
    """Display call for labeling."""
    print("\n" + "=" * 60)
    print(f"Call: {item['call_id']} ({index + 1}/{total})")
    print("=" * 60)
    print(item.get("preview_text", "No preview available"))
    print("=" * 60)
    print("Who is the AGENT/CALLER?")
    print("  [0] SPEAKER_00    [1] SPEAKER_01    [s] Skip    [b] Back    [q] Quit")

def main(queue_path: str, labels_path: str):
    """Run labeling CLI."""
    items = load_queue(queue_path)
    labeled = load_existing_labels(labels_path)

    # Filter to unlabeled
    to_label = [i for i in items if i["call_id"] not in labeled]
    print(f"Loaded {len(items)} items, {len(to_label)} remaining to label")

    history = []
    i = 0

    while i < len(to_label):
        item = to_label[i]
        display_call(item, i, len(to_label))

        choice = input("> ").strip().lower()

        if choice == "0":
            save_label(labels_path, item["video_id"], item["call_id"], "SPEAKER_00")
            history.append(i)
            i += 1
        elif choice == "1":
            save_label(labels_path, item["video_id"], item["call_id"], "SPEAKER_01")
            history.append(i)
            i += 1
        elif choice == "s":
            i += 1
        elif choice == "b" and history:
            i = history.pop()
        elif choice == "q":
            break
        else:
            print("Invalid choice. Use 0, 1, s, b, or q")

    print(f"\nLabeling complete. {len(labeled) + len(history)} total labeled.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python label_roles.py <queue.jsonl> <labels.csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
```

```bash
git add scripts/whisperx_pipeline/label_roles.py
git commit -m "feat(whisperx): add role labeling CLI"
```

---

### Task 13: Create Role Classifier Training Script

**Files:**
- Create: `scripts/whisperx_pipeline/train_role_model.py`

**Implementation:**

```python
# scripts/whisperx_pipeline/train_role_model.py
"""Train role classifier from labeled data."""
import json
import csv
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from .role_features import extract_call_features

FEATURE_COLUMNS = [
    "diff_talk_time", "diff_turn_count", "diff_avg_turn_duration",
    "diff_question_turn_rate", "diff_avg_words_per_turn",
    "diff_long_turn_count", "diff_first_speaker"
]

def load_labels(labels_path: str) -> dict:
    """Load labels as {call_id: agent_spk}."""
    labels = {}
    with open(labels_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["call_id"]] = row["agent_spk"]
    return labels

def train_model(features_data: list, labels: dict, output_dir: str):
    """Train and save role classifier."""
    # Build feature matrix
    X = []
    y = []
    video_ids = []

    for item in features_data:
        call_id = item["call_id"]
        if call_id not in labels:
            continue

        X.append([item[col] for col in FEATURE_COLUMNS])
        y.append(1 if labels[call_id] == "SPEAKER_00" else 0)
        video_ids.append(item.get("video_id", call_id.split("_")[0]))

    X = np.array(X)
    y = np.array(y)

    # Video-level split
    unique_videos = list(set(video_ids))
    train_videos, val_videos = train_test_split(unique_videos, test_size=0.2, random_state=42)

    train_mask = np.array([v in train_videos for v in video_ids])
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[~train_mask], y[~train_mask]

    # Train
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression())
    ])
    model.fit(X_train, y_train)

    # Calibrate threshold
    probas = model.predict_proba(X_val)[:, 1]
    confidence_gaps = np.abs(probas - 0.5) * 2
    pred_is_spk0 = (probas >= 0.5).astype(int)

    best_threshold = 0.25
    best_coverage = 0

    for threshold in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        high_conf = confidence_gaps >= threshold
        if high_conf.sum() == 0:
            continue
        accuracy = (pred_is_spk0[high_conf] == y_val[high_conf]).mean()
        coverage = high_conf.mean()

        if accuracy >= 0.97 and coverage > best_coverage:
            best_threshold = threshold
            best_coverage = coverage

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "role_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(output_dir / "feature_schema.json", "w") as f:
        json.dump({"features": FEATURE_COLUMNS}, f)

    metadata = {
        "role_model_version": "1.0.0",
        "trained_at": datetime.now().isoformat(),
        "training_samples": len(y_train),
        "gap_threshold": best_threshold,
        "val_accuracy": float((pred_is_spk0[confidence_gaps >= best_threshold] ==
                               y_val[confidence_gaps >= best_threshold]).mean()),
        "val_coverage": float(best_coverage),
        "features": FEATURE_COLUMNS
    }
    with open(output_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved to {output_dir}")
    print(f"Threshold: {best_threshold}, Accuracy: {metadata['val_accuracy']:.3f}, Coverage: {metadata['val_coverage']:.3f}")
```

```bash
git add scripts/whisperx_pipeline/train_role_model.py
git commit -m "feat(whisperx): add role classifier training script"
```

---

### Task 14: Create Role Predictor Module

**Files:**
- Create: `scripts/whisperx_pipeline/role_predictor.py`
- Create: `tests/whisperx_pipeline/test_role_predictor.py`

**Implementation:**

```python
# scripts/whisperx_pipeline/role_predictor.py
"""Predict speaker roles using trained classifier."""
import json
import pickle
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

from .role_features import extract_call_features

@dataclass
class RolePrediction:
    agent_spk: str
    user_spk: str
    confidence_gap: float
    decision_features: Dict[str, float]
    route_to: str  # "auto_complete" or "llm_fallback"

class RolePredictor:
    """Predict speaker roles for a call."""

    def __init__(self, model_dir: str):
        model_dir = Path(model_dir)

        with open(model_dir / "role_model.pkl", "rb") as f:
            self.model = pickle.load(f)

        with open(model_dir / "training_metadata.json") as f:
            self.metadata = json.load(f)

        # Read feature schema from model directory (authoritative source)
        with open(model_dir / "feature_schema.json") as f:
            schema = json.load(f)
            self.feature_columns: List[str] = schema["features"]

        self.threshold = self.metadata.get("gap_threshold", 0.25)

    def predict(self, spk_turns: list, call_start_abs: float) -> RolePrediction:
        """Predict roles for a call."""
        features = extract_call_features("temp", spk_turns, call_start_abs)

        X = [[features[col] for col in self.feature_columns]]
        proba = self.model.predict_proba(X)[0][1]  # P(SPEAKER_00 is agent)

        confidence_gap = abs(proba - 0.5) * 2

        if proba >= 0.5:
            agent_spk = "SPEAKER_00"
            user_spk = "SPEAKER_01"
        else:
            agent_spk = "SPEAKER_01"
            user_spk = "SPEAKER_00"

        route_to = "auto_complete" if confidence_gap >= self.threshold else "llm_fallback"

        return RolePrediction(
            agent_spk=agent_spk,
            user_spk=user_spk,
            confidence_gap=confidence_gap,
            decision_features={k: features[k] for k in self.feature_columns},
            route_to=route_to
        )
```

```bash
git add scripts/whisperx_pipeline/role_predictor.py
git commit -m "feat(whisperx): add role predictor module"
```

---

## Phase 3: Docker & AWS Batch (Tasks 15-18)

### Task 15: Create Dockerfile

**Files:**
- Create: `docker/whisperx/Dockerfile`
- Create: `docker/whisperx/requirements.txt`

```dockerfile
# docker/whisperx/Dockerfile
# Use PyTorch base image to avoid CUDA/torch version mismatches
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Unified cache directories (critical for consistent caching behavior)
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch
ENV XDG_CACHE_HOME=/app/.cache
ENV PYTHONPATH=/app

# Install CUDA-specific wheels for torch audio (must match base image CUDA)
RUN pip install --no-cache-dir \
    torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

COPY docker/whisperx/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Silero VAD model during build (avoids runtime network calls in Batch)
RUN python -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)"

COPY scripts/whisperx_pipeline/ /app/whisperx_pipeline/
COPY config/ /app/config/

# NOTE: Don't assert CUDA at build time (CI may not have GPU)
# Runtime smoke test command instead:
#   python -c "import torch; assert torch.cuda.is_available()"

ENTRYPOINT ["python", "-m", "whisperx_pipeline.cli"]
```

```text
# docker/whisperx/requirements.txt
# Note: torch is provided by base image, do NOT specify here
whisperx>=3.1.0
soundfile>=0.12.0
boto3>=1.28.0
numpy>=1.24.0
scikit-learn>=1.3.0
```

```bash
mkdir -p docker/whisperx
git add docker/whisperx/
git commit -m "feat(whisperx): add Dockerfile and requirements"
```

---

### Task 16: Create CLI Entry Point

**Files:**
- Create: `scripts/whisperx_pipeline/cli.py`

```python
# scripts/whisperx_pipeline/cli.py
"""CLI entry point for WhisperX pipeline."""
import argparse
import json
import os
import sys
import numpy as np
import boto3

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
    result = model.transcribe(dummy_audio, batch_size=1)
    print(f"   Transcription: OK (detected language: {result.get('language', 'unknown')})")

    # 6. Run diarization on dummy audio (catches pyannote auth/model issues)
    print("\n6. Running diarization on dummy audio...")
    diarize_result = diarize(dummy_audio, min_speakers=1, max_speakers=2)
    segment_count = len(list(diarize_result.itertracks()))
    print(f"   Diarization: OK ({segment_count} segments, 0 expected for silence)")

    print("\n=== Smoke Test PASSED ===")
    print("Pipeline is ready to process videos.")

def main():
    parser = argparse.ArgumentParser(description="WhisperX Transcription Pipeline")
    parser.add_argument("--video-id", help="Video ID to process")
    parser.add_argument("--run-id", help="Run ID (auto-generated if not provided)")
    parser.add_argument("--audio-key", help="S3 key for audio (default: audio/{video_id}.mp3)")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick validation of GPU/models")

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
        run_id=args.run_id
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
```

```bash
git add scripts/whisperx_pipeline/cli.py
git commit -m "feat(whisperx): add CLI entry point"
```

---

### Task 17: Create AWS Batch Job Definition (Terraform/CloudFormation)

**Files:**
- Create: `infra/batch/whisperx-job.json`

```json
{
  "jobDefinitionName": "whisperx-pipeline",
  "type": "container",
  "containerProperties": {
    "image": "${AWS_ACCOUNT_ID}.dkr.ecr.us-east-2.amazonaws.com/whisperx-pipeline:latest",
    "resourceRequirements": [
      {"type": "GPU", "value": "1"},
      {"type": "VCPU", "value": "4"},
      {"type": "MEMORY", "value": "16384"}
    ],
    "secrets": [
      {"name": "HF_TOKEN", "valueFrom": "arn:aws:secretsmanager:us-east-2:${AWS_ACCOUNT_ID}:secret:hf-token"}
    ],
    "environment": [
      {"name": "S3_BUCKET", "value": "rezora-data-pipeline-864981718771"}
    ],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/aws/batch/whisperx-pipeline",
        "awslogs-region": "us-east-2",
        "awslogs-stream-prefix": "whisperx"
      }
    },
    "command": ["--video-id", "Ref::video_id", "--run-id", "Ref::run_id"]
  },
  "timeout": {"attemptDurationSeconds": 7200},
  "retryStrategy": {
    "attempts": 2,
    "evaluateOnExit": [
      {"onExitCode": "137", "action": "RETRY"},
      {"onExitCode": "1", "action": "EXIT"}
    ]
  }
}
```

```bash
mkdir -p infra/batch
git add infra/batch/
git commit -m "feat(whisperx): add AWS Batch job definition"
```

---

### Task 18: Create Job Submission Script

**Files:**
- Create: `scripts/submit_whisperx_jobs.py`

```python
# scripts/submit_whisperx_jobs.py
"""Submit WhisperX transcription jobs to AWS Batch."""
import argparse
import boto3
from datetime import datetime

def submit_job(video_id: str, queue: str = "whisperx-transcription-queue") -> str:
    """Submit a single transcription job."""
    batch = boto3.client("batch", region_name="us-east-2")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    response = batch.submit_job(
        jobName=f"whisperx-{video_id}-{run_id}",
        jobQueue=queue,
        jobDefinition="whisperx-pipeline",
        parameters={
            "video_id": video_id,
            "run_id": run_id
        }
    )

    return response["jobId"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_ids", nargs="+", help="Video IDs to process")
    parser.add_argument("--queue", default="whisperx-transcription-queue")
    args = parser.parse_args()

    for video_id in args.video_ids:
        job_id = submit_job(video_id, args.queue)
        print(f"Submitted {video_id}: {job_id}")

if __name__ == "__main__":
    main()
```

```bash
git add scripts/submit_whisperx_jobs.py
git commit -m "feat(whisperx): add job submission script"
```

---

## Summary

**Phase 1 (Tasks 1-10):** Core pipeline modules
- Config, audio preprocessing, VAD chunking
- Quality metrics (narrator_ratio, speaker_flip)
- Fuzzy lexical matcher, call splitter, turn builder
- Main pipeline orchestrator

**Phase 2 (Tasks 11-14):** Role classifier
- Feature extraction, labeling CLI
- Training script, predictor module

**Phase 3 (Tasks 15-18):** Deployment
- Dockerfile, CLI entry point
- AWS Batch job definition, submission script

**Next Steps:**
1. Build Docker image and push to ECR
2. Create AWS Batch compute environment and queue
3. Run pipeline on 50 videos to generate training data
4. Label 200-500 calls using CLI
5. Train role classifier
6. Evaluate on 100-call test set
