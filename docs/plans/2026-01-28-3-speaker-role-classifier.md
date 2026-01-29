# 3-Speaker Role Classifier Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update the role classifier to handle 3+ speakers by dynamically selecting the top 2 speakers by turn count, ignoring narrators.

**Architecture:** Add a `get_top_2_speakers()` function that counts turns per speaker in the first 60 seconds and returns the two most active speakers. This preserves the existing binary classifier (difference features) while making speaker selection dynamic instead of hardcoded.

**Tech Stack:** Python, pytest, sklearn (existing), no new dependencies

---

## Critical Invariants (Do Not Break)

1. **2-speaker calls must work identically** - when only SPEAKER_00 and SPEAKER_01 exist, behavior is unchanged
2. **Feature structure unchanged** - `diff_*` features, same columns, same model format
3. **Labels format unchanged** - CSV with `call_id, agent_spk` (agent_spk is now dynamic, not always SPEAKER_00/01)
4. **Model is still binary** - y=1 means "spk0 is agent", y=0 means "spk1 is agent"

---

## Task 1: Add `get_top_2_speakers()` Function

**Files:**
- Modify: `scripts/whisperx_pipeline/role_features.py:70-84`
- Test: `tests/whisperx_pipeline/test_role_features.py`

**Step 1: Write the failing tests**

Add to `tests/whisperx_pipeline/test_role_features.py`:

```python
def test_get_top_2_speakers_basic():
    """Returns 2 speakers with most turns in first 60s."""
    turns = [
        {"spk": "SPEAKER_00", "t0_abs": 0, "t1_abs": 5, "text": "Hello"},
        {"spk": "SPEAKER_01", "t0_abs": 5, "t1_abs": 6, "text": "Hi"},
        {"spk": "SPEAKER_00", "t0_abs": 6, "t1_abs": 10, "text": "How are you?"},
        {"spk": "SPEAKER_01", "t0_abs": 10, "t1_abs": 12, "text": "Good thanks"},
    ]
    from scripts.whisperx_pipeline.role_features import get_top_2_speakers
    spk0, spk1 = get_top_2_speakers(turns, call_start_abs=0)
    assert spk0 == "SPEAKER_00"  # 2 turns
    assert spk1 == "SPEAKER_01"  # 2 turns (tied, but 00 spoke first)


def test_get_top_2_speakers_3_speakers():
    """With 3 speakers, returns top 2 by turn count, ignoring narrator."""
    turns = [
        {"spk": "SPEAKER_00", "t0_abs": 0, "t1_abs": 5, "text": "Hi is this John?"},
        {"spk": "SPEAKER_01", "t0_abs": 5, "t1_abs": 6, "text": "Yes"},
        {"spk": "SPEAKER_02", "t0_abs": 6, "t1_abs": 8, "text": "Great opener here"},  # Narrator
        {"spk": "SPEAKER_00", "t0_abs": 8, "t1_abs": 12, "text": "I noticed your listing"},
        {"spk": "SPEAKER_01", "t0_abs": 12, "t1_abs": 14, "text": "Oh yeah"},
        {"spk": "SPEAKER_00", "t0_abs": 14, "t1_abs": 18, "text": "Would you be interested?"},
        {"spk": "SPEAKER_01", "t0_abs": 18, "t1_abs": 20, "text": "Maybe"},
    ]
    from scripts.whisperx_pipeline.role_features import get_top_2_speakers
    spk0, spk1 = get_top_2_speakers(turns, call_start_abs=0)
    # SPEAKER_00: 3 turns, SPEAKER_01: 3 turns, SPEAKER_02: 1 turn (ignored)
    assert spk0 == "SPEAKER_00"
    assert spk1 == "SPEAKER_01"
    assert "SPEAKER_02" not in (spk0, spk1)


def test_get_top_2_speakers_respects_60s_window():
    """Only counts turns in first 60s."""
    turns = [
        {"spk": "SPEAKER_00", "t0_abs": 0, "t1_abs": 5, "text": "Hello"},
        {"spk": "SPEAKER_01", "t0_abs": 5, "t1_abs": 10, "text": "Hi"},
        # After 60s - should be ignored
        {"spk": "SPEAKER_02", "t0_abs": 65, "t1_abs": 70, "text": "Turn 1"},
        {"spk": "SPEAKER_02", "t0_abs": 70, "t1_abs": 75, "text": "Turn 2"},
        {"spk": "SPEAKER_02", "t0_abs": 75, "t1_abs": 80, "text": "Turn 3"},
    ]
    from scripts.whisperx_pipeline.role_features import get_top_2_speakers
    spk0, spk1 = get_top_2_speakers(turns, call_start_abs=0)
    # Only SPEAKER_00 and SPEAKER_01 have turns in first 60s
    assert set([spk0, spk1]) == {"SPEAKER_00", "SPEAKER_01"}


def test_get_top_2_speakers_single_speaker():
    """Returns (speaker, None) for monologue."""
    turns = [
        {"spk": "SPEAKER_00", "t0_abs": 0, "t1_abs": 5, "text": "Hello"},
        {"spk": "SPEAKER_00", "t0_abs": 5, "t1_abs": 10, "text": "Anyone there?"},
    ]
    from scripts.whisperx_pipeline.role_features import get_top_2_speakers
    spk0, spk1 = get_top_2_speakers(turns, call_start_abs=0)
    assert spk0 == "SPEAKER_00"
    assert spk1 is None
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/whisperx_pipeline/test_role_features.py::test_get_top_2_speakers_basic -v
```

Expected: `ImportError: cannot import name 'get_top_2_speakers'`

**Step 3: Implement `get_top_2_speakers()`**

Add to `scripts/whisperx_pipeline/role_features.py` before `extract_call_features()`:

```python
def get_top_2_speakers(
    all_turns: List[dict],
    call_start_abs: float,
    window_s: float = 60.0
) -> tuple:
    """
    Get the 2 speakers with most turns in first 60s.

    Returns:
        (spk0, spk1) - spk0 has most turns (or tied, spoke first)
        (spk0, None) - if only 1 speaker
        (None, None) - if no turns
    """
    # Filter to first 60s
    first_60s = [t for t in all_turns if t["t0_abs"] < call_start_abs + window_s]

    if not first_60s:
        return (None, None)

    # Count turns per speaker
    counts = {}
    first_seen = {}
    for i, t in enumerate(first_60s):
        spk = t["spk"]
        counts[spk] = counts.get(spk, 0) + 1
        if spk not in first_seen:
            first_seen[spk] = i

    # Sort by count (desc), then by first appearance (asc) for tie-breaking
    sorted_spks = sorted(
        counts.keys(),
        key=lambda s: (-counts[s], first_seen[s])
    )

    if len(sorted_spks) >= 2:
        return (sorted_spks[0], sorted_spks[1])
    elif len(sorted_spks) == 1:
        return (sorted_spks[0], None)
    else:
        return (None, None)
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/whisperx_pipeline/test_role_features.py::test_get_top_2_speakers_basic tests/whisperx_pipeline/test_role_features.py::test_get_top_2_speakers_3_speakers tests/whisperx_pipeline/test_role_features.py::test_get_top_2_speakers_respects_60s_window tests/whisperx_pipeline/test_role_features.py::test_get_top_2_speakers_single_speaker -v
```

Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add scripts/whisperx_pipeline/role_features.py tests/whisperx_pipeline/test_role_features.py
git commit -m "feat(role): add get_top_2_speakers() for dynamic speaker selection"
```

---

## Task 2: Update `extract_call_features()` to Use Dynamic Speakers

**Files:**
- Modify: `scripts/whisperx_pipeline/role_features.py:70-84`
- Test: `tests/whisperx_pipeline/test_role_features.py`

**Step 1: Write the failing test**

Add to `tests/whisperx_pipeline/test_role_features.py`:

```python
def test_extract_call_features_3_speakers():
    """Call features work with 3 speakers, ignoring narrator."""
    turns = [
        {"spk": "SPEAKER_00", "t0_abs": 0, "t1_abs": 5, "text": "Hi is this John?"},
        {"spk": "SPEAKER_01", "t0_abs": 5, "t1_abs": 6, "text": "Yes"},
        {"spk": "SPEAKER_02", "t0_abs": 6, "t1_abs": 8, "text": "Great opener"},  # Narrator
        {"spk": "SPEAKER_00", "t0_abs": 8, "t1_abs": 12, "text": "How are you?"},
        {"spk": "SPEAKER_01", "t0_abs": 12, "t1_abs": 14, "text": "Good"},
    ]
    features = extract_call_features("call_1", turns, call_start_abs=0)

    # Should use SPEAKER_00 and SPEAKER_01 (top 2 by turns)
    assert features["spk0"] == "SPEAKER_00"
    assert features["spk1"] == "SPEAKER_01"
    assert "diff_talk_time" in features
    # SPEAKER_00: 9s, SPEAKER_01: 3s -> diff = 6
    assert features["diff_talk_time"] == 6.0


def test_extract_call_features_2_speakers_unchanged():
    """2-speaker calls work exactly as before."""
    turns = [
        {"spk": "SPEAKER_00", "t0_abs": 0, "t1_abs": 5, "text": "Hello there"},
        {"spk": "SPEAKER_01", "t0_abs": 5, "t1_abs": 6, "text": "Hi"},
    ]
    features = extract_call_features("call_1", turns, call_start_abs=0)

    assert features["spk0"] == "SPEAKER_00"
    assert features["spk1"] == "SPEAKER_01"
    assert features["diff_talk_time"] == 4.0  # 5s - 1s (unchanged behavior)
```

**Step 2: Run test to verify behavior**

```bash
pytest tests/whisperx_pipeline/test_role_features.py::test_extract_call_features_3_speakers -v
```

Note: This might pass or fail depending on current behavior - we're testing to confirm.

**Step 3: Update `extract_call_features()`**

Replace the function in `scripts/whisperx_pipeline/role_features.py`:

```python
def extract_call_features(
    call_id: str,
    spk_turns: List[dict],
    call_start_abs: float
) -> Dict[str, float]:
    """Extract call-level difference features.

    Dynamically selects top 2 speakers by turn count in first 60s.
    This handles 3+ speaker calls by ignoring low-activity speakers (narrators).
    """
    # Dynamic speaker selection (replaces hardcoded SPEAKER_00/01)
    spk0, spk1 = get_top_2_speakers(spk_turns, call_start_abs)

    # Handle edge cases
    if spk0 is None:
        # No speakers - return zeros
        return {"call_id": call_id, "spk0": None, "spk1": None,
                **{f"diff_{k}": 0.0 for k in ["talk_time", "turn_count", "avg_turn_duration",
                                               "median_turn_duration", "max_turn_duration",
                                               "question_turn_count", "question_turn_rate",
                                               "long_turn_count", "short_turn_count",
                                               "word_count", "avg_words_per_turn", "first_speaker"]}}

    if spk1 is None:
        # Only 1 speaker (monologue) - spk1 features are zeros
        feat0 = extract_speaker_features(spk_turns, spk0, call_start_abs)
        diff = {f"diff_{k}": feat0[k] for k in feat0}  # diff = feat0 - 0
        return {"call_id": call_id, "spk0": spk0, "spk1": None, **diff}

    # Normal case: 2+ speakers
    feat0 = extract_speaker_features(spk_turns, spk0, call_start_abs)
    feat1 = extract_speaker_features(spk_turns, spk1, call_start_abs)

    diff = {f"diff_{k}": feat0[k] - feat1[k] for k in feat0}
    return {"call_id": call_id, "spk0": spk0, "spk1": spk1, **diff}
```

**Step 4: Run all role_features tests**

```bash
pytest tests/whisperx_pipeline/test_role_features.py -v
```

Expected: All tests PASS (including existing tests - backward compatible)

**Step 5: Commit**

```bash
git add scripts/whisperx_pipeline/role_features.py tests/whisperx_pipeline/test_role_features.py
git commit -m "feat(role): update extract_call_features to use dynamic speaker selection"
```

---

## Task 3: Update `label_roles.py` for Dynamic Speakers

**Files:**
- Modify: `scripts/whisperx_pipeline/label_roles.py`
- No automated tests (CLI tool)

**Step 1: Review current hardcoding**

Lines 48, 69-74 hardcode SPEAKER_00 and SPEAKER_01.

**Step 2: Update to show dynamic speakers**

Replace `scripts/whisperx_pipeline/label_roles.py`:

```python
# scripts/whisperx_pipeline/label_roles.py
"""CLI for labeling speaker roles."""
import json
import csv
from pathlib import Path
from datetime import datetime
import sys

from .role_features import get_top_2_speakers


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


def get_speakers_from_turns(turns: list, call_start_abs: float = 0.0) -> tuple:
    """Get top 2 speakers from turns data."""
    return get_top_2_speakers(turns, call_start_abs)


def display_call(item: dict, index: int, total: int, spk0: str, spk1: str):
    """Display call for labeling."""
    print("\n" + "=" * 60)
    print(f"Call: {item['call_id']} ({index + 1}/{total})")
    print("=" * 60)
    print(item.get("preview_text", "No preview available"))
    print("=" * 60)
    print("Who is the AGENT/CALLER?")
    # Dynamic speaker options
    if spk1:
        print(f"  [0] {spk0}    [1] {spk1}    [s] Skip    [b] Back    [q] Quit")
    else:
        print(f"  [0] {spk0}    [s] Skip    [b] Back    [q] Quit")
        print("  (Only 1 speaker detected - monologue)")


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

        # Get dynamic speakers from item's turns data
        turns = item.get("turns", [])
        call_start_abs = item.get("call_start_abs", 0.0)
        spk0, spk1 = get_speakers_from_turns(turns, call_start_abs)

        if spk0 is None:
            print(f"Skipping {item['call_id']} - no speakers detected")
            i += 1
            continue

        display_call(item, i, len(to_label), spk0, spk1)

        choice = input("> ").strip().lower()

        if choice == "0":
            save_label(labels_path, item["video_id"], item["call_id"], spk0)
            history.append(i)
            i += 1
        elif choice == "1" and spk1:
            save_label(labels_path, item["video_id"], item["call_id"], spk1)
            history.append(i)
            i += 1
        elif choice == "s":
            i += 1
        elif choice == "b" and history:
            i = history.pop()
        elif choice == "q":
            break
        else:
            valid = "0, s, b, or q" if not spk1 else "0, 1, s, b, or q"
            print(f"Invalid choice. Use {valid}")

    print(f"\nLabeling complete. {len(labeled) + len(history)} total labeled.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python label_roles.py <queue.jsonl> <labels.csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
```

**Step 3: Verify import works**

```bash
python -c "from scripts.whisperx_pipeline.label_roles import get_speakers_from_turns; print('OK')"
```

Expected: `OK`

**Step 4: Commit**

```bash
git add scripts/whisperx_pipeline/label_roles.py
git commit -m "feat(role): update label_roles CLI for dynamic speaker selection"
```

---

## Task 4: Update `train_role_model.py` for Dynamic Labels

**Files:**
- Modify: `scripts/whisperx_pipeline/train_role_model.py:46`
- Test: Manual verification (training requires labeled data)

**Step 1: Review current hardcoding**

Line 46: `y.append(1 if labels[call_id] == "SPEAKER_00" else 0)`

This assumes label is always "SPEAKER_00" or "SPEAKER_01". Now labels contain the actual speaker ID (e.g., could be "SPEAKER_02" if that was top speaker).

**Step 2: Update label encoding**

Replace line 46 in `scripts/whisperx_pipeline/train_role_model.py`:

```python
# OLD:
y.append(1 if labels[call_id] == "SPEAKER_00" else 0)

# NEW:
# Label is now the actual speaker ID (could be any SPEAKER_XX)
# y=1 if agent is spk0 (first in feature dict), y=0 if agent is spk1
agent_spk = labels[call_id]
y.append(1 if agent_spk == item["spk0"] else 0)
```

**Step 3: Add validation for unexpected labels**

Add validation after line 42:

```python
for item in features_data:
    call_id = item["call_id"]
    if call_id not in labels:
        continue

    # Validate label matches one of the speakers
    agent_spk = labels[call_id]
    if agent_spk not in (item["spk0"], item["spk1"]):
        print(f"WARNING: Label '{agent_spk}' not in speakers ({item['spk0']}, {item['spk1']}) for {call_id}")
        continue

    X.append([item[col] for col in FEATURE_COLUMNS])
    y.append(1 if agent_spk == item["spk0"] else 0)
    video_ids.append(item.get("video_id", call_id.split("_")[0]))
```

**Step 4: Verify syntax**

```bash
python -c "from scripts.whisperx_pipeline.train_role_model import train_model; print('OK')"
```

Expected: `OK`

**Step 5: Commit**

```bash
git add scripts/whisperx_pipeline/train_role_model.py
git commit -m "feat(role): update train_role_model for dynamic speaker labels"
```

---

## Task 5: Update `role_predictor.py` to Use Dynamic Speakers

**Files:**
- Modify: `scripts/whisperx_pipeline/role_predictor.py:40-64`
- Test: `tests/whisperx_pipeline/test_role_predictor.py`

**Step 1: Write the failing test**

Add to `tests/whisperx_pipeline/test_role_predictor.py`:

```python
def test_role_prediction_3_speakers():
    """RolePrediction works with any speaker IDs."""
    prediction = RolePrediction(
        agent_spk="SPEAKER_02",  # Not hardcoded to 00/01
        user_spk="SPEAKER_00",
        confidence_gap=0.85,
        decision_features={"diff_talk_time": 5.0},
        route_to="auto_complete"
    )
    assert prediction.agent_spk == "SPEAKER_02"
    assert prediction.user_spk == "SPEAKER_00"
```

**Step 2: Run test (should pass - dataclass is already flexible)**

```bash
pytest tests/whisperx_pipeline/test_role_predictor.py -v
```

Expected: PASS

**Step 3: Update `predict()` method**

Replace `predict()` in `scripts/whisperx_pipeline/role_predictor.py`:

```python
def predict(self, spk_turns: list, call_start_abs: float) -> RolePrediction:
    """Predict roles for a call.

    Uses dynamic speaker selection - works with any number of speakers.
    """
    features = extract_call_features("temp", spk_turns, call_start_abs)

    spk0 = features["spk0"]
    spk1 = features["spk1"]

    # Handle edge cases
    if spk0 is None:
        return RolePrediction(
            agent_spk=None,
            user_spk=None,
            confidence_gap=0.0,
            decision_features={},
            route_to="llm_fallback"
        )

    if spk1 is None:
        # Monologue - assume single speaker is agent
        return RolePrediction(
            agent_spk=spk0,
            user_spk=None,
            confidence_gap=0.5,  # Medium confidence
            decision_features={k: features[k] for k in self.feature_columns},
            route_to="llm_fallback"
        )

    # Normal prediction
    X = [[features[col] for col in self.feature_columns]]
    proba = self.model.predict_proba(X)[0][1]  # P(spk0 is agent)

    confidence_gap = abs(proba - 0.5) * 2

    if proba >= 0.5:
        agent_spk = spk0
        user_spk = spk1
    else:
        agent_spk = spk1
        user_spk = spk0

    route_to = "auto_complete" if confidence_gap >= self.threshold else "llm_fallback"

    return RolePrediction(
        agent_spk=agent_spk,
        user_spk=user_spk,
        confidence_gap=confidence_gap,
        decision_features={k: features[k] for k in self.feature_columns},
        route_to=route_to
    )
```

**Step 4: Update RolePrediction dataclass for Optional fields**

Update the dataclass at the top of `role_predictor.py`:

```python
from typing import Dict, List, Optional

@dataclass
class RolePrediction:
    agent_spk: Optional[str]
    user_spk: Optional[str]
    confidence_gap: float
    decision_features: Dict[str, float]
    route_to: str  # "auto_complete" or "llm_fallback"
```

**Step 5: Run all tests**

```bash
pytest tests/whisperx_pipeline/test_role_predictor.py tests/whisperx_pipeline/test_role_features.py -v
```

Expected: All PASS

**Step 6: Commit**

```bash
git add scripts/whisperx_pipeline/role_predictor.py tests/whisperx_pipeline/test_role_predictor.py
git commit -m "feat(role): update role_predictor for dynamic speaker selection"
```

---

## Task 6: Update Queue Generation to Include Turns

**Files:**
- This depends on how the queue is generated (not in provided files)
- The queue JSONL must include `turns` field for `label_roles.py` to work

**Step 1: Document queue format requirement**

The queue JSONL items must now include:

```json
{
  "call_id": "video123_0_60000",
  "video_id": "video123",
  "preview_text": "SPEAKER_00: Hi is this...",
  "turns": [
    {"spk": "SPEAKER_00", "t0_abs": 0, "t1_abs": 5, "text": "Hi is this John?"},
    {"spk": "SPEAKER_01", "t0_abs": 5, "t1_abs": 6, "text": "Yes"}
  ],
  "call_start_abs": 0.0
}
```

**Step 2: Create queue generation script (if not exists)**

If queue generation doesn't exist, create `scripts/whisperx_pipeline/generate_label_queue.py`:

```python
# scripts/whisperx_pipeline/generate_label_queue.py
"""Generate labeling queue from extracted calls."""
import json
import boto3
from pathlib import Path

from .config import S3_BUCKET, AWS_REGION
from .db import get_calls_for_labeling


def generate_preview(turns: list, max_turns: int = 10) -> str:
    """Generate text preview for labeling UI."""
    lines = []
    for t in turns[:max_turns]:
        lines.append(f"{t['spk']}: {t['text']}")
    if len(turns) > max_turns:
        lines.append(f"... ({len(turns) - max_turns} more turns)")
    return "\n".join(lines)


def generate_queue(conversation_type: str, output_path: str, limit: int = 100):
    """Generate labeling queue JSONL."""
    s3 = boto3.client("s3", region_name=AWS_REGION)
    calls = get_calls_for_labeling(conversation_type, limit=limit)

    with open(output_path, "w") as f:
        for call in calls:
            # Load spk_turns.json from S3
            try:
                response = s3.get_object(Bucket=S3_BUCKET, Key=call["s3_key"])
                data = json.loads(response["Body"].read())
                turns = data.get("turns", [])
            except Exception as e:
                print(f"Error loading {call['s3_key']}: {e}")
                continue

            item = {
                "call_id": call["call_id"],
                "video_id": call["video_id"],
                "preview_text": generate_preview(turns),
                "turns": turns,
                "call_start_abs": data.get("call_start_abs", 0.0)
            }
            f.write(json.dumps(item) + "\n")

    print(f"Generated queue with {len(calls)} items: {output_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python generate_label_queue.py <conversation_type> <output.jsonl>")
        sys.exit(1)
    generate_queue(sys.argv[1], sys.argv[2])
```

**Step 3: Commit**

```bash
git add scripts/whisperx_pipeline/generate_label_queue.py
git commit -m "feat(role): add queue generation script with turns data"
```

---

## Task 7: Integration Test - Full Pipeline

**Files:**
- Test: Manual end-to-end verification

**Step 1: Run all unit tests**

```bash
pytest tests/whisperx_pipeline/ -v
```

Expected: All PASS

**Step 2: Test feature extraction with 3-speaker data**

```bash
python -c "
from scripts.whisperx_pipeline.role_features import extract_call_features

# Simulate 3-speaker call
turns = [
    {'spk': 'SPEAKER_00', 't0_abs': 0, 't1_abs': 5, 'text': 'Hi is this John?'},
    {'spk': 'SPEAKER_01', 't0_abs': 5, 't1_abs': 6, 'text': 'Yes'},
    {'spk': 'SPEAKER_02', 't0_abs': 6, 't1_abs': 8, 'text': 'Great opener'},
    {'spk': 'SPEAKER_00', 't0_abs': 8, 't1_abs': 12, 'text': 'I noticed your listing expired'},
    {'spk': 'SPEAKER_01', 't0_abs': 12, 't1_abs': 14, 'text': 'Oh yeah that'},
]

features = extract_call_features('test_call', turns, 0)
print(f'spk0: {features[\"spk0\"]}')
print(f'spk1: {features[\"spk1\"]}')
print(f'diff_talk_time: {features[\"diff_talk_time\"]}')
print('SPEAKER_02 (narrator) correctly ignored!' if 'SPEAKER_02' not in [features['spk0'], features['spk1']] else 'ERROR')
"
```

Expected:
```
spk0: SPEAKER_00
spk1: SPEAKER_01
diff_talk_time: 7.0
SPEAKER_02 (narrator) correctly ignored!
```

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat(role): complete 3-speaker support for role classifier

- Add get_top_2_speakers() for dynamic speaker selection
- Update extract_call_features() to use top 2 speakers by turn count
- Update label_roles.py CLI for dynamic speakers
- Update train_role_model.py for dynamic label encoding
- Update role_predictor.py for dynamic speaker assignment
- Add queue generation script with turns data

3+ speaker calls now correctly ignore low-activity speakers (narrators).
Existing 2-speaker calls work identically (backward compatible).
"
```

---

## Verification Checklist

After implementation, verify:

- [ ] `pytest tests/whisperx_pipeline/ -v` - All tests pass
- [ ] 2-speaker calls produce same features as before
- [ ] 3-speaker calls ignore narrator (speaker with fewest turns)
- [ ] Labels CSV can contain any SPEAKER_XX value
- [ ] Training works with mixed 2/3-speaker labeled data
- [ ] Prediction returns actual speaker IDs (not hardcoded)

---

## Notes for Training

When you label ~50 calls:

1. **Mix of 2 and 3 speaker calls is fine** - the model learns from features, not speaker counts
2. **Label the actual agent** - whatever speaker ID they have (SPEAKER_00, SPEAKER_01, or SPEAKER_02)
3. **The model will generalize** because:
   - Features are `diff_*` (differences between top 2 speakers)
   - y=1 means "first top speaker is agent"
   - Speaker IDs don't matter, only behavioral patterns

**Example**: If a 3-speaker call has SPEAKER_00 (narrator, 2 turns), SPEAKER_01 (agent, 15 turns), SPEAKER_02 (prospect, 12 turns):
- `get_top_2_speakers()` returns (SPEAKER_01, SPEAKER_02)
- You label SPEAKER_01 as agent
- Training encodes y=1 (agent is spk0)
- Model learns "spk0 pattern = agent behavior"
