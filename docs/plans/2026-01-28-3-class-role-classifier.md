# 3-Class Role Classifier Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the binary agent/prospect classifier with a 3-class learned classifier that predicts agent, prospect, and narrator roles for each speaker.

**Architecture:** Change from `diff_*` features (comparing 2 hardcoded speakers) to per-speaker features for all detected speakers. Train a multi-class LogisticRegression that learns behavioral patterns for all 3 roles from human-labeled examples.

**Tech Stack:** Python, pytest, sklearn (existing), no new dependencies

---

## Why This Change?

### Current Approach (Binary Classifier)
```
SPEAKER_00 features ─┐
                     ├─► diff_* features ─► Binary Model ─► "spk0 is agent? yes/no"
SPEAKER_01 features ─┘
```
- **Hardcoded to 2 speakers** - ignores SPEAKER_02+
- **Can't detect narrators** - must filter with heuristics

### New Approach (3-Class Classifier)
```
SPEAKER_00 features ─► Model ─► "agent" (0.85)
SPEAKER_01 features ─► Model ─► "prospect" (0.72)
SPEAKER_02 features ─► Model ─► "narrator" (0.91)
```
- **Handles any number of speakers** - each classified independently
- **Learns narrator patterns** - same approach as agent/prospect

---

## Critical Invariants (Do Not Break)

1. **Existing 2-speaker calls must work** - backward compatible prediction output
2. **Model learns from data** - no hardcoded rules or keyword matching
3. **Same feature categories** - talk time, turn count, questions, etc. (just not diff_*)
4. **Confidence-based routing** - low confidence → LLM fallback

---

## Task 1: Add Per-Speaker Feature Extraction

**Files:**
- Modify: `scripts/whisperx_pipeline/role_features.py`
- Test: `tests/whisperx_pipeline/test_role_features.py`

**Step 1: Write the failing test**

Add to `tests/whisperx_pipeline/test_role_features.py`:

```python
def test_extract_all_speaker_features():
    """Extract features for all speakers, not just SPEAKER_00/01."""
    turns = [
        {"spk": "SPEAKER_00", "t0_abs": 0, "t1_abs": 10, "text": "Hi, this is Mike calling about your listing. How are you today?"},
        {"spk": "SPEAKER_01", "t0_abs": 10, "t1_abs": 12, "text": "Good thanks"},
        {"spk": "SPEAKER_02", "t0_abs": 12, "t1_abs": 25, "text": "So you can see here the agent is building rapport. This is a great technique to use."},
        {"spk": "SPEAKER_00", "t0_abs": 25, "t1_abs": 32, "text": "I noticed your listing expired last week. What happened there?"},
        {"spk": "SPEAKER_01", "t0_abs": 32, "t1_abs": 34, "text": "Yeah it did"},
    ]
    from scripts.whisperx_pipeline.role_features import extract_all_speaker_features
    features = extract_all_speaker_features(turns, call_start_abs=0)

    # Should have features for all 3 speakers
    assert len(features) == 3
    assert "SPEAKER_00" in features
    assert "SPEAKER_01" in features
    assert "SPEAKER_02" in features

    # SPEAKER_00 (agent pattern): more questions, moderate talk time
    assert features["SPEAKER_00"]["question_turn_rate"] > 0  # Asked questions
    assert features["SPEAKER_00"]["turn_count"] == 2

    # SPEAKER_01 (prospect pattern): short turns, no questions
    assert features["SPEAKER_01"]["question_turn_rate"] == 0
    assert features["SPEAKER_01"]["avg_turn_duration"] < 3  # Short turns

    # SPEAKER_02 (narrator pattern): long turns, no questions, doesn't interact
    assert features["SPEAKER_02"]["turn_count"] == 1
    assert features["SPEAKER_02"]["talk_time"] > 10  # Long monologue
    assert features["SPEAKER_02"]["avg_turn_duration"] > 10


def test_extract_all_speaker_features_2_speakers():
    """Works with 2 speakers (backward compatible)."""
    turns = [
        {"spk": "SPEAKER_00", "t0_abs": 0, "t1_abs": 5, "text": "Hello there"},
        {"spk": "SPEAKER_01", "t0_abs": 5, "t1_abs": 7, "text": "Hi"},
    ]
    from scripts.whisperx_pipeline.role_features import extract_all_speaker_features
    features = extract_all_speaker_features(turns, call_start_abs=0)

    assert len(features) == 2
    assert "SPEAKER_00" in features
    assert "SPEAKER_01" in features
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/whisperx_pipeline/test_role_features.py::test_extract_all_speaker_features -v
```

Expected: `ImportError: cannot import name 'extract_all_speaker_features'`

**Step 3: Implement `extract_all_speaker_features()`**

Add to `scripts/whisperx_pipeline/role_features.py`:

```python
def extract_all_speaker_features(
    all_turns: List[dict],
    call_start_abs: float,
    window_s: float = 60.0
) -> Dict[str, Dict[str, float]]:
    """
    Extract features for ALL speakers in the call.

    Returns: {speaker_id: {feature_name: value, ...}, ...}

    This replaces the hardcoded SPEAKER_00/01 approach.
    Each speaker gets their own feature vector for classification.
    """
    # Find all unique speakers
    speakers = set(t["spk"] for t in all_turns)

    # Extract features for each speaker
    result = {}
    for speaker in speakers:
        features = extract_speaker_features(all_turns, speaker, call_start_abs, window_s)
        result[speaker] = features

    return result
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/whisperx_pipeline/test_role_features.py::test_extract_all_speaker_features tests/whisperx_pipeline/test_role_features.py::test_extract_all_speaker_features_2_speakers -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add scripts/whisperx_pipeline/role_features.py tests/whisperx_pipeline/test_role_features.py
git commit -m "feat(role): add extract_all_speaker_features for multi-speaker support"
```

---

## Task 2: Create Multi-Class Label Format

**Files:**
- Create: `scripts/whisperx_pipeline/label_roles_v2.py`
- Test: Manual CLI testing

**Step 1: Define new label format**

Old format (`labels.csv`):
```csv
call_id,agent_spk,labeled_at
video123_0_60000,SPEAKER_00,2026-01-28T10:00:00
```

New format (`speaker_labels.csv`):
```csv
call_id,speaker_id,role,labeled_at
video123_0_60000,SPEAKER_00,agent,2026-01-28T10:00:00
video123_0_60000,SPEAKER_01,prospect,2026-01-28T10:00:00
video123_0_60000,SPEAKER_02,narrator,2026-01-28T10:00:00
```

**Step 2: Create new labeling CLI**

Create `scripts/whisperx_pipeline/label_roles_v2.py`:

```python
# scripts/whisperx_pipeline/label_roles_v2.py
"""CLI for labeling speaker roles (3-class: agent/prospect/narrator)."""
import json
import csv
from pathlib import Path
from datetime import datetime
import sys
from typing import List, Dict, Set


ROLES = ["agent", "prospect", "narrator"]


def load_queue(queue_path: str) -> List[dict]:
    """Load labeling queue from JSONL."""
    items = []
    with open(queue_path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def load_existing_labels(labels_path: str) -> Dict[str, Dict[str, str]]:
    """Load existing labels as {call_id: {speaker_id: role}}."""
    labels = {}
    if Path(labels_path).exists():
        with open(labels_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                call_id = row["call_id"]
                if call_id not in labels:
                    labels[call_id] = {}
                labels[call_id][row["speaker_id"]] = row["role"]
    return labels


def get_labeled_calls(labels_path: str) -> Set[str]:
    """Get call_ids that are fully labeled (all speakers have roles)."""
    # For simplicity, we track which calls have been processed
    # A more robust approach would check if all speakers are labeled
    labeled = set()
    if Path(labels_path).exists():
        with open(labels_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                labeled.add(row["call_id"])
    return labeled


def save_labels(labels_path: str, video_id: str, call_id: str, speaker_roles: Dict[str, str]):
    """Save labels for all speakers in a call."""
    file_exists = Path(labels_path).exists()
    with open(labels_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["video_id", "call_id", "speaker_id", "role", "labeled_at"])
        timestamp = datetime.now().isoformat()
        for speaker_id, role in speaker_roles.items():
            writer.writerow([video_id, call_id, speaker_id, role, timestamp])


def display_call(item: dict, index: int, total: int):
    """Display call transcript for labeling."""
    print("\n" + "=" * 70)
    print(f"Call: {item['call_id']} ({index + 1}/{total})")
    print("=" * 70)

    # Display preview with speaker labels
    turns = item.get("turns", [])
    for i, turn in enumerate(turns[:15]):  # Show first 15 turns
        spk = turn.get("spk", "?")
        text = turn.get("text", "")[:80]  # Truncate long text
        print(f"  {spk}: {text}")

    if len(turns) > 15:
        print(f"  ... ({len(turns) - 15} more turns)")

    print("=" * 70)


def get_speakers_from_item(item: dict) -> List[str]:
    """Get unique speakers from item's turns."""
    turns = item.get("turns", [])
    speakers = []
    seen = set()
    for turn in turns:
        spk = turn.get("spk")
        if spk and spk not in seen:
            speakers.append(spk)
            seen.add(spk)
    return speakers


def label_speakers(speakers: List[str]) -> Dict[str, str]:
    """Prompt user to label each speaker."""
    print("\nLabel each speaker:")
    print("  [a] = agent (the cold caller)")
    print("  [p] = prospect (the person being called)")
    print("  [n] = narrator (YouTube host, commentator)")
    print("  [s] = skip this call")
    print()

    roles = {}
    for spk in speakers:
        while True:
            choice = input(f"  {spk}: ").strip().lower()
            if choice == "s":
                return {}  # Signal to skip
            elif choice == "a":
                roles[spk] = "agent"
                break
            elif choice == "p":
                roles[spk] = "prospect"
                break
            elif choice == "n":
                roles[spk] = "narrator"
                break
            else:
                print("    Invalid. Use [a]gent, [p]rospect, [n]arrator, or [s]kip")

    return roles


def main(queue_path: str, labels_path: str):
    """Run labeling CLI."""
    items = load_queue(queue_path)
    labeled_calls = get_labeled_calls(labels_path)

    # Filter to unlabeled
    to_label = [i for i in items if i["call_id"] not in labeled_calls]
    print(f"Loaded {len(items)} items, {len(to_label)} remaining to label")
    print(f"Roles: agent, prospect, narrator")
    print()

    i = 0
    history = []

    while i < len(to_label):
        item = to_label[i]
        speakers = get_speakers_from_item(item)

        if not speakers:
            print(f"Skipping {item['call_id']} - no speakers detected")
            i += 1
            continue

        display_call(item, i, len(to_label))
        print(f"Speakers detected: {', '.join(speakers)}")

        roles = label_speakers(speakers)

        if not roles:
            # User chose to skip
            print("Skipped.")
            i += 1
            continue

        # Validate: must have at least one agent and one prospect
        role_counts = {r: sum(1 for v in roles.values() if v == r) for r in ROLES}

        if role_counts["agent"] == 0:
            print("  WARNING: No agent labeled. Are you sure? [y/n]")
            if input("  ").strip().lower() != "y":
                continue

        if role_counts["prospect"] == 0:
            print("  WARNING: No prospect labeled. Are you sure? [y/n]")
            if input("  ").strip().lower() != "y":
                continue

        # Save
        save_labels(labels_path, item.get("video_id", ""), item["call_id"], roles)
        print(f"  Saved: {roles}")
        history.append(i)
        i += 1

    labeled_total = len(labeled_calls) + len(history)
    print(f"\nLabeling complete. {labeled_total} calls labeled.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python label_roles_v2.py <queue.jsonl> <speaker_labels.csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
```

**Step 3: Verify CLI syntax**

```bash
python -c "from scripts.whisperx_pipeline.label_roles_v2 import main; print('OK')"
```

Expected: `OK`

**Step 4: Commit**

```bash
git add scripts/whisperx_pipeline/label_roles_v2.py
git commit -m "feat(role): add label_roles_v2 CLI for 3-class labeling"
```

---

## Task 3: Train Multi-Class Model

**Files:**
- Create: `scripts/whisperx_pipeline/train_role_model_v2.py`
- Test: `tests/whisperx_pipeline/test_train_role_model_v2.py`

**Step 1: Write the failing test**

Create `tests/whisperx_pipeline/test_train_role_model_v2.py`:

```python
# tests/whisperx_pipeline/test_train_role_model_v2.py
"""Tests for 3-class role model training."""
import tempfile
import json
from pathlib import Path


def test_train_model_v2_basic():
    """Train a 3-class model from labeled data."""
    from scripts.whisperx_pipeline.train_role_model_v2 import train_model_v2

    # Synthetic training data (features for each speaker)
    features_data = []

    # Call 1: 3 speakers (agent, prospect, narrator pattern)
    features_data.append({
        "call_id": "call_001",
        "video_id": "video_A",
        "speakers": {
            "SPEAKER_00": {  # Agent pattern: questions, moderate talk time
                "talk_time": 25.0, "turn_count": 8, "avg_turn_duration": 3.1,
                "question_turn_rate": 0.5, "avg_words_per_turn": 12.0,
                "long_turn_count": 2, "first_speaker": 1, "short_turn_count": 1,
                "median_turn_duration": 2.5, "max_turn_duration": 8.0,
                "question_turn_count": 4, "word_count": 96
            },
            "SPEAKER_01": {  # Prospect pattern: short turns, reactive
                "talk_time": 8.0, "turn_count": 6, "avg_turn_duration": 1.3,
                "question_turn_rate": 0.0, "avg_words_per_turn": 4.0,
                "long_turn_count": 0, "first_speaker": 0, "short_turn_count": 3,
                "median_turn_duration": 1.2, "max_turn_duration": 2.5,
                "question_turn_count": 0, "word_count": 24
            },
            "SPEAKER_02": {  # Narrator pattern: long monologue
                "talk_time": 15.0, "turn_count": 1, "avg_turn_duration": 15.0,
                "question_turn_rate": 0.0, "avg_words_per_turn": 50.0,
                "long_turn_count": 1, "first_speaker": 0, "short_turn_count": 0,
                "median_turn_duration": 15.0, "max_turn_duration": 15.0,
                "question_turn_count": 0, "word_count": 50
            }
        }
    })

    # Add more synthetic calls with similar patterns...
    for i in range(2, 20):
        features_data.append({
            "call_id": f"call_{i:03d}",
            "video_id": f"video_{chr(65 + i % 5)}",  # Rotate through videos A-E
            "speakers": {
                "SPEAKER_00": {
                    "talk_time": 20 + i, "turn_count": 7 + i % 3, "avg_turn_duration": 3.0,
                    "question_turn_rate": 0.4 + (i % 3) * 0.1, "avg_words_per_turn": 10.0,
                    "long_turn_count": 2, "first_speaker": 1, "short_turn_count": 1,
                    "median_turn_duration": 2.5, "max_turn_duration": 7.0,
                    "question_turn_count": 3, "word_count": 80
                },
                "SPEAKER_01": {
                    "talk_time": 6 + i % 4, "turn_count": 5 + i % 3, "avg_turn_duration": 1.5,
                    "question_turn_rate": 0.0, "avg_words_per_turn": 5.0,
                    "long_turn_count": 0, "first_speaker": 0, "short_turn_count": 2,
                    "median_turn_duration": 1.3, "max_turn_duration": 3.0,
                    "question_turn_count": 0, "word_count": 30
                }
            }
        })

    # Labels: specify role for each speaker in each call
    labels = {
        "call_001": {"SPEAKER_00": "agent", "SPEAKER_01": "prospect", "SPEAKER_02": "narrator"}
    }
    for i in range(2, 20):
        labels[f"call_{i:03d}"] = {"SPEAKER_00": "agent", "SPEAKER_01": "prospect"}

    with tempfile.TemporaryDirectory() as tmpdir:
        train_model_v2(features_data, labels, tmpdir)

        # Verify outputs
        model_path = Path(tmpdir) / "role_model_v2.pkl"
        assert model_path.exists()

        metadata_path = Path(tmpdir) / "training_metadata.json"
        assert metadata_path.exists()

        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["num_classes"] == 3
        assert metadata["classes"] == ["agent", "narrator", "prospect"]
        assert metadata["training_samples"] > 0


def test_train_model_v2_predicts_correctly():
    """Model correctly predicts role from features."""
    from scripts.whisperx_pipeline.train_role_model_v2 import train_model_v2
    import pickle

    # Create training data with clear patterns
    features_data = []
    labels = {}

    # Generate 30 training examples with distinct patterns
    for i in range(30):
        call_id = f"call_{i:03d}"
        video_id = f"video_{chr(65 + i % 5)}"

        features_data.append({
            "call_id": call_id,
            "video_id": video_id,
            "speakers": {
                "SPEAKER_00": {  # Agent: high question rate, speaks first
                    "talk_time": 25 + i % 10, "turn_count": 8, "avg_turn_duration": 3.0,
                    "question_turn_rate": 0.5, "avg_words_per_turn": 12.0,
                    "long_turn_count": 2, "first_speaker": 1, "short_turn_count": 1,
                    "median_turn_duration": 2.5, "max_turn_duration": 8.0,
                    "question_turn_count": 4, "word_count": 96
                },
                "SPEAKER_01": {  # Prospect: no questions, short turns
                    "talk_time": 8, "turn_count": 6, "avg_turn_duration": 1.3,
                    "question_turn_rate": 0.0, "avg_words_per_turn": 4.0,
                    "long_turn_count": 0, "first_speaker": 0, "short_turn_count": 3,
                    "median_turn_duration": 1.2, "max_turn_duration": 2.5,
                    "question_turn_count": 0, "word_count": 24
                }
            }
        })
        labels[call_id] = {"SPEAKER_00": "agent", "SPEAKER_01": "prospect"}

    # Add some narrator examples
    for i in range(10):
        call_id = f"narr_{i:03d}"
        features_data.append({
            "call_id": call_id,
            "video_id": f"video_{chr(70 + i % 3)}",
            "speakers": {
                "SPEAKER_00": {  # Agent
                    "talk_time": 20, "turn_count": 6, "avg_turn_duration": 3.3,
                    "question_turn_rate": 0.5, "avg_words_per_turn": 11.0,
                    "long_turn_count": 2, "first_speaker": 1, "short_turn_count": 1,
                    "median_turn_duration": 3.0, "max_turn_duration": 7.0,
                    "question_turn_count": 3, "word_count": 66
                },
                "SPEAKER_01": {  # Prospect
                    "talk_time": 6, "turn_count": 4, "avg_turn_duration": 1.5,
                    "question_turn_rate": 0.0, "avg_words_per_turn": 5.0,
                    "long_turn_count": 0, "first_speaker": 0, "short_turn_count": 2,
                    "median_turn_duration": 1.5, "max_turn_duration": 2.0,
                    "question_turn_count": 0, "word_count": 20
                },
                "SPEAKER_02": {  # Narrator: long monologue, no interaction
                    "talk_time": 20 + i * 2, "turn_count": 1 + i % 2, "avg_turn_duration": 15.0,
                    "question_turn_rate": 0.0, "avg_words_per_turn": 60.0,
                    "long_turn_count": 1, "first_speaker": 0, "short_turn_count": 0,
                    "median_turn_duration": 15.0, "max_turn_duration": 20.0,
                    "question_turn_count": 0, "word_count": 120
                }
            }
        })
        labels[call_id] = {"SPEAKER_00": "agent", "SPEAKER_01": "prospect", "SPEAKER_02": "narrator"}

    with tempfile.TemporaryDirectory() as tmpdir:
        train_model_v2(features_data, labels, tmpdir)

        # Load model and test prediction
        with open(Path(tmpdir) / "role_model_v2.pkl", "rb") as f:
            model = pickle.load(f)

        # Test: agent-like features should predict agent
        agent_features = [25.0, 8, 3.0, 0.5, 12.0, 2, 1, 1, 2.5, 8.0, 4, 96]
        pred = model.predict([agent_features])[0]
        assert pred == "agent"

        # Test: narrator-like features should predict narrator
        narrator_features = [20.0, 1, 20.0, 0.0, 60.0, 1, 0, 0, 20.0, 20.0, 0, 60]
        pred = model.predict([narrator_features])[0]
        assert pred == "narrator"
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/whisperx_pipeline/test_train_role_model_v2.py -v
```

Expected: `ImportError: cannot import name 'train_model_v2'`

**Step 3: Implement training script**

Create `scripts/whisperx_pipeline/train_role_model_v2.py`:

```python
# scripts/whisperx_pipeline/train_role_model_v2.py
"""Train 3-class role classifier (agent/prospect/narrator)."""
import json
import csv
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Per-speaker feature columns (same as extract_speaker_features output)
FEATURE_COLUMNS = [
    "talk_time", "turn_count", "avg_turn_duration",
    "question_turn_rate", "avg_words_per_turn",
    "long_turn_count", "first_speaker", "short_turn_count",
    "median_turn_duration", "max_turn_duration",
    "question_turn_count", "word_count"
]

ROLE_CLASSES = ["agent", "narrator", "prospect"]  # Alphabetical for consistency


def load_labels_v2(labels_path: str) -> Dict[str, Dict[str, str]]:
    """Load labels as {call_id: {speaker_id: role}}."""
    labels = {}
    with open(labels_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            call_id = row["call_id"]
            if call_id not in labels:
                labels[call_id] = {}
            labels[call_id][row["speaker_id"]] = row["role"]
    return labels


def train_model_v2(
    features_data: List[dict],
    labels: Dict[str, Dict[str, str]],
    output_dir: str
):
    """
    Train 3-class role classifier.

    Args:
        features_data: List of {call_id, video_id, speakers: {spk_id: {features}}}
        labels: {call_id: {speaker_id: role}}
        output_dir: Directory to save model artifacts
    """
    # Build feature matrix: one row per speaker
    X = []
    y = []
    video_ids = []

    for item in features_data:
        call_id = item["call_id"]
        if call_id not in labels:
            continue

        call_labels = labels[call_id]
        speakers_data = item.get("speakers", {})

        for speaker_id, speaker_features in speakers_data.items():
            if speaker_id not in call_labels:
                continue

            role = call_labels[speaker_id]
            if role not in ROLE_CLASSES:
                print(f"WARNING: Unknown role '{role}' for {call_id}/{speaker_id}, skipping")
                continue

            # Extract feature vector
            feature_vec = [speaker_features.get(col, 0.0) for col in FEATURE_COLUMNS]
            X.append(feature_vec)
            y.append(role)
            video_ids.append(item.get("video_id", call_id.split("_")[0]))

    if len(X) < 10:
        raise ValueError(f"Not enough labeled data: {len(X)} samples (need at least 10)")

    X = np.array(X)
    y = np.array(y)

    # Video-level split to avoid leakage
    unique_videos = list(set(video_ids))
    if len(unique_videos) < 2:
        # Fallback to random split if only 1 video
        train_idx, val_idx = train_test_split(range(len(X)), test_size=0.2, random_state=42)
        train_mask = np.zeros(len(X), dtype=bool)
        train_mask[train_idx] = True
    else:
        train_videos, val_videos = train_test_split(unique_videos, test_size=0.2, random_state=42)
        train_mask = np.array([v in train_videos for v in video_ids])

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[~train_mask], y[~train_mask]

    # Train multi-class classifier
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(multi_class="multinomial", max_iter=1000))
    ])
    model.fit(X_train, y_train)

    # Evaluate
    val_accuracy = model.score(X_val, y_val) if len(X_val) > 0 else 0.0
    train_accuracy = model.score(X_train, y_train)

    # Compute per-class accuracy
    y_val_pred = model.predict(X_val) if len(X_val) > 0 else []
    per_class_acc = {}
    for cls in ROLE_CLASSES:
        mask = y_val == cls
        if mask.sum() > 0:
            per_class_acc[cls] = (y_val_pred[mask] == y_val[mask]).mean()
        else:
            per_class_acc[cls] = None

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "role_model_v2.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(output_dir / "feature_schema.json", "w") as f:
        json.dump({"features": FEATURE_COLUMNS, "version": "2.0"}, f)

    metadata = {
        "role_model_version": "2.0.0",
        "trained_at": datetime.now().isoformat(),
        "training_samples": len(y_train),
        "validation_samples": len(y_val),
        "num_classes": 3,
        "classes": ROLE_CLASSES,
        "train_accuracy": float(train_accuracy),
        "val_accuracy": float(val_accuracy),
        "per_class_accuracy": {k: float(v) if v is not None else None for k, v in per_class_acc.items()},
        "features": FEATURE_COLUMNS,
        "class_distribution_train": {cls: int((y_train == cls).sum()) for cls in ROLE_CLASSES},
        "class_distribution_val": {cls: int((y_val == cls).sum()) for cls in ROLE_CLASSES}
    }
    with open(output_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved to {output_dir}")
    print(f"Training accuracy: {train_accuracy:.3f}, Validation accuracy: {val_accuracy:.3f}")
    print(f"Per-class accuracy: {per_class_acc}")
    print(f"Class distribution (train): {metadata['class_distribution_train']}")
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/whisperx_pipeline/test_train_role_model_v2.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add scripts/whisperx_pipeline/train_role_model_v2.py tests/whisperx_pipeline/test_train_role_model_v2.py
git commit -m "feat(role): add train_role_model_v2 for 3-class classification"
```

---

## Task 4: Create Multi-Class Predictor

**Files:**
- Create: `scripts/whisperx_pipeline/role_predictor_v2.py`
- Test: `tests/whisperx_pipeline/test_role_predictor_v2.py`

**Step 1: Write the failing test**

Create `tests/whisperx_pipeline/test_role_predictor_v2.py`:

```python
# tests/whisperx_pipeline/test_role_predictor_v2.py
"""Tests for 3-class role predictor."""
import tempfile
import json
import pickle
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np


def test_role_predictor_v2_predict():
    """Predictor returns role for each speaker."""
    from scripts.whisperx_pipeline.role_predictor_v2 import RolePredictorV2, RolePredictionV2

    # Create mock model directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create mock model that returns predictable results
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array(["agent"])
        mock_model.predict_proba.return_value = np.array([[0.1, 0.1, 0.8]])  # agent, narrator, prospect
        mock_model.classes_ = np.array(["agent", "narrator", "prospect"])

        with open(tmpdir / "role_model_v2.pkl", "wb") as f:
            pickle.dump(mock_model, f)

        with open(tmpdir / "feature_schema.json", "w") as f:
            json.dump({"features": [
                "talk_time", "turn_count", "avg_turn_duration",
                "question_turn_rate", "avg_words_per_turn",
                "long_turn_count", "first_speaker", "short_turn_count",
                "median_turn_duration", "max_turn_duration",
                "question_turn_count", "word_count"
            ], "version": "2.0"}, f)

        with open(tmpdir / "training_metadata.json", "w") as f:
            json.dump({"classes": ["agent", "narrator", "prospect"]}, f)

        # Test prediction
        predictor = RolePredictorV2(str(tmpdir))

        turns = [
            {"spk": "SPEAKER_00", "t0_abs": 0, "t1_abs": 5, "text": "Hello?"},
            {"spk": "SPEAKER_01", "t0_abs": 5, "t1_abs": 7, "text": "Hi there"},
        ]

        prediction = predictor.predict(turns, call_start_abs=0)

        assert isinstance(prediction, RolePredictionV2)
        assert "SPEAKER_00" in prediction.speaker_roles
        assert "SPEAKER_01" in prediction.speaker_roles


def test_role_prediction_v2_dataclass():
    """RolePredictionV2 stores per-speaker predictions."""
    from scripts.whisperx_pipeline.role_predictor_v2 import RolePredictionV2

    prediction = RolePredictionV2(
        speaker_roles={"SPEAKER_00": "agent", "SPEAKER_01": "prospect", "SPEAKER_02": "narrator"},
        confidence_scores={"SPEAKER_00": 0.85, "SPEAKER_01": 0.72, "SPEAKER_02": 0.91},
        route_to="auto_complete"
    )

    assert prediction.speaker_roles["SPEAKER_00"] == "agent"
    assert prediction.speaker_roles["SPEAKER_02"] == "narrator"
    assert prediction.confidence_scores["SPEAKER_00"] == 0.85

    # Helper methods
    assert prediction.get_agent() == "SPEAKER_00"
    assert prediction.get_prospect() == "SPEAKER_01"
    assert prediction.get_narrators() == ["SPEAKER_02"]
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/whisperx_pipeline/test_role_predictor_v2.py -v
```

Expected: `ImportError: cannot import name 'RolePredictorV2'`

**Step 3: Implement predictor**

Create `scripts/whisperx_pipeline/role_predictor_v2.py`:

```python
# scripts/whisperx_pipeline/role_predictor_v2.py
"""Predict speaker roles using 3-class classifier."""
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

from .role_features import extract_all_speaker_features


@dataclass
class RolePredictionV2:
    """Prediction result for all speakers in a call."""
    speaker_roles: Dict[str, str]  # {speaker_id: role}
    confidence_scores: Dict[str, float]  # {speaker_id: confidence}
    route_to: str  # "auto_complete" or "llm_fallback"

    def get_agent(self) -> Optional[str]:
        """Get the speaker predicted as agent."""
        for spk, role in self.speaker_roles.items():
            if role == "agent":
                return spk
        return None

    def get_prospect(self) -> Optional[str]:
        """Get the speaker predicted as prospect."""
        for spk, role in self.speaker_roles.items():
            if role == "prospect":
                return spk
        return None

    def get_narrators(self) -> List[str]:
        """Get all speakers predicted as narrator."""
        return [spk for spk, role in self.speaker_roles.items() if role == "narrator"]


class RolePredictorV2:
    """Predict roles for all speakers in a call."""

    def __init__(self, model_dir: str, confidence_threshold: float = 0.6):
        model_dir_path = Path(model_dir)

        with open(model_dir_path / "role_model_v2.pkl", "rb") as f:
            self.model = pickle.load(f)

        with open(model_dir_path / "feature_schema.json") as f:
            schema = json.load(f)
            self.feature_columns: List[str] = schema["features"]

        with open(model_dir_path / "training_metadata.json") as f:
            self.metadata = json.load(f)

        self.classes = self.metadata.get("classes", ["agent", "narrator", "prospect"])
        self.confidence_threshold = confidence_threshold

    def predict(self, spk_turns: list, call_start_abs: float) -> RolePredictionV2:
        """Predict roles for all speakers."""
        # Extract features for all speakers
        all_features = extract_all_speaker_features(spk_turns, call_start_abs)

        if not all_features:
            return RolePredictionV2(
                speaker_roles={},
                confidence_scores={},
                route_to="llm_fallback"
            )

        speaker_roles = {}
        confidence_scores = {}
        min_confidence = 1.0

        for speaker_id, features in all_features.items():
            # Build feature vector
            X = [[features.get(col, 0.0) for col in self.feature_columns]]

            # Predict
            pred = self.model.predict(X)[0]
            proba = self.model.predict_proba(X)[0]

            # Get confidence (max probability)
            confidence = float(proba.max())

            speaker_roles[speaker_id] = pred
            confidence_scores[speaker_id] = confidence
            min_confidence = min(min_confidence, confidence)

        # Route based on minimum confidence across all speakers
        route_to = "auto_complete" if min_confidence >= self.confidence_threshold else "llm_fallback"

        return RolePredictionV2(
            speaker_roles=speaker_roles,
            confidence_scores=confidence_scores,
            route_to=route_to
        )
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/whisperx_pipeline/test_role_predictor_v2.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add scripts/whisperx_pipeline/role_predictor_v2.py tests/whisperx_pipeline/test_role_predictor_v2.py
git commit -m "feat(role): add RolePredictorV2 for 3-class prediction"
```

---

## Task 5: Generate Labeling Queue with All Speaker Features

**Files:**
- Modify/Create: `scripts/whisperx_pipeline/generate_label_queue.py`
- Test: Manual verification

**Step 1: Create queue generation script**

Create `scripts/whisperx_pipeline/generate_label_queue.py`:

```python
# scripts/whisperx_pipeline/generate_label_queue.py
"""Generate labeling queue from extracted calls."""
import json
import boto3
from pathlib import Path
from typing import List
import sys

from .config import S3_BUCKET, AWS_REGION


def generate_preview(turns: List[dict], max_turns: int = 12) -> str:
    """Generate text preview for labeling UI."""
    lines = []
    for t in turns[:max_turns]:
        text = t.get("text", "")[:60]
        lines.append(f"{t['spk']}: {text}")
    if len(turns) > max_turns:
        lines.append(f"... ({len(turns) - max_turns} more turns)")
    return "\n".join(lines)


def get_speakers_summary(turns: List[dict]) -> str:
    """Get summary of speakers in the call."""
    speakers = {}
    for t in turns:
        spk = t["spk"]
        speakers[spk] = speakers.get(spk, 0) + 1
    return ", ".join(f"{spk} ({count} turns)" for spk, count in sorted(speakers.items()))


def generate_queue(s3_prefix: str, output_path: str, limit: int = 100):
    """
    Generate labeling queue JSONL from S3 extracted calls.

    Args:
        s3_prefix: S3 prefix to scan (e.g., "runs/VIDEO_ID/RUN_ID/calls/")
        output_path: Local path for queue.jsonl
        limit: Max items to include
    """
    s3 = boto3.client("s3", region_name=AWS_REGION)

    # List all spk_turns.json files
    paginator = s3.get_paginator("list_objects_v2")
    items = []

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("spk_turns.json"):
                items.append(key)

    print(f"Found {len(items)} calls under {s3_prefix}")

    queue = []
    for key in items[:limit]:
        try:
            response = s3.get_object(Bucket=S3_BUCKET, Key=key)
            data = json.loads(response["Body"].read())
            turns = data.get("turns", [])

            # Extract call_id and video_id from path
            # Format: runs/{video_id}/{run_id}/calls/{call_id}/spk_turns.json
            parts = key.split("/")
            video_id = parts[1] if len(parts) > 1 else "unknown"
            call_id = parts[4] if len(parts) > 4 else key

            item = {
                "call_id": call_id,
                "video_id": video_id,
                "s3_key": key,
                "preview_text": generate_preview(turns),
                "speakers_summary": get_speakers_summary(turns),
                "turns": turns,
                "call_start_abs": data.get("call_start_abs", 0.0),
                "num_turns": len(turns),
                "num_speakers": len(set(t["spk"] for t in turns))
            }
            queue.append(item)

        except Exception as e:
            print(f"Error loading {key}: {e}")

    # Sort by number of speakers (prioritize 3+ speaker calls for labeling)
    queue.sort(key=lambda x: (-x["num_speakers"], x["call_id"]))

    with open(output_path, "w") as f:
        for item in queue:
            f.write(json.dumps(item) + "\n")

    print(f"Generated queue with {len(queue)} items: {output_path}")
    print(f"  3+ speakers: {sum(1 for q in queue if q['num_speakers'] >= 3)}")
    print(f"  2 speakers: {sum(1 for q in queue if q['num_speakers'] == 2)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_label_queue.py <s3_prefix> <output.jsonl>")
        print("Example: python generate_label_queue.py runs/abc123/ queue.jsonl")
        sys.exit(1)
    generate_queue(sys.argv[1], sys.argv[2])
```

**Step 2: Verify syntax**

```bash
python -c "from scripts.whisperx_pipeline.generate_label_queue import generate_queue; print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
git add scripts/whisperx_pipeline/generate_label_queue.py
git commit -m "feat(role): add generate_label_queue for labeling queue generation"
```

---

## Task 6: Integration Test - Full Workflow

**Files:**
- Test: Manual end-to-end verification

**Step 1: Run all unit tests**

```bash
pytest tests/whisperx_pipeline/test_role_features.py tests/whisperx_pipeline/test_train_role_model_v2.py tests/whisperx_pipeline/test_role_predictor_v2.py -v
```

Expected: All PASS

**Step 2: Test feature extraction with 3 speakers**

```bash
python -c "
from scripts.whisperx_pipeline.role_features import extract_all_speaker_features

turns = [
    {'spk': 'SPEAKER_00', 't0_abs': 0, 't1_abs': 8, 'text': 'Hi, this is Mike calling about your listing. Is this John?'},
    {'spk': 'SPEAKER_01', 't0_abs': 8, 't1_abs': 10, 'text': 'Yes it is'},
    {'spk': 'SPEAKER_02', 't0_abs': 10, 't1_abs': 22, 'text': 'So you can see here the agent introduced himself and asked a qualifying question. This is a great technique.'},
    {'spk': 'SPEAKER_00', 't0_abs': 22, 't1_abs': 30, 'text': 'I noticed your listing on Maple Street expired last week. What happened there?'},
    {'spk': 'SPEAKER_01', 't0_abs': 30, 't1_abs': 33, 'text': 'Yeah, the buyers fell through'},
]

features = extract_all_speaker_features(turns, 0)

print('Speaker features:')
for spk, feat in sorted(features.items()):
    print(f'{spk}:')
    print(f'  talk_time: {feat[\"talk_time\"]:.1f}s')
    print(f'  turn_count: {feat[\"turn_count\"]}')
    print(f'  question_rate: {feat[\"question_turn_rate\"]:.2f}')
    print(f'  avg_turn: {feat[\"avg_turn_duration\"]:.1f}s')
"
```

Expected output showing distinct patterns:
- SPEAKER_00 (agent): moderate talk time, questions
- SPEAKER_01 (prospect): short turns, no questions
- SPEAKER_02 (narrator): long monologue, no questions

**Step 3: Commit integration test**

```bash
git add -A
git commit -m "feat(role): complete 3-class role classifier infrastructure

- Add extract_all_speaker_features() for multi-speaker support
- Add label_roles_v2.py CLI for 3-class labeling
- Add train_role_model_v2.py for multi-class training
- Add RolePredictorV2 for 3-class prediction
- Add generate_label_queue.py for queue generation

The classifier now learns patterns for agent, prospect, AND narrator roles
from human-labeled examples, rather than using heuristics.
"
```

---

## Task 7: Update Pipeline Integration (Optional - After Labeling)

After collecting ~50 labeled examples using `label_roles_v2.py`:

**Step 1: Extract features from labeled calls**

```bash
python -c "
from scripts.whisperx_pipeline.role_features import extract_all_speaker_features
import json

# Load queue and extract features
with open('queue.jsonl') as f:
    queue = [json.loads(line) for line in f]

features_data = []
for item in queue:
    features = extract_all_speaker_features(item['turns'], item.get('call_start_abs', 0))
    features_data.append({
        'call_id': item['call_id'],
        'video_id': item['video_id'],
        'speakers': features
    })

with open('features.json', 'w') as f:
    json.dump(features_data, f, indent=2)

print(f'Extracted features for {len(features_data)} calls')
"
```

**Step 2: Train model**

```bash
python -c "
from scripts.whisperx_pipeline.train_role_model_v2 import train_model_v2, load_labels_v2
import json

with open('features.json') as f:
    features_data = json.load(f)

labels = load_labels_v2('speaker_labels.csv')
train_model_v2(features_data, labels, 'models/role_v2')
"
```

**Step 3: Test prediction**

```bash
python -c "
from scripts.whisperx_pipeline.role_predictor_v2 import RolePredictorV2

predictor = RolePredictorV2('models/role_v2')

# Test on a sample call
turns = [
    {'spk': 'SPEAKER_00', 't0_abs': 0, 't1_abs': 5, 'text': 'Hi is this John?'},
    {'spk': 'SPEAKER_01', 't0_abs': 5, 't1_abs': 7, 'text': 'Yes'},
    {'spk': 'SPEAKER_02', 't0_abs': 7, 't1_abs': 15, 'text': 'Great example of building rapport'},
]

result = predictor.predict(turns, 0)
print(f'Roles: {result.speaker_roles}')
print(f'Confidence: {result.confidence_scores}')
print(f'Agent: {result.get_agent()}')
print(f'Narrators: {result.get_narrators()}')
"
```

---

## Verification Checklist

After implementation, verify:

- [ ] `pytest tests/whisperx_pipeline/test_role_features.py -v` - All pass
- [ ] `pytest tests/whisperx_pipeline/test_train_role_model_v2.py -v` - All pass
- [ ] `pytest tests/whisperx_pipeline/test_role_predictor_v2.py -v` - All pass
- [ ] `extract_all_speaker_features()` returns features for ALL speakers
- [ ] `label_roles_v2.py` allows labeling each speaker as agent/prospect/narrator
- [ ] `train_model_v2()` trains a 3-class LogisticRegression
- [ ] `RolePredictorV2.predict()` returns role for each speaker
- [ ] Existing 2-speaker calls work correctly (backward compatible)

---

## Migration Notes

### Existing Labels (labels.csv)
The old format `call_id,agent_spk` is incompatible with the new 3-class model. Options:
1. **Start fresh** - Collect new labels using `label_roles_v2.py`
2. **Convert** - Write migration script that assumes non-agent = prospect (loses narrator info)

### Existing Model (role_model.pkl)
The v1 binary model is incompatible with v2 infrastructure. Keep both:
- `role_model.pkl` - Binary model (legacy)
- `role_model_v2.pkl` - 3-class model (new)

### Pipeline Integration
After training v2 model, update `pipeline.py` to use `RolePredictorV2` instead of `RolePredictor`.

---

## What the Model Will Learn

After training on ~50 labeled examples, the model should learn these patterns:

| Feature | Agent | Prospect | Narrator |
|---------|-------|----------|----------|
| `talk_time` | Moderate-High | Low | High (monologue) |
| `turn_count` | High | Moderate | Low (few long turns) |
| `avg_turn_duration` | 2-5 seconds | 1-3 seconds | 10+ seconds |
| `question_turn_rate` | High (0.3-0.5) | Low (0-0.1) | Low (0) |
| `first_speaker` | Often 1 | Usually 0 | Varies |
| `word_count` | High | Low | Very High |

The key distinguishing features for narrators:
- **Long monologues** (`avg_turn_duration > 10s`)
- **No questions** (`question_turn_rate = 0`)
- **No interaction** (doesn't respond to other speakers)
- **Meta-commentary** (talks ABOUT the call, not IN the call)
