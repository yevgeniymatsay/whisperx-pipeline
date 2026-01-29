# scripts/whisperx_pipeline/train_role_model_v2.py
"""Train 3-class role classifier (agent/user/narrator)."""
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

ROLE_CLASSES = ["agent", "narrator", "user"]  # Alphabetical for consistency


def load_labels_v2(labels_path: str) -> Dict[str, Dict[str, str]]:
    """Load labels as {call_id: {speaker_id: role}}."""
    labels: Dict[str, Dict[str, str]] = {}
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
) -> None:
    """
    Train 3-class role classifier.

    Args:
        features_data: List of {call_id, video_id, speakers: {spk_id: {features}}}
        labels: {call_id: {speaker_id: role}}
        output_dir: Directory to save model artifacts
    """
    # Build feature matrix: one row per speaker
    X: List[List[float]] = []
    y: List[str] = []
    video_ids: List[str] = []

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
            feature_vec = [float(speaker_features.get(col, 0.0)) for col in FEATURE_COLUMNS]
            X.append(feature_vec)
            y.append(role)
            video_ids.append(item.get("video_id", call_id.split("_")[0]))

    if len(X) < 10:
        raise ValueError(f"Not enough labeled data: {len(X)} samples (need at least 10)")

    X_arr = np.array(X)
    y_arr = np.array(y)

    # Video-level split to avoid leakage
    unique_videos = list(set(video_ids))
    if len(unique_videos) < 2:
        # Fallback to random split if only 1 video
        train_idx, _ = train_test_split(range(len(X_arr)), test_size=0.2, random_state=42)
        train_mask = np.zeros(len(X_arr), dtype=bool)
        train_mask[train_idx] = True
    else:
        train_videos, _ = train_test_split(unique_videos, test_size=0.2, random_state=42)
        train_mask = np.array([v in train_videos for v in video_ids])

    X_train, y_train = X_arr[train_mask], y_arr[train_mask]
    X_val, y_val = X_arr[~train_mask], y_arr[~train_mask]

    # Train multi-class classifier
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(multi_class="multinomial", max_iter=1000))
    ])
    model.fit(X_train, y_train)

    # Evaluate
    val_accuracy = float(model.score(X_val, y_val)) if len(X_val) > 0 else 0.0
    train_accuracy = float(model.score(X_train, y_train))

    # Compute per-class accuracy
    y_val_pred = model.predict(X_val) if len(X_val) > 0 else np.array([])
    per_class_acc: Dict[str, float | None] = {}
    for cls in ROLE_CLASSES:
        mask = y_val == cls
        if mask.sum() > 0:
            per_class_acc[cls] = float((y_val_pred[mask] == y_val[mask]).mean())
        else:
            per_class_acc[cls] = None

    # Save
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    with open(output_dir_path / "role_model_v2.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(output_dir_path / "feature_schema.json", "w") as f:
        json.dump({"features": FEATURE_COLUMNS, "version": "2.0"}, f)

    metadata = {
        "role_model_version": "2.0.0",
        "trained_at": datetime.now().isoformat(),
        "training_samples": len(y_train),
        "validation_samples": len(y_val),
        "num_classes": 3,
        "classes": ROLE_CLASSES,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "per_class_accuracy": {k: v for k, v in per_class_acc.items()},
        "features": FEATURE_COLUMNS,
        "class_distribution_train": {cls: int((y_train == cls).sum()) for cls in ROLE_CLASSES},
        "class_distribution_val": {cls: int((y_val == cls).sum()) for cls in ROLE_CLASSES}
    }
    with open(output_dir_path / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved to {output_dir_path}")
    print(f"Training accuracy: {train_accuracy:.3f}, Validation accuracy: {val_accuracy:.3f}")
    print(f"Per-class accuracy: {per_class_acc}")
    print(f"Class distribution (train): {metadata['class_distribution_train']}")


def main() -> None:
    """CLI entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train 3-class role classifier (agent/user/narrator)"
    )
    parser.add_argument(
        "--features",
        required=True,
        help="Path to features.json (from extract_training_features.py)"
    )
    parser.add_argument(
        "--labels",
        required=True,
        help="Path to speaker_labels.csv (from label_roles_v2.py)"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save model artifacts"
    )

    args = parser.parse_args()

    # Load data
    with open(args.features) as f:
        features_data = json.load(f)

    labels = load_labels_v2(args.labels)

    print(f"Loaded {len(features_data)} calls with features")
    print(f"Loaded labels for {len(labels)} calls")

    train_model_v2(features_data, labels, args.output_dir)


if __name__ == "__main__":
    main()
