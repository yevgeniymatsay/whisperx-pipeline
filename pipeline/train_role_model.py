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
    best_coverage = 0.0

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
                               y_val[confidence_gaps >= best_threshold]).mean()) if best_coverage > 0 else 0.0,
        "val_coverage": float(best_coverage),
        "features": FEATURE_COLUMNS
    }
    with open(output_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved to {output_dir}")
    print(f"Threshold: {best_threshold}, Accuracy: {metadata['val_accuracy']:.3f}, Coverage: {metadata['val_coverage']:.3f}")
