# tests/whisperx_pipeline/test_train_role_model_v2.py
"""Tests for 3-class role model training."""
import tempfile
import json
from pathlib import Path


def test_train_model_v2_basic():
    """Train a 3-class model from labeled data."""
    from pipeline.train_role_model_v2 import train_model_v2

    # Synthetic training data (features for each speaker)
    features_data = []

    # Call 1: 3 speakers (agent, user, narrator pattern)
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
        "call_001": {"SPEAKER_00": "agent", "SPEAKER_01": "user", "SPEAKER_02": "narrator"}
    }
    for i in range(2, 20):
        labels[f"call_{i:03d}"] = {"SPEAKER_00": "agent", "SPEAKER_01": "user"}

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
        assert metadata["classes"] == ["agent", "narrator", "user"]
        assert metadata["training_samples"] > 0


def test_train_model_v2_predicts_correctly():
    """Model correctly predicts role from features."""
    from pipeline.train_role_model_v2 import train_model_v2
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
        labels[call_id] = {"SPEAKER_00": "agent", "SPEAKER_01": "user"}

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
        labels[call_id] = {"SPEAKER_00": "agent", "SPEAKER_01": "user", "SPEAKER_02": "narrator"}

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
