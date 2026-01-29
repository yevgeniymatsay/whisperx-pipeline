# tests/whisperx_pipeline/test_role_predictor_v2.py
"""Tests for 3-class role predictor."""
import tempfile
import json
import pickle
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def _create_test_model():
    """Create a simple trained model for testing."""
    # Create training data with distinct patterns
    X = np.array([
        # Agent patterns: high question rate, first speaker
        [25, 8, 3, 0.5, 12, 2, 1, 1, 2.5, 8, 4, 96],
        [22, 7, 3.1, 0.4, 11, 2, 1, 1, 2.3, 7, 3, 77],
        # Prospect patterns: short turns, no questions
        [8, 6, 1.3, 0, 4, 0, 0, 3, 1.2, 2.5, 0, 24],
        [6, 5, 1.2, 0, 5, 0, 0, 2, 1.1, 2, 0, 25],
        # Narrator patterns: long monologue
        [20, 1, 20, 0, 60, 1, 0, 0, 20, 20, 0, 60],
        [25, 2, 12.5, 0, 50, 2, 0, 0, 12, 15, 0, 100],
    ])
    y = np.array(["agent", "agent", "user", "user", "narrator", "narrator"])

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    model.fit(X, y)
    return model


def test_role_predictor_v2_predict():
    """Predictor returns role for each speaker."""
    from pipeline.role_predictor_v2 import RolePredictorV2, RolePredictionV2

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create and save a real model
        model = _create_test_model()
        with open(tmpdir_path / "role_model_v2.pkl", "wb") as f:
            pickle.dump(model, f)

        with open(tmpdir_path / "feature_schema.json", "w") as f:
            json.dump({"features": [
                "talk_time", "turn_count", "avg_turn_duration",
                "question_turn_rate", "avg_words_per_turn",
                "long_turn_count", "first_speaker", "short_turn_count",
                "median_turn_duration", "max_turn_duration",
                "question_turn_count", "word_count"
            ], "version": "2.0"}, f)

        with open(tmpdir_path / "training_metadata.json", "w") as f:
            json.dump({"classes": ["agent", "narrator", "user"]}, f)

        # Test prediction
        predictor = RolePredictorV2(str(tmpdir_path))

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
    from pipeline.role_predictor_v2 import RolePredictionV2

    prediction = RolePredictionV2(
        speaker_roles={"SPEAKER_00": "agent", "SPEAKER_01": "user", "SPEAKER_02": "narrator"},
        confidence_scores={"SPEAKER_00": 0.85, "SPEAKER_01": 0.72, "SPEAKER_02": 0.91},
        route_to="auto_complete"
    )

    assert prediction.speaker_roles["SPEAKER_00"] == "agent"
    assert prediction.speaker_roles["SPEAKER_02"] == "narrator"
    assert prediction.confidence_scores["SPEAKER_00"] == 0.85

    # Helper methods
    assert prediction.get_agent() == "SPEAKER_00"
    assert prediction.get_user() == "SPEAKER_01"
    assert prediction.get_narrators() == ["SPEAKER_02"]


def test_role_predictor_v2_empty_turns():
    """Predictor handles empty turns gracefully."""
    from pipeline.role_predictor_v2 import RolePredictorV2

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create and save a real model
        model = _create_test_model()
        with open(tmpdir_path / "role_model_v2.pkl", "wb") as f:
            pickle.dump(model, f)

        with open(tmpdir_path / "feature_schema.json", "w") as f:
            json.dump({"features": [
                "talk_time", "turn_count", "avg_turn_duration",
                "question_turn_rate", "avg_words_per_turn",
                "long_turn_count", "first_speaker", "short_turn_count",
                "median_turn_duration", "max_turn_duration",
                "question_turn_count", "word_count"
            ], "version": "2.0"}, f)

        with open(tmpdir_path / "training_metadata.json", "w") as f:
            json.dump({"classes": ["agent", "narrator", "user"]}, f)

        predictor = RolePredictorV2(str(tmpdir_path))
        prediction = predictor.predict([], call_start_abs=0)

        assert prediction.speaker_roles == {}
        assert prediction.route_to == "llm_fallback"
