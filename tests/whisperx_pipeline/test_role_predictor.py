# tests/whisperx_pipeline/test_role_predictor.py
import pytest
from pipeline.role_predictor import RolePrediction


def test_role_prediction_dataclass():
    """RolePrediction has required fields."""
    prediction = RolePrediction(
        agent_spk="SPEAKER_00",
        user_spk="SPEAKER_01",
        confidence_gap=0.85,
        decision_features={"diff_talk_time": 5.0},
        route_to="auto_complete"
    )
    assert prediction.agent_spk == "SPEAKER_00"
    assert prediction.route_to == "auto_complete"
