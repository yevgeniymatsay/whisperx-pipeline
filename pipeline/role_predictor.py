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
        model_dir_path = Path(model_dir)

        with open(model_dir_path / "role_model.pkl", "rb") as f:
            self.model = pickle.load(f)

        with open(model_dir_path / "training_metadata.json") as f:
            self.metadata = json.load(f)

        # Read feature schema from model directory (authoritative source)
        with open(model_dir_path / "feature_schema.json") as f:
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
