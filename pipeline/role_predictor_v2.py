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

    def get_user(self) -> Optional[str]:
        """Get the speaker predicted as user."""
        for spk, role in self.speaker_roles.items():
            if role == "user":
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

        self.classes = self.metadata.get("classes", ["agent", "narrator", "user"])
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

        speaker_roles: Dict[str, str] = {}
        confidence_scores: Dict[str, float] = {}
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


def main() -> None:
    """CLI entry point for batch prediction."""
    import argparse
    import json
    import boto3
    from .config import S3_BUCKET, AWS_REGION

    parser = argparse.ArgumentParser(
        description="Predict roles for calls using trained 3-class model"
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Directory containing role_model_v2.pkl"
    )
    parser.add_argument(
        "--s3-prefix",
        help="S3 prefix to scan for calls (e.g., 'runs/VIDEO_ID/')"
    )
    parser.add_argument(
        "--input-file",
        help="Local JSON file with calls (alternative to S3)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for predictions JSON"
    )
    parser.add_argument(
        "--bucket",
        default=S3_BUCKET,
        help=f"S3 bucket (default: {S3_BUCKET})"
    )

    args = parser.parse_args()

    if not args.s3_prefix and not args.input_file:
        parser.error("Either --s3-prefix or --input-file is required")

    predictor = RolePredictorV2(args.model_dir)
    predictions = []

    if args.input_file:
        # Load from local file
        with open(args.input_file) as f:
            calls = json.load(f)

        for call in calls:
            turns = call.get("turns", [])
            call_start = call.get("call_start_abs", 0.0)
            result = predictor.predict(turns, call_start)
            predictions.append({
                "call_id": call.get("call_id", "unknown"),
                "speaker_roles": result.speaker_roles,
                "confidence_scores": result.confidence_scores,
                "route_to": result.route_to,
                "agent": result.get_agent(),
                "user": result.get_user(),
                "narrators": result.get_narrators()
            })
    else:
        # Load from S3
        s3 = boto3.client("s3", region_name=AWS_REGION)
        paginator = s3.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=args.bucket, Prefix=args.s3_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith("spk_turns.json"):
                    continue

                try:
                    response = s3.get_object(Bucket=args.bucket, Key=key)
                    data = json.loads(response["Body"].read())
                    turns = data.get("turns", [])
                    call_start = data.get("call_start_abs", 0.0)

                    result = predictor.predict(turns, call_start)

                    # Extract call_id from path
                    parts = key.split("/")
                    call_id = parts[4] if len(parts) > 4 else key

                    predictions.append({
                        "call_id": call_id,
                        "s3_key": key,
                        "speaker_roles": result.speaker_roles,
                        "confidence_scores": result.confidence_scores,
                        "route_to": result.route_to,
                        "agent": result.get_agent(),
                        "user": result.get_user(),
                        "narrators": result.get_narrators()
                    })
                except Exception as e:
                    print(f"Error processing {key}: {e}")

    with open(args.output, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Predicted roles for {len(predictions)} calls")
    print(f"  Auto-complete: {sum(1 for p in predictions if p['route_to'] == 'auto_complete')}")
    print(f"  LLM fallback: {sum(1 for p in predictions if p['route_to'] == 'llm_fallback')}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
