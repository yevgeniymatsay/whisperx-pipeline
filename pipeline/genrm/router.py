"""Route judged calls to S3 buckets based on GenRM scores."""
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from enum import Enum

import boto3

from .config import GenRMConfig, DEFAULT_CONFIG
from .scorer import aggregate_scores

logger = logging.getLogger(__name__)


class RoutingDecision(Enum):
    """Possible routing decisions."""
    ACCEPT = "ACCEPT"
    REVIEW = "REVIEW"
    REJECT = "REJECT"
    ROLE_FALLBACK = "ROLE_FALLBACK"


@dataclass
class GenRMJudgment:
    """Complete judgment result for a call."""
    decision: RoutingDecision
    aggregate_score: float
    principle_scores: Dict[str, Dict[str, float]]
    judged_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        # Handle mixed types in principle_scores (strings and floats)
        serialized_scores = {}
        for k, v in self.principle_scores.items():
            serialized_scores[k] = {}
            for sk, sv in v.items():
                if isinstance(sv, float):
                    serialized_scores[k][sk] = round(sv, 4)
                else:
                    serialized_scores[k][sk] = sv

        return {
            "decision": self.decision.value,
            "aggregate_score": round(self.aggregate_score, 4),
            "principle_scores": serialized_scores,
            "judged_at": self.judged_at,
            "issues": self.issues,
        }


class GenRMRouter:
    """Route calls based on GenRM judgment scores."""

    def __init__(
        self,
        config: GenRMConfig = DEFAULT_CONFIG,
        s3_bucket: Optional[str] = None,
    ):
        self.config = config
        config.validate()

        # Import S3 bucket from whisperx config if not provided
        if s3_bucket is None:
            from ..config import S3_BUCKET
            s3_bucket = S3_BUCKET
        self.s3_bucket = s3_bucket

        self.s3_client = boto3.client("s3")

    def compute_judgment(
        self,
        principle_scores: Dict[str, Dict[str, float]],
        validation_issues: Optional[List[str]] = None,
    ) -> GenRMJudgment:
        """
        Compute routing decision from principle scores.

        Args:
            principle_scores: {principle: {"reward": float, "p_yes": float}}
            validation_issues: Any issues from conversation validation

        Returns:
            GenRMJudgment with decision and scores
        """
        issues = validation_issues or []

        # Calculate aggregate score
        aggregate = aggregate_scores(principle_scores, self.config.principle_weights)

        # Determine decision based on thresholds
        if aggregate >= self.config.accept_threshold:
            decision = RoutingDecision.ACCEPT
        elif aggregate >= self.config.review_threshold:
            decision = RoutingDecision.REVIEW
        else:
            decision = RoutingDecision.REJECT

        # Check for any principle that completely fails
        for principle, scores in principle_scores.items():
            p_yes = scores.get("p_yes", 0.5)
            if p_yes < 0.3:
                issues.append(f"Low {principle} score: {p_yes:.2f}")
                # Downgrade to at most REVIEW if any principle is very low
                if decision == RoutingDecision.ACCEPT:
                    decision = RoutingDecision.REVIEW

        return GenRMJudgment(
            decision=decision,
            aggregate_score=aggregate,
            principle_scores=principle_scores,
            issues=issues,
        )

    def decide_for_role_fallback(self) -> GenRMJudgment:
        """
        Create judgment for calls that need LLM role fallback.

        These calls couldn't be role-assigned automatically, so they
        go to a separate bucket for manual review or LLM processing.
        """
        return GenRMJudgment(
            decision=RoutingDecision.ROLE_FALLBACK,
            aggregate_score=0.0,
            principle_scores={},
            issues=["Role prediction required LLM fallback"],
        )

    def get_s3_prefix(self, decision: RoutingDecision) -> str:
        """Get S3 prefix for a routing decision."""
        prefix_map = {
            RoutingDecision.ACCEPT: self.config.s3_prefix_accepted,
            RoutingDecision.REVIEW: self.config.s3_prefix_review,
            RoutingDecision.REJECT: self.config.s3_prefix_rejected,
            RoutingDecision.ROLE_FALLBACK: self.config.s3_prefix_role_fallback,
        }
        return prefix_map[decision]

    def upload_result(
        self,
        call_data: Dict[str, Any],
        judgment: GenRMJudgment,
        call_id: str,
        dry_run: bool = False,
    ) -> str:
        """
        Upload judged call to appropriate S3 bucket.

        Args:
            call_data: Original call data to augment with judgment
            judgment: GenRM judgment result
            call_id: Unique call identifier
            dry_run: If True, don't actually upload

        Returns:
            S3 key where data was (or would be) uploaded
        """
        # Add judgment to call data
        output_data = {
            **call_data,
            "genrm_judgment": judgment.to_dict(),
        }

        prefix = self.get_s3_prefix(judgment.decision)
        s3_key = f"{prefix}{call_id}.json"

        if dry_run:
            logger.info(f"[DRY RUN] Would upload to s3://{self.s3_bucket}/{s3_key}")
            return s3_key

        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=s3_key,
            Body=json.dumps(output_data, indent=2),
            ContentType="application/json",
        )

        logger.info(f"Uploaded to s3://{self.s3_bucket}/{s3_key}")
        return s3_key

    def route_call(
        self,
        call_data: Dict[str, Any],
        principle_scores: Dict[str, Dict[str, float]],
        call_id: str,
        route_to: str = "auto_complete",
        validation_issues: Optional[List[str]] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Route a single call based on GenRM scores.

        Args:
            call_data: Original call data
            principle_scores: Scores from GenRM evaluation
            call_id: Unique call identifier
            route_to: Role prediction route ("auto_complete" or "llm_fallback")
            validation_issues: Any conversation validation issues
            dry_run: If True, don't upload to S3

        Returns:
            Result dict with decision, scores, and S3 key
        """
        # Handle role fallback case
        if route_to == "llm_fallback":
            judgment = self.decide_for_role_fallback()
        else:
            judgment = self.compute_judgment(principle_scores, validation_issues)

        # Upload to S3
        s3_key = self.upload_result(call_data, judgment, call_id, dry_run=dry_run)

        return {
            "call_id": call_id,
            "decision": judgment.decision.value,
            "aggregate_score": judgment.aggregate_score,
            "principle_scores": judgment.principle_scores,
            "s3_key": s3_key,
            "issues": judgment.issues,
        }
