#!/usr/bin/env python3
"""CLI for GenRM judge pipeline.

Usage:
    # Single call (debugging)
    python -m pipeline.genrm.cli judge \
        --s3-key runs/VIDEO/RUN/calls/CALL/spk_turns.json \
        --model-dir /path/to/role_model

    # Batch processing
    python -m pipeline.genrm.cli batch \
        --s3-prefix runs/ \
        --model-dir /path/to/role_model \
        --limit 100

    # Dry run (no S3 uploads)
    python -m pipeline.genrm.cli batch \
        --s3-prefix runs/VIDEO/ \
        --model-dir /path/to/role_model \
        --dry-run
"""
import argparse
import json
import logging
import sys
from typing import Dict, Optional, Any

import boto3

from .config import GenRMConfig
from .vllm_client import VLLMClient
from .principles import get_all_principles
from .prompt_formatter import (
    format_conversation_for_genrm,
    merge_consecutive_same_role,
    validate_conversation,
)
from .scorer import judgment_to_score
from .router import GenRMRouter
from ..role_predictor_v2 import RolePredictorV2, RolePredictionV2
from ..config import S3_BUCKET, AWS_REGION


def simple_role_heuristic(turns: list) -> RolePredictionV2:
    """
    Simple heuristic for role assignment when no model is available.

    Assumes:
    - First speaker (usually SPEAKER_00) is the agent
    - All other speakers are users
    - No narrator detection

    This is a fallback for testing. Production should use the trained model.
    """
    speakers = set()
    for turn in turns:
        spk = turn.get("spk") or turn.get("speaker")
        if spk:
            speakers.add(spk)

    if not speakers:
        return RolePredictionV2(
            speaker_roles={},
            confidence_scores={},
            route_to="llm_fallback"
        )

    # Sort speakers - first one is typically agent (called first)
    sorted_speakers = sorted(speakers)
    agent = sorted_speakers[0]

    speaker_roles = {}
    confidence_scores = {}
    for spk in sorted_speakers:
        speaker_roles[spk] = "agent" if spk == agent else "user"
        confidence_scores[spk] = 0.7  # Lower confidence since it's a heuristic

    return RolePredictionV2(
        speaker_roles=speaker_roles,
        confidence_scores=confidence_scores,
        route_to="auto_complete"  # Proceed with heuristic
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_call_from_s3(s3_key: str, bucket: str = S3_BUCKET) -> Dict[str, Any]:
    """Load spk_turns.json from S3."""
    s3 = boto3.client("s3", region_name=AWS_REGION)
    response = s3.get_object(Bucket=bucket, Key=s3_key)
    return json.loads(response["Body"].read())


def extract_call_id(s3_key: str) -> str:
    """Extract call ID from S3 key.

    Expected format: runs/{video_id}/{run_id}/calls/{call_id}/spk_turns.json
    Returns: The call_id directory name (already includes video_id prefix)
    """
    parts = s3_key.split("/")
    # Find 'calls' index and get the call_id after it
    try:
        calls_idx = parts.index("calls")
        call_id = parts[calls_idx + 1]
        return call_id
    except (ValueError, IndexError):
        # Fallback: use parent directory
        return parts[-2] if len(parts) >= 2 else parts[-1].replace(".json", "")


def judge_single_call(
    s3_key: str,
    model_dir: Optional[str],
    config: GenRMConfig,
    dry_run: bool = False,
    bucket: str = S3_BUCKET,
) -> Dict[str, Any]:
    """
    Judge a single call through the full pipeline.

    Steps:
    1. Load spk_turns.json from S3
    2. Predict roles with RolePredictorV2 (or heuristic if no model)
    3. Format conversation for GenRM
    4. Evaluate each principle
    5. Compute scores and route

    Args:
        s3_key: S3 key to spk_turns.json
        model_dir: Path to role model, or None to use heuristic
        config: GenRM configuration
        dry_run: If True, don't upload to S3
        bucket: S3 bucket name

    Returns:
        Result dict with judgment and routing info
    """
    call_id = extract_call_id(s3_key)
    logger.info(f"Processing call: {call_id}")

    # Step 1: Load call data
    logger.info("Loading from S3...")
    call_data = load_call_from_s3(s3_key, bucket)
    turns = call_data.get("turns", [])
    call_start_abs = call_data.get("call_start_abs", 0.0)

    if not turns:
        return {
            "call_id": call_id,
            "status": "skipped",
            "reason": "No turns in call data",
        }

    # Step 2: Predict roles
    if model_dir:
        logger.info("Predicting roles with trained model...")
        predictor = RolePredictorV2(model_dir)
        role_result = predictor.predict(turns, call_start_abs)
    else:
        logger.info("Using simple role heuristic (no model provided)...")
        role_result = simple_role_heuristic(turns)

    # Check for LLM fallback case
    if role_result.route_to == "llm_fallback":
        logger.info("Role prediction requires LLM fallback")
        router = GenRMRouter(config, bucket)
        return router.route_call(
            call_data={
                "original_s3_key": s3_key,
                "turns": turns,
                "role_assignment": {
                    "speaker_roles": role_result.speaker_roles,
                    "confidence_scores": role_result.confidence_scores,
                    "route_to": role_result.route_to,
                },
            },
            principle_scores={},
            call_id=call_id,
            route_to="llm_fallback",
            dry_run=dry_run,
        )

    # Step 3: Format conversation
    logger.info("Formatting conversation...")
    messages, format_metadata = format_conversation_for_genrm(turns, role_result)
    messages = merge_consecutive_same_role(messages)

    # Validate conversation
    is_valid, validation_issues = validate_conversation(messages)
    if not is_valid and not messages:
        return {
            "call_id": call_id,
            "status": "skipped",
            "reason": "Conversation validation failed",
            "issues": validation_issues,
        }

    # Step 4: Check vLLM health
    logger.info("Checking vLLM server...")
    vllm_client = VLLMClient(config)
    if not vllm_client.health_check():
        raise RuntimeError(
            f"vLLM server not responding at {config.vllm_base_url}. "
            "Make sure the server is running on the EC2 instance."
        )

    # Step 5: Evaluate each principle
    logger.info("Evaluating principles...")
    principles = get_all_principles()
    principle_scores: Dict[str, Dict[str, Any]] = {}

    for principle_name, principle_text in principles.items():
        logger.info(f"  Evaluating: {principle_name}")

        # Use GenRM-Principle format (principle role + conversation)
        response = vllm_client.evaluate_principle(
            conversation_messages=messages,
            principle_text=principle_text,
            principle_name=principle_name,
        )

        # Convert judgment to score
        score = judgment_to_score(response.judgment)

        principle_scores[principle_name] = {
            "judgment": response.judgment,
            "score": score,
            "reasoning_length": len(response.reasoning),
        }
        logger.info(f"    judgment={response.judgment}, score={score:.2f}")

    # Step 6: Route based on scores
    logger.info("Routing...")
    router = GenRMRouter(config, bucket)

    result = router.route_call(
        call_data={
            "video_id": s3_key.split("/")[1] if "runs" in s3_key else "unknown",
            "call_id": call_id,
            "original_s3_key": s3_key,
            "turns": messages,  # Use formatted turns
            "role_assignment": {
                "speaker_roles": role_result.speaker_roles,
                "confidence_scores": role_result.confidence_scores,
                "route_to": role_result.route_to,
            },
            "format_metadata": format_metadata,
        },
        principle_scores=principle_scores,
        call_id=call_id,
        route_to=role_result.route_to,
        validation_issues=validation_issues if validation_issues else None,
        dry_run=dry_run,
    )

    return result


def batch_judge(
    s3_prefix: str,
    model_dir: Optional[str],
    config: GenRMConfig,
    limit: Optional[int] = None,
    dry_run: bool = False,
    bucket: str = S3_BUCKET,
) -> Dict[str, Any]:
    """
    Judge multiple calls from an S3 prefix.

    Scans for spk_turns.json files and processes each.
    """
    s3 = boto3.client("s3", region_name=AWS_REGION)
    paginator = s3.get_paginator("list_objects_v2")

    # Find all spk_turns.json files
    call_keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("spk_turns.json"):
                call_keys.append(key)

    if not call_keys:
        logger.warning(f"No spk_turns.json files found under s3://{bucket}/{s3_prefix}")
        return {"status": "no_calls", "processed": 0}

    # Apply limit
    if limit:
        call_keys = call_keys[:limit]

    logger.info(f"Found {len(call_keys)} calls to process")

    stats = {
        "total": len(call_keys),
        "accept": 0,
        "review": 0,
        "reject": 0,
        "role_fallback": 0,
        "skipped": 0,
        "error": 0,
    }
    results = []

    for i, s3_key in enumerate(call_keys, 1):
        logger.info(f"\n[{i}/{len(call_keys)}] Processing: {s3_key}")

        try:
            result = judge_single_call(
                s3_key=s3_key,
                model_dir=model_dir,
                config=config,
                dry_run=dry_run,
                bucket=bucket,
            )

            results.append(result)

            # Update stats
            if result.get("status") == "skipped":
                stats["skipped"] += 1
            else:
                decision = result.get("decision", "").upper()
                if decision == "ACCEPT":
                    stats["accept"] += 1
                elif decision == "REVIEW":
                    stats["review"] += 1
                elif decision == "REJECT":
                    stats["reject"] += 1
                elif decision == "ROLE_FALLBACK":
                    stats["role_fallback"] += 1

            # Log result
            decision = result.get("decision", result.get("status", "unknown"))
            score = result.get("aggregate_score")
            score_str = f" ({score:.3f})" if score else ""
            logger.info(f"  -> {decision}{score_str}")

        except Exception as e:
            logger.error(f"  -> ERROR: {e}")
            stats["error"] += 1
            results.append({
                "s3_key": s3_key,
                "status": "error",
                "error": str(e),
            })

    # Print summary
    print("\n" + "=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)
    print(f"  Total processed: {stats['total']}")
    print(f"  ACCEPT: {stats['accept']}")
    print(f"  REVIEW: {stats['review']}")
    print(f"  REJECT: {stats['reject']}")
    print(f"  ROLE_FALLBACK: {stats['role_fallback']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Errors: {stats['error']}")
    print("=" * 60)

    return {
        "stats": stats,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="GenRM judge pipeline for WhisperX calls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Judge single call
    judge_parser = subparsers.add_parser("judge", help="Judge a single call")
    judge_parser.add_argument(
        "--s3-key",
        required=True,
        help="S3 key to spk_turns.json file",
    )
    judge_parser.add_argument(
        "--model-dir",
        help="Path to role model directory (uses heuristic if not provided)",
    )
    judge_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't upload results to S3",
    )
    judge_parser.add_argument(
        "--bucket",
        default=S3_BUCKET,
        help=f"S3 bucket (default: {S3_BUCKET})",
    )
    judge_parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000/v1",
        help="vLLM server URL",
    )

    # Batch processing
    batch_parser = subparsers.add_parser("batch", help="Batch process calls")
    batch_parser.add_argument(
        "--s3-prefix",
        required=True,
        help="S3 prefix to scan for calls",
    )
    batch_parser.add_argument(
        "--model-dir",
        help="Path to role model directory (uses heuristic if not provided)",
    )
    batch_parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of calls to process",
    )
    batch_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't upload results to S3",
    )
    batch_parser.add_argument(
        "--bucket",
        default=S3_BUCKET,
        help=f"S3 bucket (default: {S3_BUCKET})",
    )
    batch_parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000/v1",
        help="vLLM server URL",
    )

    # Health check
    health_parser = subparsers.add_parser("health", help="Check vLLM server health")
    health_parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000/v1",
        help="vLLM server URL",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Create config with vLLM URL if provided
    config = GenRMConfig()
    if hasattr(args, "vllm_url"):
        config.vllm_base_url = args.vllm_url

    if args.command == "health":
        client = VLLMClient(config)
        if client.health_check():
            model_id = client.get_model_id()
            print(f"vLLM server is healthy at {config.vllm_base_url}")
            print(f"Model: {model_id}")
            sys.exit(0)
        else:
            print(f"vLLM server not responding at {config.vllm_base_url}")
            sys.exit(1)

    elif args.command == "judge":
        result = judge_single_call(
            s3_key=args.s3_key,
            model_dir=args.model_dir,
            config=config,
            dry_run=args.dry_run,
            bucket=args.bucket,
        )
        print(json.dumps(result, indent=2))

    elif args.command == "batch":
        result = batch_judge(
            s3_prefix=args.s3_prefix,
            model_dir=args.model_dir,
            config=config,
            limit=args.limit,
            dry_run=args.dry_run,
            bucket=args.bucket,
        )
        # Results already printed in function


if __name__ == "__main__":
    main()
