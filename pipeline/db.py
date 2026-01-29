# scripts/whisperx_pipeline/db.py
"""DynamoDB helper functions for tracking videos and calls."""
from datetime import datetime
from typing import Optional, List, Dict, Any
import boto3

from .config import (
    AWS_REGION,
    DYNAMODB_VIDEOS_TABLE,
    DYNAMODB_CALLS_TABLE,
    CONVERSATION_TYPES,
)


def _get_dynamodb():
    """Get DynamoDB resource."""
    return boto3.resource("dynamodb", region_name=AWS_REGION)


def _get_table(table_name: str):
    """Get DynamoDB table."""
    return _get_dynamodb().Table(table_name)


# ============================================================================
# Video Operations
# ============================================================================


def register_video(
    video_id: str,
    conversation_type: str,
    source_url: str,
    audio_s3_key: str,
) -> Dict[str, Any]:
    """
    Register a new video in the database.

    Args:
        video_id: YouTube video ID
        conversation_type: Type of conversation (pretraining, expired_listing)
        source_url: Original YouTube URL
        audio_s3_key: S3 key for the audio file

    Returns:
        The created video record

    Raises:
        ValueError: If conversation_type is invalid
    """
    if conversation_type not in CONVERSATION_TYPES:
        raise ValueError(
            f"Invalid conversation_type '{conversation_type}'. "
            f"Must be one of: {CONVERSATION_TYPES}"
        )

    now = datetime.now().isoformat()
    item = {
        "video_id": video_id,
        "conversation_type": conversation_type,
        "source_url": source_url,
        "audio_s3_key": audio_s3_key,
        "status": "pending",
        "latest_run_id": None,
        "call_count": 0,
        "created_at": now,
        "updated_at": now,
    }

    table = _get_table(DYNAMODB_VIDEOS_TABLE)
    table.put_item(Item=item)
    return item


def get_video(video_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a video by ID.

    Args:
        video_id: YouTube video ID

    Returns:
        Video record or None if not found
    """
    table = _get_table(DYNAMODB_VIDEOS_TABLE)
    response = table.get_item(Key={"video_id": video_id})
    return response.get("Item")


def update_video_status(
    video_id: str,
    status: str,
    run_id: Optional[str] = None,
    call_count: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Update video processing status.

    Args:
        video_id: YouTube video ID
        status: New status (pending, processing, completed, failed)
        run_id: Optional run ID to set as latest
        call_count: Optional call count to update

    Returns:
        Updated video record
    """
    table = _get_table(DYNAMODB_VIDEOS_TABLE)

    update_expr = "SET #status = :status, updated_at = :updated_at"
    expr_names = {"#status": "status"}
    expr_values: Dict[str, Any] = {
        ":status": status,
        ":updated_at": datetime.now().isoformat(),
    }

    if run_id is not None:
        update_expr += ", latest_run_id = :run_id"
        expr_values[":run_id"] = run_id

    if call_count is not None:
        update_expr += ", call_count = :call_count"
        expr_values[":call_count"] = call_count

    response = table.update_item(
        Key={"video_id": video_id},
        UpdateExpression=update_expr,
        ExpressionAttributeNames=expr_names,
        ExpressionAttributeValues=expr_values,
        ReturnValues="ALL_NEW",
    )
    return response["Attributes"]


def get_videos_by_type(
    conversation_type: str,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Get all videos of a specific conversation type.

    Args:
        conversation_type: Type to filter by
        status: Optional status filter

    Returns:
        List of video records
    """
    table = _get_table(DYNAMODB_VIDEOS_TABLE)

    # Full scan with filter (no GSI on conversation_type for videos table)
    filter_expr = "conversation_type = :type"
    expr_values = {":type": conversation_type}

    if status:
        filter_expr += " AND #status = :status"
        expr_values[":status"] = status
        response = table.scan(
            FilterExpression=filter_expr,
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues=expr_values,
        )
    else:
        response = table.scan(
            FilterExpression=filter_expr,
            ExpressionAttributeValues=expr_values,
        )

    return response.get("Items", [])


def get_videos_by_status(status: str) -> List[Dict[str, Any]]:
    """
    Get all videos with a specific status.

    Args:
        status: Status to filter by

    Returns:
        List of video records
    """
    table = _get_table(DYNAMODB_VIDEOS_TABLE)
    response = table.scan(
        FilterExpression="#status = :status",
        ExpressionAttributeNames={"#status": "status"},
        ExpressionAttributeValues={":status": status},
    )
    return response.get("Items", [])


# ============================================================================
# Call Operations
# ============================================================================


def register_call(
    call_id: str,
    video_id: str,
    run_id: str,
    s3_key: str,
    turn_count: int,
    speaker_count: int,
    conversation_type: str,
) -> Dict[str, Any]:
    """
    Register a new call in the database.

    Args:
        call_id: Unique call identifier
        video_id: Parent video ID
        run_id: WhisperX run that created this call
        s3_key: S3 key for spk_turns.json
        turn_count: Number of turns in the call
        speaker_count: Number of distinct speakers
        conversation_type: Type inherited from video

    Returns:
        The created call record
    """
    now = datetime.now().isoformat()
    item = {
        "call_id": call_id,
        "video_id": video_id,
        "conversation_type": conversation_type,
        "run_id": run_id,
        "s3_key": s3_key,
        "turn_count": turn_count,
        "speaker_count": speaker_count,
        "role_status": "pending",
        "agent_spk": None,
        "user_spk": None,
        "quality_status": "pending",
        "created_at": now,
    }

    table = _get_table(DYNAMODB_CALLS_TABLE)
    table.put_item(Item=item)
    return item


def get_call(call_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a call by ID.

    Args:
        call_id: Call identifier

    Returns:
        Call record or None if not found
    """
    table = _get_table(DYNAMODB_CALLS_TABLE)
    response = table.get_item(Key={"call_id": call_id})
    return response.get("Item")


def get_calls_for_video(video_id: str) -> List[Dict[str, Any]]:
    """
    Get all calls for a specific video.

    Args:
        video_id: Video ID

    Returns:
        List of call records
    """
    table = _get_table(DYNAMODB_CALLS_TABLE)
    response = table.query(
        IndexName="video-index",
        KeyConditionExpression="video_id = :vid",
        ExpressionAttributeValues={":vid": video_id},
    )
    return response.get("Items", [])


def update_call_roles(
    call_id: str,
    agent_spk: str,
    user_spk: str,
    role_status: str,
) -> Dict[str, Any]:
    """
    Update call role assignments.

    Args:
        call_id: Call identifier
        agent_spk: Speaker ID assigned as agent
        user_spk: Speaker ID assigned as user
        role_status: New role status (pending, labeled, predicted, reviewed)

    Returns:
        Updated call record
    """
    table = _get_table(DYNAMODB_CALLS_TABLE)
    response = table.update_item(
        Key={"call_id": call_id},
        UpdateExpression=(
            "SET agent_spk = :agent, user_spk = :user, role_status = :status"
        ),
        ExpressionAttributeValues={
            ":agent": agent_spk,
            ":user": user_spk,
            ":status": role_status,
        },
        ReturnValues="ALL_NEW",
    )
    return response["Attributes"]


def update_call_roles_v2(
    call_id: str,
    speaker_roles: Dict[str, str],
    role_status: str,
    confidence_scores: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """
    Update call role assignments (v2: supports 3+ speakers with narrator).

    Args:
        call_id: Call identifier
        speaker_roles: Dict mapping speaker_id to role (agent/user/narrator)
        role_status: New role status (pending, labeled, predicted, reviewed)
        confidence_scores: Optional dict mapping speaker_id to confidence score

    Returns:
        Updated call record
    """
    table = _get_table(DYNAMODB_CALLS_TABLE)

    # Extract agent and user for backward compatibility
    agent_spk = None
    user_spk = None
    narrator_spks = []

    for spk, role in speaker_roles.items():
        if role == "agent":
            agent_spk = spk
        elif role == "user":
            user_spk = spk
        elif role == "narrator":
            narrator_spks.append(spk)

    update_expr = (
        "SET role_status = :status, "
        "speaker_roles = :roles"
    )
    expr_values: Dict[str, Any] = {
        ":status": role_status,
        ":roles": speaker_roles,
    }

    # Add backward-compatible fields
    if agent_spk:
        update_expr += ", agent_spk = :agent"
        expr_values[":agent"] = agent_spk
    if user_spk:
        update_expr += ", user_spk = :user"
        expr_values[":user"] = user_spk
    if narrator_spks:
        update_expr += ", narrator_spks = :narrators"
        expr_values[":narrators"] = narrator_spks
    if confidence_scores:
        update_expr += ", role_confidence = :confidence"
        expr_values[":confidence"] = confidence_scores

    response = table.update_item(
        Key={"call_id": call_id},
        UpdateExpression=update_expr,
        ExpressionAttributeValues=expr_values,
        ReturnValues="ALL_NEW",
    )
    return response["Attributes"]


def update_call_quality(
    call_id: str,
    quality_status: str,
) -> Dict[str, Any]:
    """
    Update call quality status.

    Args:
        call_id: Call identifier
        quality_status: New quality status (pending, accepted, rejected, review)

    Returns:
        Updated call record
    """
    table = _get_table(DYNAMODB_CALLS_TABLE)
    response = table.update_item(
        Key={"call_id": call_id},
        UpdateExpression="SET quality_status = :status",
        ExpressionAttributeValues={":status": quality_status},
        ReturnValues="ALL_NEW",
    )
    return response["Attributes"]


def get_calls_for_labeling(
    conversation_type: str,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Get calls needing role labeling.

    Args:
        conversation_type: Type to filter by
        limit: Maximum number of calls to return

    Returns:
        List of call records with role_status='pending'
    """
    table = _get_table(DYNAMODB_CALLS_TABLE)
    response = table.query(
        IndexName="type-index",
        KeyConditionExpression="conversation_type = :type",
        FilterExpression="role_status = :status",
        ExpressionAttributeValues={
            ":type": conversation_type,
            ":status": "pending",
        },
        Limit=limit,
    )
    return response.get("Items", [])


def get_calls_by_type(
    conversation_type: str,
    role_status: Optional[str] = None,
    quality_status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Get all calls of a specific conversation type.

    Args:
        conversation_type: Type to filter by
        role_status: Optional role status filter
        quality_status: Optional quality status filter

    Returns:
        List of call records
    """
    table = _get_table(DYNAMODB_CALLS_TABLE)

    key_condition = "conversation_type = :type"
    expr_values = {":type": conversation_type}
    filter_parts = []

    if role_status:
        filter_parts.append("role_status = :role_status")
        expr_values[":role_status"] = role_status

    if quality_status:
        filter_parts.append("quality_status = :quality_status")
        expr_values[":quality_status"] = quality_status

    kwargs = {
        "IndexName": "type-index",
        "KeyConditionExpression": key_condition,
        "ExpressionAttributeValues": expr_values,
    }

    if filter_parts:
        kwargs["FilterExpression"] = " AND ".join(filter_parts)

    response = table.query(**kwargs)
    return response.get("Items", [])


# ============================================================================
# Statistics
# ============================================================================


def get_pipeline_stats() -> Dict[str, Any]:
    """
    Get overall pipeline statistics.

    Returns:
        Dict with counts by type, status, etc.
    """
    videos_table = _get_table(DYNAMODB_VIDEOS_TABLE)
    calls_table = _get_table(DYNAMODB_CALLS_TABLE)

    # Scan both tables (consider using GSI counts for large datasets)
    videos = videos_table.scan()["Items"]
    calls = calls_table.scan()["Items"]

    stats = {
        "videos": {
            "total": len(videos),
            "by_type": {},
            "by_status": {},
        },
        "calls": {
            "total": len(calls),
            "by_type": {},
            "by_role_status": {},
            "by_quality_status": {},
        },
    }

    # Count videos
    for v in videos:
        conv_type = v.get("conversation_type", "unknown")
        status = v.get("status", "unknown")

        stats["videos"]["by_type"][conv_type] = (
            stats["videos"]["by_type"].get(conv_type, 0) + 1
        )
        stats["videos"]["by_status"][status] = (
            stats["videos"]["by_status"].get(status, 0) + 1
        )

    # Count calls
    for c in calls:
        conv_type = c.get("conversation_type", "unknown")
        role_status = c.get("role_status", "unknown")
        quality_status = c.get("quality_status", "unknown")

        stats["calls"]["by_type"][conv_type] = (
            stats["calls"]["by_type"].get(conv_type, 0) + 1
        )
        stats["calls"]["by_role_status"][role_status] = (
            stats["calls"]["by_role_status"].get(role_status, 0) + 1
        )
        stats["calls"]["by_quality_status"][quality_status] = (
            stats["calls"]["by_quality_status"].get(quality_status, 0) + 1
        )

    return stats
