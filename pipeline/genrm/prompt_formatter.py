"""Format WhisperX spk_turns data for GenRM evaluation."""
from typing import Dict, List, Optional, Tuple

from ..role_predictor_v2 import RolePredictionV2


def format_conversation_for_genrm(
    spk_turns: List[Dict],
    role_prediction: RolePredictionV2,
) -> Tuple[List[Dict[str, str]], Dict]:
    """
    Convert spk_turns with role predictions to GenRM conversation format.

    Args:
        spk_turns: List of turns from whisperx output
            [{speaker: "SPEAKER_00", text: "...", start: 0.0, end: 1.5}, ...]
        role_prediction: Role prediction result with speaker_roles mapping

    Returns:
        (messages, metadata) where:
        - messages: List of {"role": "user"|"assistant", "content": "..."}
        - metadata: Stats about the formatting (filtered turns, etc.)
    """
    messages = []
    metadata = {
        "total_turns": len(spk_turns),
        "included_turns": 0,
        "filtered_narrator_turns": 0,
        "filtered_empty_turns": 0,
        "unknown_speaker_turns": 0,
    }

    speaker_roles = role_prediction.speaker_roles

    for turn in spk_turns:
        # Handle both "spk" (whisperx format) and "speaker" (legacy format)
        speaker_id = turn.get("spk") or turn.get("speaker", "")
        text = turn.get("text", "").strip()

        # Skip empty turns
        if not text:
            metadata["filtered_empty_turns"] += 1
            continue

        # Get role for this speaker
        role = speaker_roles.get(speaker_id)

        if role is None:
            metadata["unknown_speaker_turns"] += 1
            continue

        # Filter narrator turns - they're not part of the conversation
        if role == "narrator":
            metadata["filtered_narrator_turns"] += 1
            continue

        # Map to GenRM roles (agent -> assistant, user -> user)
        genrm_role = "assistant" if role == "agent" else "user"

        messages.append({
            "role": genrm_role,
            "content": text,
        })
        metadata["included_turns"] += 1

    return messages, metadata


def merge_consecutive_same_role(
    messages: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """
    Merge consecutive turns from the same role.

    Some transcripts have multiple consecutive turns from one speaker
    due to diarization errors. GenRM expects alternating turns.

    Args:
        messages: List of {"role": str, "content": str}

    Returns:
        Merged messages with no consecutive same-role turns
    """
    if not messages:
        return []

    merged = []
    current_role = None
    current_content = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == current_role:
            # Same role - accumulate content
            current_content.append(content)
        else:
            # Different role - flush previous and start new
            if current_content:
                merged.append({
                    "role": current_role,
                    "content": " ".join(current_content),
                })
            current_role = role
            current_content = [content]

    # Flush final turn
    if current_content:
        merged.append({
            "role": current_role,
            "content": " ".join(current_content),
        })

    return merged


def validate_conversation(messages: List[Dict[str, str]]) -> Tuple[bool, List[str]]:
    """
    Validate that a conversation is suitable for GenRM evaluation.

    Args:
        messages: Formatted conversation messages

    Returns:
        (is_valid, issues) - is_valid is True if conversation passes checks
    """
    issues = []

    if not messages:
        issues.append("No messages after formatting")
        return False, issues

    if len(messages) < 2:
        issues.append(f"Too few turns ({len(messages)}), need at least 2")

    # Check for reasonable turn count
    if len(messages) > 200:
        issues.append(f"Excessive turns ({len(messages)}), may be concatenated calls")

    # Check for minimum content
    total_chars = sum(len(m["content"]) for m in messages)
    if total_chars < 50:
        issues.append(f"Very short conversation ({total_chars} chars)")

    # Check for some back-and-forth
    roles = set(m["role"] for m in messages)
    if len(roles) < 2:
        issues.append("Single speaker only - not a conversation")

    return len(issues) == 0, issues


def format_for_sft_output(
    spk_turns: List[Dict],
    role_prediction: RolePredictionV2,
    system_prompt: Optional[str] = None,
) -> Dict:
    """
    Format conversation for final SFT training output.

    Args:
        spk_turns: Raw speaker turns
        role_prediction: Role assignments
        system_prompt: Optional system message for the training data

    Returns:
        SFT-ready dict with "messages" key
    """
    messages, _ = format_conversation_for_genrm(spk_turns, role_prediction)
    messages = merge_consecutive_same_role(messages)

    sft_messages = []

    if system_prompt:
        sft_messages.append({
            "role": "system",
            "content": system_prompt,
        })

    sft_messages.extend(messages)

    return {"messages": sft_messages}
