"""Labels CSV handling."""
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Set, Any

LABELS_FILE = Path("speaker_labels.csv")
PROGRESS_FILE = Path("labeler_progress.json")


def load_progress() -> Dict[str, Any]:
    """Load progress tracking."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"labeled": [], "skipped": [], "current_index": 0}


def save_progress(progress: Dict[str, Any]) -> None:
    """Save progress tracking."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def get_labeled_call_ids() -> Set[str]:
    """Get set of already labeled call IDs."""
    progress = load_progress()
    return set(progress.get("labeled", []))


def get_skipped_call_ids() -> Set[str]:
    """Get set of skipped call IDs."""
    progress = load_progress()
    return set(progress.get("skipped", []))


def save_labels(
    video_id: str,
    call_id: str,
    speaker_roles: Dict[str, str],
) -> None:
    """Save labels for a call and update progress."""
    # Append to CSV
    file_exists = LABELS_FILE.exists()
    with open(LABELS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["video_id", "call_id", "speaker_id", "role", "labeled_at"])

        timestamp = datetime.now().isoformat()
        for speaker_id, role in speaker_roles.items():
            writer.writerow([video_id, call_id, speaker_id, role, timestamp])

    # Update progress (don't increment index - frontend handles navigation)
    progress = load_progress()
    if call_id not in progress["labeled"]:
        progress["labeled"].append(call_id)
    save_progress(progress)


def skip_call(call_id: str) -> None:
    """Mark a call as skipped (don't increment index - frontend handles navigation)."""
    progress = load_progress()
    if call_id not in progress["skipped"]:
        progress["skipped"].append(call_id)
    save_progress(progress)


def get_current_index() -> int:
    """Get current position in queue."""
    progress = load_progress()
    return progress.get("current_index", 0)


def set_current_index(index: int) -> None:
    """Set current position in queue."""
    progress = load_progress()
    progress["current_index"] = index
    save_progress(progress)
