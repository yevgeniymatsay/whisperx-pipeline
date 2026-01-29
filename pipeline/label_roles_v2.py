# scripts/whisperx_pipeline/label_roles_v2.py
"""CLI for labeling speaker roles (3-class: agent/user/narrator)."""
import json
import csv
from pathlib import Path
from datetime import datetime
import sys
from typing import List, Dict, Set


ROLES = ["agent", "user", "narrator"]


def load_queue(queue_path: str) -> List[dict]:
    """Load labeling queue from JSONL."""
    items = []
    with open(queue_path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def load_existing_labels(labels_path: str) -> Dict[str, Dict[str, str]]:
    """Load existing labels as {call_id: {speaker_id: role}}."""
    labels: Dict[str, Dict[str, str]] = {}
    if Path(labels_path).exists():
        with open(labels_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                call_id = row["call_id"]
                if call_id not in labels:
                    labels[call_id] = {}
                labels[call_id][row["speaker_id"]] = row["role"]
    return labels


def get_labeled_calls(labels_path: str) -> Set[str]:
    """Get call_ids that are fully labeled (all speakers have roles)."""
    # For simplicity, we track which calls have been processed
    # A more robust approach would check if all speakers are labeled
    labeled: Set[str] = set()
    if Path(labels_path).exists():
        with open(labels_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                labeled.add(row["call_id"])
    return labeled


def save_labels(labels_path: str, video_id: str, call_id: str, speaker_roles: Dict[str, str]) -> None:
    """Save labels for all speakers in a call."""
    file_exists = Path(labels_path).exists()
    with open(labels_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["video_id", "call_id", "speaker_id", "role", "labeled_at"])
        timestamp = datetime.now().isoformat()
        for speaker_id, role in speaker_roles.items():
            writer.writerow([video_id, call_id, speaker_id, role, timestamp])


def display_call(item: dict, index: int, total: int) -> None:
    """Display call transcript for labeling."""
    print("\n" + "=" * 70)
    print(f"Call: {item['call_id']} ({index + 1}/{total})")
    print("=" * 70)

    # Display preview with speaker labels
    turns = item.get("turns", [])
    for turn in turns[:15]:  # Show first 15 turns
        spk = turn.get("spk", "?")
        text = turn.get("text", "")[:80]  # Truncate long text
        print(f"  {spk}: {text}")

    if len(turns) > 15:
        print(f"  ... ({len(turns) - 15} more turns)")

    print("=" * 70)


def get_speakers_from_item(item: dict) -> List[str]:
    """Get unique speakers from item's turns."""
    turns = item.get("turns", [])
    speakers: List[str] = []
    seen: Set[str] = set()
    for turn in turns:
        spk = turn.get("spk")
        if spk and spk not in seen:
            speakers.append(spk)
            seen.add(spk)
    return speakers


def label_speakers(speakers: List[str]) -> Dict[str, str]:
    """Prompt user to label each speaker."""
    print("\nLabel each speaker:")
    print("  [a] = agent (the cold caller)")
    print("  [u] = user (the person being called)")
    print("  [n] = narrator (YouTube host, commentator)")
    print("  [s] = skip this call")
    print()

    roles: Dict[str, str] = {}
    for spk in speakers:
        while True:
            choice = input(f"  {spk}: ").strip().lower()
            if choice == "s":
                return {}  # Signal to skip
            elif choice == "a":
                roles[spk] = "agent"
                break
            elif choice == "u":
                roles[spk] = "user"
                break
            elif choice == "n":
                roles[spk] = "narrator"
                break
            else:
                print("    Invalid. Use [a]gent, [u]ser, [n]arrator, or [s]kip")

    return roles


def main(queue_path: str, labels_path: str) -> None:
    """Run labeling CLI."""
    items = load_queue(queue_path)
    labeled_calls = get_labeled_calls(labels_path)

    # Filter to unlabeled
    to_label = [i for i in items if i["call_id"] not in labeled_calls]
    print(f"Loaded {len(items)} items, {len(to_label)} remaining to label")
    print(f"Roles: agent, user, narrator")
    print()

    i = 0
    history: List[int] = []

    while i < len(to_label):
        item = to_label[i]
        speakers = get_speakers_from_item(item)

        if not speakers:
            print(f"Skipping {item['call_id']} - no speakers detected")
            i += 1
            continue

        display_call(item, i, len(to_label))
        print(f"Speakers detected: {', '.join(speakers)}")

        roles = label_speakers(speakers)

        if not roles:
            # User chose to skip
            print("Skipped.")
            i += 1
            continue

        # Validate: must have at least one agent and one user
        role_counts = {r: sum(1 for v in roles.values() if v == r) for r in ROLES}

        if role_counts["agent"] == 0:
            print("  WARNING: No agent labeled. Are you sure? [y/n]")
            if input("  ").strip().lower() != "y":
                continue

        if role_counts["user"] == 0:
            print("  WARNING: No user labeled. Are you sure? [y/n]")
            if input("  ").strip().lower() != "y":
                continue

        # Save
        save_labels(labels_path, item.get("video_id", ""), item["call_id"], roles)
        print(f"  Saved: {roles}")
        history.append(i)
        i += 1

    labeled_total = len(labeled_calls) + len(history)
    print(f"\nLabeling complete. {labeled_total} calls labeled.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python label_roles_v2.py <queue.jsonl> <speaker_labels.csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
