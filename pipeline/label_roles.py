# scripts/whisperx_pipeline/label_roles.py
"""CLI for labeling speaker roles."""
import json
import csv
from pathlib import Path
from datetime import datetime
import sys


def load_queue(queue_path: str) -> list:
    """Load labeling queue."""
    items = []
    with open(queue_path) as f:
        for line in f:
            items.append(json.loads(line))
    return items


def load_existing_labels(labels_path: str) -> set:
    """Load already-labeled call IDs."""
    labeled = set()
    if Path(labels_path).exists():
        with open(labels_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                labeled.add(row["call_id"])
    return labeled


def save_label(labels_path: str, video_id: str, call_id: str, agent_spk: str):
    """Append label to CSV."""
    file_exists = Path(labels_path).exists()
    with open(labels_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["video_id", "call_id", "agent_spk", "labeled_at"])
        writer.writerow([video_id, call_id, agent_spk, datetime.now().isoformat()])


def display_call(item: dict, index: int, total: int):
    """Display call for labeling."""
    print("\n" + "=" * 60)
    print(f"Call: {item['call_id']} ({index + 1}/{total})")
    print("=" * 60)
    print(item.get("preview_text", "No preview available"))
    print("=" * 60)
    print("Who is the AGENT/CALLER?")
    print("  [0] SPEAKER_00    [1] SPEAKER_01    [s] Skip    [b] Back    [q] Quit")


def main(queue_path: str, labels_path: str):
    """Run labeling CLI."""
    items = load_queue(queue_path)
    labeled = load_existing_labels(labels_path)

    # Filter to unlabeled
    to_label = [i for i in items if i["call_id"] not in labeled]
    print(f"Loaded {len(items)} items, {len(to_label)} remaining to label")

    history = []
    i = 0

    while i < len(to_label):
        item = to_label[i]
        display_call(item, i, len(to_label))

        choice = input("> ").strip().lower()

        if choice == "0":
            save_label(labels_path, item["video_id"], item["call_id"], "SPEAKER_00")
            history.append(i)
            i += 1
        elif choice == "1":
            save_label(labels_path, item["video_id"], item["call_id"], "SPEAKER_01")
            history.append(i)
            i += 1
        elif choice == "s":
            i += 1
        elif choice == "b" and history:
            i = history.pop()
        elif choice == "q":
            break
        else:
            print("Invalid choice. Use 0, 1, s, b, or q")

    print(f"\nLabeling complete. {len(labeled) + len(history)} total labeled.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python label_roles.py <queue.jsonl> <labels.csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
