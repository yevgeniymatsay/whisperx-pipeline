# scripts/whisperx_pipeline/turn_builder.py
"""Build speaker turns from word stream."""
from dataclasses import dataclass
from typing import List


@dataclass
class Turn:
    turn_id: int
    spk: str
    text: str
    t0_abs: float
    t1_abs: float
    word_span_start: int  # inclusive
    word_span_end: int    # exclusive (half-open)


class TurnBuilder:
    """Build turns by grouping consecutive same-speaker words."""

    def __init__(self, max_gap_s: float = 0.3):
        """
        Args:
            max_gap_s: If same speaker resumes within this gap, merge into same turn.
                       Prevents overly-aggressive turn splitting from ASR segmentation weirdness.
        """
        self.max_gap_s = max_gap_s

    def build_turns(self, words: List[dict]) -> List[Turn]:
        """Build turns from word list with gap-based merging."""
        if not words:
            return []

        turns = []
        current_spk = words[0]["spk"]
        current_texts = [words[0]["text"]]
        current_start = words[0]["t0_abs"]
        current_end = words[0]["t1_abs"]
        span_start = 0

        for i, word in enumerate(words[1:], start=1):
            same_speaker = word["spk"] == current_spk
            gap = word["t0_abs"] - current_end

            # Merge if same speaker AND gap is small enough
            # Large gaps (even same speaker) indicate diarization artifacts or long pauses
            if same_speaker and gap <= self.max_gap_s:
                # Continue current turn
                current_texts.append(word["text"])
                current_end = word["t1_abs"]
            else:
                # Save current turn
                turns.append(Turn(
                    turn_id=len(turns),
                    spk=current_spk,
                    text=" ".join(current_texts),
                    t0_abs=current_start,
                    t1_abs=current_end,
                    word_span_start=span_start,
                    word_span_end=i
                ))

                # Start new turn
                current_spk = word["spk"]
                current_texts = [word["text"]]
                current_start = word["t0_abs"]
                current_end = word["t1_abs"]
                span_start = i

        # Add final turn
        turns.append(Turn(
            turn_id=len(turns),
            spk=current_spk,
            text=" ".join(current_texts),
            t0_abs=current_start,
            t1_abs=current_end,
            word_span_start=span_start,
            word_span_end=len(words)
        ))

        return turns
