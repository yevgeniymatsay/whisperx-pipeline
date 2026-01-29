# scripts/whisperx_pipeline/call_splitter.py
"""Split merged word stream into individual calls."""
from dataclasses import dataclass
from typing import List, Dict, Any

from .config import SplitCallsConfig
from .lexical_matcher import FuzzyLexicalMatcher


@dataclass
class CallBoundary:
    call_id: str
    start_abs: float
    end_abs: float
    boundary_confidence: float
    start_evidence: Dict[str, Any]
    end_evidence: Dict[str, Any]


class CallSplitter:
    """Split word stream into calls using silence + lexical resets."""

    def __init__(self, config: SplitCallsConfig, video_id: str):
        self.config = config
        self.video_id = video_id
        self.matcher = FuzzyLexicalMatcher(
            patterns=config.greeting_tokens,
            threshold=config.fuzzy_match_threshold
        )

    def find_boundaries(self, words: List[dict]) -> List[CallBoundary]:
        """Find call boundaries in word stream."""
        if not words:
            return []

        # Find potential split points
        split_points = []

        # 1. Silence gaps
        for i in range(1, len(words)):
            gap = words[i]["t0_abs"] - words[i-1]["t1_abs"]
            if gap >= self.config.silence_threshold_s:
                split_points.append({
                    "t_abs": words[i]["t0_abs"],
                    "type": "silence_gap",
                    "gap_s": gap,
                    "confidence": min(1.0, gap / 5.0),
                    "word_index": i
                })

        # 2. Greeting resets (after initial 30s)
        greeting_matches = self.matcher.find_greeting_resets(
            words,
            min_gap_from_start_s=30.0
        )
        for match in greeting_matches:
            # Check if there isn't already a silence split nearby
            is_near_silence = any(
                abs(sp["t_abs"] - match["t_abs"]) < 5.0
                for sp in split_points if sp["type"] == "silence_gap"
            )
            # Check if there isn't already a greeting reset nearby (dedup overlapping matches)
            is_near_greeting = any(
                abs(sp["t_abs"] - match["t_abs"]) < 2.0
                for sp in split_points if sp["type"] == "greeting_reset"
            )
            if not is_near_silence and not is_near_greeting:
                split_points.append({
                    "t_abs": match["t_abs"],
                    "type": "greeting_reset",
                    "phrase_debug": " ".join(match["pattern"]),
                    "confidence": match["confidence"],
                    "word_index": match["word_index"]
                })

        # Sort split points by time
        split_points.sort(key=lambda x: x["t_abs"])

        # Build call boundaries
        boundaries = []
        current_start = words[0]["t0_abs"]
        current_start_evidence: Dict[str, Any] = {"type": "start_of_audio"}

        for sp in split_points:
            # End current call
            end_t = words[sp["word_index"] - 1]["t1_abs"] if sp["word_index"] > 0 else sp["t_abs"]
            boundaries.append(self._make_boundary(
                start=current_start,
                end=end_t,
                start_evidence=current_start_evidence,
                end_evidence=sp,
                confidence=sp["confidence"]
            ))

            # Start new call
            current_start = sp["t_abs"]
            current_start_evidence = sp

        # Add final call
        if words:
            boundaries.append(self._make_boundary(
                start=current_start,
                end=words[-1]["t1_abs"],
                start_evidence=current_start_evidence,
                end_evidence={"type": "end_of_audio"},
                confidence=1.0
            ))

        return boundaries

    def _make_boundary(
        self,
        start: float,
        end: float,
        start_evidence: dict,
        end_evidence: dict,
        confidence: float
    ) -> CallBoundary:
        """Create CallBoundary with computed ID."""
        start_ms = int(start * 1000)
        end_ms = int(end * 1000)
        return CallBoundary(
            call_id=f"{self.video_id}_{start_ms}_{end_ms}",
            start_abs=start,
            end_abs=end,
            boundary_confidence=confidence,
            start_evidence=start_evidence,
            end_evidence=end_evidence
        )

    def split_words(
        self,
        words: List[dict],
        boundaries: List[CallBoundary]
    ) -> List[List[dict]]:
        """Split word list according to boundaries."""
        calls = []
        for boundary in boundaries:
            call_words = [
                w for w in words
                if boundary.start_abs <= w["t0_abs"] < boundary.end_abs
            ]
            calls.append(call_words)
        return calls
