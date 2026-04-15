"""Temporal OCR fusion — vote best plate text across multiple frames."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class FusionResult:
    """Result of temporal OCR fusion."""
    final_text: str
    confidence: float
    total_frames: int
    votes: dict[str, int]


@dataclass
class OCRFusion:
    """
    Fuse OCR results across multiple frames to improve accuracy.

    Strategy: collect N reads, then take the most common result.
    Optionally weighted by confidence.
    """
    vote_frames: int = 5
    min_votes: int = 2

    _readings: list[tuple[str, float]] = field(default_factory=list)

    def add_reading(self, text: str, confidence: float) -> None:
        """Add an OCR reading."""
        if text and len(text) >= 3:  # ignore very short/empty readings
            self._readings.append((text, confidence))

    def fuse(self) -> FusionResult:
        """
        Compute the fused result from collected readings.
        Resets readings after fusion.
        """
        if not self._readings:
            return FusionResult(final_text="", confidence=0.0, total_frames=0, votes={})

        # Weighted vote: each reading contributes confidence points
        vote_scores: dict[str, float] = {}
        for text, conf in self._readings:
            vote_scores[text] = vote_scores.get(text, 0.0) + conf

        best_text = max(vote_scores, key=vote_scores.get)  # type: ignore
        best_score = vote_scores[best_text]

        total_conf = sum(c for _, c in self._readings)
        avg_conf = total_conf / len(self._readings) if self._readings else 0.0
        final_conf = (best_score / total_conf) * avg_conf if total_conf > 0 else 0.0

        votes = {t: sum(1 for txt, _ in self._readings if txt == t) for t in vote_scores}

        self._readings.clear()
        return FusionResult(
            final_text=best_text,
            confidence=min(final_conf, 1.0),
            total_frames=len(self._readings) if not self._readings else len(vote_scores),
            votes=votes,
        )

    @staticmethod
    def format_result(result: FusionResult) -> str:
        """Format fusion result as a readable string."""
        if not result.final_text:
            return "No plate detected"
        return (
            f"Plate: {result.final_text} "
            f"(conf: {result.confidence:.2f}, "
            f"votes: {dict(sorted(result.votes.items(), key=lambda x: -x[1])[:3])})"
        )
