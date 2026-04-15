"""Traffic light state estimator via color histogram in ROI."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np


class LightColor(Enum):
    """Traffic light color."""
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    UNKNOWN = "unknown"


@dataclass
class LightState:
    """State of traffic light at a given frame."""
    color: LightColor
    confidence: float
    frame_idx: int


class TrafficLightEstimator:
    """Estimate traffic light state from ROI using HSV color analysis."""

    HSV_RANGES = {
        LightColor.RED: [
            ((0, 100, 100), (10, 255, 255)),   # lower red
            ((160, 100, 100), (180, 255, 255)), # upper red
        ],
        LightColor.YELLOW: [
            ((15, 100, 100), (35, 255, 255)),
        ],
        LightColor.GREEN: [
            ((40, 50, 50), (80, 255, 255)),
        ],
    }
    # Lower thresholds for darker/lower-resolution traffic lights
    HSV_RANGES_SENSITIVE = {
        LightColor.RED: [
            ((0, 80, 80), (10, 255, 255)),
            ((160, 80, 80), (180, 255, 255)),
        ],
        LightColor.YELLOW: [
            ((15, 80, 80), (35, 255, 255)),
        ],
        LightColor.GREEN: [
            ((40, 40, 40), (80, 255, 255)),
        ],
    }

    def __init__(
        self,
        roi: list[list[int]],
        sensitive: bool = True,
    ) -> None:
        self.roi = np.array(roi, dtype=np.int32)
        self.sensitive = sensitive
        self._last_state = LightColor.UNKNOWN

    def estimate(self, frame: np.ndarray, frame_idx: int) -> LightState:
        """Estimate light state from the current frame."""
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [self.roi], 255)
        roi_frame = cv2.bitwise_and(frame, frame, mask=mask)

        hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        ranges = self.HSV_RANGES_SENSITIVE if self.sensitive else self.HSV_RANGES

        scores = {}
        for color, hsv_ranges in ranges.items():
            total_pixels = 0
            for (lower, upper) in hsv_ranges:
                lo = np.array(lower, dtype=np.uint8)
                up = np.array(upper, dtype=np.uint8)
                mask_c = cv2.inRange(hsv, lo, up)
                total_pixels += cv2.countNonZero(mask_c)
            scores[color] = total_pixels

        best_color = LightColor.UNKNOWN
        best_score = 50  # minimum pixel count threshold
        for color, score in scores.items():
            if score > best_score:
                best_score = score
                best_color = color

        if best_color == LightColor.UNKNOWN:
            return LightState(color=self._last_state, confidence=0.0, frame_idx=frame_idx)

        confidence = min(best_score / 500, 1.0)
        self._last_state = best_color
        return LightState(color=best_color, confidence=confidence, frame_idx=frame_idx)

    @staticmethod
    def draw_roi(frame: np.ndarray, roi: list[list[int]], light_state: LightColor) -> None:
        """Draw traffic light ROI with state label."""
        pts = np.array(roi, dtype=np.int32)
        color_map = {
            LightColor.RED: (0, 0, 255),
            LightColor.YELLOW: (0, 255, 255),
            LightColor.GREEN: (0, 255, 0),
            LightColor.UNKNOWN: (128, 128, 128),
        }
        clr = color_map.get(light_state, (128, 128, 128))
        cv2.polylines(frame, [pts], isClosed=True, color=clr, thickness=2)
        x, y = int(np.mean(pts[:, 0])), int(np.min(pts[:, 1])) - 10
        cv2.putText(frame, f"LIGHT: {light_state.value}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, clr, 2)
