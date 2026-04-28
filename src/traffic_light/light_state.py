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
            fallback = self._estimate_overexposed_bulb(frame, mask, frame_idx)
            if fallback.color != LightColor.UNKNOWN:
                self._last_state = fallback.color
                return fallback
            return LightState(color=self._last_state, confidence=0.0, frame_idx=frame_idx)

        confidence = min(best_score / 500, 1.0)
        self._last_state = best_color
        return LightState(color=best_color, confidence=confidence, frame_idx=frame_idx)

    def _estimate_overexposed_bulb(
        self, frame: np.ndarray, roi_mask: np.ndarray, frame_idx: int
    ) -> LightState:
        """
        Fallback for CCTV footage where the active lamp is overexposed white.

        Fixed traffic lights usually arrange bulbs vertically: red on top, yellow
        in the middle, green at the bottom. If color thresholding fails, detect
        the brightest blob inside the ROI and infer state from its vertical
        position.
        """
        x, y, w, h = cv2.boundingRect(self.roi)
        if w <= 0 or h <= 0:
            return LightState(LightColor.UNKNOWN, 0.0, frame_idx)

        crop = frame[y:y + h, x:x + w]
        crop_mask = roi_mask[y:y + h, x:x + w]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2]

        bright = cv2.inRange(value, 185, 255)
        bright = cv2.bitwise_and(bright, bright, mask=crop_mask)
        bright = cv2.morphologyEx(
            bright,
            cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
        )

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bright)
        best_idx = None
        best_area = 0
        for idx in range(1, num_labels):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area > best_area:
                best_area = area
                best_idx = idx

        if best_idx is None or best_area < 8:
            return LightState(LightColor.UNKNOWN, 0.0, frame_idx)

        _, cy = centroids[best_idx]
        rel_y = cy / max(h, 1)
        if rel_y < 0.38:
            color = LightColor.RED
        elif rel_y < 0.65:
            color = LightColor.YELLOW
        else:
            color = LightColor.GREEN

        confidence = min(best_area / 120.0, 1.0)
        return LightState(color=color, confidence=confidence, frame_idx=frame_idx)

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
