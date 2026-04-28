"""License plate detector using YOLO with a local heuristic fallback."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class PlateDetection:
    """Single license plate detection."""

    bbox: tuple[int, int, int, int]
    confidence: float
    source: str = "yolo"


class PlateDetector:
    """Detect license plate boxes.

    If a trained plate model is available, YOLO is used. If the model path is
    missing, the class falls back to a simple color/geometry heuristic so Layer 4
    can still save plate crops for evidence.
    """

    def __init__(
        self,
        model_path: str | None = "plate_detector.pt",
        conf_threshold: float = 0.4,
        enable_fallback: bool = True,
    ) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.enable_fallback = enable_fallback
        self.model = None
        self.backend = "heuristic"

        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
            self.backend = "yolo"
        elif model_path and model_path.startswith("yolo") and model_path.endswith(".pt"):
            print(
                "[WARN] Built-in COCO YOLO weights do not include a license-plate class; "
                "using heuristic plate crop fallback."
            )
        elif model_path:
            print(f"[WARN] Plate detector model not found: {model_path}; using heuristic fallback.")

    def detect(
        self,
        frame: np.ndarray,
        vehicle_bbox: tuple[int, int, int, int] | None = None,
    ) -> list[tuple[int, int, int, int]]:
        """Detect plate bounding boxes in a frame."""
        return [d.bbox for d in self.detect_with_scores(frame, vehicle_bbox)]

    def detect_with_scores(
        self,
        frame: np.ndarray,
        vehicle_bbox: tuple[int, int, int, int] | None = None,
    ) -> list[PlateDetection]:
        """Detect plates and keep confidence/source metadata."""
        if self.model is not None:
            detections = self._detect_yolo(frame, vehicle_bbox)
            if detections:
                return detections

        if self.enable_fallback:
            return self._detect_heuristic(frame, vehicle_bbox)
        return []

    def best_detection(
        self,
        frame: np.ndarray,
        vehicle_bbox: tuple[int, int, int, int] | None = None,
    ) -> PlateDetection | None:
        """Return the highest-confidence detection, if any."""
        detections = self.detect_with_scores(frame, vehicle_bbox)
        if not detections:
            return None
        return max(detections, key=lambda d: d.confidence)

    def _detect_yolo(
        self,
        frame: np.ndarray,
        vehicle_bbox: tuple[int, int, int, int] | None = None,
    ) -> list[PlateDetection]:
        search_frame, offset = self._search_frame(frame, vehicle_bbox)
        results = self.model(search_frame, conf=self.conf_threshold, verbose=False)
        detections: list[PlateDetection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                ox, oy = offset
                detections.append(
                    PlateDetection(
                        bbox=self._clamp_bbox((x1 + ox, y1 + oy, x2 + ox, y2 + oy), frame),
                        confidence=conf,
                        source="yolo",
                    )
                )
        return sorted(detections, key=lambda d: d.confidence, reverse=True)

    def _detect_heuristic(
        self,
        frame: np.ndarray,
        vehicle_bbox: tuple[int, int, int, int] | None = None,
    ) -> list[PlateDetection]:
        """Find blue/green plate-like rectangles, then fallback to lower vehicle crop."""
        search_frame, (ox, oy) = self._search_frame(frame, vehicle_bbox)
        if search_frame.size == 0:
            return []

        hsv = cv2.cvtColor(search_frame, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, np.array([90, 45, 35]), np.array([135, 255, 255]))
        green = cv2.inRange(hsv, np.array([35, 35, 35]), np.array([90, 255, 255]))
        mask = cv2.bitwise_or(blue, green)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        detections: list[PlateDetection] = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        search_h, search_w = search_frame.shape[:2]
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue
            area = w * h
            aspect = w / h
            if area < 90 or w < 18 or h < 6 or aspect < 1.5 or aspect > 6.8:
                continue
            if area > search_w * search_h * 0.35:
                continue
            rel_center_y = (y + h / 2) / search_h if search_h else 0.0
            if vehicle_bbox is not None and not 0.42 <= rel_center_y <= 0.96:
                continue
            mask_area = float(cv2.contourArea(contour))
            fill = mask_area / area if area else 0.0
            if fill < 0.18:
                continue
            global_bbox = self._clamp_bbox((x + ox, y + oy, x + w + ox, y + h + oy), frame)
            vertical_bonus = max(0.0, 1.0 - abs(rel_center_y - 0.78) / 0.36) * 0.18
            confidence = min(0.95, 0.20 + fill + min(area / 4500.0, 0.30) + vertical_bonus)
            detections.append(PlateDetection(global_bbox, confidence, "color_heuristic"))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        if detections:
            return detections[:5]

        if vehicle_bbox is None:
            return []

        fallback = self.vehicle_plate_region(frame, vehicle_bbox)
        if fallback is None:
            return []
        return [PlateDetection(fallback, 0.05, "vehicle_lower_crop")]

    @staticmethod
    def crop_plate(
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        padding: float = 0.08,
    ) -> np.ndarray:
        """Crop plate region from frame."""
        x1, y1, x2, y2 = PlateDetector._pad_bbox(bbox, frame, padding)
        return frame[y1:y2, x1:x2]

    @staticmethod
    def vehicle_plate_region(
        frame: np.ndarray,
        vehicle_bbox: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int] | None:
        """Return a conservative lower-middle crop from a vehicle bbox."""
        h_img, w_img = frame.shape[:2]
        x1, y1, x2, y2 = vehicle_bbox
        x1 = max(0, min(w_img - 1, x1))
        x2 = max(0, min(w_img, x2))
        y1 = max(0, min(h_img - 1, y1))
        y2 = max(0, min(h_img, y2))
        width = x2 - x1
        height = y2 - y1
        if width < 20 or height < 20:
            return None
        crop_x1 = int(x1 + width * 0.20)
        crop_x2 = int(x2 - width * 0.20)
        crop_y1 = int(y1 + height * 0.58)
        crop_y2 = int(y2 - height * 0.02)
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return None
        return (crop_x1, crop_y1, crop_x2, crop_y2)

    @staticmethod
    def draw_detections(frame: np.ndarray, detections: list[PlateDetection]) -> None:
        """Draw plate detections in-place."""
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                frame,
                f"plate {det.confidence:.2f}",
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

    @staticmethod
    def _search_frame(
        frame: np.ndarray,
        vehicle_bbox: tuple[int, int, int, int] | None,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        if vehicle_bbox is None:
            return frame, (0, 0)
        x1, y1, x2, y2 = PlateDetector._pad_bbox(vehicle_bbox, frame, padding=0.08)
        return frame[y1:y2, x1:x2], (x1, y1)

    @staticmethod
    def _pad_bbox(
        bbox: tuple[int, int, int, int],
        frame: np.ndarray,
        padding: float,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        pad_x = int(max(1, (x2 - x1) * padding))
        pad_y = int(max(1, (y2 - y1) * padding))
        return PlateDetector._clamp_bbox((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), frame)

    @staticmethod
    def _clamp_bbox(
        bbox: tuple[int, int, int, int],
        frame: np.ndarray,
    ) -> tuple[int, int, int, int]:
        h_img, w_img = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(w_img - 1, int(x1)))
        x2 = max(0, min(w_img, int(x2)))
        y1 = max(0, min(h_img - 1, int(y1)))
        y2 = max(0, min(h_img, int(y2)))
        if x2 <= x1:
            x2 = min(w_img, x1 + 1)
        if y2 <= y1:
            y2 = min(h_img, y1 + 1)
        return x1, y1, x2, y2
