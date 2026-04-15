"""Plate detector using YOLO."""
from __future__ import annotations

import numpy as np
from ultralytics import YOLO


class PlateDetector:
    """YOLO-based license plate detector."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.4,
    ) -> None:
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Detect plate bounding boxes in a frame.

        Returns:
            List of (x1, y1, x2, y2) bounding boxes
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        bboxes = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bboxes.append((x1, y1, x2, y2))
        return bboxes

    @staticmethod
    def crop_plate(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        """Crop plate region from frame."""
        x1, y1, x2, y2 = bbox
        return frame[y1:y2, x1:x2]
