"""Vehicle detector using YOLO (ultralytics)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class DetectedVehicle:
    """Single vehicle detection."""
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str


class VehicleDetector:
    """YOLO-based vehicle detector. Filters only car class by default."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.4,
        allowed_classes: Optional[list[int]] = None,
    ) -> None:
        # COCO class 2 = car; class 3 = motorcycle; class 5 = bus; class 7 = truck
        self.allowed_classes = allowed_classes or [2, 5, 7]  # car, bus, truck
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = YOLO(model_name)

    def detect(self, frame: np.ndarray) -> list[DetectedVehicle]:
        """Run detection on a single frame. Returns list of vehicles."""
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        vehicles = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id not in self.allowed_classes:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_name = result.names[cls_id]
                vehicles.append(
                    DetectedVehicle(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=cls_name,
                    )
                )
        return vehicles

    @staticmethod
    def draw_boxes(
        frame: np.ndarray, vehicles: list[DetectedVehicle], track_ids: Optional[dict] = None
    ) -> np.ndarray:
        """Draw bounding boxes on frame."""
        for v in vehicles:
            x1, y1, x2, y2 = v.bbox
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{v.class_name} {v.confidence:.2f}"
            if track_ids and v.bbox in track_ids:
                label = f"ID{track_ids[v.bbox]} {label}"
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame
