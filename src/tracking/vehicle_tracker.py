"""Vehicle tracker using ByteTrack."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from bytetrack.byte_track import BYTETracker
except ImportError:
    BYTETracker = None  # type: ignore

from src.detection.vehicle_detector import DetectedVehicle


@dataclass
class TrackedVehicle:
    """Vehicle with assigned track ID."""
    track_id: int
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_name: str
    frame_id: int
    vel_x: float = 0.0
    vel_y: float = 0.0


@dataclass
class TrackingState:
    """State for managing all active tracks across frames."""
    active_tracks: dict[int, TrackedVehicle] = field(default_factory=dict)
    all_tracks: list[TrackedVehicle] = field(default_factory=list)


class VehicleTracker:
    """Wrapper around ByteTrack for multi-object tracking."""

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        fps: float = 30,
        match_thresh: float = 0.8,
        min_box_area: int = 1500,
    ) -> None:
        if BYTETracker is None:
            raise ImportError(
                "ByteTrack not installed. Run: "
                "git clone https://github.com/ifzhang/ByteTrack.git && "
                "cd ByteTrack && pip install -r requirements.txt"
            )
        self.tracker = BYTETracker(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            fps=fps,
            match_thresh=match_thresh,
            min_box_area=min_box_area,
        )
        self.frame_id = 0

    def update(
        self, detections: list[DetectedVehicle]
    ) -> list[TrackedVehicle]:
        """Update tracker with current frame detections. Returns tracked vehicles."""
        # Convert to ByteTrack format: [x1, y1, x2, y2, score, class_id]
        dets = np.array(
            [
                [*d.bbox, d.confidence, float(d.class_id)]
                for d in detections
            ],
            dtype=np.float64,
        )
        if dets.size == 0:
            dets = np.empty((0, 6), dtype=np.float64)

        tracks = self.tracker.update(dets, None, None)
        self.frame_id += 1

        tracked = []
        for track in tracks:
            x1, y1, x2, y2, score, cls_id, track_id = track
            cls_name = detections[0].class_name if detections else "car"
            for d in detections:
                if d.bbox == (int(x1), int(y1), int(x2), int(y2)):
                    cls_name = d.class_name
                    break
            tracked.append(
                TrackedVehicle(
                    track_id=int(track_id),
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=float(score),
                    class_name=cls_name,
                    frame_id=self.frame_id,
                )
            )
        return tracked

    @staticmethod
    def bottom_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
        """Return the bottom-center point of a bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, y2)

    @staticmethod
    def draw_tracks(
        frame, tracked: list[TrackedVehicle], next_event_ids: Optional[set] = None
    ) -> None:
        """Draw track IDs and bboxes on frame (in-place)."""
        import cv2
        for t in tracked:
            x1, y1, x2, y2 = t.bbox
            color = (0, 0, 255) if (next_event_ids and t.track_id in next_event_ids) else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cx, cy = VehicleTracker.bottom_center(t.bbox)
            cv2.putText(
                frame, f"ID{t.track_id}", (cx - 20, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )
