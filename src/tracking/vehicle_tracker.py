"""Vehicle tracker using ByteTrack with a small IoU fallback for MVP debugging."""
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
    """Wrapper around ByteTrack, falling back to IoU matching if ByteTrack is absent."""

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        fps: float = 30,
        match_thresh: float = 0.8,
        min_box_area: int = 1500,
    ) -> None:
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_box_area = min_box_area
        self.frame_id = 0
        self.backend = "bytetrack" if BYTETracker is not None else "iou"
        self._next_track_id = 1
        self._active_tracks: dict[int, TrackedVehicle] = {}
        self._missed_frames: dict[int, int] = {}

        if BYTETracker is not None:
            self.tracker = BYTETracker(
                track_thresh=track_thresh,
                track_buffer=track_buffer,
                fps=fps,
                match_thresh=match_thresh,
                min_box_area=min_box_area,
            )
        else:
            self.tracker = None
            print("[WARN] ByteTrack not installed; using simple IoU tracker fallback.")

    def update(
        self, detections: list[DetectedVehicle]
    ) -> list[TrackedVehicle]:
        """Update tracker with current frame detections. Returns tracked vehicles."""
        if self.backend == "iou":
            return self._update_iou(detections)

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

    def _update_iou(self, detections: list[DetectedVehicle]) -> list[TrackedVehicle]:
        """Simple greedy IoU tracker used only as a baseline fallback."""
        self.frame_id += 1

        detections = [
            d for d in detections
            if self._area(d.bbox) >= self.min_box_area and d.confidence >= self.track_thresh
        ]

        unmatched_track_ids = set(self._active_tracks)
        unmatched_det_idxs = set(range(len(detections)))
        matches: list[tuple[int, int]] = []
        # The fallback tracker is intentionally permissive. It is a Layer 2
        # debug aid, not a replacement for ByteTrack.
        iou_threshold = min(max(self.match_thresh, 0.1), 0.25)

        while unmatched_track_ids and unmatched_det_idxs:
            best_pair: tuple[int, int] | None = None
            best_iou = 0.0
            for tid in unmatched_track_ids:
                prev = self._active_tracks[tid]
                for det_idx in unmatched_det_idxs:
                    score = self._iou(prev.bbox, detections[det_idx].bbox)
                    if score > best_iou:
                        best_iou = score
                        best_pair = (tid, det_idx)

            if best_pair is None or best_iou < iou_threshold:
                break

            tid, det_idx = best_pair
            matches.append((tid, det_idx))
            unmatched_track_ids.remove(tid)
            unmatched_det_idxs.remove(det_idx)

        updated: dict[int, TrackedVehicle] = {}

        for tid, det_idx in matches:
            det = detections[det_idx]
            prev = self._active_tracks[tid]
            prev_cx, prev_cy = self.bottom_center(prev.bbox)
            curr_cx, curr_cy = self.bottom_center(det.bbox)
            updated[tid] = TrackedVehicle(
                track_id=tid,
                bbox=det.bbox,
                confidence=det.confidence,
                class_name=det.class_name,
                frame_id=self.frame_id,
                vel_x=float(curr_cx - prev_cx),
                vel_y=float(curr_cy - prev_cy),
            )
            self._missed_frames[tid] = 0

        for det_idx in unmatched_det_idxs:
            det = detections[det_idx]
            tid = self._next_track_id
            self._next_track_id += 1
            updated[tid] = TrackedVehicle(
                track_id=tid,
                bbox=det.bbox,
                confidence=det.confidence,
                class_name=det.class_name,
                frame_id=self.frame_id,
            )
            self._missed_frames[tid] = 0

        for tid in unmatched_track_ids:
            self._missed_frames[tid] = self._missed_frames.get(tid, 0) + 1
            if self._missed_frames[tid] <= self.track_buffer:
                updated[tid] = self._active_tracks[tid]

        self._active_tracks = updated
        return list(updated.values())

    @staticmethod
    def bottom_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
        """Return the bottom-center point of a bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, y2)

    @staticmethod
    def _area(bbox: tuple[int, int, int, int]) -> int:
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @staticmethod
    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = VehicleTracker._area(a) + VehicleTracker._area(b) - inter
        return inter / union if union > 0 else 0.0

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
