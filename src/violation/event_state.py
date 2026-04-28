"""Violation event state machine."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EventState(Enum):
    """State of a violation event."""
    PENDING = "pending"     # stop line crossed while light is red
    CONFIRMED = "confirmed" # OCR confirmed, evidence saved
    DISMISSED = "dismissed"  # track lost before confirmation


@dataclass
class ViolationEvent:
    """A single violation event."""
    event_id: str
    track_id: int
    frame_idx: int
    state: EventState
    plate_text: Optional[str] = None
    plate_confidence: Optional[float] = None
    plate_bbox: Optional[tuple[int, int, int, int]] = None
    plate_detection_confidence: Optional[float] = None
    plate_detection_source: Optional[str] = None
    ocr_backend: Optional[str] = None
    ocr_votes: Optional[dict[str, int]] = None
    ocr_frame_count: int = 0
    evidence_image_path: Optional[str] = None
    plate_crop_path: Optional[str] = None
    evidence_clip_path: Optional[str] = None
    camera_id: str = "CAM_01"
    violation_type: str = "red_light_crossing"
    vehicle_type: str = "car"
    timestamp: Optional[str] = None
    light_state: Optional[str] = None
    light_confidence: Optional[float] = None
    crossing_direction: Optional[str] = None
    point_before: Optional[tuple[int, int]] = None
    point_after: Optional[tuple[int, int]] = None
    vehicle_bbox: Optional[tuple[int, int, int, int]] = None
    notes: str = ""

    def to_dict(self) -> dict:
        """Serialize event to JSON-safe metadata."""
        return {
            "event_id": self.event_id,
            "track_id": self.track_id,
            "frame_idx": self.frame_idx,
            "state": self.state.value,
            "plate_text": self.plate_text,
            "plate_confidence": self.plate_confidence,
            "plate_bbox": list(self.plate_bbox) if self.plate_bbox else None,
            "plate_detection_confidence": self.plate_detection_confidence,
            "plate_detection_source": self.plate_detection_source,
            "ocr_backend": self.ocr_backend,
            "ocr_votes": self.ocr_votes,
            "ocr_frame_count": self.ocr_frame_count,
            "evidence_image_path": self.evidence_image_path,
            "plate_crop_path": self.plate_crop_path,
            "evidence_clip_path": self.evidence_clip_path,
            "camera_id": self.camera_id,
            "violation_type": self.violation_type,
            "vehicle_type": self.vehicle_type,
            "timestamp": self.timestamp,
            "light_state": self.light_state,
            "light_confidence": self.light_confidence,
            "crossing_direction": self.crossing_direction,
            "point_before": list(self.point_before) if self.point_before else None,
            "point_after": list(self.point_after) if self.point_after else None,
            "vehicle_bbox": list(self.vehicle_bbox) if self.vehicle_bbox else None,
            "notes": self.notes,
        }


@dataclass
class EventStateManager:
    """Manages state for all active and historical violation events."""
    pending_events: dict[int, ViolationEvent] = field(default_factory=dict)
    confirmed_events: list[ViolationEvent] = field(default_factory=list)
    dismissed_events: list[ViolationEvent] = field(default_factory=list)
    _event_counter: int = 0

    def new_event_id(self) -> str:
        self._event_counter += 1
        return f"EVT_{self._event_counter:04d}"

    def create_pending(
        self, track_id: int, frame_idx: int, camera_id: str = "CAM_01", **metadata
    ) -> ViolationEvent:
        """Create a new pending violation event."""
        event = ViolationEvent(
            event_id=self.new_event_id(),
            track_id=track_id,
            frame_idx=frame_idx,
            state=EventState.PENDING,
            camera_id=camera_id,
        )
        for key, value in metadata.items():
            if hasattr(event, key):
                setattr(event, key, value)
        self.pending_events[track_id] = event
        return event

    def confirm(self, track_id: int, **updates) -> Optional[ViolationEvent]:
        """Move a pending event to confirmed."""
        if track_id not in self.pending_events:
            return None
        event = self.pending_events.pop(track_id)
        for k, v in updates.items():
            if hasattr(event, k):
                setattr(event, k, v)
        event.state = EventState.CONFIRMED
        self.confirmed_events.append(event)
        return event

    def dismiss(self, track_id: int) -> Optional[ViolationEvent]:
        """Move a pending event to dismissed (track lost)."""
        if track_id not in self.pending_events:
            return None
        event = self.pending_events.pop(track_id)
        event.state = EventState.DISMISSED
        self.dismissed_events.append(event)
        return event

    @property
    def all_events(self) -> list[ViolationEvent]:
        return self.confirmed_events + self.dismissed_events
