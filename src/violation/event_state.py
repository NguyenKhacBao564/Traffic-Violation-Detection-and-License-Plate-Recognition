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
    evidence_image_path: Optional[str] = None
    evidence_clip_path: Optional[str] = None
    camera_id: str = "CAM_01"
    violation_type: str = "red_light_crossing"
    vehicle_type: str = "car"
    timestamp: Optional[str] = None
    notes: str = ""


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
        self, track_id: int, frame_idx: int, camera_id: str = "CAM_01"
    ) -> ViolationEvent:
        """Create a new pending violation event."""
        event = ViolationEvent(
            event_id=self.new_event_id(),
            track_id=track_id,
            frame_idx=frame_idx,
            state=EventState.PENDING,
            camera_id=camera_id,
        )
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
