"""Event store — save and load violation events from JSON."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.violation.event_state import ViolationEvent, EventState


class EventStore:
    """Persist and query violation events."""

    def __init__(self, output_dir: str = "outputs/events") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.output_dir / "events_index.json"
        self._load_index()

    def _load_index(self) -> None:
        if self._index_path.exists():
            with open(self._index_path) as f:
                self._index = json.load(f)
        else:
            self._index: dict[str, dict] = {}

    def _save_index(self) -> None:
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    def save(self, event: ViolationEvent) -> None:
        """Save event to its own folder + update index."""
        event_dir = self.output_dir / event.event_id
        event_dir.mkdir(parents=True, exist_ok=True)

        meta_path = event_dir / "event.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self._event_to_dict(event), f, ensure_ascii=False, indent=2)

        self._index[event.event_id] = self._event_to_dict(event)
        self._save_index()

    def load(self, event_id: str) -> Optional[ViolationEvent]:
        """Load an event by ID."""
        if event_id not in self._index:
            event_path = self.output_dir / event_id / "event.json"
            if not event_path.exists():
                return None
            with open(event_path) as f:
                data = json.load(f)
        else:
            data = self._index[event_id]
        return self._dict_to_event(data)

    def all_events(self) -> list[ViolationEvent]:
        """Load all stored events."""
        events = []
        for event_id in self._index:
            ev = self.load(event_id)
            if ev:
                events.append(ev)
        return events

    def confirmed_events(self) -> list[ViolationEvent]:
        return [e for e in self.all_events() if e.state == EventState.CONFIRMED]

    def _event_to_dict(self, event: ViolationEvent) -> dict:
        return {
            "event_id": event.event_id,
            "track_id": event.track_id,
            "frame_idx": event.frame_idx,
            "state": event.state.value,
            "plate_text": event.plate_text,
            "plate_confidence": event.plate_confidence,
            "evidence_image_path": event.evidence_image_path,
            "evidence_clip_path": event.evidence_clip_path,
            "camera_id": event.camera_id,
            "violation_type": event.violation_type,
            "vehicle_type": event.vehicle_type,
            "timestamp": event.timestamp,
        }

    def _dict_to_event(self, data: dict) -> ViolationEvent:
        return ViolationEvent(
            event_id=data["event_id"],
            track_id=data["track_id"],
            frame_idx=data["frame_idx"],
            state=EventState(data["state"]),
            plate_text=data.get("plate_text"),
            plate_confidence=data.get("plate_confidence"),
            evidence_image_path=data.get("evidence_image_path"),
            evidence_clip_path=data.get("evidence_clip_path"),
            camera_id=data.get("camera_id", "CAM_01"),
            violation_type=data.get("violation_type", "red_light_crossing"),
            vehicle_type=data.get("vehicle_type", "car"),
            timestamp=data.get("timestamp"),
        )
