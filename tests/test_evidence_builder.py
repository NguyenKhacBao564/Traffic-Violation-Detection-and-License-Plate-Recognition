"""Tests for evidence/evidence_builder.py."""
import json

import numpy as np

from src.evidence.evidence_builder import EvidenceBuilder
from src.violation.event_state import EventState, ViolationEvent


def test_save_event_serializes_json_schema(tmp_path):
    """Event evidence JSON should be loadable and keep state as a string."""
    event = ViolationEvent(
        event_id="EVT_0001",
        track_id=7,
        frame_idx=123,
        state=EventState.PENDING,
        light_state="red",
        crossing_direction="backward",
    )
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    plate = np.zeros((5, 10, 3), dtype=np.uint8)

    builder = EvidenceBuilder(output_dir=str(tmp_path))
    builder.save_event(event, frame, plate)

    event_json = tmp_path / "EVT_0001" / "event.json"
    data = json.loads(event_json.read_text(encoding="utf-8"))

    assert data["state"] == "pending"
    assert data["evidence_image_path"].endswith("frame.jpg")
    assert data["plate_crop_path"].endswith("plate.jpg")
    assert data["evidence_image_path"] != data["plate_crop_path"]
    assert (tmp_path / "EVT_0001" / "frame.jpg").exists()
    assert (tmp_path / "EVT_0001" / "plate.jpg").exists()


def test_save_clip_writes_mp4(tmp_path):
    builder = EvidenceBuilder(output_dir=str(tmp_path))
    frames = [np.zeros((20, 30, 3), dtype=np.uint8) for _ in range(3)]

    path = builder.save_clip("EVT_0002", frames, fps=10)

    assert path is not None
    assert (tmp_path / "EVT_0002" / "clip.mp4").exists()
