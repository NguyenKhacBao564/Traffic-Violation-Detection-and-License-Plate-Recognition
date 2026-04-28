"""Tests for violation/violation_engine.py."""
import numpy as np

from src.geometry.line_crossing import StopLineChecker
from src.traffic_light.light_state import LightColor, LightState
from src.tracking.vehicle_tracker import TrackedVehicle
from src.violation.event_state import EventState
from src.violation.violation_engine import ViolationEngine, ViolationEngineConfig


class FakeLightEstimator:
    def estimate(self, frame, frame_idx):
        return LightState(color=LightColor.RED, confidence=1.0, frame_idx=frame_idx)


def test_backward_crossing_creates_pending_event():
    """Violation engine should respect backward scene direction."""
    checker = StopLineChecker(
        stop_line=[[500, 600], [1000, 600]],
        direction="backward",
    )
    engine = ViolationEngine(
        light_estimator=FakeLightEstimator(),
        stop_line_checker=checker,
        tracker=None,
        config=ViolationEngineConfig(min_tracking_frames=2),
    )
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    engine.process_frame(
        frame,
        [TrackedVehicle(1, (550, 560, 650, 650), 0.9, "car", 1)],
        frame_idx=1,
    )
    events = engine.process_frame(
        frame,
        [TrackedVehicle(1, (550, 460, 650, 550), 0.9, "car", 2)],
        frame_idx=2,
    )

    assert len(events) == 1
    assert events[0].state == EventState.PENDING
    assert events[0].track_id == 1
