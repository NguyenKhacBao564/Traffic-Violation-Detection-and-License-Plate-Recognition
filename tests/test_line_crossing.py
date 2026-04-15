"""Tests for geometry/line_crossing.py"""
from src.geometry.line_crossing import StopLineChecker, CrossingDirection


def test_no_crossing_when_parallel():
    """Vehicle moving parallel to stop line should not trigger crossing."""
    checker = StopLineChecker(
        stop_line=[[500, 600], [1000, 600]],
        direction="forward",
    )
    event = checker.check_crossing(
        track_id=1,
        prev_point=(600, 500),
        curr_point=(700, 510),  # slight movement down
        frame_idx=10,
    )
    assert event is None


def test_forward_crossing_detected():
    """Forward crossing (Y increases) should trigger."""
    checker = StopLineChecker(
        stop_line=[[500, 600], [1000, 600]],
        direction="forward",
    )
    event = checker.check_crossing(
        track_id=1,
        prev_point=(600, 550),  # above line
        curr_point=(600, 650),  # below line
        frame_idx=10,
    )
    assert event is not None
    assert event.track_id == 1
    assert event.direction == CrossingDirection.FORWARD


def test_backward_movement_not_flagged():
    """Backward movement should trigger BACKWARD direction."""
    checker = StopLineChecker(
        stop_line=[[500, 600], [1000, 600]],
        direction="forward",
    )
    event = checker.check_crossing(
        track_id=1,
        prev_point=(600, 650),  # below line
        curr_point=(600, 550),  # above line (going backward)
        frame_idx=10,
    )
    # Direction is BACKWARD because Y decreased
    assert event is not None
    assert event.direction == CrossingDirection.BACKWARD
