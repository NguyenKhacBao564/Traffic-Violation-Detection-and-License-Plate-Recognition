"""Tests for violation/event_state.py"""
from src.violation.event_state import EventStateManager, EventState


def test_create_pending_event():
    """Should create a pending event with auto-increment ID."""
    mgr = EventStateManager()
    ev = mgr.create_pending(track_id=5, frame_idx=100)
    assert ev.event_id == "EVT_0001"
    assert ev.track_id == 5
    assert ev.state == EventState.PENDING
    assert 5 in mgr.pending_events


def test_confirm_pending_event():
    """Confirming should move event from pending to confirmed."""
    mgr = EventStateManager()
    mgr.create_pending(track_id=5, frame_idx=100)
    confirmed = mgr.confirm(5, plate_text="51H-12345", plate_confidence=0.92)
    assert confirmed is not None
    assert confirmed.state == EventState.CONFIRMED
    assert confirmed.plate_text == "51H-12345"
    assert 5 not in mgr.pending_events
    assert confirmed in mgr.confirmed_events


def test_dismiss_pending_event():
    """Dismiss should move event to dismissed."""
    mgr = EventStateManager()
    mgr.create_pending(track_id=5, frame_idx=100)
    dismissed = mgr.dismiss(5)
    assert dismissed is not None
    assert dismissed.state == EventState.DISMISSED
    assert 5 not in mgr.pending_events


def test_confirm_nonexistent_returns_none():
    """Confirming a non-existent track_id should return None."""
    mgr = EventStateManager()
    assert mgr.confirm(999) is None
