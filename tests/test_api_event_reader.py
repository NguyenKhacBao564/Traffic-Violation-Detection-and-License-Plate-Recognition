import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.app import app
from src.api.event_reader import list_events, read_event, update_review


def make_event(events_dir: Path, event_id: str = "EVT_0001", plate_text: str = "京A12345") -> Path:
    event_dir = events_dir / event_id
    event_dir.mkdir(parents=True)
    (event_dir / "frame.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (event_dir / "plate.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    payload = {
        "event_id": event_id,
        "track_id": 7,
        "frame_idx": 123,
        "state": "confirmed",
        "plate_text": plate_text,
        "plate_confidence": 0.82,
        "evidence_image_path": str(event_dir / "frame.jpg"),
        "plate_crop_path": str(event_dir / "plate.jpg"),
        "evidence_clip_path": None,
        "camera_id": "CAM_01",
        "violation_type": "red_light_crossing",
        "light_state": "red",
    }
    (event_dir / "event.json").write_text(json.dumps(payload), encoding="utf-8")
    return event_dir


def test_event_reader_lists_events_and_review_status(tmp_path):
    make_event(tmp_path, "EVT_0001", "京A12345")
    make_event(tmp_path, "EVT_0002", "沪B54321")

    event = read_event(tmp_path, "EVT_0001")
    assert event["event_id"] == "EVT_0001"
    assert event["plate_text"] == "京A12345"
    assert event["review_status"] == "pending"
    assert event["frame_path"].endswith("frame.jpg")

    reviewed = update_review(
        tmp_path,
        "EVT_0001",
        review_status="confirmed",
        review_note="Clear red-light crossing.",
    )
    assert reviewed["review_status"] == "confirmed"
    assert reviewed["review_note"] == "Clear red-light crossing."

    filtered = list_events(tmp_path, review_status="confirmed")
    assert [item["event_id"] for item in filtered] == ["EVT_0001"]

    plate_filtered = list_events(tmp_path, plate_text="沪B")
    assert [item["event_id"] for item in plate_filtered] == ["EVT_0002"]


def test_api_lists_events_and_updates_review(tmp_path):
    make_event(tmp_path, "EVT_0001", "京A12345")
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    listed = client.get("/api/v1/events", params={"events_dir": str(tmp_path)})
    assert listed.status_code == 200
    body = listed.json()
    assert body["count"] == 1
    assert body["events"][0]["event_id"] == "EVT_0001"

    patched = client.patch(
        "/api/v1/events/EVT_0001/review",
        params={"events_dir": str(tmp_path)},
        json={"review_status": "rejected", "review_note": "Not a violation."},
    )
    assert patched.status_code == 200
    assert patched.json()["review_status"] == "rejected"

    filtered = client.get(
        "/api/v1/events",
        params={"events_dir": str(tmp_path), "review_status": "rejected"},
    )
    assert filtered.status_code == 200
    assert filtered.json()["count"] == 1


def test_api_rejects_invalid_review_status(tmp_path):
    make_event(tmp_path, "EVT_0001", "京A12345")
    client = TestClient(app)

    response = client.patch(
        "/api/v1/events/EVT_0001/review",
        params={"events_dir": str(tmp_path)},
        json={"review_status": "invalid"},
    )

    assert response.status_code == 400
