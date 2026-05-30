"""Read and update local violation event evidence folders."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ALLOWED_REVIEW_STATUSES = {"pending", "confirmed", "rejected", "uncertain"}
EVENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class EventReadError(ValueError):
    """Raised when an event folder cannot be read safely."""


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def validate_event_id(event_id: str) -> None:
    if not EVENT_ID_PATTERN.match(event_id):
        raise EventReadError(f"Invalid event_id: {event_id}")


def resolve_events_dir(events_dir: str | Path) -> Path:
    path = Path(events_dir)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _event_dir(events_dir: Path, event_id: str) -> Path:
    validate_event_id(event_id)
    return events_dir / event_id


def _path_from_event(event_dir: Path, raw: dict[str, Any], key: str, fallback_name: str) -> Path | None:
    raw_value = raw.get(key)
    candidates: list[Path] = []
    if raw_value:
        raw_path = Path(raw_value)
        candidates.append(raw_path if raw_path.is_absolute() else ROOT / raw_path)
        candidates.append(event_dir / raw_path.name)
    candidates.append(event_dir / fallback_name)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_review(event_dir: Path) -> dict[str, Any]:
    review_path = event_dir / "review.json"
    if not review_path.exists():
        return {"review_status": "pending", "review_note": None, "review_updated_at": None}
    review = _read_json(review_path)
    return {
        "review_status": review.get("review_status", "pending"),
        "review_note": review.get("review_note"),
        "review_updated_at": review.get("review_updated_at"),
    }


def read_event(events_dir: str | Path, event_id: str) -> dict[str, Any] | None:
    """Read one event folder and return review-ready metadata."""
    root = resolve_events_dir(events_dir)
    event_dir = _event_dir(root, event_id)
    event_json = event_dir / "event.json"
    if not event_json.exists():
        return None

    raw = _read_json(event_json)
    frame_path = _path_from_event(event_dir, raw, "evidence_image_path", "frame.jpg")
    plate_path = _path_from_event(event_dir, raw, "plate_crop_path", "plate.jpg")
    clip_path = _path_from_event(event_dir, raw, "evidence_clip_path", "clip.mp4")
    review = _load_review(event_dir)

    return {
        "event_id": raw.get("event_id", event_id),
        "timestamp": raw.get("timestamp"),
        "frame_idx": raw.get("frame_idx"),
        "violation_type": raw.get("violation_type"),
        "state": raw.get("state"),
        "track_id": raw.get("track_id"),
        "camera_id": raw.get("camera_id"),
        "vehicle_type": raw.get("vehicle_type"),
        "plate_text": raw.get("plate_text"),
        "ocr_confidence": raw.get("plate_confidence"),
        "ocr_backend": raw.get("ocr_backend"),
        "light_state": raw.get("light_state"),
        "light_confidence": raw.get("light_confidence"),
        "crossing_direction": raw.get("crossing_direction"),
        "frame_path": repo_relative(frame_path) if frame_path else None,
        "plate_path": repo_relative(plate_path) if plate_path else None,
        "clip_path": repo_relative(clip_path) if clip_path else None,
        "review_status": review["review_status"],
        "review_note": review["review_note"],
        "review_updated_at": review["review_updated_at"],
        "event_json_path": repo_relative(event_json),
        "raw": raw,
    }


def list_events(
    events_dir: str | Path,
    *,
    limit: int = 100,
    plate_text: str | None = None,
    review_status: str | None = None,
) -> list[dict[str, Any]]:
    """List event folders, optionally filtering by plate text or review status."""
    root = resolve_events_dir(events_dir)
    if not root.exists():
        return []

    events: list[dict[str, Any]] = []
    for event_json in sorted(root.glob("*/event.json")):
        event_id = event_json.parent.name
        event = read_event(root, event_id)
        if not event:
            continue
        if plate_text and plate_text.lower() not in str(event.get("plate_text") or "").lower():
            continue
        if review_status and event.get("review_status") != review_status:
            continue
        events.append(event)

    events.sort(key=lambda item: (item.get("frame_idx") is None, item.get("frame_idx") or 0, item["event_id"]))
    return events[: max(0, limit)]


def media_path(events_dir: str | Path, event_id: str, media_type: str) -> Path | None:
    """Return an existing frame/plate/clip path for an event."""
    if media_type not in {"frame", "plate", "clip"}:
        raise EventReadError(f"Unsupported media_type: {media_type}")

    root = resolve_events_dir(events_dir)
    event_dir = _event_dir(root, event_id)
    event_json = event_dir / "event.json"
    if event_json.exists():
        raw = _read_json(event_json)
    else:
        raw = {}

    if media_type == "frame":
        return _path_from_event(event_dir, raw, "evidence_image_path", "frame.jpg")
    if media_type == "plate":
        return _path_from_event(event_dir, raw, "plate_crop_path", "plate.jpg")
    return _path_from_event(event_dir, raw, "evidence_clip_path", "clip.mp4")


def update_review(
    events_dir: str | Path,
    event_id: str,
    *,
    review_status: str,
    review_note: str | None = None,
) -> dict[str, Any]:
    """Write a transparent review.json beside event.json and return the event."""
    if review_status not in ALLOWED_REVIEW_STATUSES:
        allowed = ", ".join(sorted(ALLOWED_REVIEW_STATUSES))
        raise EventReadError(f"review_status must be one of: {allowed}")

    root = resolve_events_dir(events_dir)
    event_dir = _event_dir(root, event_id)
    if not (event_dir / "event.json").exists():
        raise FileNotFoundError(event_id)

    payload = {
        "event_id": event_id,
        "review_status": review_status,
        "review_note": review_note,
        "review_updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(event_dir / "review.json", payload)
    event = read_event(root, event_id)
    if event is None:
        raise FileNotFoundError(event_id)
    return event

