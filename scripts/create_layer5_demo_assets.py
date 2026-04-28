#!/usr/bin/env python3
"""Create redacted Layer 5 demo assets from event evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-dir", type=Path, default=Path("outputs/events_layer4"))
    parser.add_argument("--contact", type=Path, default=Path("outputs/reports/layer5_demo_contact_redacted.jpg"))
    parser.add_argument("--video", type=Path, default=Path("outputs/debug_videos/layer5_demo_redacted.mp4"))
    parser.add_argument("--fps", type=float, default=2.0)
    args = parser.parse_args()

    event_rows = []
    video_frames = []
    for event_dir in sorted(args.events_dir.glob("EVT_*")):
        event_path = event_dir / "event.json"
        frame_path = event_dir / "frame.jpg"
        plate_path = event_dir / "plate.jpg"
        if not event_path.exists() or not frame_path.exists():
            continue
        event = json.loads(event_path.read_text(encoding="utf-8"))
        frame = cv2.imread(str(frame_path))
        plate = cv2.imread(str(plate_path)) if plate_path.exists() else None
        if frame is None:
            continue
        redacted = redact_frame(frame, event)
        event_rows.append(make_contact_tile(redacted, plate, event))
        video_frames.extend([cv2.resize(redacted, (960, 720)) for _ in range(2)])

    args.contact.parent.mkdir(parents=True, exist_ok=True)
    args.video.parent.mkdir(parents=True, exist_ok=True)

    if event_rows:
        contact = np.hstack(event_rows)
        cv2.imwrite(str(args.contact), contact)
        write_video(args.video, video_frames, args.fps)
    print(f"Contact: {args.contact}")
    print(f"Video: {args.video}")
    return 0


def redact_frame(frame: np.ndarray, event: dict) -> np.ndarray:
    out = frame.copy()
    for key in ("plate_bbox", "vehicle_bbox"):
        bbox = event.get(key)
        if bbox:
            blur_bbox(out, bbox, heavy=(key == "plate_bbox"))
    return out


def blur_bbox(frame: np.ndarray, bbox: list[int], heavy: bool = True) -> None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    pad = 8 if heavy else 0
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    kernel = 51 if heavy else 25
    if roi.shape[0] < 10 or roi.shape[1] < 10:
        kernel = 15
    if kernel % 2 == 0:
        kernel += 1
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kernel, kernel), 0)


def make_contact_tile(frame: np.ndarray, plate: np.ndarray | None, event: dict) -> np.ndarray:
    frame_small = cv2.resize(frame, (320, 240))
    if plate is None:
        plate_small = np.zeros((80, 320, 3), dtype=np.uint8)
    else:
        plate_small = cv2.resize(plate, (320, 80), interpolation=cv2.INTER_CUBIC)
        plate_small = cv2.GaussianBlur(plate_small, (31, 31), 0)

    label = f"{event['event_id']} {event.get('state')} text_redacted"
    cv2.putText(frame_small, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return np.vstack([frame_small, plate_small])


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


if __name__ == "__main__":
    raise SystemExit(main())

