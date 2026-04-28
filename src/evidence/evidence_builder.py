"""Evidence builder — save frame images, plate crops, and video clips."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.violation.event_state import ViolationEvent


@dataclass
class EvidenceConfig:
    """Configuration for evidence generation."""
    clip_duration_sec: float = 5.0
    clip_fps: int = 15
    save_plate_crop: bool = True


class EvidenceBuilder:
    """Build and save evidence for violation events."""

    def __init__(
        self,
        output_dir: str = "outputs/events",
        config: EvidenceConfig | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.config = config or EvidenceConfig()

    def save_event(
        self,
        event: ViolationEvent,
        full_frame: np.ndarray,
        plate_crop: Optional[np.ndarray],
    ) -> None:
        """
        Save evidence for a violation event.

        Creates: outputs/events/EVT_XXXX/
          - frame.jpg     ← full frame
          - plate.jpg     ← cropped plate (if available)
          - event.json    ← metadata
        """
        event_dir = self.output_dir / event.event_id
        event_dir.mkdir(parents=True, exist_ok=True)

        # Save full frame
        frame_path = event_dir / "frame.jpg"
        cv2.imwrite(str(frame_path), full_frame)
        event.evidence_image_path = str(frame_path)

        # Save plate crop
        if plate_crop is not None and self.config.save_plate_crop:
            plate_path = event_dir / "plate.jpg"
            cv2.imwrite(str(plate_path), plate_crop)
            event.plate_crop_path = str(plate_path)

        # Save metadata
        meta = event.to_dict()
        meta_path = event_dir / "event.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def append_clip_frames(
        self,
        event_id: str,
        frames: list[np.ndarray],
        fps: float,
    ) -> Optional[str]:
        """
        Append frames to a clip for a violation event.
        Call multiple times, then finalize_clip() to close.

        Returns path to the clip file.
        """
        if not frames:
            return None

        clip_dir = self.output_dir / event_id
        clip_dir.mkdir(parents=True, exist_ok=True)
        clip_path = clip_dir / "clip.mp4"

        if not (clip_dir / ".clip_init").exists():
            # First call — init writer
            h, w = frames[0].shape[:2]
            writer = cv2.VideoWriter(
                str(clip_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                min(fps, 30),
                (w, h),
            )
            (clip_dir / ".clip_init").write_text(str(writer))
        else:
            writer_path = (clip_dir / ".clip_init").read_text()
            writer = cv2.VideoWriter.__new__(cv2.VideoWriter)
            writer.open(
                str(clip_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                min(fps, 30),
                frames[0].shape[2],  # width  # type: ignore
                frames[0].shape[:2][::-1],  # height  # type: ignore
            )

        for frame in frames:
            writer.write(frame)
        writer.release()

        return str(clip_path)

    def save_clip(
        self,
        event_id: str,
        frames: list[np.ndarray],
        fps: float,
    ) -> Optional[str]:
        """Save a short evidence clip for one event."""
        if not frames:
            return None

        event_dir = self.output_dir / event_id
        event_dir.mkdir(parents=True, exist_ok=True)
        clip_path = event_dir / "clip.mp4"

        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(clip_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            min(fps, self.config.clip_fps),
            (w, h),
        )
        for frame in frames:
            writer.write(frame)
        writer.release()
        return str(clip_path)

    @staticmethod
    def save_debug_frame(
        output_dir: str, frame_idx: int, frame: np.ndarray
    ) -> None:
        """Save a single debug frame."""
        dbg_dir = Path(output_dir)
        dbg_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dbg_dir / f"frame_{frame_idx:06d}.jpg"), frame)
