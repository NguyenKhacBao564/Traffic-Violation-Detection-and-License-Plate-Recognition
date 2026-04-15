"""Video reader — wrapper around cv2.VideoCapture."""
from __future__ import annotations

import cv2
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional


@dataclass
class VideoInfo:
    """Metadata about the input video."""
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float


class VideoReader:
    """Efficient video frame reader."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {self.path}")
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.path}")

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *args) -> None:
        self.cap.release()

    @property
    def info(self) -> VideoInfo:
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        fc = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return VideoInfo(
            path=str(self.path),
            width=w,
            height=h,
            fps=fps,
            frame_count=fc,
            duration_sec=fc / fps if fps > 0 else 0.0,
        )

    def read(self) -> tuple[bool, Optional[object]]:
        """Read next frame. Returns (success, frame)."""
        ret, frame = self.cap.read()
        return ret, frame

    def iter_frames(
        self, skip_frames: int = 0
    ) -> Generator[tuple[int, object], None, None]:
        """Yield (frame_idx, frame) generator. Optionally skip every N frames."""
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Seek forward by skip_frames if needed
            for _ in range(skip_frames):
                self.cap.grab()

            yield frame_idx, frame
            frame_idx += 1

    def seek(self, frame_idx: int) -> bool:
        """Seek to specific frame index. Returns True if successful."""
        return self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    def reset(self) -> None:
        """Reset to first frame."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
