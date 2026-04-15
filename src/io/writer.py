"""Video writer — write frames to MP4 with bounding box overlays."""
from __future__ import annotations

import cv2
from pathlib import Path
from typing import Optional


class VideoWriter:
    """Write frames to a video file, optionally with overlay annotations."""

    def __init__(
        self,
        output_path: str,
        fps: float,
        frame_size: tuple[int, int],
        codec: str = "mp4v",
    ) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.frame_size = frame_size  # (width, height)

        self._writer: Optional[cv2.VideoWriter] = cv2.VideoWriter(
            str(self.output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            frame_size,
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot open VideoWriter for: {self.output_path}")

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def write(self, frame) -> None:
        """Write a single frame."""
        if self._writer is None:
            raise RuntimeError("VideoWriter is closed")
        self._writer.write(frame)

    def close(self) -> None:
        """Close the writer and finalize the file."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
