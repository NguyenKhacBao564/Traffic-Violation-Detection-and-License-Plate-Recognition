"""Stop line crossing logic using computational geometry."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from shapely.geometry import LineString, Point


class CrossingDirection(Enum):
    """Direction a vehicle crosses the stop line."""
    FORWARD = "forward"   # going forward (into intersection)
    BACKWARD = "backward" # going backward


@dataclass
class LineCrossingEvent:
    """Record of a stop-line crossing event."""
    track_id: int
    frame_idx: int
    point_before: tuple[int, int]
    point_after: tuple[int, int]
    direction: CrossingDirection


class StopLineChecker:
    """Detect when a vehicle crosses the stop line."""

    def __init__(
        self,
        stop_line: list[list[int]],
        direction: str = "forward",
        line_thickness: int = 10,
    ) -> None:
        self.stop_line = LineString(stop_line)
        self.direction = direction
        self.line_thickness = line_thickness

    def check_crossing(
        self,
        track_id: int,
        prev_point: tuple[int, int],
        curr_point: tuple[int, int],
        frame_idx: int,
    ) -> LineCrossingEvent | None:
        """
        Check if vehicle track crosses the stop line between two points.

        Args:
            track_id: vehicle track ID
            prev_point: bottom-center of bbox in previous frame
            curr_point: bottom-center of bbox in current frame
            frame_idx: current frame index

        Returns:
            LineCrossingEvent if crossed, None otherwise
        """
        p1 = Point(prev_point)
        p2 = Point(curr_point)
        movement_line = LineString([p1, p2])

        if not movement_line.intersects(self.stop_line):
            return None

        if self.direction == "forward":
            expected_dir = CrossingDirection.FORWARD
        else:
            expected_dir = CrossingDirection.BACKWARD

        # For forward: expect Y increase (moving down in image)
        # Y increases downward in image coordinates
        y_movement = curr_point[1] - prev_point[1]
        direction = CrossingDirection.FORWARD if y_movement > 0 else CrossingDirection.BACKWARD

        return LineCrossingEvent(
            track_id=track_id,
            frame_idx=frame_idx,
            point_before=prev_point,
            point_after=curr_point,
            direction=direction,
        )

    @staticmethod
    def draw_stop_line(
        frame, stop_line: list[list[int]], color=(0, 255, 255), thickness: int = 3
    ) -> None:
        """Draw stop line on frame."""
        import cv2
        pts = np.array(stop_line, dtype=np.int32)
        cv2.line(frame, tuple(pts[0]), tuple(pts[1]), color, thickness)
        mid = ((pts[0][0] + pts[1][0]) // 2, (pts[0][1] + pts[1][1]) // 2)
        cv2.putText(frame, "STOP", (mid[0] - 30, mid[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
