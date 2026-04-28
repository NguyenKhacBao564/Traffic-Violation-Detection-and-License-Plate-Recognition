"""Violation detection engine — ties together light state, tracking, and line crossing."""
from __future__ import annotations

from dataclasses import dataclass

from src.geometry.line_crossing import StopLineChecker, CrossingDirection, LineCrossingEvent
from src.traffic_light.light_state import TrafficLightEstimator, LightColor, LightState
from src.tracking.vehicle_tracker import VehicleTracker, TrackedVehicle
from src.violation.event_state import EventStateManager, EventState, ViolationEvent


@dataclass
class ViolationEngineConfig:
    """Configuration for the violation engine."""
    min_tracking_frames: int = 5
    violation_window_frames: int = 30
    require_red_light: bool = True


class ViolationEngine:
    """
    Core violation detection logic.

    Evaluates each frame:
      1. Check light state — if red, consider active violations
      2. For each tracked vehicle, check stop line crossing
      3. If crossing + red light → create pending event
      4. Track lifecycle: pending → confirmed → saved
    """

    def __init__(
        self,
        light_estimator: TrafficLightEstimator,
        stop_line_checker: StopLineChecker,
        tracker: VehicleTracker,
        config: ViolationEngineConfig | None = None,
        camera_id: str = "CAM_01",
    ) -> None:
        self.light_estimator = light_estimator
        self.stop_line_checker = stop_line_checker
        self.tracker = tracker
        self.config = config or ViolationEngineConfig()
        self.event_manager = EventStateManager()
        self.camera_id = camera_id
        self.last_light_state: LightState | None = None

        # Track previous positions for each track_id
        self._prev_positions: dict[int, tuple[int, int]] = {}
        # Track how many frames each track has been seen
        self._track_seen_frames: dict[int, int] = {}

    def process_frame(
        self,
        frame,
        tracked_vehicles: list[TrackedVehicle],
        frame_idx: int,
    ) -> list[ViolationEvent]:
        """
        Process one frame. Returns list of newly confirmed events this frame.
        """
        newly_confirmed: list[ViolationEvent] = []

        # 1. Estimate light state
        light_state = self.light_estimator.estimate(frame, frame_idx)
        self.last_light_state = light_state
        is_red = light_state.color == LightColor.RED

        active_track_ids = {t.track_id for t in tracked_vehicles}

        # 2. Age out tracks not seen this frame
        tracks_to_dismiss = set(self._prev_positions.keys()) - active_track_ids
        for tid in tracks_to_dismiss:
            self._prev_positions.pop(tid, None)
            if tid in self.event_manager.pending_events:
                self.event_manager.dismiss(tid)

        # 3. Check crossing for each active track
        for vehicle in tracked_vehicles:
            tid = vehicle.track_id
            curr_pos = VehicleTracker.bottom_center(vehicle.bbox)

            # Increment seen frames
            self._track_seen_frames[tid] = self._track_seen_frames.get(tid, 0) + 1

            prev_pos = self._prev_positions.get(tid)
            if prev_pos is None:
                self._prev_positions[tid] = curr_pos
                continue

            # 4. Check stop-line crossing
            crossing = self.stop_line_checker.check_crossing(
                tid, prev_pos, curr_pos, frame_idx
            )

            if crossing and crossing.direction == self.stop_line_checker.expected_direction:
                # Only flag if this track hasn't already been flagged
                if tid not in self.event_manager.pending_events:
                    # Only create event if track has been seen long enough
                    if self._track_seen_frames[tid] >= self.config.min_tracking_frames:
                        if is_red or not self.config.require_red_light:
                            event = self.event_manager.create_pending(
                                tid,
                                frame_idx,
                                self.camera_id,
                                light_state=light_state.color.value,
                                light_confidence=light_state.confidence,
                                crossing_direction=crossing.direction.value,
                                point_before=crossing.point_before,
                                point_after=crossing.point_after,
                                vehicle_bbox=vehicle.bbox,
                            )
                            newly_confirmed.append(event)

            self._prev_positions[tid] = curr_pos

        # 5. Age out old pending events (track lost too long)
        # Clean up confirmed events from pending dict
        confirmed_ids = {e.track_id for e in self.event_manager.confirmed_events}
        for tid in confirmed_ids:
            self.event_manager.pending_events.pop(tid, None)

        return newly_confirmed

    def confirm_event(
        self, track_id: int, plate_text: str, plate_confidence: float,
        evidence_image_path: str, evidence_clip_path: str,
    ) -> ViolationEvent | None:
        """Manually confirm a pending event after OCR succeeds."""
        return self.event_manager.confirm(
            track_id,
            plate_text=plate_text,
            plate_confidence=plate_confidence,
            evidence_image_path=evidence_image_path,
            evidence_clip_path=evidence_clip_path,
        )

    @property
    def pending_count(self) -> int:
        return len(self.event_manager.pending_events)

    @property
    def confirmed_count(self) -> int:
        return len(self.event_manager.confirmed_events)
