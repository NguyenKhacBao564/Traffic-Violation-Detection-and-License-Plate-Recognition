#!/usr/bin/env python3
"""
Traffic Violation Detection & LPR — Main Pipeline

Usage:
    python src/main.py --video data/raw_videos/clips/cam_01_clip_001.mp4 \\
                       --camera configs/cameras/cam_01.json \\
                       --output outputs/debug_videos/demo.mp4

Pipeline:
    Video → YOLO Detection → ByteTrack → Light State →
    Stop Line Check → Violation Event → Plate OCR → Evidence
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import yaml

from src.io.video_reader import VideoReader
from src.io.writer import VideoWriter
from src.detection.vehicle_detector import VehicleDetector
from src.tracking.vehicle_tracker import VehicleTracker
from src.traffic_light.light_state import TrafficLightEstimator
from src.geometry.line_crossing import StopLineChecker
from src.violation.violation_engine import ViolationEngine, ViolationEngineConfig
from src.evidence.evidence_builder import EvidenceBuilder
from src.storage.event_store import EventStore


def load_camera_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_default_config(path: str = "configs/default.yaml") -> dict:
    if not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _validate_points(name: str, value: object, min_points: int) -> None:
    if not isinstance(value, list) or len(value) < min_points:
        raise ValueError(f"{name} must be a list with at least {min_points} points")
    for idx, point in enumerate(value):
        if (
            not isinstance(point, list)
            or len(point) != 2
            or not all(isinstance(coord, int) for coord in point)
        ):
            raise ValueError(f"{name}[{idx}] must be [x, y] integer coordinates")


def validate_camera_config(cam_cfg: dict) -> None:
    """Validate scene config before expensive model initialization."""
    required = [
        "camera_id",
        "resolution",
        "stop_line",
        "traffic_light_roi",
        "road_roi",
        "direction",
    ]
    missing = [key for key in required if key not in cam_cfg]
    if missing:
        raise ValueError(f"Camera config missing required keys: {', '.join(missing)}")

    resolution = cam_cfg["resolution"]
    if not isinstance(resolution, dict):
        raise ValueError("resolution must be an object with width and height")
    if not isinstance(resolution.get("width"), int) or not isinstance(resolution.get("height"), int):
        raise ValueError("resolution.width and resolution.height must be integers")

    _validate_points("stop_line", cam_cfg["stop_line"], 2)
    if len(cam_cfg["stop_line"]) != 2:
        raise ValueError("stop_line must contain exactly 2 points")
    _validate_points("traffic_light_roi", cam_cfg["traffic_light_roi"], 4)
    _validate_points("road_roi", cam_cfg["road_roi"], 4)

    if cam_cfg["direction"] not in {"forward", "backward"}:
        raise ValueError("direction must be either 'forward' or 'backward'")


def draw_scene_overlay(frame: np.ndarray, cam_cfg: dict, light_state=None) -> None:
    """Draw static scene config overlays in-place for Layer 2 visual QA."""
    # Road ROI: green polygon.
    road_pts = np.array(cam_cfg["road_roi"], dtype=np.int32)
    cv2.polylines(frame, [road_pts], isClosed=True, color=(0, 180, 0), thickness=2)

    # Stop line: yellow line.
    StopLineChecker.draw_stop_line(frame, cam_cfg["stop_line"])

    # Traffic light ROI: state-colored box when state is available.
    if light_state is not None:
        TrafficLightEstimator.draw_roi(frame, cam_cfg["traffic_light_roi"], light_state.color)
    else:
        light_pts = np.array(cam_cfg["traffic_light_roi"], dtype=np.int32)
        cv2.polylines(frame, [light_pts], isClosed=True, color=(0, 0, 255), thickness=2)
        x, y = int(np.min(light_pts[:, 0])), int(np.min(light_pts[:, 1])) - 8
        cv2.putText(
            frame,
            "LIGHT ROI",
            (x, max(y, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )


def write_layer3_report(
    report_path: str,
    *,
    video_path: str,
    camera_path: str,
    cam_cfg: dict,
    processed_frames: int,
    elapsed_sec: float,
    tracker_backend: str,
    events: list[dict],
) -> None:
    """Write Layer 3 predicted-event report for manual review/evaluation."""
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "layer": "layer_3_violation_event_baseline",
        "video": video_path,
        "camera_config": camera_path,
        "camera_id": cam_cfg.get("camera_id"),
        "processed_frames": processed_frames,
        "elapsed_sec": elapsed_sec,
        "processing_fps": processed_frames / elapsed_sec if elapsed_sec > 0 else 0.0,
        "tracker_backend": tracker_backend,
        "predicted_event_count": len(events),
        "events": events,
        "evaluation_status": "pending_manual_ground_truth",
        "ground_truth_path": f"data/annotations/{cam_cfg.get('camera_id', 'cam_01').lower()}_events.json",
        "notes": (
            "Predicted events are pending red-light crossing candidates. "
            "Manual labels are still required before precision/recall can be trusted."
        ),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_layer4_report(
    report_path: str,
    *,
    video_path: str,
    camera_path: str,
    event_output_dir: str,
    processed_frames: int,
    elapsed_sec: float,
    tracker_backend: str,
    plate_detector_backend: str | None,
    ocr_backend: str | None,
    events: list[dict],
) -> None:
    """Write Layer 4 OCR/evidence report."""
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    confirmed = [event for event in events if event.get("state") == "confirmed"]
    with_plate_crop = [event for event in events if event.get("plate_crop_path")]
    with_plate_text = [event for event in events if event.get("plate_text")]
    data = {
        "layer": "layer_4_plate_ocr_evidence",
        "video": video_path,
        "camera_config": camera_path,
        "event_output_dir": event_output_dir,
        "processed_frames": processed_frames,
        "elapsed_sec": elapsed_sec,
        "processing_fps": processed_frames / elapsed_sec if elapsed_sec > 0 else 0.0,
        "tracker_backend": tracker_backend,
        "plate_detector_backend": plate_detector_backend,
        "ocr_backend": ocr_backend,
        "event_count": len(events),
        "events_with_plate_crop": len(with_plate_crop),
        "confirmed_event_count": len(confirmed),
        "events_with_plate_text": len(with_plate_text),
        "events": events,
        "notes": (
            "Confirmed means OCR produced a usable plate text. Events with plate crops "
            "but no text still have evidence for manual review or later OCR retraining."
        ),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Traffic Violation Detection & LPR Pipeline"
    )
    parser.add_argument(
        "--video", "-v", required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--camera", "-c", default="configs/cameras/cam_01.json",
        help="Camera scene config JSON"
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Default pipeline config YAML"
    )
    parser.add_argument(
        "--output", "-o", default="outputs/debug_videos/output.mp4",
        help="Output debug video path"
    )
    parser.add_argument(
        "--skip-frames", type=int, default=0,
        help="Skip every N frames for faster processing"
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Max frames to process (0 = all)"
    )
    parser.add_argument(
        "--no-debug-video", action="store_true",
        help="Skip writing debug video"
    )
    parser.add_argument(
        "--enable-ocr", action="store_true",
        help="Enable plate detector and PaddleOCR path. Disabled by default for Layer 2."
    )
    parser.add_argument(
        "--ocr-backend",
        choices=["auto", "paddle", "hyperlpr", "easyocr", "none"],
        default=None,
        help="OCR backend for Layer 4. Defaults to configs/default.yaml."
    )
    parser.add_argument(
        "--plate-detector-model",
        default=None,
        help="Path to trained plate detector weights. Falls back to heuristic crop if missing."
    )
    parser.add_argument(
        "--events-dir",
        default=None,
        help="Directory for event evidence. Defaults to output.output_dir in config."
    )
    parser.add_argument(
        "--layer3-report",
        default="outputs/reports/layer3_event_report.json",
        help="Path to write Layer 3 predicted-event report JSON."
    )
    parser.add_argument(
        "--layer4-report",
        default="outputs/reports/layer4_ocr_report.json",
        help="Path to write Layer 4 plate OCR/evidence report JSON."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Load configs ---
    cam_cfg = load_camera_config(args.camera)
    validate_camera_config(cam_cfg)
    cfg = load_default_config(args.config)

    print(f"[*] Video: {args.video}")
    print(f"[*] Camera: {cam_cfg.get('camera_id', 'CAM_01')}")

    # --- Init pipeline components ---
    with VideoReader(args.video) as reader:
        info = reader.info
        print(f"[*] Video info: {info.width}x{info.height} @ {info.fps:.1f} FPS, {info.frame_count} frames")
        expected_res = cam_cfg["resolution"]
        if expected_res["width"] != info.width or expected_res["height"] != info.height:
            print(
                "[WARN] Camera config resolution does not match video: "
                f"{expected_res['width']}x{expected_res['height']} vs {info.width}x{info.height}"
            )

        # Vehicle detector
        det_cfg = cfg.get("model", {})
        vehicle_detector = VehicleDetector(
            model_name=det_cfg.get("vehicle_model", "yolov8n.pt"),
            conf_threshold=det_cfg.get("vehicle_conf", 0.3),
            iou_threshold=det_cfg.get("vehicle_iou", 0.4),
            allowed_classes=det_cfg.get("vehicle_classes", [2]),
        )

        # Vehicle tracker
        trk_cfg = cfg.get("tracking", {})
        vehicle_tracker = VehicleTracker(
            track_thresh=trk_cfg.get("track_thresh", 0.5),
            track_buffer=trk_cfg.get("track_buffer", 30),
            fps=info.fps,
            match_thresh=trk_cfg.get("match_thresh", 0.8),
            min_box_area=trk_cfg.get("min_box_area", 1500),
        )
        print(f"[*] Tracker backend: {vehicle_tracker.backend}")

        # Traffic light estimator
        light_estimator = TrafficLightEstimator(
            roi=cam_cfg["traffic_light_roi"],
        )

        # Stop line checker
        stop_line_checker = StopLineChecker(
            stop_line=cam_cfg["stop_line"],
            direction=cam_cfg.get("direction", "forward"),
        )

        # Violation engine
        vio_cfg = cfg.get("violation", {})
        violation_engine = ViolationEngine(
            light_estimator=light_estimator,
            stop_line_checker=stop_line_checker,
            tracker=vehicle_tracker,
            config=ViolationEngineConfig(
                min_tracking_frames=vio_cfg.get("min_tracking_frames", 5),
                violation_window_frames=vio_cfg.get("violation_window_frames", 30),
            ),
            camera_id=cam_cfg.get("camera_id", "CAM_01"),
        )

        plate_detector = None
        plate_ocr = None
        OCRFusion = None
        if args.enable_ocr:
            from src.plate.plate_detector import PlateDetector
            from src.plate.plate_ocr import PlateOCR
            from src.plate.fusion import OCRFusion as OCRFusionClass

            ocr_cfg = cfg.get("ocr", {})
            plate_model = args.plate_detector_model or cfg.get("model", {}).get("plate_detector_model")
            plate_detector = PlateDetector(
                model_path=plate_model,
                conf_threshold=cfg.get("model", {}).get("plate_conf", 0.4),
                enable_fallback=cfg.get("model", {}).get("plate_fallback", True),
            )
            plate_ocr = PlateOCR(
                lang=ocr_cfg.get("lang", "ch"),
                use_angle_cls=ocr_cfg.get("use_angle_cls", True),
                backend=args.ocr_backend or ocr_cfg.get("backend", "auto"),
                easyocr_languages=ocr_cfg.get("easyocr_languages", ["ch_sim", "en"]),
                gpu=ocr_cfg.get("gpu", False),
            )
            OCRFusion = OCRFusionClass
            print(f"[*] Plate detector backend: {plate_detector.backend}")
            print(f"[*] OCR backend: {plate_ocr.backend_name}")
        else:
            print("[*] OCR disabled for this run. Use --enable-ocr for Layer 4.")

        # Evidence + storage
        ev_cfg = cfg.get("evidence", {})
        event_output_dir = args.events_dir or cfg.get("output", {}).get("output_dir", "outputs/events")
        evidence_builder = EvidenceBuilder(
            output_dir=event_output_dir,
        )
        event_store = EventStore(
            output_dir=event_output_dir,
        )
        violation_engine.event_manager._event_counter = event_store.max_event_number()

        # Debug video writer
        writer = None
        if not args.no_debug_video:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            writer = VideoWriter(
                args.output,
                fps=min(info.fps, 30),
                frame_size=(info.width, info.height),
            )

        # --- Main processing loop ---
        print("[*] Starting pipeline...")
        frame_idx = 0
        total_start = time.time()
        clip_buffer: dict[int, list] = {}  # track_id -> list of frames
        event_records: dict[str, dict] = {}
        ocr_fusion_cache = {}
        ocr_frame_counts: dict[int, int] = {}
        last_plate_crops: dict[int, np.ndarray] = {}
        last_plate_detections: dict[int, object] = {}
        layer4_plate_detections: list[object] = []

        while True:
            ret, frame = reader.read()
            if not ret:
                break

            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break

            frame_idx += 1
            loop_start = time.time()

            # Step 1: Detect vehicles
            detections = vehicle_detector.detect(frame)

            # Step 2: Track vehicles
            tracked = vehicle_tracker.update(detections)

            # Step 3: Process violation logic
            new_events = violation_engine.process_frame(frame, tracked, frame_idx)
            for event in new_events:
                print(
                    f"  [+PENDING] {event.event_id} | "
                    f"ID{event.track_id} | frame={event.frame_idx}"
                )

            # Step 4: OCR for pending violations
            layer4_plate_detections = []
            if args.enable_ocr and plate_detector and plate_ocr and OCRFusion:
                pending_events = list(violation_engine.event_manager.pending_events.values())
            else:
                pending_events = []

            ocr_cfg = cfg.get("ocr", {})
            ocr_min_conf = ocr_cfg.get("min_confidence", 0.15)
            ocr_frame_stride = max(1, int(ocr_cfg.get("frame_stride", 5)))
            for event in pending_events:
                tid = event.track_id
                if tid not in ocr_fusion_cache:
                    ocr_fusion_cache[tid] = OCRFusion(
                        vote_frames=ocr_cfg.get("vote_frames", 5)
                    )
                ocr_frame_counts[tid] = ocr_frame_counts.get(tid, 0) + 1

                # Find bbox for this track
                track_vehicles = [t for t in tracked if t.track_id == tid]
                if not track_vehicles:
                    continue
                bbox = track_vehicles[0].bbox
                event.vehicle_bbox = bbox

                # Buffer clip frames for later evidence.
                if tid not in clip_buffer:
                    clip_buffer[tid] = []
                clip_buffer[tid].append(frame.copy())
                max_frames_clip = int(ev_cfg.get("clip_duration_sec", 5) * info.fps)
                clip_buffer[tid] = clip_buffer[tid][-max_frames_clip:]

                if ocr_frame_counts[tid] != 1 and ocr_frame_counts[tid] % ocr_frame_stride != 0:
                    continue

                # Detect plate inside the vehicle region. If no model exists,
                # PlateDetector will use a local crop heuristic.
                plate_detections = plate_detector.detect_with_scores(frame, vehicle_bbox=bbox)
                layer4_plate_detections.extend(plate_detections)
                if plate_detections:
                    plate_det = plate_detections[0]
                    crop = plate_detector.crop_plate(frame, plate_det.bbox)
                    if crop.size:
                        last_plate_crops[tid] = crop
                        last_plate_detections[tid] = plate_det
                        event.plate_bbox = plate_det.bbox
                        event.plate_detection_confidence = plate_det.confidence
                        event.plate_detection_source = plate_det.source
                        event.ocr_backend = plate_ocr.backend_name
                        event.ocr_frame_count = ocr_frame_counts[tid]

                        if event.plate_crop_path is None:
                            evidence_builder.save_event(event, frame.copy(), crop)
                            event_store.save(event)
                            event_records[event.event_id] = event.to_dict()

                        text, conf = plate_ocr.read_and_filter(crop)
                        if text and conf >= ocr_min_conf:
                            ocr_fusion_cache[tid].add_reading(text, conf)

            # Step 5: Confirm events when enough OCR readings
            for tid, fusion in list(ocr_fusion_cache.items()):
                if fusion.is_ready:
                    result = fusion.fuse()
                    if result.final_text and result.confidence >= ocr_min_conf:
                        # Get best frame (middle of clip buffer)
                        frames = clip_buffer.get(tid, [])
                        best_frame = frames[len(frames) // 2] if frames else frame
                        plate_crop = last_plate_crops.get(tid)
                        plate_det = last_plate_detections.get(tid)

                        confirmed = violation_engine.confirm_event(
                            track_id=tid,
                            plate_text=result.final_text,
                            plate_confidence=result.confidence,
                            evidence_image_path="",
                            evidence_clip_path="",
                        )
                        if confirmed:
                            if plate_det is not None:
                                confirmed.plate_bbox = plate_det.bbox
                                confirmed.plate_detection_confidence = plate_det.confidence
                                confirmed.plate_detection_source = plate_det.source
                            confirmed.ocr_backend = plate_ocr.backend_name
                            confirmed.ocr_votes = result.votes
                            confirmed.ocr_frame_count = result.total_frames
                            clip_path = evidence_builder.save_clip(
                                confirmed.event_id,
                                frames,
                                fps=info.fps,
                            )
                            confirmed.evidence_clip_path = clip_path
                            evidence_builder.save_event(confirmed, best_frame, plate_crop)
                            event_store.save(confirmed)
                            event_records[confirmed.event_id] = confirmed.to_dict()
                            print(
                                f"  [+EVENT] {confirmed.event_id} | "
                                f"ID{confirmed.track_id} | "
                                f"{confirmed.plate_text} ({confirmed.plate_confidence:.2f})"
                            )
                        clip_buffer.pop(tid, None)
                        ocr_fusion_cache.pop(tid, None)
                        last_plate_crops.pop(tid, None)
                        last_plate_detections.pop(tid, None)

            # Step 6: Render debug overlay
            draw_scene_overlay(frame, cam_cfg, violation_engine.last_light_state)

            if tracked:
                pending_ids = set(violation_engine.event_manager.pending_events)
                vehicle_tracker.draw_tracks(frame, tracked, next_event_ids=pending_ids)

            if args.enable_ocr and plate_detector and layer4_plate_detections:
                plate_detector.draw_detections(frame, layer4_plate_detections)

            for event in new_events:
                evidence_builder.save_event(event, frame.copy(), None)
                event_store.save(event)
                event_records[event.event_id] = event.to_dict()

            if writer:
                writer.write(frame)

            loop_time = time.time() - loop_start
            if frame_idx % 30 == 0:
                light_state = violation_engine.last_light_state
                light_text = (
                    f"{light_state.color.value}:{light_state.confidence:.2f}"
                    if light_state else "n/a"
                )
                print(
                    f"[*] Frame {frame_idx}/{info.frame_count} "
                    f"| {1/loop_time:.1f} FPS "
                    f"| Det: {len(detections)} "
                    f"| Tracks: {len(tracked)} "
                    f"| Light: {light_text} "
                    f"| Pending: {violation_engine.pending_count} "
                    f"| Confirmed: {violation_engine.confirmed_count}"
                )

        if writer:
            writer.close()

        total_time = time.time() - total_start
        final_event_records = list(event_records.values())
        write_layer3_report(
            args.layer3_report,
            video_path=args.video,
            camera_path=args.camera,
            cam_cfg=cam_cfg,
            processed_frames=frame_idx,
            elapsed_sec=total_time,
            tracker_backend=vehicle_tracker.backend,
            events=final_event_records,
        )
        if args.enable_ocr:
            write_layer4_report(
                args.layer4_report,
                video_path=args.video,
                camera_path=args.camera,
                event_output_dir=event_output_dir,
                processed_frames=frame_idx,
                elapsed_sec=total_time,
                tracker_backend=vehicle_tracker.backend,
                plate_detector_backend=getattr(plate_detector, "backend", None),
                ocr_backend=getattr(plate_ocr, "backend_name", None),
                events=final_event_records,
            )
        print(f"\n[*] Done! Processed {frame_idx} frames in {total_time:.1f}s")
        print(f"[*] {len(final_event_records)} violation candidates detected")
        print(f"[*] {violation_engine.confirmed_count} confirmed violations detected")
        print(f"[*] Debug video: {args.output}")
        print(f"[*] Layer 3 report: {args.layer3_report}")
        if args.enable_ocr:
            print(f"[*] Layer 4 report: {args.layer4_report}")


if __name__ == "__main__":
    main()
