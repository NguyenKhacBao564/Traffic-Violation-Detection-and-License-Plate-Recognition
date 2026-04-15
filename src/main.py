#!/usr/bin/env python3
"""
Traffic Violation Detection & LPR — Main Pipeline

Usage:
    python src/main.py --video data/raw_videos/sample.mp4 \\
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
from src.plate.plate_detector import PlateDetector
from src.plate.plate_ocr import PlateOCR
from src.plate.fusion import OCRFusion
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
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Load configs ---
    cam_cfg = load_camera_config(args.camera)
    cfg = load_default_config(args.config)

    print(f"[*] Video: {args.video}")
    print(f"[*] Camera: {cam_cfg.get('camera_id', 'CAM_01')}")

    # --- Init pipeline components ---
    with VideoReader(args.video) as reader:
        info = reader.info
        print(f"[*] Video info: {info.width}x{info.height} @ {info.fps:.1f} FPS, {info.frame_count} frames")

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

        # Plate OCR (lazy init — PaddleOCR is slow to start)
        plate_detector = PlateDetector(
            conf_threshold=cfg.get("model", {}).get("plate_conf", 0.4),
        )
        plate_ocr = PlateOCR(
            lang=cfg.get("ocr", {}).get("lang", "vi_en"),
            use_angle_cls=cfg.get("ocr", {}).get("use_angle_cls", True),
        )

        # Evidence + storage
        ev_cfg = cfg.get("evidence", {})
        evidence_builder = EvidenceBuilder(
            output_dir=cfg.get("output", {}).get("output_dir", "outputs/events"),
        )
        event_store = EventStore(
            output_dir=cfg.get("output", {}).get("output_dir", "outputs/events"),
        )

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
            violation_engine.process_frame(frame, tracked, frame_idx)

            # Step 4: OCR for pending violations
            ocr_fusion_cache: dict[int, OCRFusion] = {}
            for event in violation_engine.event_manager.pending_events.values():
                tid = event.track_id
                if tid not in ocr_fusion_cache:
                    ocr_fusion_cache[tid] = OCRFusion(
                        vote_frames=cfg.get("ocr", {}).get("vote_frames", 5)
                    )

                # Find bbox for this track
                track_vehicles = [t for t in tracked if t.track_id == tid]
                if not track_vehicles:
                    continue
                bbox = track_vehicles[0].bbox

                # Detect and read plate
                plate_bboxes = plate_detector.detect(frame)
                for pbox in plate_bboxes:
                    # Only process plates in the vehicle region
                    px1, py1, px2, py2 = pbox
                    bx1, by1, bx2, by2 = bbox
                    if (abs(px1 - bx1) < 200 and abs(py1 - by1) < 200
                            and abs(px2 - bx2) < 200 and abs(py2 - by2) < 200):
                        crop = PlateDetector.crop_plate(frame, pbox)
                        text, conf = plate_ocr.read_and_filter(crop)
                        if text:
                            ocr_fusion_cache[tid].add_reading(text, conf)

                # Buffer clip frames
                if tid not in clip_buffer:
                    clip_buffer[tid] = []
                clip_buffer[tid].append(frame.copy())
                # Keep only last N seconds
                max_frames_clip = int(ev_cfg.get("clip_duration_sec", 5) * info.fps)
                clip_buffer[tid] = clip_buffer[tid][-max_frames_clip:]

            # Step 5: Confirm events when enough OCR readings
            for tid, fusion in ocr_fusion_cache.items():
                if len(fusion._readings) >= fusion.vote_frames:
                    result = fusion.fuse()
                    if result.final_text and result.confidence > 0.3:
                        # Get best frame (middle of clip buffer)
                        frames = clip_buffer.get(tid, [])
                        best_frame = frames[len(frames) // 2] if frames else frame
                        plate_crop = None  # Could store plate crop from last detection

                        confirmed = violation_engine.confirm_event(
                            track_id=tid,
                            plate_text=result.final_text,
                            plate_confidence=result.confidence,
                            evidence_image_path="",
                            evidence_clip_path="",
                        )
                        if confirmed:
                            evidence_builder.save_event(confirmed, best_frame, plate_crop)
                            event_store.save(confirmed)
                            print(
                                f"  [+EVENT] {confirmed.event_id} | "
                                f"ID{confirmed.track_id} | "
                                f"{confirmed.plate_text} ({confirmed.plate_confidence:.2f})"
                            )
                        clip_buffer.pop(tid, None)

            # Step 6: Render debug overlay
            StopLineChecker.draw_stop_line(frame, cam_cfg["stop_line"])

            if tracked:
                vehicle_tracker.draw_tracks(frame, tracked)

            if writer:
                writer.write(frame)

            loop_time = time.time() - loop_start
            if frame_idx % 30 == 0:
                print(
                    f"[*] Frame {frame_idx}/{info.frame_count} "
                    f"| {1/loop_time:.1f} FPS "
                    f"| Pending: {violation_engine.pending_count} "
                    f"| Confirmed: {violation_engine.confirmed_count}"
                )

        if writer:
            writer.close()

        total_time = time.time() - total_start
        print(f"\n[*] Done! Processed {frame_idx} frames in {total_time:.1f}s")
        print(f"[*] {violation_engine.confirmed_count} violations detected")
        print(f"[*] Debug video: {args.output}")


if __name__ == "__main__":
    main()
