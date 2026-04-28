# Plan — Traffic Violation LPR MVP

## 1. Project Goal

Build an offline fixed-camera traffic video pipeline that can:

1. Detect cars.
2. Track cars across frames.
3. Estimate traffic light state.
4. Detect stop-line crossing.
5. Create violation events.
6. Read license plates.
7. Save evidence and JSON metadata.
8. Report real metrics on annotated clips.

The project is scoped as a portfolio MVP, not a production traffic enforcement system.

---

## 2. MVP Scope

| Item | Scope |
|---|---|
| Camera | One fixed traffic camera scene first |
| Video | Offline daytime clips |
| Vehicle type | Cars first |
| Violation type | Red-light / stop-line crossing |
| Storage | JSON files |
| Public demo | Blur/anonymize plates before sharing |

Out of scope for MVP:

- Real-time streaming.
- Multi-camera deployment.
- Motorbike-specific logic.
- Night/rain robustness.
- Real enforcement integration.
- Training all models from scratch.

---

## 3. Current Data Layers

### Layer 1 — Raw Video Preparation ✅ Done

Purpose: make raw videos usable for development.

Completed outputs:

```text
data/raw_videos/clips/cam_01_clip_001.mp4
data/raw_videos/clips/cam_02_clip_001.mp4
data/raw_videos/clips/cam_03_clip_001.mp4
data/frames/references/layer1_contact_sheet.jpg
data/frames/references/cam_01_scene_overlay.jpg
data/annotations/layer1_inventory.json
data/annotations/cam_01_events_template.json
configs/cameras/cam_01.json
configs/cameras/cam_02.json
configs/cameras/cam_03.json
```

Layer 1 does not prove the AI pipeline yet. It only ensures the videos are readable, clipped, documented, and ready for annotation/debugging.

### Layer 2 — Detection, Tracking & Scene Overlay ✅ Baseline done

Purpose: run computer vision on the Layer 1 clips and visually verify the scene logic.

Tasks:

- Fix crossing logic so `direction = backward` works.
- Validate camera config before pipeline starts.
- Run YOLO vehicle detection.
- Run tracking and draw `track_id`.
- Draw stop line, road ROI, traffic light ROI.
- Produce a debug video.

### Layer 3 — Violation Event Baseline ✅ Baseline done

Purpose: create `pending` events when tracked vehicles cross the stop line under red light.

Tasks:

- Verify traffic light HSV state.
- Compare vehicle movement against stop line.
- Create event state `pending`.
- Save pending event JSON and snapshot frame.
- Create a predicted-event report.
- Create a manual annotation file for review.

Current status:

```text
outputs/events/EVT_0001/event.json
outputs/events/EVT_0002/event.json
outputs/events/EVT_0003/event.json
outputs/events/EVT_0004/event.json
outputs/reports/layer3_event_report.json
outputs/reports/layer3_event_contact.jpg
data/annotations/cam_01_events.json
```

The predicted Layer 3 events have been reviewed under the configured stop-line rule. Recall still needs full timeline ground truth because candidate-only review cannot count missed violations.

### Layer 4 — Plate OCR & Evidence ✅ Improved baseline done

Purpose: attach plate text and evidence files to confirmed events.

Dataset status:

```text
data/processed/ccpd_layer4/ccpd_plate.yaml
data/processed/ccpd_layer4/manifest.json
outputs/reports/ccpd_layer4_sample_contact.jpg
```

CCPD2019 is only needed as a source for this compact subset. After conversion, the full raw folder/archive is not needed for later layers.

Tasks:

- Avoid using COCO YOLO as a plate detector.
- Fine-tune a one-class YOLO plate detector on the compact CCPD subset.
- Use a heuristic vehicle/plate crop fallback when the trained plate model is missing.
- Run OCR with HyperLPR3 for the current Chinese plate video.
- Fuse OCR readings across frames.
- Save `frame.jpg`, `plate.jpg`, `clip.mp4`, and `event.json`.

Current result on the full 60-second CAM_01 clip:

```text
outputs/events_layer4/
outputs/reports/layer4_ocr_report.json
outputs/reports/layer4_evidence_contact.jpg
```

Summary: 5 predicted events, 5 plate crops, 4 OCR texts, ~19.5 FPS. The trained local plate detector exported to `models/plate_detector/ccpd_yolov8n_best.pt` and reached mAP50 ~0.994 on the compact CCPD validation split.

### Layer 5 — Evaluation & Portfolio Polish

Purpose: make the project CV-ready.

Tasks:

- Measure evidence readiness and FPS from Layer 4 outputs.
- Create manual review template for Event Precision and Plate Accuracy.
- Create anonymized demo image/video.
- Update README with results.
- Keep raw/private video files out of Git.

Current outputs:

```text
data/annotations/cam_01_layer5_review.json
outputs/reports/layer5_evaluation_report.json
outputs/reports/layer5_demo_contact_redacted.jpg
outputs/debug_videos/layer5_demo_redacted.mp4
```

Current reviewed status:

- Event Precision: 5/5 = 1.0 under the configured stop-line rule.
- Plate Accuracy: pending because the event plate crops are visible but too small/blurred for reliable human plate-text ground truth.
- Event Recall: pending until a full timeline ground truth file exists.

---

## 4. Pipeline Architecture

```text
Input Video
  -> Frame Reader
  -> Vehicle Detector
  -> Vehicle Tracker
  -> Traffic Light State Estimator
  -> Stop-Line Crossing Logic
  -> Violation Event Trigger
  -> Plate Detector + OCR
  -> Evidence Builder
  -> Event Store
```

---

## 5. Tech Stack

| Layer | Tool | Reason |
|---|---|---|
| Video I/O | OpenCV | Standard video/frame processing |
| Vehicle detection | Ultralytics YOLO | Fast pretrained baseline |
| Tracking | ByteTrack / tracker wrapper | Track IDs across frames |
| Traffic light | ROI + HSV | Simple and explainable for fixed camera |
| Geometry | Shapely / line crossing | Stop-line crossing logic |
| OCR | HyperLPR3 / PaddleOCR / EasyOCR | Configurable OCR; HyperLPR3 fits the current Chinese plate demo |
| Storage | JSON | Simple MVP evidence format |

---

## 6. Immediate Next Steps

1. Create full timeline ground truth if event recall is required.
2. Improve OCR/plate readability, either by better plate crop selection or a dedicated OCR dataset.
3. Replace the IoU tracker fallback with ByteTrack/BoT-SORT for fewer ID-switch risks.
4. Add lane-specific logic for right-turn permissions before claiming legal traffic-enforcement accuracy.

Already completed plate detector training command:

```bash
python scripts/train_plate_detector.py \
  --epochs 6 \
  --patience 3 \
  --device mps
```

---

## 7. Success Metrics

| Metric | Target v1 |
|---|---:|
| Event Precision | > 80% |
| Event Recall | > 70% |
| Plate Accuracy | > 85% |
| Processing FPS | >= 15 |
| False positives per clip | <= 3 |

Metrics must be reported from manually annotated clips.
