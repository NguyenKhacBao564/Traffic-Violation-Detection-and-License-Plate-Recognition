# Traffic Violation Detection & License Plate Recognition

Fixed-camera traffic video pipeline for detecting red-light/stop-line violations, reading license plates, and saving evidence.

## MVP Scope

- One fixed traffic camera scene.
- Offline daytime video.
- Cars only for the first MVP.
- Rule-based traffic light state from ROI + HSV.
- Stop-line crossing from tracked vehicle movement.
- Evidence saved as image/JSON first; clip output can be added after baseline.

## Current Status

Layer 5 event review is complete on `cam_01_clip_001.mp4` under the configured stop-line rule.

| Layer | Status | Notes |
|---|---|---|
| Layer 1 — Data preparation | Done | Raw videos converted to 60s clips, reference frames, overlay images, inventory JSON |
| Layer 2 — Detection/tracking/scene logic | Baseline done | YOLO + IoU tracker fallback debug run on CAM_01 |
| Layer 3 — Violation baseline | Baseline reviewed | Pending events saved with snapshots and reviewed under the configured stop-line rule |
| Layer 4 — Plate OCR/evidence | Improved baseline done | Fine-tuned YOLO plate detector + HyperLPR evidence run produced 5 event candidates, 5 plate crops, 4 OCR texts |
| Layer 5 — Evaluation/polish | Event review done | Candidate precision is reviewed under the configured stop-line rule; plate accuracy waits for readable plate ground truth |

## Layer 1 Outputs

Raw videos are stored locally under `data/raw_videos/` and are ignored by Git.

Generated Layer 1 files:

```text
data/raw_videos/clips/cam_01_clip_001.mp4
data/raw_videos/clips/cam_02_clip_001.mp4
data/raw_videos/clips/cam_03_clip_001.mp4
data/frames/references/layer1_contact_sheet.jpg
data/frames/references/cam_01_scene_overlay.jpg
data/frames/references/cam_02_scene_overlay.jpg
data/frames/references/cam_03_scene_overlay.jpg
data/annotations/layer1_inventory.json
data/annotations/cam_01_events_template.json
configs/cameras/cam_01.json
configs/cameras/cam_02.json
configs/cameras/cam_03.json
```

Important scene detail: near-lane vehicles move upward in image coordinates, so camera configs currently use:

```json
"direction": "backward"
```

The violation/crossing logic now respects this direction setting.

## Tech Stack

| Stage | Tool |
|---|---|
| Video I/O | OpenCV |
| Vehicle Detection | YOLO via Ultralytics |
| Vehicle Tracking | ByteTrack / tracker wrapper |
| Traffic Light State | ROI + HSV threshold |
| Stop-Line Logic | Geometry with bottom-center bbox |
| Plate OCR | HyperLPR3 for current Chinese plates; PaddleOCR/EasyOCR optional backends |
| Storage | JSON |

## Pipeline

```text
Input Video
  -> Frame Reader
  -> Vehicle Detector
  -> Vehicle Tracker
  -> Traffic Light State Estimator
  -> Stop-Line Crossing Logic
  -> Violation Event State Machine
  -> Plate Detector + OCR
  -> Evidence Builder
  -> Event Store
```

## Layer 2 Debug Run

Latest verified command:

```bash
python src/main.py \
  --video data/raw_videos/clips/cam_01_clip_001.mp4 \
  --camera configs/cameras/cam_01.json \
  --output outputs/debug_videos/cam_01_layer2_debug.mp4 \
  --max-frames 900
```

Generated local outputs:

```text
outputs/debug_videos/cam_01_layer2_debug.mp4
outputs/debug_videos/cam_01_layer2_debug_contact.jpg
```

Notes:

- OCR is disabled by default for Layer 2. Use `--enable-ocr` later in Layer 4.
- ByteTrack is not installed locally yet, so the run uses a simple IoU tracker fallback.
- Traffic light detection uses HSV first, then an overexposed-bulb fallback for this CCTV video.

## Layer 3 Event Baseline

Latest verified command:

```bash
python src/main.py \
  --video data/raw_videos/clips/cam_01_clip_001.mp4 \
  --camera configs/cameras/cam_01.json \
  --output outputs/debug_videos/cam_01_layer3_debug.mp4 \
  --max-frames 900 \
  --layer3-report outputs/reports/layer3_event_report.json
```

Generated local outputs:

```text
outputs/events/EVT_0001/event.json
outputs/events/EVT_0001/frame.jpg
outputs/events/EVT_0002/event.json
outputs/events/EVT_0003/event.json
outputs/events/EVT_0004/event.json
outputs/reports/layer3_event_report.json
outputs/reports/layer3_event_contact.jpg
data/annotations/cam_01_events.json
```

Layer 3 produced 4 pending red-light crossing candidates on the first 900 frames. `data/annotations/cam_01_events.json` has been reviewed under the configured stop-line rule. Recall still needs full timeline ground truth.

## Layer 4 Plate OCR & Evidence

Latest verified command:

```bash
python src/main.py \
  --video data/raw_videos/clips/cam_01_clip_001.mp4 \
  --camera configs/cameras/cam_01.json \
  --output outputs/debug_videos/layer4_yolo_plate.mp4 \
  --events-dir outputs/events_layer4 \
  --enable-ocr \
  --ocr-backend hyperlpr \
  --layer3-report outputs/reports/layer4_event_report.json \
  --layer4-report outputs/reports/layer4_ocr_report.json
```

Verified result on the full 60-second CAM_01 clip:

| Result | Count |
|---|---:|
| Violation candidates | 5 |
| Events with `plate.jpg` | 5 |
| Confirmed OCR events | 4 |
| Processing FPS | ~19.5 |
| Plate detector backend | YOLO fine-tuned on CCPD |

Generated local outputs:

```text
outputs/events_layer4/EVT_0001/{frame.jpg,plate.jpg,clip.mp4,event.json}
outputs/events_layer4/EVT_0002/{frame.jpg,plate.jpg,clip.mp4,event.json}
outputs/events_layer4/EVT_0003/{frame.jpg,plate.jpg,clip.mp4,event.json}
outputs/events_layer4/EVT_0004/{frame.jpg,plate.jpg,event.json}
outputs/events_layer4/EVT_0005/{frame.jpg,plate.jpg,clip.mp4,event.json}
outputs/reports/layer4_ocr_report.json
outputs/reports/layer4_evidence_contact.jpg
outputs/debug_videos/layer4_yolo_plate.mp4
```

Notes:

- `PlateDetector` uses `models/plate_detector/ccpd_yolov8n_best.pt` when the local trained weight exists. If the weight is missing, it falls back to a color/lower-vehicle crop heuristic.
- OCR backend is configurable: `auto`, `paddle`, `hyperlpr`, `easyocr`, or `none`.
- HyperLPR3 is the best current backend for this Chinese traffic video. The OCR confidence is still modest, and exact plate accuracy is not reported until readable manual plate ground truth exists.

## Layer 4 Dataset Prep

CCPD2019 was converted into a compact local subset for plate detection/OCR:

```text
data/processed/ccpd_layer4/ccpd_plate.yaml
data/processed/ccpd_layer4/images/{train,val,test}/
data/processed/ccpd_layer4/labels/{train,val,test}/
data/processed/ccpd_layer4/ocr_crops/{train,val,test}/
data/processed/ccpd_layer4/ocr_labels/{train,val,test}.txt
data/processed/ccpd_layer4/manifest.json
outputs/reports/ccpd_layer4_sample_contact.jpg
```

Subset size: 8,598 images total, including 8,000 positive plate samples and 598 no-plate negative samples. The full raw CCPD folder/archive is not needed after this conversion.

Training command used for the current local detector:

```bash
python scripts/train_plate_detector.py \
  --epochs 6 \
  --patience 3 \
  --device mps
```

Best validation result from the current run: precision `0.989`, recall `0.998`, mAP50 `0.994`, mAP50-95 `0.73`. The exported local weight is `models/plate_detector/ccpd_yolov8n_best.pt`; model weights are intentionally ignored by Git.

## Layer 5 Evaluation & Portfolio Assets

Latest verified commands:

```bash
python scripts/evaluate_layer5.py \
  --events-dir outputs/events_layer4 \
  --layer4-report outputs/reports/layer4_ocr_report.json \
  --review data/annotations/cam_01_layer5_review.json \
  --output outputs/reports/layer5_evaluation_report.json

python scripts/create_layer5_demo_assets.py \
  --events-dir outputs/events_layer4 \
  --contact outputs/reports/layer5_demo_contact_redacted.jpg \
  --video outputs/debug_videos/layer5_demo_redacted.mp4
```

Current evidence metrics:

| Metric | Result |
|---|---:|
| Events with full frame | 5/5 |
| Events with plate crop | 5/5 |
| Events with evidence clip | 4/5 |
| Events with OCR text | 4/5 |
| Processing FPS | ~19.5 |
| Plate detector backend | YOLO |

Reviewed quality metrics:

- Event Precision: 5/5 = 1.0 under the configured stop-line rule.
- Plate Accuracy: pending because all event plate crops are too small/blurred for reliable manual `plate_text_gt`.
- Event Recall: requires full timeline ground truth, not only predicted-event review.

Important scope note: the current event precision is a technical metric for the configured stop-line + red-light ROI rule. It is not a legal traffic-enforcement metric because lane-specific right-turn permissions are not modeled yet.

Redacted demo asset:

```text
outputs/reports/layer5_demo_contact_redacted.jpg
outputs/debug_videos/layer5_demo_redacted.mp4
```

## Documentation

- [Plan.md](./Plan.md): roadmap and layer definitions.
- [MILESTONES.md](./MILESTONES.md): current progress.
- [DATASET_SOURCING.md](./DATASET_SOURCING.md): Layer 1 data status and dataset policy.
- [PROJECT_GUIDE.md](./PROJECT_GUIDE.md): detailed learning guide for the full pipeline.
- [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md): baseline runnable engineering checklist.
- [evaluate.md](./evaluate.md): Layer 5 evaluation policy and commands.

## Success Metrics

Final MVP should report:

| Metric | Target |
|---|---:|
| Event Precision | > 80% |
| Event Recall | > 70% |
| Plate Accuracy | > 85% |
| Processing FPS | >= 15 |

Metrics must be measured on manually annotated clips, not estimated.
