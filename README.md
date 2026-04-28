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

Layer 4 baseline is complete on `cam_01_clip_001.mp4`.

| Layer | Status | Notes |
|---|---|---|
| Layer 1 — Data preparation | Done | Raw videos converted to 60s clips, reference frames, overlay images, inventory JSON |
| Layer 2 — Detection/tracking/scene logic | Baseline done | YOLO + IoU tracker fallback debug run on CAM_01 |
| Layer 3 — Violation baseline | Baseline done | Pending events saved with snapshots and report; manual GT review still needed |
| Layer 4 — Plate OCR/evidence | Baseline done | Plate crops saved for 4 events; HyperLPR confirmed 3 plate texts; YOLO plate fine-tune remains an optional robustness upgrade |
| Layer 5 — Evaluation/polish | Pending | Metrics, demo assets, README polish |

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

The next code task is to make the violation/crossing logic respect that direction.

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

Layer 3 produced 4 pending red-light crossing candidates on the first 900 frames. The annotation file is intentionally marked `needs_manual_review`; precision/recall should not be reported until those labels are filled.

## Layer 4 Plate OCR & Evidence

Latest verified command:

```bash
python src/main.py \
  --video data/raw_videos/clips/cam_01_clip_001.mp4 \
  --camera configs/cameras/cam_01.json \
  --output outputs/debug_videos/cam_01_layer4_debug.mp4 \
  --events-dir outputs/events_layer4 \
  --max-frames 900 \
  --enable-ocr \
  --ocr-backend hyperlpr \
  --layer3-report outputs/reports/layer4_event_report.json \
  --layer4-report outputs/reports/layer4_ocr_report.json
```

Verified result on the first 900 frames:

| Result | Count |
|---|---:|
| Violation candidates | 4 |
| Events with `plate.jpg` | 4 |
| Confirmed OCR events | 3 |
| Processing FPS | ~16.3 |

Generated local outputs:

```text
outputs/events_layer4/EVT_0001/{frame.jpg,plate.jpg,clip.mp4,event.json}
outputs/events_layer4/EVT_0002/{frame.jpg,plate.jpg,clip.mp4,event.json}
outputs/events_layer4/EVT_0003/{frame.jpg,plate.jpg,clip.mp4,event.json}
outputs/events_layer4/EVT_0004/{frame.jpg,plate.jpg,event.json}
outputs/reports/layer4_ocr_report.json
outputs/reports/layer4_evidence_contact.jpg
outputs/debug_videos/cam_01_layer4_debug.mp4
```

Notes:

- `PlateDetector` no longer uses COCO YOLO as a fake plate detector. If no trained plate model exists, it falls back to a color/lower-vehicle crop heuristic.
- OCR backend is configurable: `auto`, `paddle`, `hyperlpr`, `easyocr`, or `none`.
- HyperLPR3 is the best current backend for this Chinese traffic video. The OCR confidence is still modest, so manual review remains required before reporting plate accuracy.

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

Training command target:

```bash
yolo detect train \
  model=yolov8n.pt \
  data=data/processed/ccpd_layer4/ccpd_plate.yaml \
  epochs=30 \
  imgsz=640
```

## Documentation

- [Plan.md](./Plan.md): roadmap and layer definitions.
- [MILESTONES.md](./MILESTONES.md): current progress.
- [DATASET_SOURCING.md](./DATASET_SOURCING.md): Layer 1 data status and dataset policy.
- [PROJECT_GUIDE.md](./PROJECT_GUIDE.md): detailed learning guide for the full pipeline.
- [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md): baseline runnable engineering checklist.

## Success Metrics

Final MVP should report:

| Metric | Target |
|---|---:|
| Event Precision | > 80% |
| Event Recall | > 70% |
| Plate Accuracy | > 85% |
| Processing FPS | >= 15 |

Metrics must be measured on manually annotated clips, not estimated.
