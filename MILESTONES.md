# Traffic Violation LPR — Milestones

## Muc tieu chung

Hoan thien mot MVP co the demo tren video giao thong co dinh:

- Detect xe.
- Track xe qua nhieu frame.
- Xac dinh den giao thong va stop line.
- Phat hien xe vuot vach khi den do.
- Doc bien so va luu bang chung.
- Bao cao metric that tren subset annotate.

---

## Layer 1 — Data Preparation ✅ Done

### Muc tieu

Bien raw video thanh input ngan, sach, co metadata va san sang annotate.

### Tasks

- [x] Dat raw videos vao `data/raw_videos/`.
- [x] Kiem tra OpenCV mo duoc video.
- [x] Lay metadata: resolution, FPS, frame count, duration.
- [x] Cat 3 clip demo 60 giay, 30 FPS.
- [x] Trich reference frames de annotate scene.
- [x] Tao contact sheet de review nhanh.
- [x] Tao overlay stop line / road ROI / traffic light ROI.
- [x] Tao `layer1_inventory.json`.
- [x] Tao `cam_01_events_template.json`.
- [x] Cap nhat `configs/cameras/cam_01.json`.
- [x] Tao `configs/cameras/cam_02.json` va `cam_03.json`.

### Deliverables

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

### Ghi chu

Scene config hien tai la uoc luong ban dau. `direction` dang la `backward` vi xe o lane gan camera di len tren anh, tuc toa do `y` giam.

---

## Layer 2 — Detection, Tracking & Scene Logic ✅ Baseline Done

### Muc tieu

Chay pipeline tren clip Layer 1 va tao debug video co overlay.

### Tasks

- [x] Sua crossing logic de ton trong `direction = backward`.
- [x] Validate camera config truoc khi chay pipeline.
- [x] Cai dat dependency runtime can thiet: OpenCV, Ultralytics, Shapely, PyYAML.
- [x] Chay vehicle detection tren `cam_01_clip_001.mp4`.
- [x] Chay tracking va ve `track_id`.
- [x] Ve stop line, road ROI, traffic light ROI len debug video.
- [x] Log light state moi 30 frame.
- [x] Review debug contact sheet: bbox, ROI va stop line hien dung.

### Ghi chu

ByteTrack chua duoc cai, nen Layer 2 baseline dang dung simple IoU tracker fallback. Fallback du de debug overlay va event crossing, nhung neu muon track ID on dinh hon thi can cai ByteTrack/BoT-SORT o buoc sau.

### Deliverable

```text
outputs/debug_videos/cam_01_layer2_debug.mp4
outputs/debug_videos/cam_01_layer2_debug_contact.jpg
```

---

## Layer 3 — Violation Event Baseline ✅ Baseline Done

### Muc tieu

Tao duoc event `pending` khi xe cat stop line trong dieu kien den do.

### Tasks

- [x] Xac dinh light state bang ROI + HSV + fallback den bi overexposed.
- [x] Kiem tra crossing theo bottom-center bbox.
- [x] Tao event state `pending`.
- [x] Luu pending event thanh `event.json`.
- [x] Luu snapshot `frame.jpg` cho moi event.
- [x] Tao Layer 3 report.
- [x] Tao event contact sheet de review.
- [x] Tao va dien file annotation `cam_01_events.json` theo configured stop-line rule.
- [ ] Dien ground truth thu cong va tinh precision/recall.

### Deliverable

```text
outputs/events/EVT_0001/event.json
outputs/reports/layer3_event_report.json
outputs/reports/layer3_event_contact.jpg
data/annotations/cam_01_events.json
```

---

## Layer 4 — Plate OCR & Evidence ✅ Baseline Done

### Muc tieu

Doc bien so xe vi pham va luu bang chung.

### Dataset status

- [x] Kiem tra CCPD2019.
- [x] Tao compact subset `data/processed/ccpd_layer4`.
- [x] Sinh YOLO labels cho bien so.
- [x] Sinh OCR plate crops va `ocr_labels`.
- [x] Tao `manifest.json` va sample contact sheet.
- [x] Xac dinh raw CCPD full khong can giu sau khi da convert.

### Tasks

- [x] Fix plate detector path, khong dung YOLO COCO mac dinh cho bien so.
- [x] Them fallback crop heuristic theo bbox xe.
- [x] Them OCR backend optional: PaddleOCR / HyperLPR3 / EasyOCR.
- [x] Dung HyperLPR3 cho video bien so Trung Quoc hien tai.
- [x] Fusion OCR qua nhieu frame.
- [x] Luu `frame.jpg`, `plate.jpg`, `event.json`.
- [x] Luu `clip.mp4` cho event da confirmed.
- [x] Tao `outputs/reports/layer4_ocr_report.json`.
- [x] Tao `outputs/reports/layer4_evidence_contact.jpg`.
- [x] Train/fine-tune YOLO plate detector tu `data/processed/ccpd_layer4/ccpd_plate.yaml` de tang do ben.

### Deliverable

```text
outputs/events_layer4/EVT_0001/
  frame.jpg
  plate.jpg
  clip.mp4
  event.json
outputs/reports/layer4_ocr_report.json
outputs/reports/layer4_evidence_contact.jpg
```

### Ket qua baseline

Tren full 60s `cam_01_clip_001.mp4` sau khi dung YOLO plate detector da fine-tune:

| Metric | Ket qua |
|---|---:|
| Violation candidates | 5 |
| Events co `plate.jpg` | 5 |
| OCR confirmed | 4 |
| Processing FPS | ~19.5 |
| Plate detector backend | YOLO |

Training detector bien so tren compact CCPD dat precision `0.989`, recall `0.998`, mAP50 `0.994`, mAP50-95 `0.73`. Luu y: model weight local nam o `models/plate_detector/ccpd_yolov8n_best.pt` va bi ignore khoi Git.

Luu y: OCR confidence con thap va crop bien so tren video that qua nho/mo, nen Plate Accuracy exact van can ground truth doc duoc bang mat.

---

## Layer 5 — Evaluation & Portfolio Polish ✅ Baseline Done

### Muc tieu

Bien project thanh main project CV co demo va metric that.

### Tasks

- [x] Tao `scripts/evaluate_layer5.py`.
- [x] Tao `data/annotations/cam_01_layer5_review.json`.
- [x] Tao `outputs/reports/layer5_evaluation_report.json`.
- [x] Do processing FPS tu Layer 4 report.
- [x] Tao demo contact/video da blur bien so.
- [x] Cap nhat README voi lenh chay va ket qua hien co.
- [x] Dam bao test suite pass trong moi truong hien tai.
- [x] Dien manual review de tinh Event Precision theo configured stop-line rule.
- [ ] Dien manual plate ground truth de tinh Plate Accuracy.
- [ ] Tao full timeline ground truth neu muon tinh Event Recall.

### Deliverable

```text
data/annotations/cam_01_layer5_review.json
outputs/reports/layer5_evaluation_report.json
outputs/reports/layer5_demo_contact_redacted.jpg
outputs/debug_videos/layer5_demo_redacted.mp4
```

### Ket qua baseline

| Metric | Ket qua |
|---|---:|
| Events co full frame | 5/5 |
| Events co plate crop | 5/5 |
| Events co clip | 4/5 |
| Events co OCR text | 4/5 |
| Processing FPS | ~19.5 |
| Event Precision | 5/5 = 1.0 |

Event Precision o day chi tinh theo configured stop-line rule cua project, khong phai ket luan phap ly vi chua model lane-specific right-turn permissions. Plate Accuracy chua report vi plate crop qua nho/mo de tao manual `plate_text_gt` dang tin. Event Recall van can full timeline ground truth.

---

## Progress Log

| Ngay | Ghi chu |
|---|---|
| 2026-04-15 | Tao repo, viet Plan.md, scaffold source tree |
| 2026-04-27 | Hoan thanh Layer 1: raw videos, clips, reference frames, overlay, inventory, camera configs |
| 2026-04-27 | Hoan thanh Layer 2 baseline: YOLO detection, IoU tracking fallback, scene overlay, light state fallback, debug video |
| 2026-04-28 | Hoan thanh Layer 3 baseline: pending event JSON, event snapshots, report, annotation file can review |
| 2026-04-28 | Chuan hoa CCPD2019 thanh compact dataset cho Layer 4 plate detection/OCR |
| 2026-04-28 | Hoan thanh Layer 4 baseline: plate crop, HyperLPR OCR, evidence clip/json, Layer 4 report |
| 2026-04-28 | Hoan thanh Layer 5 baseline: evaluation script, review template, redacted demo assets, evidence metrics |
| 2026-04-28 | Fine-tune YOLO plate detector tren compact CCPD va review Layer 5 event precision theo configured stop-line rule |
