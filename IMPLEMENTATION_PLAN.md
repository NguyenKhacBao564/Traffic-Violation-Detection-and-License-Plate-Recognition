# Implementation Plan — Baseline Runnable

Muc tieu cua phase nay: bien repo thanh baseline chay duoc tren clip Layer 1, co debug video de kiem tra detection, tracking, traffic light ROI va stop-line crossing.

Layer 1 da xong, nen file nay khong con tap trung vao viec tim video nua. Viec tiep theo la lam cho pipeline chay that tren:

```text
data/raw_videos/clips/cam_01_clip_001.mp4
configs/cameras/cam_01.json
```

---

## 1. Trang thai hien tai

### Da xong

- Raw videos doc duoc bang OpenCV.
- Da cat 3 clip 60 giay, 30 FPS.
- Da tao reference frames va scene overlays.
- Da tao `layer1_inventory.json`.
- Da tao camera configs cho `cam_01`, `cam_02`, `cam_03`.
- Camera configs da la JSON hop le.

### Chua xong

- Chua chay full pipeline YOLO/tracking.
- Chua co debug video tu model.
- Chua verify light state bang HSV.
- Chua tao event `pending`.
- Chua fix plate detector/OCR.

---

## 2. Ordered Work Items

### Step 1 — Config validation

**Muc tieu:** pipeline fail early neu camera config sai.

Can lam:

- Validate co du key: `camera_id`, `resolution`, `stop_line`, `traffic_light_roi`, `road_roi`, `direction`.
- Validate `stop_line` co 2 diem.
- Validate `traffic_light_roi` va `road_roi` co it nhat 4 diem.
- Validate `direction` chi nhan `forward` hoac `backward`.

Tieu chi xong:

```bash
python -m json.tool configs/cameras/cam_01.json
```

va pipeline co error message ro neu config sai.

---

### Step 2 — Fix crossing direction

**Muc tieu:** stop-line crossing logic phai ton trong `direction = backward`.

Hien tai xe o lane gan camera di len tren anh, tuc la `y` giam. Config da dat:

```json
"direction": "backward"
```

Nhung `ViolationEngine` hien chi flag crossing khi direction la `FORWARD`. Can sua de:

- Neu config `direction = forward`, flag crossing forward.
- Neu config `direction = backward`, flag crossing backward.

Tieu chi xong:

- Unit test co case `backward` crossing.
- Event logic khong hard-code `CrossingDirection.FORWARD`.

---

### Step 3 — Runtime dependencies

**Muc tieu:** moi truong co du dependency de chay pipeline.

Can co:

- `opencv-python` hoac `opencv-python-headless`.
- `ultralytics`.
- tracker dependency hoac fallback tracker.
- `shapely`.
- `pyyaml`.

Luu y: da tao `.venv` local de xu ly video Layer 1 va cai `opencv-python-headless`. `.venv` duoc gitignore.

---

### Step 4 — Layer 2 debug run

**Muc tieu:** tao debug video co overlay.

Command muc tieu:

```bash
python src/main.py \
  --video data/raw_videos/clips/cam_01_clip_001.mp4 \
  --camera configs/cameras/cam_01.json \
  --output outputs/debug_videos/cam_01_layer2_debug.mp4 \
  --max-frames 900
```

Can verify trong debug video:

- Bbox xe dung.
- Track ID on dinh.
- Stop line dung vi tri.
- Traffic light ROI dung vi tri.
- FPS/log khong qua cham.

---

### Step 5 — Light state verification

**Muc tieu:** biet HSV ROI co doc duoc den khong.

Can log moi 30 frame:

```text
frame=300 light_state=red confidence=0.82
```

Tieu chi tam chap nhan:

- Light state khong bi `unknown` qua nhieu.
- Neu ROI nho/sai, dieu chinh `traffic_light_roi` truoc khi doi model.

---

### Step 6 — Event baseline ✅ Done

**Muc tieu:** tao event `pending` khi xe cat vach trong dieu kien den do.

Da lam:

- Chay crossing logic tren tracked vehicles.
- Tao pending event.
- Log event id, track id, frame index.
- Luu `event.json` va `frame.jpg` cho moi pending event.
- Tao `outputs/reports/layer3_event_report.json`.
- Tao `outputs/reports/layer3_event_contact.jpg`.
- Tao `data/annotations/cam_01_events.json` de review tay.

Ket qua hien tai:

```text
EVT_0001 frame=98
EVT_0002 frame=162
EVT_0003 frame=625
EVT_0004 frame=805
```

Precision/recall chua duoc tinh vi can dien ground truth thu cong truoc.

---

### Step 7 — Plate/OCR fix sau baseline ✅ Done

**Muc tieu:** sau khi detection/tracking/event logic chay, moi fix OCR.

Van de da xu ly:

- `PlateDetector` khong con dung `yolov8n.pt` nhu plate detector gia.
- Neu chua co weight plate detector, pipeline dung crop heuristic theo bbox xe.
- `main.py` da co `--enable-ocr`, `--ocr-backend`, `--plate-detector-model`, `--events-dir`, `--layer4-report`.
- Evidence da luu duoc `frame.jpg`, `plate.jpg`, `clip.mp4`, `event.json`.
- HyperLPR3 doc duoc 3/4 event candidate tren 900 frame dau.

Dataset da san sang:

```text
data/processed/ccpd_layer4/ccpd_plate.yaml
data/processed/ccpd_layer4/ocr_labels/train.txt
data/processed/ccpd_layer4/manifest.json
```

Ket qua hien tai:

```text
outputs/events_layer4/
outputs/reports/layer4_ocr_report.json
outputs/reports/layer4_evidence_contact.jpg
```

Huong nang cap:

1. Train/fine-tune YOLO plate detector tren compact CCPD subset.
2. Gan weight plate detector vao `configs/default.yaml`.
3. Tang manual labels de tinh Plate Accuracy that.

---

## 3. File Patch Plan

| File | Viec can lam | Uu tien |
|---|---|---|
| `src/geometry/line_crossing.py` | Dam bao direction forward/backward duoc tinh ro | High |
| `src/violation/violation_engine.py` | Khong hard-code crossing direction la forward | High |
| `src/main.py` | Validate config, log debug info, ve ROI | High |
| `src/evidence/evidence_builder.py` | Serialize event JSON tường minh, khong overwrite frame path | Medium |
| `src/plate/plate_detector.py` | Nhan model path dung cho plate | Medium |
| `src/plate/fusion.py` | Fix `total_frames` semantics | Low |

---

## 4. Acceptance Criteria

Baseline phase duoc xem la xong khi:

| # | Criteria |
|---|---|
| AC1 | Pipeline chay tren `cam_01_clip_001.mp4` khong crash |
| AC2 | Tao duoc debug video |
| AC3 | Debug video co bbox + track_id |
| AC4 | Stop line va traffic light ROI hien dung vi tri |
| AC5 | Direction `backward` duoc xu ly dung |
| AC6 | Light state duoc log |
| AC7 | Co event pending neu co xe cat vach luc den do |
| AC8 | Event JSON load duoc va `state` la string |

---

## 5. Decision Gates

### Tracker

Neu track ID nhay lien tuc:

1. Kiem tra format input/output cua tracker.
2. Kiem tra FPS va min box area.
3. Neu ByteTrack kho cai/chay, dung fallback tracker IoU don gian de co baseline.

### Traffic light

Neu light state sai:

1. Sua `traffic_light_roi`.
2. Dieu chinh HSV threshold.
3. Chi dung dataset/model traffic light khi HSV khong du.

### Plate OCR

Neu OCR blank:

1. Thu crop vehicle bbox.
2. Thu crop vung duoi bbox.
3. Thu backend phu hop domain: HyperLPR3 cho bien Trung Quoc, PaddleOCR/EasyOCR cho fallback.
4. Sau do moi fine-tune plate detector rieng de crop on dinh hon.

---

## 6. Khong lam luc nay

- Chua train YOLO tu dau.
- Chua them multi-camera logic.
- Chua them xe may.
- Chua realtime streaming.
- Chua chuyen JSON sang database.
- Chua toi uu FPS truoc khi co baseline.
