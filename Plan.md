# Plan — Traffic Violation LPR MVP

## 1. Project Overview

**Name:** Traffic Violation Detection and License Plate Recognition from Fixed-Camera Video
**Short:** traffic-violation-lpr
**Goal:** Xây dựng hệ thống xử lý video từ camera giao thông cố định, phát hiện ô tô vượt đèn đỏ, đọc biển số, lưu bằng chứng. Phục vụ portfolio AI Engineer Intern.

---

## 2. Mục tiêu MVP (v1 — chỉ ô tô, 1 camera, offline)

| STT | Mục tiêu |
|-----|----------|
| 1 | Detect ô tô trong video (YOLO) |
| 2 | Track xe qua nhiều frame (ByteTrack / BoT-SORT) |
| 3 | Xác định trạng thái đèn giao thông (màu đỏ/vàng/xanh) |
| 4 | Phát hiện xe vượt stop line khi đèn đỏ |
| 5 | Đọc biển số xe vi phạm (YOLO plate + PaddleOCR) |
| 6 | Lưu ảnh/clip bằng chứng + metadata JSON |
| 7 | Đo lường: Event Precision, Event Recall, Plate Accuracy |

---

## 3. Tech Stack

| Layer | Tool | Lý do |
|-------|------|-------|
| Detection (xe) | YOLO (ultralytics) | Sẵn có, nhanh, dễ train thêm |
| Tracking | ByteTrack | Mã nguồn mở, hiệu quả, dễ tích hợp |
| Plate Detection | YOLO custom | Fine-tune trên dữ liệu biển số VN |
| OCR | PaddleOCR | Tốt cho tiếng Việt, dễ cài |
| Video I/O | OpenCV | Tiêu chuẩn công nghiệp |
| Storage | JSON (v1) → SQLite/PostgreSQL (v2) | Đơn giản, mở rộng được |
| Env management | conda/pip | Dễ reproduce |

---

## 4. Pipeline Architecture

```
Input Video
    ↓
Frame Reader (cv2.VideoCapture)
    ↓
Vehicle Detector (YOLO — car class only)
    ↓
Vehicle Tracker (ByteTrack — assign track_id)
    ↓
Traffic Light State Estimator (color histogram / rule-based)
    ↓
Stop Line Crossing Logic (geometry check)
    ↓
Violation Event Trigger (red light + crossed = violation)
    ↓
Plate Detector (YOLO custom — run ONLY on violation frames)
    ↓
Plate OCR (PaddleOCR — temporal fusion / voting)
    ↓
Evidence Builder (crop plate image, save clip, save JSON)
    ↓
Event Store (JSON/DB)
```

---

## 5. Implementation Phases (8 tuần)

### Phase 1 — Skeleton & I/O (Tuần 1–2)
- [ ] Scaffold repo (đã làm)
- [ ] Video reader: đọc frame, FPS, resolution
- [ ] Video writer: ghi debug video với bbox
- [ ] Scene config JSON cho 1 camera mẫu
- [ ] Chạy thử end-to-end với dummy detections

### Phase 2 — Detection + Tracking (Tuần 3–4)
- [ ] Tích hợp YOLO (vehicle detection)
- [ ] Filter chỉ giữ class "car"
- [ ] Tích hợp ByteTrack
- [ ] Debug: vẽ track_id lên frame
- [ ] Đánh giá: track smoothness, ID switch rate

### Phase 3 — Traffic Light + Stop Line (Tuần 5–6)
- [ ] Traffic light ROI + color extraction (HSV histogram)
- [ ] Stop line định nghĩa trong scene config (coordinates)
- [ ] Line crossing logic: bottom-center of bbox crosses stop line
- [ ] Violation event state machine (pending → confirmed → dismissed)
- [ ] Debug: visualize stop line, light state

### Phase 4 — Plate OCR + Evidence (Tuần 7)
- [ ] Tích hợp YOLO plate detection (pretrained hoặc fine-tune đơn giản)
- [ ] Tích hợp PaddleOCR
- [ ] Temporal fusion: đọc nhiều frame → vote kết quả tốt nhất
- [ ] Evidence builder: ảnh full-frame, ảnh crop plate, clip 3–8s
- [ ] Lưu event JSON

### Phase 5 — Evaluation + Polish (Tuần 8)
- [ ] Tính Event Precision, Recall, Plate Accuracy trên video mẫu
- [ ] Tối ưu tốc độ (bỏ qua frame không cần)
- [ ] Viết README + demo GIF
- [ ] Dọn code, thêm type hints, docstrings
- [ ] Push lên GitHub

---

## 6. Data Layers

### Layer A — Raw Videos
- 5–10 video ban ngày, 1–5 phút/video
- Camera cố định, stop line rõ, đèn giao thông thấy được
- Biển số không quá nhỏ
- Nguồn: ghi hình thực tế hoặc dataset công khai (AI City, MTSD,...)

### Layer B — Scene Annotations (JSON)
```json
{
  "camera_id": "CAM_01",
  "stop_line": [[x1, y1], [x2, y2]],
  "road_roi": [[x,y], ...],
  "traffic_light_roi": [[x,y], ...],
  "direction": "forward"
}
```
→ Đánh dấu thủ công, không cần AI.

### Layer C — Object/Event Annotations (tùy chọn v2)
- bbox xe, bbox biển số, label vi phạm/không
- Cần cho việc đánh giá recall

---

## 7. Success Metrics

| Metric | Target v1 |
|--------|-----------|
| Event Precision | > 80% |
| Event Recall | > 70% (trên video đã test) |
| Plate Accuracy (on true events) | > 85% |
| FPS (processing) | ≥ 15 FPS |
| False Positive per video | ≤ 3 |

Công thức:
- `Event Precision = TP / (TP + FP)`
- `Event Recall = TP / (TP + FN)`
- `Plate Accuracy = plates_correct / true_violation_events`

---

## 8. Scope Control (Out-of-Scope ngay)

- Real-time streaming
- Nhiều camera
- Xe máy
- Ban đêm / mưa lớn
- Vượt tốc độ / sai làn
- Nhận diện người lái
- Tích hợp hệ thống xử phạt thật
- Training YOLO từ đầu (dùng pretrained)

---

## 9. File Structure (target)

```
traffic-violation-lpr/
├── configs/
│   ├── cameras/
│   │   └── cam_01.json
│   └── default.yaml
├── data/
│   ├── raw_videos/        ← gitignore
│   ├── frames/            ← gitignore
│   ├── annotations/
│   └── samples/
├── models/
│   ├── detectors/         ← weights gitignore
│   ├── plate_detector/    ← weights gitignore
│   └── ocr/
├── src/
│   ├── io/
│   │   ├── video_reader.py
│   │   └── writer.py
│   ├── detection/
│   │   └── vehicle_detector.py
│   ├── tracking/
│   │   └── vehicle_tracker.py
│   ├── traffic_light/
│   │   └── light_state.py
│   ├── geometry/
│   │   └── line_crossing.py
│   ├── violation/
│   │   ├── event_state.py
│   │   └── violation_engine.py
│   ├── plate/
│   │   ├── plate_detector.py
│   │   ├── plate_ocr.py
│   │   └── fusion.py
│   ├── evidence/
│   │   └── evidence_builder.py
│   ├── storage/
│   │   └── event_store.py
│   └── main.py
├── outputs/
│   ├── debug_videos/      ← gitignore
│   ├── events/            ← gitignore
│   └── logs/              ← gitignore
├── notebooks/
├── tests/
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── MILESTONES.md
└── Plan.md
```

---

## 10. Next Steps (First Week Actions)

1. Chuẩn bị 3–5 video mẫu (quay thực tế hoặc cắt từ YouTube giao thông VN)
2. Định nghĩa stop line + light ROI cho scene đầu tiên → `configs/cameras/cam_01.json`
3. Cài đặt môi trường: `conda create -n lpr python=3.10 && pip install ultralytics ByteTrack opencv-python paddleocr`
4. Chạy thử video reader + YOLO inference đơn giản
5. Verify: bounding box xe xuất hiện đúng trên video
