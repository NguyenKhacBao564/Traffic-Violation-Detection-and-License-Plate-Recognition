# Traffic Violation LPR — Milestones

## Mục tiêu chung
Portfolio AI Engineer cho vị trí Intern tại TP.HCM.
Hoàn thiện MVP end-to-end trong **8 tuần**, demo được trên GitHub.

---

## Milestone 1 — Skeleton & Environment (Tuần 1–2) ✅ **Đang làm**

### Mục tiêu
Repo scaffold sạch, chạy được end-to-end với dummy data.

### Tasks
- [x] Tạo cấu trúc folder `src/`, `configs/`, `models/`, `outputs/`
- [x] Viết `Plan.md` hoàn chỉnh
- [ ] Cài đặt môi trường conda/pip
- [ ] Video reader: đọc frame, FPS, resolution
- [ ] Video writer: ghi debug video với bbox
- [ ] Scene config JSON cho 1 camera mẫu (`configs/cameras/cam_01.json`)
- [ ] Test: chạy đọc video + ghi ra frame mẫu

### Deliverable
Repo có thể `git clone && pip install -r requirements.txt && python src/main.py --video data/raw_videos/sample.mp4`

---

## Milestone 2 — Vehicle Detection + Tracking (Tuần 3–4)

### Mục tiêu
YOLO detect xe ô tô → ByteTrack gán track_id ổn định.

### Tasks
- [ ] Tích hợp YOLO (ultralytics), filter class "car"
- [ ] Tích hợp ByteTrack / BoT-SORT
- [ ] Debug: vẽ bbox + track_id lên frame
- [ ] Đánh giá: track smoothness, ID switch rate
- [ ] Lưu ảnh debug xuống `outputs/debug_videos/`

### Deliverable
Video debug có bounding box màu theo từng track, track_id hiển thị.

---

## Milestone 3 — Traffic Light State + Stop Line Logic (Tuần 5–6)

### Mục tiêu
Hệ thống xác định đèn đỏ và phát hiện xe vượt stop line.

### Tasks
- [ ] Traffic light ROI — trích xuất màu đèn bằng HSV histogram
- [ ] Stop line coordinates trong scene config
- [ ] Line crossing logic (bottom-center bbox crosses stop line)
- [ ] Violation event state machine: `pending → confirmed → dismissed`
- [ ] Debug: visualize stop line, light state trên frame

### Deliverable
Chạy trên video mẫu, ghi log từng frame: light_state, tracks đang active.

---

## Milestone 4 — Plate OCR + Evidence (Tuần 7)

### Mục tiêu
Đọc biển số xe vi phạm, lưu đầy đủ bằng chứng.

### Tasks
- [ ] Plate detector (YOLO pretrained hoặc fine-tune đơn giản)
- [ ] PaddleOCR integration
- [ ] Temporal OCR fusion — đọc nhiều frame → vote kết quả tốt nhất
- [ ] Evidence builder: ảnh full-frame, crop plate, clip 3–8s
- [ ] Lưu event JSON với đầy đủ metadata

### Deliverable
`outputs/events/EVT_0001/`: `frame.jpg`, `plate.jpg`, `clip.mp4`, `event.json`

---

## Milestone 5 — Evaluation + Polish (Tuần 8)

### Mục tiêu
Đo metrics thực tế, hoàn thiện portfolio.

### Tasks
- [ ] Tính Event Precision, Recall, Plate Accuracy
- [ ] Tối ưu FPS (bỏ qua frame không cần thiết)
- [ ] Viết README.md đẹp với demo GIF/video
- [ ] Thêm type hints, docstrings, comments
- [ ] Đẩy lên GitHub, verify git clone hoạt động

### Deliverable
GitHub repo sạch, README có demo, code có thể chạy được.

---

## Timeline Overview

| Tuần | Milestone | Trọng tâm |
|------|-----------|-----------|
| 1–2 | Milestone 1 | Skeleton, I/O, environment |
| 3–4 | Milestone 2 | Detection + Tracking |
| 5–6 | Milestone 3 | Traffic Light + Stop Line |
| 7 | Milestone 4 | Plate OCR + Evidence |
| 8 | Milestone 5 | Evaluation + Polish |

---

## Progress Log

| Ngày | Ghi chú |
|------|---------|
| 2026-04-15 | Tạo repo, viết Plan.md, scaffold src/, tạo .gitignore, MILESTONES.md |
