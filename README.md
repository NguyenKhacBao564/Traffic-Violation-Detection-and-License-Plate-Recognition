# Traffic Violation Detection & License Plate Recognition
# From Fixed-Camera Video

**Short:** traffic-violation-lpr

---

## Mục tiêu

Xây dựng hệ thống xử lý video từ camera giao thông cố định:
- Phát hiện ô tô vượt đèn đỏ / vượt vạch dừng
- Đọc biển số xe vi phạm
- Lưu bằng chứng (ảnh, clip, JSON metadata)

MVP chỉ hỗ trợ: ô tô, 1 camera, video ban ngày, offline.

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Vehicle Detection | YOLO (ultralytics) |
| Tracking | ByteTrack |
| Plate Detection | YOLO custom |
| OCR | PaddleOCR |
| Video I/O | OpenCV |
| Storage | JSON (v1) |

---

## Architecture

```
Input Video
  └─ Frame Reader
       └─ Vehicle Detector (YOLO)
            └─ Vehicle Tracker (ByteTrack)
                 └─ Traffic Light State Estimator
                      └─ Stop Line Crossing Logic
                           └─ Violation Event Trigger
                                └─ Plate Detector + OCR
                                     └─ Evidence Builder
                                          └─ Event Store (JSON)
```

Chi tiết: xem [Plan.md](./Plan.md)

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/NguyenKhacBao564/Traffic-Violation-Detection-and-License-Plate-Recognition.git
cd traffic-violation-lpr

# 2. Tạo môi trường
conda create -n lpr python=3.10 -y
conda activate lpr
pip install -r requirements.txt

# 3. Cài model weights (tự động tải khi chạy đầu tiên)
# YOLO sẽ tự tải yolov8n.pt khi cần

# 4. Chạy
python src/main.py --video data/raw_videos/sample.mp4 --camera configs/cameras/cam_01.json

# 5. Output
# outputs/events/EVT_XXXX/frame.jpg  ← ảnh bằng chứng
# outputs/events/EVT_XXXX/clip.mp4   ← clip bằng chứng
# outputs/events/EVT_XXXX/event.json  ← metadata
```

---

## Project Structure

```
traffic-violation-lpr/
├── configs/          # Scene config JSON
├── data/             # Raw videos, annotations
├── models/           # Model weights (gitignored)
├── src/
│   ├── io/          # Video reader/writer
│   ├── detection/   # Vehicle detector
│   ├── tracking/     # Vehicle tracker
│   ├── traffic_light/# Light state estimator
│   ├── geometry/     # Stop line logic
│   ├── violation/   # Event trigger
│   ├── plate/       # Plate detector + OCR
│   ├── evidence/    # Evidence builder
│   └── storage/     # Event store
├── outputs/         # Debug videos, events (gitignored)
├── notebooks/       # Jupyter notebooks
├── requirements.txt
└── README.md
```

---

## Milestones

Xem [MILESTONES.md](./MILESTONES.md) để theo dõi tiến độ.

---

## Success Metrics (MVP v1)

| Metric | Target |
|--------|--------|
| Event Precision | > 80% |
| Event Recall | > 70% |
| Plate Accuracy | > 85% |
| Processing FPS | >= 15 |

---

## License

MIT
