# Traffic Violation Detection & License Plate Recognition

## Tài liệu hướng dẫn chi tiết dự án

> **Mục tiêu:** Giúp bạn hiểu rõ dự án đang giải quyết bài toán gì, pipeline xử lý video hoạt động ra sao, vì sao chọn các công nghệ hiện tại, và cần hoàn thiện từng bước nào để biến repo thành project chính đưa vào CV.

---

## Trạng thái hiện tại

Dự án đã hoàn thành baseline cho tất cả 5 Layer trên video `cam_01_clip_001.mp4`.

- **Layer 1 ✅** — 3 clip 60s, reference frames, camera config.
- **Layer 2 ✅** — YOLO detection + IoU tracker + debug video.
- **Layer 3 ✅** — 4 pending red-light crossing events, report, annotation.
- **Layer 4 ✅** — Fine-tuned YOLO plate detector, 5 events, 4 OCR texts, ~19.5 FPS.
- **Layer 5 ✅** — Evaluation script, review template, demo đã blur. Event Precision 5/5.

---

## 1. Tổng quan dự án

### 1.1. Bài toán cần giải quyết

Dự án xây dựng hệ thống xử lý video từ camera giao thông cố định với ba nhiệm vụ:

1. **Phát hiện xe ô tô** trong video.
2. **Xác định xe có vượt vạch dừng khi đèn đỏ** hay không.
3. **Đọc biển số xe vi phạm** và lưu bằng chứng (ảnh, clip, metadata).

Nói đơn giản: hệ thống xem từng frame video ngã tư, tìm xe, theo dõi xe qua thời gian, kiểm tra đèn giao thông, kiểm tra xe có cắt qua vạch dừng lúc đèn đỏ không, rồi crop biển số và lưu bằng chứng.

### 1.2. Phạm vi MVP

| Hạng mục | Phạm vi |
|---|---|
| Camera | 1 camera cố định |
| Video | Offline, ban ngày |
| Đối tượng | Ô tô |
| Vi phạm | Vượt đèn đỏ / vượt vạch dừng |
| Đầu ra | Ảnh, clip, JSON metadata |

Không nằm trong MVP: realtime, multi-camera, xe máy, đêm/mưa, hệ thống phạt nguội thật, training model lớn từ đầu.

Đây là scope đúng cho portfolio: chứng minh bạn hiểu pipeline AI end-to-end, không phải làm production system.

---

## 2. Vì sao dự án này hợp để đưa vào CV?

Dự án này mạnh hơn project AI cơ bản vì kết hợp nhiều nhóm kỹ năng:

| Nhóm kỹ năng | Thể hiện trong dự án |
|---|---|
| Computer Vision | Vehicle detection, plate detection, traffic light ROI |
| Multi-object Tracking | Gán `track_id` cho xe qua nhiều frame |
| Xử lý video | Đọc frame, ghi debug video, cắt clip bằng chứng |
| Geometry | Kiểm tra xe cắt qua stop line (Shapely) |
| State machine | Quản lý event `pending → confirmed → dismissed` |
| OCR | Đọc biển số bằng HyperLPR3/PaddleOCR/EasyOCR |
| Data engineering | Lưu event JSON, metadata, folder output |
| Software engineering | Module hoá code, config riêng, unit test |

Phù hợp CV cho: AI Engineer Intern, CV Intern, ML Engineer Intern.

---

## 3. Kiến trúc tổng thể

```text
Input Video (cam_01_clip_001.mp4)
  │
  ▼
Frame Reader ─── đọc từng frame từ video
  │
  ▼
Vehicle Detector (YOLO) ─── phát hiện xe trong frame
  │
  ▼
Vehicle Tracker (ByteTrack/IoU) ─── gán track_id, theo dõi xe
  │
  ▼
Traffic Light Estimator (HSV) ─── xác định đèn đỏ/xanh/vàng
  │
  ▼
Stop Line Crossing Logic ─── kiểm tra xe cắt qua vạch dừng
  │
  ▼
Violation Engine ─── đèn đỏ + cắt vạch → tạo event PENDING
  │
  ▼
Plate Detector + OCR ─── crop biển số, đọc ký tự
  │
  ▼
OCR Fusion ─── vote kết quả OCR nhiều frame → biển số tốt nhất
  │
  ▼
Evidence Builder ─── lưu frame.jpg, plate.jpg, clip.mp4
  │
  ▼
Event Store ─── lưu event.json + events_index.json
```

Đây là **multi-stage pipeline**: chia bài toán thành nhiều bước nhỏ, mỗi bước có input/output rõ ràng. Chia nhỏ giúp debug dễ (biết stage nào sai), test riêng từng phần, và thay thế component mà không ảnh hưởng phần còn lại.

---

## 4. Giải thích từng thành phần

### 4.1. Frame Reader — đọc video thành frame

**Module:** `src/io/video_reader.py`

Video là chuỗi ảnh liên tiếp. Mỗi ảnh = 1 frame. Video 30 FPS = 30 frame/giây.

Frame Reader dùng `cv2.VideoCapture` để mở video, đọc metadata (width, height, FPS), và trả từng frame cho pipeline.

**Lưu ý:** FPS cao → mượt nhưng nặng. Trong AI video có thể skip frame để tối ưu tốc độ.

---

### 4.2. Vehicle Detector — phát hiện xe bằng YOLO

**Module:** `src/detection/vehicle_detector.py`

Trả lời: **Trong frame có xe nào, ở đâu?**

Output: danh sách bounding box `[x1, y1, x2, y2]` + confidence + class (`car`/`bus`/`truck`).

**YOLO** (You Only Look Once): nhìn cả ảnh một lần, dự đoán trực tiếp tất cả bbox + class. Nhanh, phù hợp video.

Dự án dùng `yolov8n.pt` (nano) — nhỏ, nhanh, pretrained trên COCO (80 class). Lọc chỉ lấy class `car` (class ID = 2).

**Lưu ý:** COCO **không có** class `license_plate` → cần model riêng cho plate detection.

---

### 4.3. Vehicle Tracker — theo dõi xe qua frame

**Module:** `src/tracking/vehicle_tracker.py`

Trả lời: **Xe này ở frame hiện tại có phải xe đã thấy ở frame trước?**

```text
Frame 10: car bbox → track_id = 3
Frame 11: car bbox → track_id = 3  (cùng xe)
Frame 13: new car  → track_id = 4  (xe mới)
```

**Vì sao cần tracking?** Vi phạm là sự kiện theo thời gian — cần biết đường đi của xe (trước vạch → sau vạch). Không có tracking = không biết xe di chuyển thế nào. Tracking cũng giúp gom OCR nhiều frame cho cùng xe.

**ByteTrack:** tracker multi-object, liên kết bbox qua frame dựa trên vị trí + confidence + chuyển động. Hiện đang dùng IoU tracker fallback (đơn giản hơn, dễ đổi ID hơn).

---

### 4.4. Traffic Light Estimator — xác định màu đèn

**Module:** `src/traffic_light/light_state.py`

Trả lời: **Đèn đang đỏ, vàng, xanh hay không rõ?**

Cách làm: ROI + HSV threshold (không dùng deep learning).

**ROI** (Region of Interest): vùng ảnh đèn giao thông, đánh dấu sẵn trong config vì camera cố định.

**HSV** (Hue-Saturation-Value): không gian màu dễ lọc hơn BGR. Đếm pixel đỏ/xanh/vàng trong ROI → màu nào nhiều nhất = trạng thái đèn.

Ưu điểm: đơn giản, nhanh, không cần dataset. Nhược điểm: phụ thuộc ROI đúng, dễ sai khi nắng/bóng.

---

### 4.5. Stop Line Crossing — kiểm tra xe cắt vạch

**Module:** `src/geometry/line_crossing.py`

Trả lời: **Xe có đi qua stop line không?**

Lấy **bottom-center** của bbox xe (gần mặt đường nhất) làm đại diện vị trí. So sánh vị trí qua 2 frame:

```text
Frame trước:  bottom-center ở trên vạch (y=534)
Frame hiện tại: bottom-center ở dưới vạch (y=529)
Stop line y ≈ 530 → đường đi cắt qua vạch → crossing!
```

Shapely kiểm tra 2 đoạn thẳng (stop line + movement line) có cắt nhau không.

**Direction:** Trong ảnh, y tăng từ trên xuống. Video hiện tại xe đi lên → `y giảm` → `direction = backward`.

---

### 4.6. Violation Engine — quản lý sự kiện vi phạm

**Module:** `src/violation/violation_engine.py`, `src/violation/event_state.py`

Điều kiện vi phạm: **xe cắt stop line + đèn đỏ + track đủ frame**.

**State machine:**

```text
PENDING ──(OCR thành công)──→ CONFIRMED
PENDING ──(mất track/timeout)──→ DISMISSED
```

| Trạng thái | Ý nghĩa |
|---|---|
| `pending` | Nghi vi phạm, chờ OCR/evidence |
| `confirmed` | Đã có biển số / bằng chứng đủ |
| `dismissed` | Không xác nhận được, loại bỏ |

State machine giúp quản lý rõ ràng: event nào đang chờ, xong, hay bị huỷ.

---

### 4.7. Plate Detector — phát hiện biển số

**Module:** `src/plate/plate_detector.py`

```text
Full frame → crop vehicle bbox → detect plate → crop plate → OCR
```

Cách hoạt động hiện tại:
1. Nếu có weight `models/plate_detector/ccpd_yolov8n_best.pt` → YOLO custom.
2. Nếu không → heuristic crop vùng dưới vehicle bbox.

Fine-tune trên CCPD subset (8,000 ảnh): mAP50 = 0.994.

---

### 4.8. Plate OCR — đọc ký tự biển số

**Module:** `src/plate/plate_ocr.py`

```text
Ảnh biển số → OCR backend → "浙B56061" + confidence
```

Backend: `hyperlpr` (tốt nhất cho biển TQ hiện tại), `paddle`, `easyocr`, `none`.

Khó khăn: biển nhỏ, motion blur, góc nghiêng, ánh sáng, ký tự nhầm (0/O, 1/I, 5/S).

---

### 4.9. OCR Fusion — hợp nhất OCR nhiều frame

**Module:** `src/plate/fusion.py`

Đọc biển số nhiều frame rồi **vote**:

```text
Frame 162: "京B56061"    Frame 163: "浙B56061"
Frame 164: "京554641"    Frame 165: "京556861"
→ Vote: "浙B56061" (confidence cao nhất) → Final text
```

**Temporal fusion** biến OCR ~70% single frame thành ~90%+ trên chuỗi frame.

---

### 4.10. Evidence Builder — lưu bằng chứng

**Module:** `src/evidence/evidence_builder.py`

```text
outputs/events_layer4/EVT_0001/
├── frame.jpg   — ảnh full frame
├── plate.jpg   — crop biển số
├── clip.mp4    — clip ngắn quanh thời điểm vi phạm
└── event.json  — metadata (event_id, track_id, plate_text, light_state...)
```

Cần evidence để: debug khi sai, giải thích cho người, làm demo CV, tính metric.

---

### 4.11. Event Store

**Module:** `src/storage/event_store.py`

Lưu event ra JSON + cập nhật `events_index.json`. MVP dùng JSON, sau có thể mở rộng sang SQLite/PostgreSQL/API.

---

## 5. Công nghệ sử dụng

| Công nghệ | Vai trò |
|---|---|
| **Python** | Ngôn ngữ chính — hệ sinh thái AI/CV mạnh |
| **OpenCV** | Đọc/ghi video, vẽ debug, crop, convert HSV |
| **Ultralytics YOLO** | Detect xe + fine-tune detect biển số |
| **ByteTrack / IoU** | Tracking xe qua frame |
| **HyperLPR3** | OCR biển số Trung Quốc (backend chính hiện tại) |
| **PaddleOCR / EasyOCR** | OCR fallback/mở rộng |
| **Shapely** | Geometry — kiểm tra line crossing |
| **JSON / YAML** | Config + metadata storage |

---

## 6. Cấu trúc thư mục

```text
traffic-violation-lpr/
├── configs/                    — cấu hình pipeline + camera
├── data/                       — video, annotations, processed dataset
├── models/                     — model weights (gitignored)
├── scripts/                    — train, evaluate, demo scripts
├── src/
│   ├── main.py                 — pipeline orchestrator
│   ├── io/                     — đọc/ghi video
│   ├── detection/              — phát hiện xe (YOLO)
│   ├── tracking/               — theo dõi xe
│   ├── traffic_light/          — trạng thái đèn
│   ├── geometry/               — stop line crossing
│   ├── violation/              — event state machine
│   ├── plate/                  — plate detect, OCR, fusion
│   ├── evidence/               — lưu bằng chứng
│   └── storage/                — lưu event JSON
├── tests/                      — unit tests
├── outputs/                    — events, reports, debug video (gitignored)
├── README.md / Plan.md / MILESTONES.md / ...
└── requirements.txt
```

---

## 7. File config quan trọng

### `configs/default.yaml` — tham số pipeline

```yaml
model:
  vehicle_model: yolov8n.pt     # model detect xe
  vehicle_conf: 0.3             # confidence tối thiểu
  vehicle_classes: [2]          # class 2 = car (COCO)
tracking:
  track_thresh: 0.5
  track_buffer: 30              # giữ track bao nhiêu frame khi mất detect
ocr:
  vote_frames: 5                # số frame OCR trước khi fusion
```

### `configs/cameras/cam_01.json` — scene config camera

```json
{
  "camera_id": "CAM_01",
  "stop_line": [[850, 680], [1150, 680]],
  "traffic_light_roi": [[1000, 200], [1150, 200], [1150, 380], [1000, 380]],
  "direction": "backward"
}
```

**Nếu stop line hoặc traffic light ROI sai → cả pipeline sẽ sai** dù model tốt đến đâu. Phải dùng reference frame để click toạ độ thật.

---

## 8. Các Layer triển khai

| Layer | Mục tiêu | Kỹ năng chính |
|---|---|---|
| **1** | Chuẩn bị video, config, reference frames | Data engineering |
| **2** | Detect xe + tracking + debug video | YOLO, tracking, OpenCV |
| **3** | Phát hiện vi phạm (đèn đỏ + cắt vạch) | Geometry, state machine, HSV |
| **4** | OCR biển số + lưu bằng chứng | Plate detection, OCR, fusion |
| **5** | Evaluation, metric, demo, polish | Evaluation, documentation |

Layer 5 quan trọng nhất cho CV: nhà tuyển dụng xem bạn có demo, metric, README rõ không — không chỉ xem dùng YOLO.

---

## 9. Luồng xử lý chi tiết (ví dụ frame 162)

1. **Đọc frame** — frame_idx=162, 1342×1008 pixel.
2. **Detect xe** — bbox=(1107,378,1254,474), conf=0.88, class=car.
3. **Tracking** — track_id=5 (cùng xe từ frame trước).
4. **Đèn giao thông** — light_state=RED, conf=0.60.
5. **Stop line** — bottom-center di chuyển từ y=534 → y=529, cắt qua vạch y≈530 → **crossing detected**.
6. **Tạo event** — EVT_0002, state=PENDING.
7. **Plate OCR** — crop xe → detect plate → HyperLPR3 → "浙B56061".
8. **Fusion** — 5 frame vote → final "浙B56061".
9. **Confirm** — state=CONFIRMED, lưu frame.jpg + plate.jpg + clip.mp4 + event.json.

---

## 10. Metric đánh giá

| Metric | Công thức | Target | Hiện tại |
|---|---|---:|---|
| **Event Precision** | TP/(TP+FP) | >80% | 5/5=100% (theo rule) |
| **Event Recall** | TP/(TP+FN) | >70% | Chưa (cần full GT) |
| **Plate Accuracy** | đúng/tổng readable | >85% | Chưa (crop quá nhỏ) |
| **Processing FPS** | frames/time | ≥15 | ~19.5 |

Không điền số ảo. Event Precision hiện tại chỉ là candidate precision theo rule kỹ thuật.

---

## 11. Rủi ro kỹ thuật

1. **Plate detector cần model riêng** — COCO YOLO không có license plate. Đã fix bằng fine-tune + heuristic fallback.
2. **OCR phụ thuộc chất lượng crop** — crop sai/nhỏ → OCR không cứu được.
3. **HSV dễ sai** nếu ROI sai hoặc ánh sáng bất thường.
4. **Tracking ID switch** → event không confirm hoặc trùng lặp.
5. **Config camera là yếu tố sống còn** — stop line/ROI sai = pipeline sai.

---

## 12. Checklist CV-ready

### Bắt buộc

- [x] Video sample 30-60s + config camera.
- [x] Chạy end-to-end không crash.
- [x] Debug video có bbox, track_id, stop line.
- [x] Ít nhất 1 event output + `event.json` rõ ràng.
- [x] OCR có kết quả thực.
- [x] Demo asset đã blur biển số.
- [ ] Metric đầy đủ (Plate Accuracy, Event Recall).
- [x] `python -m pytest` pass.

### Nên có

- [ ] So sánh trước/sau OCR fusion.
- [ ] Bảng lỗi thường gặp kèm giải thích.
- [ ] Ảnh minh hoạ pipeline trong README.
- [ ] Hướng dẫn annotate stop line/ROI.

---

## 13. Cách kể trong CV

**Tiếng Anh:**

```text
Traffic Violation Detection & License Plate Recognition
- Built an offline fixed-camera CV pipeline using YOLOv8, tracking, and OCR
  to detect red-light violations and read license plates.
- Designed 8-stage pipeline: detection → tracking → light estimation →
  stop-line crossing → state machine → plate detection → OCR fusion → evidence.
- Fine-tuned YOLOv8 plate detector on CCPD subset (mAP50 = 0.994).
- 100% candidate precision, ~19.5 FPS on test video.
```

**Tiếng Việt (phỏng vấn):**

```text
Em xây dựng pipeline xử lý video giao thông từ camera cố định. Hệ thống dùng
YOLO detect xe, tracker gán ID qua frame, ROI+HSV xác định đèn, geometry kiểm
tra cắt vạch. Nếu vi phạm → crop biển số, OCR + fusion nhiều frame, lưu ảnh/
clip/JSON làm bằng chứng.
```

---

## 14. Câu hỏi phỏng vấn

**Q: Vì sao cần tracking?**
Vi phạm là sự kiện theo thời gian — cần biết cùng xe đi từ đâu đến đâu. Tracking cũng giúp gom OCR nhiều frame.

**Q: Vì sao dùng HSV thay vì train model đèn?**
Camera cố định, ROI đánh dấu sẵn, HSV đủ cho MVP ban ngày. Nếu mở rộng → thay bằng classification model.

**Q: Điểm yếu lớn nhất?**
OCR phụ thuộc chất lượng crop. Crop nhỏ/mờ → accuracy thấp. Tracking ID switch cũng là rủi ro.

**Q: Giảm false positive thế nào?**
Tăng confidence threshold, yêu cầu track đủ frame, kiểm tra direction, thêm road ROI, thêm lane-specific rule.

**Q: "Tích hợp model" khác "train model" ở đâu?**
Tích hợp = nối output model A → input model B + logic nghiệp vụ. Train = thu thập data, gán nhãn, train, evaluate. Dự án có cả hai (pipeline + fine-tune plate detector).

---

## 15. Lộ trình học

1. **Python + NumPy** — cơ bản về array, shape ảnh.
2. **OpenCV** — đọc video, vẽ bbox, crop, BGR/HSV.
3. **Object Detection** — bbox, confidence, NMS, YOLO.
4. **Tracking** — track ID, vì sao cần tracking video.
5. **Geometry** — line crossing, Shapely.
6. **OCR** — OCR là gì, fusion nhiều frame.
7. **Evaluation** — precision, recall, cách ghi metric thật.
8. **Software Engineering** — module, config, test, README.

---

## 16. Thuật ngữ

| Thuật ngữ | Giải thích |
|---|---|
| Frame | Một ảnh trong video |
| FPS | Số frame mỗi giây |
| Bounding box | Hình chữ nhật bao quanh vật thể |
| Track ID | Mã định danh vật thể qua video |
| ROI | Vùng ảnh quan tâm |
| HSV | Không gian màu (Hue, Saturation, Value) |
| OCR | Nhận diện chữ trong ảnh |
| Temporal fusion | Hợp nhất kết quả qua nhiều frame |
| State machine | Mô hình trạng thái (pending → confirmed) |
| Precision | Tỷ lệ đúng trong các case hệ thống báo có |
| Recall | Tỷ lệ phát hiện trong các case thực sự có |
| NMS | Loại bỏ bbox trùng lặp |
| mAP | Mean Average Precision — metric đánh giá detection |

---

## 17. Kết luận

Dự án này có nên làm main project CV không? **Có.**

Giá trị lớn nhất không nằm ở việc "dùng YOLO", mà ở việc bạn xây được pipeline nhiều thành phần: detection, tracking, traffic light estimation, geometry, OCR, fusion, evidence, storage và evaluation.

Nếu bạn hiểu và giải thích được từng stage trong tài liệu này, bạn đã có nền tảng rất tốt để trình bày project trong phỏng vấn AI/CV Intern.
