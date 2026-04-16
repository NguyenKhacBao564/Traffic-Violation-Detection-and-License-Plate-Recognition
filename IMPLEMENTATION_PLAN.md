# Implementation Plan — traffic-violation-lpr
## Phase: Baseline Runnable

### Mục tiêu của phase này

Biến repo từ **scaffold đúng hướng, chưa validate** thành **baseline chạy được trên 1 video test ngắn (30s–1p)**.

Phạm vi phase:
- Chỉ fix bugs và design issues đã verified từ static review
- Chưa tối ưu model
- Chưa mở rộng scope (nhiều camera, xe máy, v.v.)
- Chưa thay stack lớn nếu chưa có runtime evidence

---

## 1. Goal of this phase

| Mục tiêu | Định nghĩa "xong" |
|-----------|-------------------|
| Baseline chạy end-to-end trên 1 video ngắn | Không crash, tạo output |
| Mỗi pipeline stage có log để verify | Track ID, light state, plate text xuất hiện trong log |
| Tạo debug video có overlay | Visual verification của detection + tracking |
| Tạo event JSON hợp lệ | Schema đúng, `state` là string |
| Biết tracker path có vấn đề không | Có/th không → quyết định ByteTrack tiếp |

Phase này **không** yêu cầu:
- OCR accuracy cao
- Zero false positive
- Đúng hết violation thật
- Clip video hoàn chỉnh

---

## 2. Ordered work items

### Step 1 — Config + data readiness
*Trước khi chạm bất kỳ code pipeline nào*

**Mục tiêu:** Đảm bảo input hợp lệ, có video test, có config đúng.

```
1a. Chuẩn bị 1 video test ngắn (30s–1p)
    → Đặt vào data/raw_videos/sample.mp4
    → Cam kết: camera cố định, thấy rõ stop line, thấy đèn giao thông

1b. Cập nhật configs/cameras/cam_01.json với tọa độ thực
    → Dùng notebook 01 để click xác định stop_line và light ROI trên video
    → Validate: stop_line là 2 điểm, light_roi là 4 điểm

1c. Validate camera config keys
    → Kiểm tra có đủ: camera_id, stop_line, traffic_light_roi, direction
    → Fail early với clear error nếu thiếu (P3 trong evaluate.md)
```

**Đầu ra mong muốn:**
- File video tồn tại tại `data/raw_videos/sample.mp4`
- `configs/cameras/cam_01.json` có coordinates thực cho video đó

**Tiêu chí hoàn thành:** Chạy `python -c "import json; c=json.load(open('configs/cameras/cam_01.json')); assert 'stop_line' in c and 'traffic_light_roi' in c"` → không error.

---

### Step 2 — Plate detector fix
*Fix P1 trong evaluate.md*

**Mục tiêu:** Plate path không còn là no-op. OCR được gọi với input hợp lý.

```
2a. Chọn approach:
    Option A (nhanh nhất): Dùng PaddleOCR trực tiếp trên vehicle crop region
      → Crop: vùng phía dưới của vehicle bbox (y2-40px → y2+20px)
      → Không cần plate detector riêng
      → PaddleOCR.ocr() vừa detect text bbox VỪA recognize

    Option B: Truyền pretrained plate model path vào PlateDetector
      → Tìm pretrained YOLO plate detector (Roboflow / HuggingFace community)
      → Config: plate_detector_model: <path_to_model>

2b. Cập nhật main.py: truyền plate_crop thực vào save_event()
    → Không truyền None nữa
    → Lưu crop vào trường riêng, không overwrite frame_path

2c. Thêm log: print plate text mỗi khi OCR đọc được
    → Track xem OCR path có được gọi không
```

**Đầu ra mong muốn:**
- `plate_crop` trong evidence không còn luôn `None`
- Log có dòng `plate_text: 51H-12345` hoặc `plate_text:` (rỗng = cần điều chỉnh crop region)

**Tiêu chí hoàn thành:** Chạy pipeline 10 frame đầu → kiểm tra log → OCR được gọi ít nhất 1 lần cho 1 vehicle.

---

### Step 3 — Evidence schema fix
*Fix P3 trong evaluate.md*

**Mục tiêu:** Event JSON schema nhất quán. `state` là string.

```
3a. Thay asdict(event) → serialize tường minh
    → state: event.state.value
    → Tất cả Optional fields: None nếu None, không serialize Enum

3b. Tách evidence_image_path và plate_crop_path
    → frame.jpg → evidence_image_path
    → plate crop → plate_crop_path (trường riêng)
    → Không overwrite nữa

3c. Verify: load event.json rồi kiểm tra types
    → state là str, plate_crop_path là str hoặc None
```

**Đầu ra mong muốn:**
- `event.json` load được, `state` là string `"pending"/"confirmed"`
- Cả 2 image path đều tồn tại và khác nhau khi có plate

**Tiêu chí hoàn thành:** `python -c "import json; e=json.load(open('outputs/events/EVT_0001/event.json')); assert isinstance(e['state'], str)"` → không error.

---

### Step 4 — Runtime verification run
*Chạy pipeline với instrumentation*

**Mục tiêu:** Baseline chạy được, verify mỗi stage, biết tracker path có vấn đề không.

```
4a. Chạy pipeline: python src/main.py --video data/raw_videos/sample.mp4 \
    --camera configs/cameras/cam_01.json --max-frames 300 --no-debug-video

4b. In thêm instrumentation:
    - Mỗi frame: log track_id count
    - Mỗi 30 frame: log light_state hiện tại
    - Mỗi OCR read: log plate_text + conf
    - Mỗi event confirm: log event_id + track_id

4c. Review output:
    - Debug video có bbox và track_id overlay
    - Log có track_ids ổn định?
    - Log có light_state không toàn UNKNOWN?
    - Có plate text đọc được?
    - Có event nào confirmed?
```

**Tiêu chí hoàn thành:** Xem Phase 5 (Acceptance Criteria).

---

### Step 5 — Small cleanup sau baseline
*Sau khi baseline chạy được*

```
5a. OCRFusion total_frames semantics (DR3 trong evaluate.md)
    → total_frames = len(self._readings) thay vì len(vote_scores)

5b. Clip buffer — quyết định có cần fix không
    → Nếu event_clip_path luôn rỗng và đây là requirement → fix append_clip_frames
    → Nếu không cần clip cho demo → defer

5c. DR5: Thêm config validation rõ ràng cho stop_line / light_roi coords
    → (Đã làm ở Step 1b nhưng có thể cần defensive check trong code)
```

---

## 3. File-by-file patch plan

| File | Vấn đề | Loại việc | Ưu tiên | Ghi chú |
|------|---------|-----------|---------|---------|
| `configs/cameras/cam_01.json` | Coordinates chưa có giá trị thực | verify-first | Step 1 | Cần xác định từ video thực |
| `src/main.py:146–148` | PlateDetector không nhận model path → dùng COCO | fix-now | P1 | Truyền pretrained model HOẶC dùng OCR trực tiếp |
| `src/main.py:91` | cam_cfg không validate → crash ở line 123/128 | fix-now | P3 | Thêm assert/check cho required keys |
| `src/evidence/evidence_builder.py:65` | asdict(event) → state không phải string | fix-now | P3 | Serialize tường minh: state.value |
| `src/evidence/evidence_builder.py:56,62` | plate_path overwrite frame_path | fix-now | P3 | Tách 2 trường |
| `src/main.py:200–260` | Plate crop always None (phụ thuộc P1) | fix-now | P1 | Truyền crop thực sau khi P1 xong |
| `src/main.py:228–233` | clip_buffer không được ghi | verify-first | After baseline | Chỉ fix nếu clip là requirement |
| `src/plate/fusion.py:64` | total_frames = len(vote_scores) | cleanup-later | Step 5 | Không ảnh hưởng MVP baseline |
| `src/io/video_reader.py:63–73` | iter_frames skip logic ngược | cleanup-later | Deferred | Không ảnh hưởng main pipeline |
| `src/traffic_light/light_state.py` | HSV threshold hardcoded | verify-first | Step 4 | Điều chỉnh sau baseline nếu cần |

---

## 4. Runtime verification checklist

### 4.1 Tracker verification
```
Cần quan sát:
  - Mở debug video → cùng 1 xe đi từ đầu đến cuối → cùng 1 màu / cùng 1 track_id
  - Track ID không nhảy liên tục (bình thường nhảy 1-2 lần khi occlude)
  - Track ID không đổi khi xe chưa ra khỏi frame

Thế nào là PASS:
  - ≥80% vehicles giữ nguyên track_id từ frame xuất hiện đến khi ra khỏi frame
  - Không có track ID > 50 (quá nhiều xe cùng lúc = bình thường với giao thông đông)

Thế nào là FAIL:
  - 1 xe bất kỳ nhảy track_id > 3 lần trong 10 frame đầu
  → Cần investigate ByteTrack format
```

### 4.2 Light state verification
```
Cần quan sát:
  - Log mỗi 30 frame: LIGHT: red/yellow/green/unknown

Thế nào là PASS:
  - ≥70% frames có light_state != UNKNOWN
  - Light state thay đổi hợp lý (red → yellow → green → red) nếu video đủ dài

Thế nào là FAIL:
  - >50% frames là UNKNOWN → HSV thresholds cần điều chỉnh
  - Luôn cùng 1 màu → có thể threshold quá cao hoặc ROI sai
```

### 4.3 Plate path verification
```
Cần quan sát:
  - Log OCR reads: "plate_text: XX-XX.XXXXX" hoặc "plate_text: (blank)"

Thế nào là PASS:
  - Plate text non-empty xuất hiện ít nhất 1 lần trên video 30s–1p
  - Format hợp lý (2-3 ký tự + dash + 4-5 số)

Thế nào là FAIL:
  - 100% plate_text blank → crop region sai HOẶC plate detector path chưa fix đúng
  → Verify P1 đã được fix trước
```

### 4.4 Event detection
```
Cần quan sát:
  - Video có xe vượt đèn đỏ thật không?
  - Nếu không có → không có event là bình thường

Thế nào là PASS:
  - Nếu video có ≥1 violation thật:
    → Có ≥1 event confirmed trong log
  - Nếu video không có violation:
    → Có log pending events (nếu xe đi qua stop line khi đỏ)

Thế nào là FAIL:
  - Xe vượt đèn đỏ thật nhưng không có event confirmed
  → Violation engine hoặc tracker có vấn đề
```

### 4.5 Event JSON schema
```
Cần kiểm tra:
  - Mở outputs/events/EVT_0001/event.json
  - Load bằng Python và kiểm tra types

PASS criteria:
  - state: str ("pending" hoặc "confirmed")
  - plate_text: str hoặc None
  - evidence_image_path: str (file tồn tại)
  - track_id: int
```

---

## 5. Acceptance criteria

Phase này **hoàn thành** khi TẤT CẢ điều sau đúng:

| # | Criteria | Method |
|---|----------|--------|
| AC1 | Pipeline chạy không crash trên video 30s | `python src/main.py --max-frames 900` → exit code 0 |
| AC2 | Debug video tạo ra với overlay bbox + track_id | Mở file output bằng VLC/image viewer |
| AC3 | Log có output từ mỗi stage | Log chứa: detections count, track_id, light_state, plate_text |
| AC4 | Event JSON có schema đúng | Load JSON, `isinstance(state, str)` |
| AC5 | ByteTrack verified: track IDs ổn định | Manual review 30s debug video |
| AC6 | Plate path verified: OCR được gọi | Log chứa plate_text non-empty HOẶC có proof OCR path chạy |
| AC7 | Biết tracker path có cần đổi không | V1 checklist → có hoặc không |

**Điều kiện DỪNG:** Nếu AC1 fail (crash) → quay lại Step 1, kiểm tra video + config.

---

## 6. Risks and decision gates

### Decision Gate 1: Tracker quality
```
IF V1 (tracker verification) FAIL:
  → Ngừng tối ưu các stage khác
  → Investigate: ByteTrack format? FPS mismatch? Video quality?
  → Option A: Fix ByteTrack wrapper (nếu format đúng → bug ở integration)
  → Option B: Thử Ultralytics model.track() (nếu Option A không rõ nguyên nhân)
  → Option C: IoU-based tracker thủ công (đơn giản hơn, fallback)

IF V1 PASS:
  → Giữ ByteTrack wrapper. Tiếp tục.
```

### Decision Gate 2: Light state quality
```
IF light state UNKNOWN > 50% frames:
  → Điều chỉnh HSV thresholds cho video cụ thể
  → Hoặc dùng manual annotation của light state sequence (nếu cần)
  → KHÔNG train ML model cho light state ở phase này

IF light state UNKNOWN < 30%:
  → HSV approach đủ. Tiếp tục.
```

### Decision Gate 3: Plate path
```
IF plate text blank 100% frames sau khi P1 fixed:
  → Verify crop region
  → Thử PaddleOCR trực tiếp trên toàn bộ vehicle bbox (không chỉ vùng dưới)
  → Thử pretrained plate detector model

IF plate text xuất hiện:
  → OCR path hoạt động. Tiếp tục.
```

---

## 7. What NOT to do yet

Danh sách việc **CHƯA NÊN LÀM** trong phase này:

| # | Việc chưa làm | Lý do |
|---|---------------|-------|
| N1 | Train YOLO / fine-tune plate detector | Chưa có baseline để đánh giá pretrained đủ hay không |
| N2 | Thay ByteTrack → BoT-SORT | Chưa verify ByteTrack fail |
| N3 | Thay HSV → ML traffic light detector | HSV có thể đủ. Chỉ thay khi <50% frames có light state |
| N4 | Thêm nhiều camera / multi-camera support | MVP chỉ 1 camera |
| N5 | Tối ưu FPS / batch inference | Chưa baseline → không có baseline FPS để so sánh |
| N6 | Thêm xe máy vào scope | MVP chỉ ô tô |
| N7 | Thêm real-time streaming | MVP offline |
| N8 | Chuyển JSON → SQLite | JSON đủ cho baseline |
| N9 | Homography / camera calibration | Over-engineer |
| N10 | Refactor lớn module structure | Cấu trúc hiện tại đủ tốt. Không refactor khi chưa validate |
