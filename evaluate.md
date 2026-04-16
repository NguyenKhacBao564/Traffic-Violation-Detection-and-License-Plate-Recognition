# Tech Stack Evaluation — traffic-violation-lpr MVP (v3)
## Tài liệu quyết định dựa trên static review + runtime verification plan

> **Phương pháp:** Đọc code → phân loại mỗi nhận định theo mức độ chắc chắn.
> **Mục tiêu:** Tài liệu đủ chặt để dùng làm nền cho implementation, không overclaim, không underclaim.

---

## 1. VERIFIED FROM CODE
Những gì có thể kết luận chắc từ đọc code trực tiếp.

### 1.1 Plate detector path — chặn baseline

**File:** `src/plate/plate_detector.py` + `src/main.py:146–148`

```python
# plate_detector.py
def __init__(self, model_path: str = "yolov8n.pt", ...):  # ← COCO pretrained

# main.py
plate_detector = PlateDetector(
    conf_threshold=cfg.get("model", {}).get("plate_conf", 0.4),
)   # ← model_path NOT passed → uses COCO default
```

**Fact:** `yolov8n.pt` là COCO pretrained. COCO class IDs không có license plate. `plate_detector.detect()` sẽ luôn trả về list rỗng.

**Impact cụ thể:**
- `plate_crop` trong `main.py:243` luôn là `None`
- `plate_ocr.read_and_filter()` không bao giờ được gọi với crop hợp lệ
- Plate text không bao giờ có trong output
- Metadata `plate_crop` trong evidence không có ý nghĩa thực tế

**Diagnosis:** CHẮC CHẮN. Không cần runtime test để biết đây là vấn đề.

**Recommendation:** Xem Phần 3 (Priority Fixes).

---

### 1.2 Clip writer — dead code path không được gọi

**File:** `src/main.py` + `src/evidence/evidence_builder.py`

**Fact:** `main.py` KHÔNG gọi `append_clip_frames()` ở bất kỳ đâu. Clip frames được buffer trong `clip_buffer` dict (`main.py:228–233`) nhưng KHÔNG bao giờ được ghi ra file.

**Impact:**
- Clip bằng chứng không được tạo trong pipeline hiện tại
- `append_clip_frames()` có bug (writer re-init mỗi call → ghi đè) NHƯNG bug không bao giờ trigger vì function không được gọi
- Chỉ có ảnh `frame.jpg` và `event.json` được lưu

**Diagnosis:** CHẮC CHẮN từ code — nhưng severity giảm đáng kể vì đây là missing feature, không phải silent corruption.

---

### 1.3 `asdict(event)` — schema inconsistency trong evidence path

**File:** `src/evidence/evidence_builder.py:65`

```python
meta = asdict(event)  # event.state is EventState (Enum)
json.dump(meta, ...)  # Enum serializes as nested dict
```

**Fact:** `dataclasses.asdict()` không tự động unwrap Enum thành `.value`. Kết quả: `state` field trong JSON có thể là `{name: "CONFIRMED", value: "confirmed"}` thay vì `"confirmed"`.

**Điểm chưa chắc:** Chưa runtime-verify schema đầu ra chính xác — phụ thuộc Python version và dataclasses behavior. Nhưng code pattern này KHÔNG đảm bảo `state` là string như spec yêu cầu.

**Impact:** Event JSON schema không đồng nhất giữa `evidence_builder.save_event()` và `event_store.save()` (event_store dùng `state.value` đúng ở line 72). Nếu load event từ evidence JSON rồi so sánh với index, `state` field sẽ sai type.

**Diagnosis:** CÓ VẤN ĐỀ — cần serialize tường minh. Severity: MEDIUM (không crash, nhưng schema ambiguous).

---

### 1.4 Evidence image path overwrite

**File:** `src/evidence/evidence_builder.py:56, 62`

```python
cv2.imwrite(str(frame_path), full_frame)
event.evidence_image_path = str(frame_path)   # ← set to frame.jpg

if plate_crop is not None and self.config.save_plate_crop:
    cv2.imwrite(str(plate_path), plate_crop)
    event.evidence_image_path = str(plate_path)  # ← OVERWRITES frame_path
```

**Fact:** Khi `plate_crop` được truyền vào, `evidence_image_path` bị ghi đè từ `frame.jpg` thành `plate.jpg`. Path này sau đó được lưu trong `event.json` metadata. Kết quả: `evidence_image_path` trỏ đến plate crop, không phải full frame — dù spec Topic.md nói rõ là "1 ảnh full frame có bbox xe vi phạm".

**Diagnosis:** CHẮC CHẮN từ code. Severity: LOW (plate_crop hiện tại luôn None, nên không trigger, nhưng logic sai về mặt spec).

---

### 1.5 OCRFusion cache không được clean khi event bị dismiss

**File:** `src/main.py:200–260`

```python
ocr_fusion_cache: dict[int, OCRFusion] = {}  # ← re-init mỗi frame
for event in violation_engine.event_manager.pending_events.values():
    if tid not in ocr_fusion_cache:
        ocr_fusion_cache[tid] = OCRFusion(...)
```

**Fact:** `ocr_fusion_cache` được re-init mỗi frame (`{}`), nên không có accumulation issue. Tuy nhiên: khi `violation_engine.dismiss(tid)` được gọi (track lost), `ocr_fusion_cache` vẫn giữ Fusion object cho tid đó nếu không bị overwritten. Nhưng vì re-init `{}` mỗi frame, items cũ bị discard. ĐÂY KHÔNG PHẢI BUG với cách re-init hiện tại.

**Nhưng có vấn đề tiềm ẩn:** `OCRFusion._readings` được `clear()` chỉ khi `fuse()` được gọi (line 60 `fusion.py`). Nếu event bị dismiss TRƯỚC KHI đủ `vote_frames` readings → `fuse()` không bao giờ được gọi → `_readings` không clear. Với re-init `{}` mỗi frame, Fusion object cũ bị discard. KHÔNG LÀ BUG trong runtime hiện tại.

**Diagnosis:** Design acceptable cho MVP. Ghi chú để theo dõi nếu thấy Fusion objects accumulate.

---

### 1.6 `iter_frames()` skip logic ngược

**File:** `src/io/video_reader.py:63–73`

```python
while True:
    ret, frame = self.cap.read()   # ← đọc frame
    if not ret: break

    for _ in range(skip_frames):    # ← skip SAU khi đọc rồi
        self.cap.grab()

    yield frame_idx, frame          # ← yield frame VỪA đọc (chưa skip)
    frame_idx += 1
```

**Fact:** Frame đầu tiên luôn được yield, không skip. Chỉ skip từ frame thứ 2 trở đi.

**Impact:** Chỉ ảnh hưởng nếu `iter_frames(skip_frames=N)` được dùng (hiện tại `main.py` không dùng `iter_frames`, dùng `read()` trực tiếp).

**Diagnosis:** Design bug tồn tại nhưng không ảnh hưởng pipeline hiện tại. Severity: VERY LOW.

---

### 1.7 Module structure + architecture

**Fact:** Module-based separation, constructor DI, dataclass contracts — đúng hướng và đủ clean cho MVP. Không có cyclic import, không có hidden state ở global.

**Diagnosis:** ĐÚNG. Giữ nguyên.

---

## 2. STRONG HYPOTHESIS — NEEDS RUNTIME TEST

Những nhận định hợp lý từ pattern và general knowledge, nhưng CẦN runtime verification trước khi acting lớn.

### 2.1 ByteTrack wrapper — format tương thích

**Giả thuyết:** ByteTrack return format có thể khác với unpack hiện tại.

**Thực tế đã biết:** ByteTrack standard return = `np.array([[x1,y1,x2,y2,score,cls_id,track_id]])`. Unpack 7 giá trị → CÓ VẺ đúng.

**Không xác minh được:** ByteTrack version nào đang được dùng? Có thêm field nào (vel_x, vel_y, age) không?

**Hành động cần:** Chạy pipeline với video test 10-30s. In `track_id` ra log. Verify cùng 1 xe giữ nguyên track_id qua ≥5 frames liên tiếp.

**→ Quyết định: DEFER. Không đổi gì cho đến khi có runtime evidence.**

---

### 2.2 Ultralytics `model.track()` integration

**Giả thuyết:** Ultralytics `model.track()` tích hợp ByteTrack và có thể thay thế wrapper hiện tại đơn giản hơn.

**Chưa xác minh:** Cần test integration — format output của Ultralytics khác với ByteTrack wrapper hiện tại. Refactor có thể tốn effort.

**→ Quyết định: DEFER. Chỉ xem xét nếu ByteTrack wrapper verified fail.**

---

### 2.3 HSV traffic light accuracy

**Giả thuyết:** HSV threshold cố định có thể fragile trong điều kiện thực tế VN (nắng, flare, đèn LED nhỏ).

**Thực tế:** Code có `_last_state` persistence giúp giảm flicker. Temporal smoothing đơn giản đã có.

**Chưa xác minh:** Threshold `best_score=50` có đủ cho video cụ thể không? Camera placement và ánh sáng ảnh hưởng lớn.

**→ Hành động cần:** Chạy pipeline. Đếm % frames `light_state != UNKNOWN`. Nếu <70% → cần cải thiện. Nếu >80% → HSV approach đủ cho MVP.

---

### 2.4 Vehicle confidence threshold

**Giả thuyết:** `vehicle_conf: 0.3` trong config quá thấp → nhiều false positive.

**Thực tế:** Đây là hyperparameter. 0.3 có thể OK cho video sạch, có thể quá nhiều bbox sai cho video đông xe.

**→ Hành động cần:** Benchmark conf=0.3 vs 0.5 trên cùng video. Dựa trên output thực tế mới điều chỉnh.

---

### 2.5 OCR accuracy for Vietnamese plates

**Giả thuyết:** PaddleOCR `vi_en` đủ tốt cho biển số VN.

**Thực tế:** PaddleOCR là OCR engine phổ biến nhất trong Python ecosystem cho tiếng Việt. `vi_en` lang có pretrained support.

**Chưa xác minh:** Accuracy trên video cụ thể chưa biết. Fusion logic có giúp cải thiện không?

**→ Hành động cần:** Sau khi plate path hoạt động, đánh giá OCR output trên ground truth (nếu có) hoặc manual review.

---

## 3. DESIGN RECOMMENDATIONS
Những điểm không phải bug nhưng nên cải thiện khi có baseline chạy được.

| # | Item | File | Recommendation | Timing |
|---|------|------|----------------|--------|
| DR1 | `plate_crop` always `None` | `main.py:243` | Khi plate path được fix, cần truyền plate crop thực vào `save_event()` thay vì `None` | After plate fix |
| DR2 | Evidence image path overwrite | `evidence_builder.py:56,62` | Lưu `frame.jpg` vào `evidence_image_path`, lưu plate crop vào trường riêng | Cleanup |
| DR3 | OCRFusion `total_frames` semantics | `fusion.py:64` | `total_frames` nên là `len(self._readings)` không phải `len(vote_scores)` | Cleanup |
| DR4 | Clip buffer not written | `main.py:228–233` | Thêm logic ghi `clip_buffer` ra file hoặc tích hợp với `append_clip_frames()` | After baseline |
| DR5 | Config no validation | `main.py:91,92` | Thêm validation cho `cam_cfg` keys (`stop_line`, `traffic_light_roi`, `road_roi`) | Before first run |
| DR6 | HSV threshold hardcoded | `light_state.py:44–55` | Cân nhắc đọc threshold từ config thay vì hardcode | After HSV baseline validated |

---

## 4. DEFERRED DECISIONS
Những thay đổi stack không nên làm ở giai đoạn này.

| # | Suggestion từ v1/v2 | Lý do defer |
|---|---------------------|-------------|
| DD1 | ByteTrack → Ultralytics track mode | Chưa verify wrapper fail. Over-engineer nếu wrapper đúng. |
| DD2 | ByteTrack → BoT-SORT | Tracker SOTA hơn nhưng chưa có baseline để đánh giá ByteTrack đủ hay không. |
| DD3 | HSV → traffic light detector model | Tốn thêm model + latency. Chỉ cần nếu HSV fail >30% frames. |
| DD4 | Custom plate YOLO fine-tune | Trainplate detector từ đầu là v2 task. |
| DD5 | JSON → SQLite | JSON đủ cho <50 events. Chỉ chuyển khi cần multi-video query. |
| DD6 | Homography calibration | Over-engineer cho MVP. |
| DD7 | Batch inference | Tối ưu sớm = premature. |

---

## 5. PRIORITY FIXES

### Tier 1 — Fix trước khi chạy video đầu tiên

| # | Fix | File | Loại | Ghi chú |
|---|-----|------|------|---------|
| P1 | **Plate detector path** | `main.py:146–148` | fix-now | Truyền pretrained plate model path HOẶC dùng PaddleOCR trực tiếp trên vehicle crop |
| P2 | **Config validation** | `main.py:91` | fix-now | Validate `cam_cfg` có đủ keys: `stop_line`, `traffic_light_roi` |
| P3 | **Evidence schema consistency** | `evidence_builder.py:65` | fix-now | Serialize `state.value` tường minh, tách frame_path và plate_path |

### Tier 2 — Runtime verification sau khi baseline chạy

| # | Verify | Method | Pass criteria |
|---|--------|--------|--------------|
| V1 | ByteTrack tracking | In `track_id` mỗi frame trên log | 1 xe = 1 track_id qua ≥5 frames liên tiếp |
| V2 | HSV light state | Đếm % frames `UNKNOWN` | <30% UNKNOWN → OK. >70% → cần cải thiện |
| V3 | Plate path hoạt động | In plate text ra log | Có plate text non-empty trong output |
| V4 | YOLO detection quality | Manual review debug video 30 frames | Không quá 20% false positive |
| V5 | Event JSON schema | Load JSON rồi kiểm tra `state` type | `state` là string `"pending"/"confirmed"` |

---

## 6. RECOMMENDED NEXT STEP

**Immediate (trước khi chạy):**
1. Validate camera config keys có đủ → fail early với clear error message
2. Fix plate detector: thay bằng pretrained plate model HOẶC dùng PaddleOCR trực tiếp trên vehicle crop region (phần dưới của bbox, nơi biển số thường nằm)
3. Fix `asdict()` → serialize `state.value` tường minh
4. Thêm log/instrumentation cho `track_id`, `light_state`, `plate_text` để verify mỗi path

**Sau baseline chạy được:**
5. Đánh giá V1–V5 checklist
6. Dựa trên evidence thực tế → quyết định có cần đổi gì ở Tier 2 không

**Điều quan trọng nhất:** Không thay stack lớn cho đến khi baseline tạo ra output để so sánh.

---

## CHANGE LOG v2 → v3

| Thay đổi | Lý do |
|----------|-------|
| (A) `asdict()` → "schema inconsistency" thay vì "JSON fail" | Giảm overclaim. Mô tả vấn đề đúng hơn: không đảm bảo `state` là string. |
| (B) Bỏ benchmark/performance claims | Không có nguồn trích dẫn. Đổi thành "phổ biến" / "phù hợp MVP". |
| (C) Plate detector → phân tầng 3 path | Không còn "xóa module". Fix minimal → OCR fallback → v2+ train. |
| (D) Bổ sung `iter_frames` bug, evidence path overwrite, DR5–DR6 | Các gap còn sót từ v2. |
| (E) Phân loại rõ 3 tiers: verified / hypothesis / design rec | Mỗi nhận định có label rõ ràng. |
| Rút bớt severity clip writer | Dead code path, không phải silent corruption. Severity giảm. |
| Thêm P3 (config validation) | Blocking issue mới phát hiện — cam config thiếu keys sẽ crash ở line 123/128. |
