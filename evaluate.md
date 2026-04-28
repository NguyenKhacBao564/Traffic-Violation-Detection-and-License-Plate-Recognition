# Đánh giá dự án (Evaluation Guide)

Tài liệu này hướng dẫn cách đánh giá pipeline và tạo demo cho portfolio.

---

## 1. Tổng quan

Layer 5 biến pipeline từ "chạy được" thành "có bằng chứng đánh giá". Cần phân biệt rõ:

- **Pipeline readiness** — pipeline có tạo đúng output không? (đếm file, đo FPS)
- **Model quality** — kết quả có đúng không? (cần con người review)

---

## 2. Input hiện tại

```text
outputs/events_layer4/          — event folders (EVT_0001..EVT_0005)
outputs/reports/layer4_ocr_report.json
outputs/debug_videos/cam_01_layer4_debug.mp4
```

---

## 3. Lệnh chạy

### Tạo review template + evaluation report

```bash
python scripts/evaluate_layer5.py \
  --events-dir outputs/events_layer4 \
  --layer4-report outputs/reports/layer4_ocr_report.json \
  --review data/annotations/cam_01_layer5_review.json \
  --output outputs/reports/layer5_evaluation_report.json
```

### Tạo demo đã blur biển số

```bash
python scripts/create_layer5_demo_assets.py \
  --events-dir outputs/events_layer4 \
  --contact outputs/reports/layer5_demo_contact_redacted.jpg \
  --video outputs/debug_videos/layer5_demo_redacted.mp4
```

---

## 4. Phân loại metric

### Tính ngay được (tự động)

| Metric | Nguồn | Ý nghĩa |
|---|---|---|
| Evidence frame coverage | Đếm `frame.jpg` | Mỗi event có ảnh bằng chứng? |
| Plate crop coverage | Đếm `plate.jpg` | Mỗi event có crop biển số? |
| Clip coverage | Đếm `clip.mp4` | Mỗi event có video clip? |
| Plate text coverage | Đếm `plate_text` != null | OCR có trả kết quả? |
| Processing FPS | `layer4_ocr_report.json` | Pipeline nhanh bao nhiêu? |

### Cần review thủ công

| Metric | Cách tính | Yêu cầu |
|---|---|---|
| **Event Precision** | TP / (TP + FP) | Xem debug video, xác nhận từng event |
| **Plate Accuracy** | Đúng / tổng readable | Cần `plate_text_gt` từ reviewer |

### Cần full timeline ground truth

| Metric | Cách tính | Yêu cầu |
|---|---|---|
| **Event Recall** | TP / (TP + FN) | Annotate toàn bộ vi phạm trong video |

---

## 5. Kết quả CAM_01 hiện tại

| Metric | Kết quả |
|---|---:|
| Reviewed predicted events | 5/5 |
| Event Precision | 5/5 = 1.0 |
| Plate crops visible | 5/5 |
| Readable manual plate GT | 0/5 |
| Plate Accuracy | pending |
| Processing FPS | ~19.5 |

---

## 6. Lưu ý quan trọng

- **Event Precision** hiện tại đo theo configured stop-line rule: xe cắt vạch khi ROI đèn đỏ. Đây chưa phải metric pháp lý vì chưa model lane-specific right-turn permissions.
- **Plate Accuracy** không report cho đến khi có `plate_text_gt` từ ảnh crop đọc được hoặc dataset biển số riêng.
- **Event Recall** cần full timeline ground truth (annotate toàn bộ video), không chỉ review predicted events.
- Không nên điền số ảo vào báo cáo. Ghi rõ metric nào chưa tính được và lý do.

---

## 7. Các vấn đề kỹ thuật đã phát hiện trong code

Dưới đây là các issue đã xác nhận từ code review (tham khảo `IMPLEMENTATION_PLAN.md` để biết trạng thái fix):

| ID | Mô tả | Mức độ |
|---|---|---|
| P1 | `PlateDetector` mặc định dùng `yolov8n.pt` (COCO) — không detect biển số | **Critical** — đã fix bằng fine-tuned weight |
| P2 | `asdict(event)` serialize Enum thành dict thay vì string | Medium |
| P3 | `evidence_image_path` bị ghi đè từ frame → plate | Medium |
| P4 | `OCRFusion.fuse()` tính `total_frames` sau khi clear readings → luôn = 0 | Medium |
| P5 | `ocr_fusion_cache` khởi tạo lại mỗi frame → fusion không hoạt động | **Critical** — cần move ra ngoài loop |
| D2 | `append_clip_frames` serialize `cv2.VideoWriter` — dead code, sẽ crash nếu gọi | Low |
| D3 | `process_frame` trả `newly_confirmed` luôn rỗng | Medium |
