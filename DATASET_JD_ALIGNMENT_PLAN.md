# Dataset Roadmap Khớp JD Gameloft AI Engineer Intern

## 1. Mục tiêu của tài liệu

Tài liệu này dùng để biến phần dataset của project **Traffic Violation Detection & License Plate Recognition** thành một câu chuyện CV/phỏng vấn phù hợp với JD **AI Engineer Intern (Game Development) - Gameloft Vietnam**.

Điểm quan trọng: JD Gameloft không chỉ tìm người "train model". Vị trí này nhấn mạnh năng lực **dataset engineering**:

- Tổ chức, làm sạch và cấu trúc dataset ảnh/video lớn.
- Quản lý naming, metadata, duplicate removal và lọc mẫu kém chất lượng.
- Annotation chính xác, theo labeling standard, đảm bảo consistency.
- Theo dõi progress, coverage, versioning và changelog.
- Quality check liên tục để tăng độ tin cậy của dataset.
- Phối hợp với AI engineer để cải thiện dữ liệu dựa trên model performance và edge cases.
- Hiểu vấn đề thực tế như lighting changes, occlusion, motion blur, reflections.
- Biết dùng Python cho validation, formatting, reporting.

Vì vậy, hướng cải thiện project không phải là nói quá rằng "đã có model hoàn hảo", mà là trình bày rõ:

```text
Tôi đã xây một workflow xử lý dataset cho bài toán license plate detection:
organize -> clean -> annotate -> split -> manifest -> train -> evaluate -> QA -> improve.
```

Đây là hướng khớp với JD hơn nhiều so với chỉ nói "tôi fine-tune YOLO".

## 2. Repo Và Trạng Thái Dataset Hiện Tại

Đường dẫn repo đúng:

```text
/Users/nguyen_bao/Projects/AIproject/AI_Project/traffic-violation-lpr
```

Dataset train chính hiện tại:

```text
data/processed/ccpd_layer4
```

Dung lượng hiện tại:

| Thành phần | Dung lượng |
|---|---:|
| `data/` | khoảng 1.6GB |
| `data/processed/` | khoảng 780MB |
| `data/processed/ccpd_layer4/` | khoảng 780MB |

Quy mô dataset biển số đã xử lý:

| Split | Images | YOLO labels | OCR crops |
|---|---:|---:|---:|
| Train | 6,398 | 6,398 | 6,000 |
| Val | 1,100 | 1,100 | 1,000 |
| Test | 1,100 | 1,100 | 1,000 |
| Tổng | 8,598 | 8,598 | 8,000 |

Ghi chú: số OCR crops (8,000) ít hơn số ảnh (8,598) vì crop chỉ được sinh cho **positive samples** có biển số. 598 ảnh `ccpd_np` (negative) không có bbox nên không có crop. Tổng positive 8,000 = 6,000 train + 1,000 val + 1,000 test.

Thông tin từ `manifest.json`:

| Trường | Giá trị |
|---|---|
| Dataset | `CCPD2019 Layer 4 compact subset` |
| `record_count` | 8,598 |
| `seed` | 42 |
| Positive samples | 8,000 |
| Negative samples | 598 |
| Warnings | 0 |

Phân bổ nguồn trong manifest:

| Source | Số lượng | Ý nghĩa |
|---|---:|---|
| `ccpd_base` | 5,400 | Ảnh biển số cơ bản |
| `ccpd_blur` | 700 | Ảnh có blur |
| `ccpd_rotate` | 660 | Ảnh biển số xoay |
| `ccpd_tilt` | 660 | Ảnh nghiêng |
| `ccpd_weather` | 380 | Ảnh điều kiện thời tiết |
| `ccpd_challenge` | 200 | Ảnh khó |
| `ccpd_np` | 598 | Negative samples, không có biển số |

Mỗi record trong manifest hiện có các trường quan trọng:

```text
split, kind, source, original_path, image, label, ocr_crop,
plate_text, bbox_xyxy, crop_xyxy, image_size, brightness, blurriness
```

Đây là điểm rất có giá trị khi nói chuyện với nhà tuyển dụng, vì nó chứng minh project không chỉ có ảnh rời rạc mà đã có metadata để audit và QA.

## 3. JD Gameloft Và Cách Map Vào Project

### 3.1. Yêu cầu chính trong JD

| JD Gameloft | Cách hiểu thực tế |
|---|---|
| Build high-quality computer vision dataset at scale | Dataset phải có cấu trúc, có QA, có thể mở rộng |
| Around 1,000 real-world objects | Cần quản lý nhiều object/class, coverage và consistency |
| Diverse real-world conditions | Dữ liệu phải chịu được ánh sáng, che khuất, blur, reflection |
| Organize, clean, structure image/video datasets | Folder, naming, split, manifest, metadata phải rõ |
| Removing duplicates, filtering low-quality samples | Cần script phát hiện trùng, ảnh lỗi, ảnh quá mờ/tối/sáng |
| Annotate accurately and follow labeling standards | Label phải đúng format, bbox/mask/class không lỗi |
| Track dataset progress, coverage, versioning | Cần changelog, version dataset, coverage report |
| Perform quality checks | Cần audit report, contact sheet, QA checklist |
| Refine data based on model performance and edge cases | False positive/false negative phải quay lại cải thiện dataset |

Toàn văn JD gốc được lưu trong **Phụ lục A** ở cuối tài liệu để tham khảo.

### 3.2. Điểm project đã khớp JD

| Năng lực trong JD | Project hiện tại đã có |
|---|---|
| Dataset organization | `images/train`, `images/val`, `images/test`, `labels/*`, `ocr_crops/*` |
| Annotation format | YOLO bbox label cho class `plate` |
| Train/val/test split | 6,398 / 1,100 / 1,100 |
| Metadata | `manifest.json` có source, bbox, crop, brightness, blurriness |
| Negative samples | 598 ảnh `ccpd_np` không có biển số |
| Edge-case sources | blur, rotate, tilt, weather, challenge |
| Python scripting | `scripts/prepare_ccpd_layer4.py`, `scripts/train_plate_detector.py`, `scripts/evaluate_layer5.py` |
| Evidence review | `scripts/create_layer5_demo_assets.py` + contact sheet/event evidence report |
| Dataset cleanup | Đã dọn các dataset lớn không dùng, giữ dataset cần cho project |

### 3.3. Điểm chưa nên claim quá mức

Không nên nói:

```text
Tôi đã xây dataset 1,000 object giống Gameloft.
```

Vì project hiện tại chỉ train dataset biển số xe, không phải dataset nhiều object đồ vật.

Không nên nói:

```text
Tôi đã xử lý classification và segmentation dataset.
```

Vì project hiện tại chủ yếu là detection bbox cho biển số. OCR dùng pretrained engine, không phải tự train classifier ký tự hay segmentation mask.

Cách nói đúng hơn:

```text
I built a license plate dataset processing workflow that includes
format conversion, YOLO annotation generation, train/val/test split,
metadata manifest, negative samples, OCR crops, and QA-oriented reports.
```

## 4. Dataset Nào Được Train, Phần Nào Không Train?

Đây là phần rất quan trọng để tránh hiểu nhầm khi phỏng vấn.

| Module | Có train bằng dataset local không? | Cách project đang làm |
|---|---|---|
| Vehicle detection | Không | Dùng YOLO pretrained trên COCO |
| Vehicle tracking | Không | Dùng tracker logic, ByteTrack hoặc IoU fallback |
| Traffic light state | Không | ROI + HSV threshold |
| Stop line | Không | Cấu hình thủ công trong camera config |
| Violation logic | Không | Rule-based geometry + state machine |
| Plate detection | Có | Fine-tune YOLO trên CCPD compact subset |
| Plate OCR | Không | Dùng HyperLPR3 pretrained cho biển số Trung Quốc |
| Layer 5 evaluation | Không | Manual review + report |

Kết luận:

```text
Dataset train chính của project hiện tại là dataset biển số xe.
Các phần còn lại là pretrained model hoặc rule-based pipeline.
```

Đây không phải điểm yếu. Với một project portfolio, cách này hợp lý vì nó cho thấy bạn biết ghép các module AI/thị giác máy tính thành pipeline end-to-end thay vì cố train mọi thứ từ đầu.

## 5. Những Phương Pháp Xử Lý Dataset Đã Dùng

### 5.1. Compact subset

Thay vì giữ toàn bộ dataset rất lớn, project dùng một subset gọn hơn để phù hợp máy local:

- Đủ lớn để fine-tune plate detector.
- Đủ nhỏ để lưu trong project sau khi dọn máy.
- Có train/val/test rõ ràng.
- Có nhiều source khó như blur, rotate, tilt, weather, challenge.

Điểm có thể nói trong phỏng vấn:

```text
I selected a compact but diverse subset instead of blindly keeping the full dataset,
because local storage and iteration speed matter in practical dataset workflows.
```

### 5.2. Convert annotation sang YOLO format

Project đã chuyển thông tin bbox biển số từ format gốc của CCPD sang YOLO label:

```text
class_id x_center y_center width height
```

Ý nghĩa:

- YOLO cần label normalized theo kích thước ảnh.
- Bbox phải đúng class `plate`.
- Mỗi ảnh có label tương ứng trong cùng split.

### 5.3. OCR crop generation

Ngoài ảnh gốc và bbox, project còn sinh crop biển số:

```text
ocr_crops/train
ocr_crops/val
ocr_crops/test
```

Ý nghĩa:

- Dùng để kiểm tra khả năng đọc biển số.
- Dùng để debug xem plate detector crop có đủ tốt không.
- Giúp tách lỗi detector và lỗi OCR:
  - Nếu crop sai: lỗi plate detection hoặc crop logic.
  - Nếu crop đúng nhưng text sai: lỗi OCR hoặc ảnh quá mờ/nhỏ.

### 5.4. Negative samples

Dataset có 598 negative samples từ `ccpd_np`.

Ý nghĩa:

- Giúp detector học rằng không phải vùng nào cũng là biển số.
- Giảm false positive.
- Đây là điểm khớp JD vì dataset tốt không chỉ có positive samples đẹp.

### 5.5. Metadata manifest

`manifest.json` là phần quan trọng nhất để kể chuyện dataset engineering.

Nó giúp trả lời:

- Ảnh thuộc split nào?
- Là positive hay negative?
- Đến từ source nào?
- Label nằm ở đâu?
- Crop OCR nằm ở đâu?
- Bbox gốc là gì?
- Ảnh sáng trung bình thế nào?
- Độ blur ra sao?

Nếu đi phỏng vấn, đây là bằng chứng rằng bạn hiểu dataset không chỉ là một folder ảnh.

## 6. Gap Analysis So Với JD Gameloft

| Hạng mục | Hiện trạng | Mức độ cần cải thiện |
|---|---|---|
| Naming convention | Đã có `train_000001.jpg` | Tốt |
| Train/val/test split | Đã có | Tốt |
| YOLO label consistency | Đã có label 1-1 với ảnh | Tốt |
| Manifest | Đã có metadata cơ bản + `manifest_quality.json` | Tốt |
| Duplicate detection | Đã có duplicate report | Cần review thủ công trước khi xóa |
| Low-quality filtering | Đã có brightness/blurriness threshold report | Tốt |
| Edge-case coverage | Đã có coverage report theo source/quality flag | Tốt |
| Contact sheet theo lỗi | Đã có contact sheet theo dark/blur/small/weather/challenge/negative | Tốt |
| Annotation QA checklist | Chưa có checklist chuẩn | Cần thêm |
| Dataset changelog/versioning | Đã có `docs/DATASET_CHANGELOG.md` | Tốt |
| Model-error feedback loop | Có evaluation, nhưng chưa gom lỗi thành dataset queue | Cần thêm |
| Classification dataset | Chưa có | Không claim |
| Segmentation dataset | Chưa có | Không claim |

Nhận xét:

```text
Project hiện đã đủ để nói về dataset processing cho object detection.
Để khớp JD Gameloft hơn, cần bổ sung lớp QA/report/versioning thay vì chỉ train thêm model.
```

## 7. Roadmap Hoàn Thiện Dataset Cho CV Và Phỏng Vấn

### Bước 1: Tạo dataset audit script

File đề xuất:

```text
scripts/audit_plate_dataset.py
```

Nhiệm vụ:

- Đếm ảnh/label/crop theo split.
- Kiểm tra ảnh thiếu label.
- Kiểm tra label thiếu ảnh.
- Kiểm tra file hỏng.
- Kiểm tra bbox YOLO nằm ngoài `[0, 1]`.
- Kiểm tra bbox quá nhỏ hoặc quá lớn bất thường.
- Kiểm tra positive/negative đúng logic.

Output:

```text
outputs/reports/dataset_audit_report.json
outputs/reports/dataset_audit_report.md
```

### Bước 2: Mở rộng metadata cho từng ảnh

Thêm các field có ích cho QA:

```text
brightness_level: dark / normal / bright
blur_level: sharp / mild_blur / heavy_blur
plate_area_ratio
bbox_valid
has_ocr_crop
quality_flags
```

Ví dụ:

```json
{
  "image": "images/train/train_000001.jpg",
  "split": "train",
  "source": "ccpd_blur",
  "kind": "positive",
  "brightness": 146,
  "blurriness": 76,
  "brightness_level": "normal",
  "blur_level": "mild_blur",
  "quality_flags": ["small_plate"]
}
```

Ý nghĩa với JD:

```text
Filtering low-quality samples, metadata tracking, dataset reliability.
```

### Bước 3: Duplicate detection

Thêm kiểm tra ảnh trùng hoặc gần trùng:

- Hash file để tìm exact duplicate.
- Perceptual hash để tìm ảnh gần giống.
- Report duplicate groups.

Output đề xuất:

```text
outputs/reports/dataset_duplicates.json
outputs/reports/dataset_duplicates.md
```

Điểm phỏng vấn:

```text
I added duplicate checks because repeated images can inflate validation metrics
and make the model look better than it really is.
```

### Bước 4: Contact sheet theo nhóm lỗi

Tạo các contact sheet riêng:

```text
outputs/reports/dataset_contact_dark.jpg
outputs/reports/dataset_contact_blur.jpg
outputs/reports/dataset_contact_small_plate.jpg
outputs/reports/dataset_contact_weather.jpg
outputs/reports/dataset_contact_negative.jpg
```

Ý nghĩa:

- Review nhanh bằng mắt.
- Phát hiện label lệch.
- Phát hiện crop biển số quá sát hoặc thiếu ký tự.
- Phù hợp với công việc annotation QA trong JD.

### Bước 5: Dataset versioning và changelog

Tạo file:

```text
docs/DATASET_CHANGELOG.md
```

Ví dụ version:

| Version | Nội dung | Ghi chú |
|---|---|---|
| `v0.1` | Raw CCPD subset | Chưa audit |
| `v0.2` | YOLO conversion + split | Dùng train detector |
| `v0.3` | Add manifest + OCR crops | Dùng cho Layer 4 |
| `v0.4` | Add QA flags + duplicate report | Khớp JD dataset QA |

Điểm này rất hợp JD vì Gameloft nhắc rõ **progress, coverage, versioning, changelog**.

### Bước 6: Augmentation mô phỏng điều kiện thực tế

Không nên chỉ nói "trời tối, trời mưa làm model tệ hơn". Nên biến nó thành pipeline cụ thể:

| Điều kiện thực tế | Augmentation đề xuất | Mục đích |
|---|---|---|
| Trời tối | brightness/contrast giảm | Tăng robust low-light |
| Chói đèn/phản chiếu | glare overlay | Mô phỏng reflection |
| Mưa | rain-like streak + blur nhẹ | Mô phỏng camera giao thông ngoài trời |
| Xe chạy nhanh | motion blur | Mô phỏng tốc độ cao |
| Camera nén mạnh | JPEG compression | Mô phỏng CCTV/video platform |
| Nhiễu hình | Gaussian noise | Mô phỏng sensor noise |

Lưu ý:

```text
Augmentation không thay thế dữ liệu thật.
Nó chỉ giúp tạo thêm case khó và kiểm tra độ ổn định của model.
```

### Bước 7: Model-error feedback loop

Sau khi chạy pipeline trên video thật, gom lỗi thành nhóm:

| Loại lỗi | Nguyên nhân có thể | Cách đưa về dataset |
|---|---|---|
| Không detect biển số | Plate quá nhỏ/mờ/tối | Thêm vào hard samples |
| Detect sai vùng | Reflection hoặc vật giống biển số | Thêm negative samples |
| OCR sai text | Crop quá nhỏ/mờ | Tăng quality check crop |
| Event có xe nhưng plate crop sai | Vehicle bbox không chứa rõ plate | Điều chỉnh crop strategy |
| Đêm/mưa thất bại | Domain shift | Tag low-light/weather và test riêng |

Cách nói trong phỏng vấn:

```text
I used model failures as signals to improve the dataset, instead of treating
evaluation as the final step.
```

## 8. Trả Lời Phỏng Vấn Về Bài Toán Thực Tế

### Câu hỏi: Nếu trời tối hoặc trời mưa thì làm sao?

Câu trả lời tốt:

```text
Em sẽ không chỉ chỉnh threshold hoặc train thêm ngay. Em sẽ audit dataset trước:
dataset hiện có bao nhiêu ảnh low-light, rain, blur, reflection, small object.
Sau đó em tạo report coverage, gắn quality flags cho từng ảnh, tạo contact sheet
để review bằng mắt, rồi mới quyết định bổ sung dữ liệu thật hoặc augmentation.
Cuối cùng em evaluate riêng từng nhóm edge case để biết model yếu ở đâu.
```

### Câu hỏi: Làm sao đảm bảo annotation consistency?

Câu trả lời tốt:

```text
Em dùng một format chuẩn, ví dụ YOLO bbox normalized, rồi viết script validate:
mỗi ảnh phải có label tương ứng, bbox không vượt biên, class id hợp lệ,
positive/negative đúng logic. Với các case khó, em tạo contact sheet để review
nhanh và ghi lại checklist QA.
```

### Câu hỏi: Dataset lớn quá thì quản lý thế nào?

Câu trả lời tốt:

```text
Em không để dataset lẫn lộn trong một folder. Em chia raw/processed/splits,
dùng manifest để lưu metadata, dùng changelog để biết version nào đã được train,
và dùng report để theo dõi số lượng, chất lượng, edge-case coverage.
```

## 9. Cách Kể Chuyện Với Gameloft (CV + Phỏng Vấn)

### 9.1. Mapping Gameloft cần → project có

| Gameloft cần | Project có thể chứng minh |
|---|---|
| Quản lý ảnh/video nhiều điều kiện thực tế | Video giao thông + CCPD edge-case sources |
| Annotation consistency | YOLO labels và validation roadmap |
| Metadata | `manifest.json` |
| Quality check | Contact sheet, audit report roadmap |
| Edge cases | blur, weather, small plate, reflection, low-light roadmap |
| Python script | `scripts/prepare_ccpd_layer4.py`, `train_plate_detector.py`, `evaluate_layer5.py`, audit script (planned) |
| Làm việc tỉ mỉ với dữ liệu | split, crop, labels, negative samples |

### 9.2. Phiên bản CV (EN, ngắn)

```text
Built a license plate dataset processing workflow for a traffic violation CV pipeline,
including YOLO label conversion, train/val/test split, metadata manifest,
negative samples, OCR crop generation, and QA-oriented reporting.
```

### 9.3. Phiên bản CV (EN, nhấn JD Gameloft)

```text
Designed a computer vision dataset workflow focused on data organization,
annotation consistency, metadata tracking, quality checks, and edge-case analysis
for license plate detection under real-world CCTV conditions.
```

### 9.4. Phiên bản VI (để tự giải thích khi phỏng vấn)

```text
Em không chỉ train YOLO biển số. Em xây workflow xử lý dataset gồm chuẩn hóa folder,
convert label, chia train/val/test, tạo manifest metadata, sinh crop OCR, thêm negative
samples và lên kế hoạch QA cho các case khó như blur, low-light, weather, reflection.
```

### 9.5. Cách định vị khi bị hỏi "sao làm traffic mà ứng tuyển game?"

```text
Domain của project là traffic, nhưng kỹ năng chính em muốn chứng minh là
dataset engineering cho computer vision: tổ chức dữ liệu, kiểm tra chất lượng,
annotation consistency, metadata, edge-case analysis và feedback loop từ model lỗi.
Những kỹ năng này chuyển được sang bài toán object recognition trong game.
```

## 10. Checklist Triển Khai Tiếp Theo

Effort ước lượng cho 1 người làm, giả định không vướng bối cảnh dữ liệu mới.

| Ưu tiên | Việc cần làm | Mục đích | Effort |
|---|---|---|---|
| P0 | Tạo `scripts/audit_plate_dataset.py` | Có audit thật thay vì nói bằng lời | Đã làm |
| P0 | Xuất `outputs/reports/dataset_audit_report.md` | Dùng trực tiếp trong CV/phỏng vấn | Đã làm |
| P1 | Thêm duplicate detection (file hash + perceptual hash) | Khớp JD cleaning/duplicate removal | Đã làm |
| P1 | Thêm quality flags (brightness/blur level, plate area) | Khớp edge-case handling | Đã làm |
| P1 | Tạo contact sheet theo nhóm lỗi | Khớp QA bằng mắt | Đã làm |
| P2 | Tạo `docs/DATASET_CHANGELOG.md` | Khớp versioning/changelog | Đã làm |
| P2 | Thêm augmentation demo | Trả lời tốt câu hỏi trời tối/mưa/blur | ~3h |
| P2 | Tạo model-error queue | Biến lỗi model thành kế hoạch cải thiện data | ~2h |

P0 + P1 đã hoàn thành. Hai phần còn lại nên làm sau nếu muốn biến project thành một câu chuyện dataset/model improvement sâu hơn.

## 11. Kết Luận

Project hiện tại đã có nền tảng dataset khá tốt cho một portfolio AI intern:

- 8,598 ảnh biển số đã xử lý.
- 8,598 YOLO labels.
- 8,000 OCR crops.
- 598 negative samples.
- Manifest có metadata.
- Plate detector đã fine-tune.
- Pipeline có video thật để kiểm tra thực tế.

Nhưng để khớp mạnh hơn với JD Gameloft, phần cần nâng cấp tiếp theo là:

```text
Dataset QA + metadata + duplicate detection + edge-case coverage + versioning.
```

Nói cách khác, bước tiếp theo không phải là tải thêm hàng chục GB dataset. Bước đúng hơn là biến dataset hiện có thành một workflow chuyên nghiệp, có audit, có report, có checklist và có câu chuyện rõ ràng về cách xử lý dữ liệu thực tế.

## 12. Trạng Thái Sau Khi Triển Khai Dataset QA

Các hạng mục roadmap quan trọng đã được triển khai trong repo:

| Hạng mục | File / output | Trạng thái |
|---|---|---|
| Dataset audit script | `scripts/audit_plate_dataset.py` | Đã làm |
| Audit report JSON/MD | `outputs/reports/dataset_audit_report.*` | Đã sinh |
| Quality manifest | `data/processed/ccpd_layer4/manifest_quality.json` | Đã sinh |
| Contact sheet theo nhóm lỗi | `outputs/reports/dataset_contact_*.jpg` | Đã sinh |
| Duplicate report | `outputs/reports/dataset_duplicates_report.*` | Đã sinh |
| Dataset changelog | `docs/DATASET_CHANGELOG.md` | Đã làm |

Kết quả audit mới nhất:

| Metric | Kết quả |
|---|---:|
| Total records | 8,598 |
| Integrity issues | 0 |
| Positive samples | 8,000 |
| Negative samples | 598 |
| Exact duplicate groups | 2 |
| Perceptual-hash collision groups | 102 |
| Dark positive images | 1,955 |
| Heavy-blur positive images | 4,349 |
| Small-plate positive images | 285 |
| Avg plate area ratio | 0.0328 |

Lưu ý khi nói trong CV/phỏng vấn:

- Có thể claim chắc rằng project đã có **automated dataset audit** và **0 integrity issues** trên dataset hiện tại.
- Có thể nói đã có duplicate review, nhưng không nên nói đã xóa duplicate, vì perceptual-hash collision chỉ là candidate cần review bằng mắt.
- Có thể dùng dark/heavy-blur/small-plate statistics để trả lời câu hỏi thực tế về trời tối, mưa, blur, CCTV degradation.
- Đây vẫn là dataset workflow cho **license plate detection**, không phải 1,000-object recognition dataset như domain có thể có của Gameloft.

## Phụ lục A. JD gốc Gameloft AI Engineer Intern (Game Development)

Nguồn: bản đăng tuyển công khai của Gameloft Vietnam. Lưu lại để đối chiếu nội dung mục 3.

```text
About the job

WHAT YOU WILL BE WORKING ON

Main challenge
You'll help build a high-quality computer vision dataset at scale
(~1,000 real-world objects) that powers reliable object recognition.
The challenge is ensuring data consistency and quality across diverse
real-world conditions like lighting changes, occlusion, motion blur,
and reflections.

What it means on a daily basis
- Organize, clean, and structure large image/video datasets
  (naming, metadata, removing duplicates, filtering low-quality samples)
- Annotate data accurately and follow labeling standards to ensure
  consistency at scale
- Track dataset progress, coverage, and versioning
  (logs, changelogs, dataset updates)
- Perform quality checks and continuously improve dataset reliability
  through structured QA
- Work closely with AI engineers to refine data based on model
  performance and edge cases

Who you would be working with
- AI, Data, and programmer teams contributing to dataset quality and workflows
- Cross-functional teams, depending on project needs

WHAT YOU NEED TO SUCCEED
- Basic understanding of computer vision tasks
  (classification, detection, segmentation)
- Familiarity with handling large datasets (images/videos) and
  maintaining structured documentation
- Attention to detail in data annotation and quality control
- Basic Python knowledge (for simple scripts, validation, or formatting tasks)
- Understanding of real-world data challenges
  (lighting, occlusion, motion blur, etc.)

WHO YOU ARE
- Highly detail-oriented and patient, with the ability to maintain
  accuracy in repetitive tasks
- Organized and methodical in managing data and documentation
- Proactive in identifying data issues and suggesting improvements
- Comfortable working both independently and in a collaborative environment
- Curious about AI, computer vision, and how models improve through data

Recruitment journey: Screening call -> Test -> Interview -> Offer
Internship allowance from VND 6,000,000 gross/month.
```
