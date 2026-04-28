# Dataset Plan & Layer 1 Status

Tai lieu nay chi giu cac thong tin can thiet cho giai doan hien tai cua du an. Cac danh sach dataset mo rong da duoc bo bot vi chua can dung ngay cho MVP.

---

## 1. Ket luan ve du lieu hien co

Ban da co du lieu Layer 1 de bat dau chay pipeline:

| File goc | Resolution | FPS goc | Thoi luong | Vai tro |
|---|---:|---:|---:|---|
| `data/raw_videos/cam_01_full.mp4` | 1342x1008 | ~60 | 147.4s | Raw video camera 01 |
| `data/raw_videos/cam_02_full.mp4` | 1342x1008 | ~60 | 95.6s | Raw video camera 02 |
| `data/raw_videos/cam_03_full.mp4` | 1342x1008 | ~60 | 144.5s | Raw video camera 03 |

Day la du lieu tot cho **integration demo** vi co:

- Camera co dinh.
- Xe ro.
- Vach dung / vach nguoi di bo ro.
- Den giao thong nhin duoc.
- Bien so kha ro o vung gan camera.

Nhung day **chua phai dataset hoan chinh** de train/evaluate nghiem tuc vi:

- Chua co ground truth annotations.
- Chi la mot bo clip cung mot scene/giao lo.
- License/copyright cua raw video chua ro.
- Bien so la thong tin nhay cam, khong nen public raw frame/video.

---

## 2. Layer 1 da tao nhung gi?

Layer 1 co muc tieu bien raw video thanh input sach, nho, co metadata va san sang annotate.

### 2.1. Clip demo 60 giay

Da tao 3 clip ngan, moi clip 60 giay, 30 FPS:

| Clip | Frames | FPS | Thoi luong | Muc dich |
|---|---:|---:|---:|---|
| `data/raw_videos/clips/cam_01_clip_001.mp4` | 1800 | 30 | 60s | Clip test pipeline chinh |
| `data/raw_videos/clips/cam_02_clip_001.mp4` | 1800 | 30 | 60s | Clip test bo sung |
| `data/raw_videos/clips/cam_03_clip_001.mp4` | 1800 | 30 | 60s | Clip test bo sung |

Tai sao can cat clip?

- Chay model nhanh hon.
- Debug de hon video dai.
- De annotate event bang tay.
- De tao demo/metric nho cho CV.

### 2.2. Reference frames

Da tao reference frames trong:

```text
data/frames/references/
```

Cac frame nay dung de:

- Xac dinh `stop_line`.
- Xac dinh `traffic_light_roi`.
- Kiem tra `road_roi`.
- Tao overlay de nhin nhanh scene config dung hay sai.

File quan trong:

```text
data/frames/references/layer1_contact_sheet.jpg
data/frames/references/cam_01_scene_overlay.jpg
data/frames/references/cam_02_scene_overlay.jpg
data/frames/references/cam_03_scene_overlay.jpg
```

### 2.3. Inventory metadata

Da tao:

```text
data/annotations/layer1_inventory.json
```

File nay ghi lai:

- Source video.
- Resolution.
- FPS.
- Frame count.
- Duration.
- Clip da sinh ra.
- Reference frames da sinh ra.

Muc dich: sau nay khi review du an, ta biet du lieu Layer 1 duoc tao tu dau va co thong so gi.

### 2.4. Event annotation template

Da tao:

```text
data/annotations/cam_01_events_template.json
```

File nay la template de dien ground truth bang tay, vi du:

- Xe cat vach o frame nao.
- Luc cat vach den do hay xanh.
- Co vi pham hay khong.
- Bien so doc bang mat neu co.

---

## 3. Scene config hien tai

Da tao/cap nhat:

```text
configs/cameras/cam_01.json
configs/cameras/cam_02.json
configs/cameras/cam_03.json
```

Trong cac file nay co:

- `resolution`: 1342x1008.
- `stop_line`: toa do vach dung uoc luong.
- `traffic_light_roi`: vung den giao thong uoc luong.
- `road_roi`: vung mat duong can quan sat.
- `direction`: hien dat la `backward`.

Ly do `direction = backward`: trong lane gan camera, xe di len phia tren anh, tuc la toa do `y` giam dan. Code hien tai can duoc sua tiep de ton trong direction nay khi check crossing.

---

## 4. Task va muc dich cua Layer 1

| Task | Da lam | Muc dich |
|---|---|---|
| Kiem tra video co doc duoc khong | Co | Dam bao OpenCV co the mo raw video |
| Lay metadata video | Co | Biet resolution, FPS, duration de cau hinh pipeline |
| Cat clip 60 giay | Co | Tao input nho de debug nhanh |
| Downsample ve 30 FPS | Co | Giam chi phi xu ly, phu hop debug |
| Trich reference frames | Co | Co anh tinh de annotate stop line/ROI |
| Tao contact sheet | Co | Nhin nhanh toan bo scene va chon frame tot |
| Tao scene overlay | Co | Kiem tra truc quan stop line/ROI |
| Tao inventory JSON | Co | Luu lai thong tin du lieu da chuan bi |
| Tao event annotation template | Co | Chuan bi cho buoc danh gia event |

Layer 1 khong train model va khong yeu cau OCR. Muc tieu cua Layer 1 chi la lam cho du lieu video **sach, nho, co cau truc va san sang cho pipeline**.

---

## 5. Viec chua lam trong Layer 1

Cac viec sau chua xem la xong hoan toan:

- Chua annotate event ground truth.
- Chua verify stop line/traffic light ROI tren nhieu frame.
- Chua blur/anonymize bien so cho public demo.
- Chua chay model detection/tracking tren clip.
- Chua tinh metrics.

Nhung cac viec nay thuoc Layer 2/Layer 3, khong phai blocker cua Layer 1.

---

## 6. Nguyen tac public du lieu

Khong nen commit/public:

- Raw video trong `data/raw_videos/`.
- Clip co bien so ro neu chua blur.
- Event JSON co bien so that.
- Dataset tai tu nguon chua ro license.

Co the public:

- Code.
- Config camera.
- README.
- Anh/GIF demo da blur bien so.
- Metric tren subset annotate.
- Link nguon dataset cong khai neu license cho phep.

---

## 7. Dataset bo sung

Hien tai chi giu dataset bo sung theo nhu cau tung layer, khong giu raw dataset lon neu da convert xong.

| Dataset | Dung cho | Khi nao dung |
|---|---|---|
| CCPD | License plate detection/OCR Trung Quoc | Da convert thanh subset Layer 4 |
| UA-DETRAC | Vehicle detection/tracking | Khi can benchmark tracking |
| BSTLD/DTLD | Traffic light detection | Khi HSV ROI khong on |
| CULane/OpenLane | Lane detection | Khi muon tu dong detect lane/vach |

### CCPD2019 decision

Raw CCPD2019 day du khong can giu tiep trong workspace sau khi da tao subset Layer 4.

Da tao:

```text
data/processed/ccpd_layer4/
outputs/reports/ccpd_layer4_sample_contact.jpg
```

Noi dung subset:

| Thanh phan | So luong |
|---|---:|
| Tong image | 8,598 |
| Positive plate samples | 8,000 |
| Negative no-plate samples | 598 |
| OCR crops/labels | 8,000 |

Muc dich:

- Train/fine-tune YOLO plate detector.
- Thu HyperLPR3/PaddleOCR/EasyOCR tren plate crop.
- Giu du lieu nho, ro provenance bang `manifest.json`.
- Xoa duoc raw `CCPD2019/` va `CCPD2019.tar.xz` de tiet kiem dung luong.

---

## 8. Bước tiếp theo

Các bước dưới đây đã hoàn thành trong Layer 2–5:

- [x] Sửa `ViolationEngine` crossing direction cho `backward`.
- [x] Chạy detection/tracking trên `cam_01_clip_001.mp4`.
- [x] Tạo debug video có bbox, track_id, stop line, traffic light ROI.
- [x] Kiểm tra traffic light HSV.
- [x] Train plate detector bằng `data/processed/ccpd_layer4`.
- [x] Gắn detector/OCR vào event evidence (Layer 4).
- [x] Tạo evaluation report + demo assets (Layer 5).

Việc còn lại:

- [ ] Annotate event ground truth đầy đủ cho Event Recall.
- [ ] Cải thiện plate crop quality để tính Plate Accuracy.
- [ ] Xoá raw `CCPD2019/` và `CCPD2019.tar.xz` nếu không cần nữa.

---

## 9. Kết luận

Layer 1 đã đạt mục tiêu: raw videos → clip test, frame tham chiếu, overlay, config, metadata. Layer 4 cũng đã có compact CCPD subset để train/thử plate detection/OCR mà không cần giữ raw CCPD full.
