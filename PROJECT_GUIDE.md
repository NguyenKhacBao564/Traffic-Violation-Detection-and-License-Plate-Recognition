# Traffic Violation Detection & License Plate Recognition

## Tai lieu giai thich du an theo goc nhin giang vien

> Muc tieu cua tai lieu nay: giup nguoi moi hoc AI/CV hieu duoc du an dang giai quyet bai toan gi, pipeline xu ly video hoat dong ra sao, vi sao chon cac cong nghe hien tai, va can hoan thien tung buoc nhu the nao de bien repo thanh mot main project co the dua vao CV.

---

## 1. Tong quan du an

### 1.1. Bai toan can giai quyet

Du an **Traffic Violation Detection & License Plate Recognition** xay dung mot he thong xu ly video tu camera giao thong co dinh. He thong co ba nhiem vu chinh:

1. Phat hien xe o to trong video.
2. Xac dinh xe co vuot vach dung khi den do hay khong.
3. Doc bien so xe vi pham va luu bang chung gom anh, video clip va metadata.

Noi mot cach de hieu: neu co mot video quay nga tu, he thong se xem tung frame, tim xe, theo doi xe qua thoi gian, kiem tra trang thai den giao thong, kiem tra xe co cat qua vach dung luc den do hay khong, sau do cat bien so va luu lai bang chung.

### 1.2. Pham vi MVP

MVP la phien ban nho nhat nhung co gia tri de demo. Du an nay dang scope theo huong rat hop ly:

| Hang muc | Pham vi hien tai |
|---|---|
| Loai camera | 1 camera co dinh |
| Loai video | Offline, video ban ngay |
| Doi tuong | O to |
| Loai vi pham | Vuot den do / vuot vach dung |
| Dau ra | Anh bang chung, clip bang chung, JSON metadata |
| Luu tru | File JSON |

Nhung thu tam thoi **khong** nen lam o MVP:

- Xu ly realtime streaming.
- Nhieu camera cung luc.
- Xe may, xe dap, nguoi di bo.
- Dem, mua lon, camera rung.
- Ket noi voi he thong phat nguoi that.
- Training model lon tu dau.

Day la cach scope dung, vi voi mot du an portfolio, muc tieu khong phai la lam he thong production toan dien, ma la chung minh ban hieu pipeline AI end-to-end va co kha nang bien y tuong thanh san pham demo duoc.

---

## 2. Vi sao du an nay hop de dua vao CV?

Du an nay manh hon nhieu project AI co ban vi no khong chi dung mot model de predict anh. No yeu cau ket hop nhieu nhom ky nang:

| Nhom ky nang | The hien trong du an |
|---|---|
| Computer Vision | Vehicle detection, license plate detection, traffic light ROI |
| Multi-object tracking | Gan `track_id` cho xe qua nhieu frame |
| Xu ly video | Doc frame, ghi debug video, cat clip bang chung |
| Geometry | Kiem tra xe cat qua stop line |
| State machine | Quan ly event `pending -> confirmed -> dismissed` |
| OCR | Doc ky tu bien so bang PaddleOCR |
| Data engineering nho | Luu event JSON, metadata, folder output |
| Software engineering | Module hoa code, config rieng, test unit |

Neu hoan thien tot, day co the la main project CV cho vi tri:

- AI Engineer Intern.
- Computer Vision Intern.
- Machine Learning Engineer Intern.
- Data/AI Application Intern.

---

## 3. Kien truc tong the

Pipeline hien tai co the hinh dung nhu sau:

```text
Input Video
  |
  v
Frame Reader
  |
  v
Vehicle Detector (YOLO)
  |
  v
Vehicle Tracker (ByteTrack)
  |
  v
Traffic Light State Estimator
  |
  v
Stop Line Crossing Logic
  |
  v
Violation Event Trigger
  |
  v
Plate Detector + OCR
  |
  v
Evidence Builder
  |
  v
Event Store (JSON)
```

Mot pipeline nhu vay duoc goi la **multi-stage computer vision pipeline**. Thay vi yeu cau mot model duy nhat lam tat ca, ta chia bai toan thanh nhieu buoc nho. Moi buoc co input, output va trach nhiem ro rang.

### 3.1. Tai sao chia thanh pipeline?

Neu chi noi "phat hien xe vi pham", nghe co ve don gian. Nhung may tinh can tra loi rat nhieu cau hoi nho:

1. Trong frame nay co xe nao?
2. Xe o dau?
3. Xe nay co phai cung la xe o frame truoc khong?
4. Den giao thong dang mau gi?
5. Xe co di qua vach dung khong?
6. Luc xe di qua, den co do khong?
7. Neu vi pham, bien so cua xe la gi?
8. Can luu bang chung nao de nguoi xem co the kiem tra lai?

Moi cau hoi tren la mot stage trong pipeline.

---

## 4. Giai thich tung thanh phan trong pipeline

### 4.1. Frame Reader - doc video thanh frame

**Module lien quan:** `src/io/video_reader.py`

Video thuc chat la mot chuoi anh lien tiep. Moi anh goi la mot **frame**. Neu video co 30 FPS, nghia la moi giay co khoang 30 frame.

Frame Reader co nhiem vu:

- Mo file video bang OpenCV.
- Doc thong tin video: width, height, FPS, so frame.
- Doc tung frame de dua vao pipeline.

Vi du:

```text
sample.mp4
  -> frame 1
  -> frame 2
  -> frame 3
  -> ...
```

**Kien thuc can nam:**

- FPS cang cao thi video cang muot, nhung xu ly cang nang.
- Trong AI video, ta thuong khong can xu ly tat ca frame neu can toi uu toc do.
- Debug video la video dau ra co ve bbox, track_id, stop line de con nguoi xem lai.

---

### 4.2. Vehicle Detector - phat hien xe bang YOLO

**Module lien quan:** `src/detection/vehicle_detector.py`

Stage nay tra loi cau hoi: **Trong frame hien tai co xe nao va xe nam o dau?**

Du an dung YOLO tu thu vien `ultralytics`. YOLO la mot model object detection pho bien, co the phat hien nhieu loai vat the trong anh.

Output cua detector thuong la danh sach bbox:

```text
[
  {
    bbox: [x1, y1, x2, y2],
    confidence: 0.87,
    class_name: "car"
  },
  ...
]
```

Trong do:

- `x1, y1`: toa do goc tren trai cua bbox.
- `x2, y2`: toa do goc duoi phai cua bbox.
- `confidence`: do tu tin cua model.
- `class_name`: ten lop, vi du `car`, `bus`, `truck`.

**Ly thuyet ngan gon ve YOLO:**

YOLO la viet tat cua "You Only Look Once". Thay vi cat anh thanh nhieu vung roi phan loai tung vung, YOLO nhin ca anh mot lan va du doan truc tiep cac bounding box cung class. Uu diem cua YOLO la nhanh, phu hop cho video.

**Tai sao dung YOLO cho project nay?**

- De cai dat va dung voi `ultralytics`.
- Co model pretrained tren COCO.
- Chay nhanh tren CPU/GPU.
- De fine-tune sau nay neu co du lieu rieng.

**Luu y quan trong:**

Model YOLO COCO co class `car`, `bus`, `truck`, nhung khong co class `license plate`. Vi vay YOLO mac dinh dung tot cho phat hien xe, nhung khong du cho phat hien bien so.

---

### 4.3. Vehicle Tracker - theo doi xe qua nhieu frame

**Module lien quan:** `src/tracking/vehicle_tracker.py`

Neu chi detect tung frame rieng le, ta biet frame nay co xe, frame sau cung co xe, nhung chua biet do co phai cung mot xe hay khong.

Tracking tra loi cau hoi: **Xe nay o frame hien tai co phai la xe da xuat hien o frame truoc khong?**

Du an dung ByteTrack de gan `track_id` cho tung xe:

```text
Frame 10: car bbox -> ID 3
Frame 11: car bbox -> ID 3
Frame 12: car bbox -> ID 3
```

Neu tracking tot, cung mot xe se giu cung mot ID trong suot qua trinh di qua camera.

**Vi sao tracking quan trong voi bai toan vi pham?**

Ta can biet duong di cua mot xe:

```text
Vi tri truoc vach dung -> Vi tri sau vach dung
```

Neu khong co tracking, ta kho xac dinh mot xe da di chuyen nhu the nao qua thoi gian. Vi pham giao thong khong chi la mot khoanh khac, ma la mot **su kien theo thoi gian**.

**Ly thuyet ngan gon ve ByteTrack:**

ByteTrack la mot tracker multi-object. No nhan detection bbox moi frame, sau do lien ket bbox qua nhieu frame dua tren:

- Vi tri bbox.
- Do tu tin detection.
- Do tuong dong chuyen dong.

Uu diem cua ByteTrack la no co the tan dung ca detection confidence cao va thap de giam mat track khi vat the bi che mot phan.

---

### 4.4. Traffic Light State Estimator - xac dinh mau den giao thong

**Module lien quan:** `src/traffic_light/light_state.py`

Stage nay tra loi cau hoi: **Den giao thong trong frame dang do, vang, xanh hay khong ro?**

Trong MVP, cach lam khong dung deep learning ma dung ROI va HSV threshold.

#### ROI la gi?

ROI la viet tat cua **Region of Interest**, nghia la vung anh ma ta quan tam. Vi camera co dinh, vi tri den giao thong gan nhu khong thay doi. Ta co the danh dau san vung den trong config:

```json
"traffic_light_roi": [
  [1000, 200],
  [1150, 200],
  [1150, 380],
  [1000, 380]
]
```

He thong chi can nhin vao vung nay de xac dinh mau den.

#### HSV la gi?

HSV la khong gian mau gom:

- `H` - Hue: sac mau, vi du do, xanh, vang.
- `S` - Saturation: do bao hoa mau.
- `V` - Value: do sang.

OpenCV thuong doc anh theo BGR, sau do convert sang HSV de loc mau de hon.

Vi du logic don gian:

```text
Neu so pixel mau do trong ROI lon nhat -> den do
Neu so pixel mau xanh trong ROI lon nhat -> den xanh
Neu so pixel mau vang trong ROI lon nhat -> den vang
Neu khong ro -> unknown hoac giu trang thai cu
```

**Uu diem cua cach nay:**

- Don gian, de giai thich.
- Khong can dataset den giao thong.
- Phu hop voi camera co dinh va video ban ngay.

**Nhuoc diem:**

- De sai neu ROI sai.
- De bi anh huong boi nang, bong, den LED nho.
- Can tinh chinh threshold theo video thuc te.

---

### 4.5. Stop Line Crossing Logic - kiem tra xe co cat qua vach dung

**Module lien quan:** `src/geometry/line_crossing.py`

Stage nay tra loi cau hoi: **Xe co di qua stop line khong?**

Trong config, stop line duoc dinh nghia bang 2 diem:

```json
"stop_line": [
  [850, 680],
  [1150, 680]
]
```

Moi xe co bbox. Ta lay diem **bottom-center** cua bbox lam diem dai dien cho vi tri xe tren mat duong:

```text
x_center = (x1 + x2) / 2
y_bottom = y2
```

Ly do lay bottom-center: day la diem gan voi banh xe/mat duong hon so voi tam bbox.

Sau do so sanh:

```text
Frame truoc: xe o tren vach
Frame hien tai: xe o duoi vach
=> duong di cua xe cat qua stop line
=> crossing detected
```

**Kien thuc geometry:**

Ta co hai doan thang:

1. Stop line: doan thang co dinh tren mat duong.
2. Movement line: doan thang noi vi tri xe o frame truoc va frame hien tai.

Neu hai doan thang cat nhau, xe da di qua vach.

**Vi sao can direction?**

Trong anh, truc y tang tu tren xuong duoi. Neu xe di tu tren xuong duoi:

```text
y hien tai > y truoc do
=> chuyen dong forward
```

Neu xe di nguoc lai:

```text
y hien tai < y truoc do
=> chuyen dong backward
```

Voi camera co dinh, ta can khai bao huong di dung de tranh nham xe di nguoc chieu hoac xe o phia ben kia duong.

---

### 4.6. Violation Engine - tao va quan ly su kien vi pham

**Module lien quan:** `src/violation/violation_engine.py`, `src/violation/event_state.py`

Stage nay tra loi cau hoi: **Khi nao thi mot xe duoc xem la vi pham?**

Dieu kien MVP:

```text
Xe cat qua stop line
AND
Den giao thong dang do
AND
Track da ton tai du so frame toi thieu
```

Khi dieu kien dung, he thong tao mot event:

```text
EventState.PENDING
```

Sau do neu doc duoc bien so va luu duoc bang chung:

```text
EventState.CONFIRMED
```

Neu xe mat track hoac khong du dieu kien xac nhan:

```text
EventState.DISMISSED
```

Day la mot **state machine**.

### 4.6.1. State machine la gi?

State machine la cach mo hinh hoa mot doi tuong co nhieu trang thai va cac dieu kien de chuyen trang thai.

Trong du an:

```text
pending -> confirmed
pending -> dismissed
```

Bang giai thich:

| State | Y nghia |
|---|---|
| `pending` | Xe co dau hieu vi pham, dang cho OCR/evidence |
| `confirmed` | Da doc bien so hoac da co bang chung du de luu |
| `dismissed` | Khong xac nhan duoc, loai bo |

**Vi sao can state machine?**

Vi mot event vi pham khong ket thuc ngay tai frame cat vach. Sau khi phat hien vi pham, he thong con can doc bien so trong vai frame tiep theo, chon frame tot, luu anh/clip. State machine giup quan ly qua trinh nay ro rang hon.

---

### 4.7. Plate Detector - phat hien bien so

**Module lien quan:** `src/plate/plate_detector.py`

Stage nay tra loi cau hoi: **Bien so nam o dau trong anh xe?**

Bien so la object nho, nen khong nen OCR ca anh full-frame. Ta can cat dung vung bien so truoc.

Pipeline ly tuong:

```text
Full frame
  -> crop vehicle bbox
  -> detect license plate inside vehicle crop
  -> crop plate
  -> OCR
```

**Cong nghe du kien:** YOLO custom cho license plate.

**Diem can chu y:**

YOLO pretrained `yolov8n.pt` tren COCO khong co class license plate. Neu dung no lam plate detector, ket qua se khong dang tin. Can mot trong hai cach:

1. Dung pretrained license plate detector tu nguon khac.
2. Fine-tune YOLO tren dataset bien so.
3. Ban MVP nhanh co the crop vung duoi cua vehicle bbox roi OCR truc tiep, nhung do chinh xac se kem hon.

---

### 4.8. Plate OCR - doc ky tu bien so

**Module lien quan:** `src/plate/plate_ocr.py`

OCR la viet tat cua **Optical Character Recognition**, nghia la nhan dien chu trong anh.

Du an dung PaddleOCR de doc bien so:

```text
plate image -> PaddleOCR -> text + confidence
```

Vi du:

```text
Input: anh bien so
Output: "51H-12345", confidence = 0.92
```

**Tai sao dung PaddleOCR?**

- Manh trong OCR tieng Viet va ky tu Latin.
- Co pretrained model.
- Co the doc text trong anh khong can training lai ngay.
- Pho bien trong cac project OCR Python.

**Kho khan khi OCR bien so:**

- Bien so nho.
- Anh bi mo do chuyen dong.
- Goc camera nghieng.
- Anh sang manh/yiu.
- Ky tu de nham: `0/O`, `1/I`, `5/S`, `8/B`.

---

### 4.9. OCR Fusion - hop nhat ket qua OCR qua nhieu frame

**Module lien quan:** `src/plate/fusion.py`

Neu doc bien so o mot frame duy nhat, ket qua co the sai. Nhung trong video, cung mot xe xuat hien qua nhieu frame. Ta co the doc bien so nhieu lan roi vote.

Vi du:

```text
Frame 101: 51H-12345, conf 0.82
Frame 102: 51H-12345, conf 0.88
Frame 103: 51H-1234S, conf 0.61
Frame 104: 51H-12345, conf 0.91

Final: 51H-12345
```

Day goi la **temporal fusion**. "Temporal" nghia la theo thoi gian. Fusion nghia la hop nhat nhieu ket qua thanh mot ket qua tot hon.

**Tai sao temporal fusion quan trong?**

OCR co the sai o tung frame rieng le, nhung neu nhin nhieu frame, ket qua dung thuong xuat hien lap lai nhieu lan hon.

---

### 4.10. Evidence Builder - luu bang chung

**Module lien quan:** `src/evidence/evidence_builder.py`

Khi co event vi pham, he thong can luu bang chung de nguoi khac kiem tra lai:

```text
outputs/events/EVT_0001/
  frame.jpg
  plate.jpg
  clip.mp4
  event.json
```

Trong do:

| File | Y nghia |
|---|---|
| `frame.jpg` | Anh full frame co xe vi pham |
| `plate.jpg` | Anh crop bien so |
| `clip.mp4` | Clip ngan quanh thoi diem vi pham |
| `event.json` | Metadata: event_id, track_id, frame_idx, plate_text, confidence |

**Vi sao can evidence?**

Trong project AI thuc te, output khong chi la "co vi pham" hay "khong vi pham". Ta can co bang chung de:

- Debug khi model sai.
- Giai thich ket qua cho con nguoi.
- Lam demo cho nha tuyen dung.
- Tinh metric sau nay.

---

### 4.11. Event Store - luu va truy van event

**Module lien quan:** `src/storage/event_store.py`

Event Store quan ly viec luu event ra JSON va cap nhat index:

```text
events_index.json
EVT_0001/event.json
EVT_0002/event.json
...
```

Voi MVP, JSON la du. Sau nay neu mo rong, co the thay bang:

- SQLite.
- PostgreSQL.
- API backend.
- Dashboard web.

---

## 5. Cong nghe su dung trong du an

### 5.1. Python

Python la ngon ngu chinh vi he sinh thai AI/CV rat manh:

- OpenCV cho video/image.
- Ultralytics YOLO cho object detection.
- PaddleOCR cho OCR.
- NumPy cho xu ly ma tran/anh.
- PyYAML/JSON cho config.

### 5.2. OpenCV

OpenCV duoc dung cho:

- Doc video.
- Ghi video.
- Ve bbox, stop line, text debug.
- Crop anh.
- Convert mau BGR sang HSV.

Day la thu vien nen tang trong computer vision.

### 5.3. Ultralytics YOLO

YOLO duoc dung cho:

- Detect xe.
- Co the dung/fine-tune detect bien so neu co weights phu hop.

Trong du an, YOLO vehicle detector dung model `yolov8n.pt`. Ban `n` la nano, nho va nhanh, phu hop MVP.

### 5.4. ByteTrack

ByteTrack duoc dung de tracking xe qua frame. Dau vao la detection bbox, dau ra la tracked object co `track_id`.

Mot project video AI co tracking se thuyet phuc hon project chi detect anh, vi no xu ly duoc thong tin theo thoi gian.

### 5.5. PaddleOCR

PaddleOCR duoc dung de doc text bien so. No tra ve text va confidence. Ket qua OCR duoc dua vao fusion de tang do on dinh.

### 5.6. Shapely

Shapely duoc dung cho geometry, dac biet la kiem tra hai doan thang co cat nhau khong. Trong du an, no ho tro line crossing logic.

### 5.7. JSON/YAML

Du an dung:

- JSON cho camera scene config va event metadata.
- YAML cho default pipeline config.

Tach config ra khoi code la thoi quen tot trong software engineering. Khi thay video/camera, ta chi can sua config, khong sua logic.

---

## 6. Giai thich cau truc thu muc

```text
traffic-violation-lpr/
├── configs/
│   ├── default.yaml
│   └── cameras/cam_01.json
├── data/
│   └── README.md
├── models/
├── src/
│   ├── io/
│   ├── detection/
│   ├── tracking/
│   ├── traffic_light/
│   ├── geometry/
│   ├── violation/
│   ├── plate/
│   ├── evidence/
│   └── storage/
├── tests/
├── README.md
├── Plan.md
├── MILESTONES.md
├── IMPLEMENTATION_PLAN.md
└── requirements.txt
```

Bang giai thich:

| Thu muc/file | Vai tro |
|---|---|
| `configs/` | Chua cau hinh pipeline va camera |
| `data/` | Noi dat video, frame, annotations |
| `models/` | Noi dat model weights, thuong bi gitignore |
| `src/io/` | Doc/ghi video |
| `src/detection/` | Phat hien xe |
| `src/tracking/` | Theo doi xe qua frame |
| `src/traffic_light/` | Xac dinh trang thai den |
| `src/geometry/` | Logic hinh hoc, stop line crossing |
| `src/violation/` | Event/state machine vi pham |
| `src/plate/` | Detect bien so, OCR, fusion |
| `src/evidence/` | Luu bang chung |
| `src/storage/` | Luu event JSON/index |
| `tests/` | Unit tests |

Cau truc nay kha tot cho portfolio vi moi module co trach nhiem rieng. Nha tuyen dung co the doc repo va hieu nhanh logic tong the.

---

## 7. Giai thich cac file config

### 7.1. `configs/default.yaml`

File nay chua tham so chung:

```yaml
model:
  vehicle_model: yolov8n.pt
  vehicle_conf: 0.3
  vehicle_iou: 0.4
  vehicle_classes: [2]

tracking:
  track_thresh: 0.5
  track_buffer: 30

ocr:
  lang: vi_en
  vote_frames: 5
```

Giai thich mot so tham so:

| Tham so | Y nghia |
|---|---|
| `vehicle_conf` | Detection confidence toi thieu de chap nhan bbox |
| `vehicle_iou` | Nguong IoU cho NMS cua YOLO |
| `vehicle_classes` | Class ID muon detect, COCO class 2 la car |
| `track_thresh` | Nguong confidence cho tracking |
| `track_buffer` | So frame giu track khi doi tuong tam thoi mat detection |
| `vote_frames` | So frame OCR can doc truoc khi fusion |

### 7.2. `configs/cameras/cam_01.json`

File nay chua thong tin rieng cua camera:

```json
{
  "camera_id": "CAM_01",
  "stop_line": [[850, 680], [1150, 680]],
  "traffic_light_roi": [[1000, 200], [1150, 200], [1150, 380], [1000, 380]],
  "direction": "forward"
}
```

Day la file rat quan trong. Neu toa do stop line hoac traffic light ROI sai, ca pipeline se sai.

**Nguyen tac thuc hanh:**

- Moi video/camera nen co mot config rieng.
- Phai dung frame dau tien hoac notebook de click toa do that.
- Khong nen dung toa do mau neu video khac kich thuoc/resolution.

---

## 8. Giai thich ke hoach trien khai trong Plan/Milestones

### Phase 1 - Skeleton & I/O

**Muc tieu:** tao khung du an va doc/ghi video duoc.

Can hoan thanh:

- Cau truc folder ro rang.
- VideoReader doc duoc frame.
- VideoWriter ghi duoc debug video.
- Config camera mau.
- Chay thu voi dummy data hoac video ngan.

**Vi sao phase nay quan trong?**

Neu khong doc/ghi video on dinh, cac model AI phia sau khong co du lieu dau vao/dau ra de kiem tra. Day la nen mong cua toan bo pipeline.

**Tieu chi pass:**

```bash
python src/main.py --video data/raw_videos/sample.mp4 --camera configs/cameras/cam_01.json
```

Lenh chay khong crash va co output debug co ban.

---

### Phase 2 - Vehicle Detection + Tracking

**Muc tieu:** phat hien xe va gan track_id on dinh.

Can hoan thanh:

- YOLO detect xe.
- Loc class car.
- ByteTrack gan ID.
- Debug video ve bbox va ID.
- Quan sat xe co bi doi ID lien tuc khong.

**Tai sao tracking phai on dinh?**

Neu mot xe dang di qua vach ma tracker doi ID lien tuc, he thong co the:

- Khong nhan ra xe da cat qua line.
- Tao nhieu event trung lap.
- Khong gom duoc OCR readings cua cung mot xe.

**Metric nen co:**

- So lan ID switch.
- Ty le xe giu ID on dinh den khi roi frame.
- FPS xu ly.

---

### Phase 3 - Traffic Light + Stop Line

**Muc tieu:** biet khi nao den do va xe cat qua vach.

Can hoan thanh:

- Traffic light ROI.
- HSV threshold cho do/vang/xanh.
- Stop line coordinates.
- Logic bottom-center crossing.
- Event state `pending`.

**Day la phan bien detection thanh "hieu hanh vi".**

YOLO chi noi co xe. Tracking noi xe di dau. Stop line + den giao thong moi tra loi duoc cau hoi: xe co vi pham hay khong.

**Metric nen co:**

- Ty le frame nhan dung light state.
- So event pending dung/sai khi xe qua vach.
- So false positive khi den xanh.

---

### Phase 4 - Plate OCR + Evidence

**Muc tieu:** doc bien so xe vi pham va luu bang chung.

Can hoan thanh:

- Plate detector dung model phu hop.
- Crop plate tu vehicle/plate bbox.
- PaddleOCR doc text.
- OCR fusion qua nhieu frame.
- Save `frame.jpg`, `plate.jpg`, `clip.mp4`, `event.json`.

**Tai sao day la phase kho?**

Bien so trong video giao thong thuong nho va mo. Detection xe co the de, nhung OCR bien so chinh xac moi la phan lam project khac biet.

**Metric nen co:**

- Plate detection recall tren event frames.
- OCR accuracy.
- Ty le event co plate text hop le.

---

### Phase 5 - Evaluation + Polish

**Muc tieu:** bien project tu "code chay duoc" thanh "portfolio chuyen nghiep".

Can hoan thanh:

- Demo video/GIF trong README.
- Metrics co so lieu that.
- Huong dan setup ro rang.
- Tests pass.
- Code sach, config ro, output mau.

**Day la phase quan trong nhat neu dua vao CV.**

Nha tuyen dung khong chi xem ban dung YOLO. Ho xem:

- Ban co demo that khong?
- Ban co biet do luong ket qua khong?
- Ban co biet viet README de nguoi khac chay lai khong?
- Ban co biet giai thich trade-off khong?

---

## 9. Luong xu ly chi tiet khi chay pipeline

Gia su ta co mot frame video. Pipeline xu ly nhu sau:

### Buoc 1: Doc frame

```text
frame_idx = 120
frame = anh tai frame 120
```

### Buoc 2: Detect xe

```text
detections = [
  bbox=(500, 400, 700, 800), confidence=0.91, class="car"
]
```

### Buoc 3: Tracking

```text
tracked = [
  track_id=7, bbox=(505, 405, 705, 805)
]
```

### Buoc 4: Xac dinh den

```text
light_state = red
```

### Buoc 5: Kiem tra stop line

```text
previous bottom-center = (600, 670)
current bottom-center  = (605, 710)
stop_line y = 680

=> xe cat qua vach
```

### Buoc 6: Tao pending event

```text
EVT_0001
track_id = 7
state = pending
```

### Buoc 7: Detect/OCR bien so

```text
plate_crop -> PaddleOCR -> "51H-12345"
```

### Buoc 8: Fusion

```text
5 readings -> vote -> final_text = "51H-12345"
```

### Buoc 9: Confirm va luu bang chung

```text
state = confirmed
save frame.jpg
save plate.jpg
save clip.mp4
save event.json
```

---

## 10. Metric danh gia du an

Mot du an AI khong nen chi noi "model chay duoc". Can co metric.

### 10.1. Event Precision

Cong thuc:

```text
Precision = TP / (TP + FP)
```

Trong do:

- TP: he thong bao vi pham va thuc te co vi pham.
- FP: he thong bao vi pham nhung thuc te khong vi pham.

Precision cao nghia la it bao nham.

### 10.2. Event Recall

Cong thuc:

```text
Recall = TP / (TP + FN)
```

Trong do:

- FN: thuc te co vi pham nhung he thong bo sot.

Recall cao nghia la it bo sot.

### 10.3. Plate Accuracy

Cong thuc:

```text
Plate Accuracy = so bien so doc dung / so event that co bien so nhin duoc
```

### 10.4. Processing FPS

Cong thuc:

```text
Processing FPS = so frame xu ly / tong thoi gian xu ly
```

Neu FPS cao, pipeline nhanh hon. Voi MVP offline, FPS khong can realtime tuyet doi, nhung nen co so lieu.

### 10.5. Bang metric nen dua vao README

| Metric | Target MVP | Ket qua thuc te |
|---|---:|---:|
| Event Precision | > 80% | TBD |
| Event Recall | > 70% | TBD |
| Plate Accuracy | > 85% | TBD |
| Processing FPS | >= 15 | TBD |

Khong nen dien so ao. Hay chay tren video test va ghi dung ket qua.

---

## 11. Cac rui ro ky thuat can biet

### 11.1. Plate detector can model rieng

YOLO COCO khong co license plate. Day la blocker lon neu muon demo LPR that. Can co pretrained/fine-tuned plate detector hoac dung fallback crop heuristic.

### 11.2. OCR phu thuoc chat luong crop

Neu crop bien so sai, PaddleOCR se khong cuu duoc. Trong OCR pipeline, chat luong crop anh huong rat manh den ket qua cuoi.

### 11.3. Traffic light HSV de sai neu ROI/anh sang khong tot

HSV threshold hop voi MVP, nhung can test tren video that. Neu ROI chua dung, den se luon `unknown` hoac sai mau.

### 11.4. Tracking ID switch lam sai event

Neu ByteTrack doi ID giua chung, event co the khong duoc confirm hoac bi tao trung lap.

### 11.5. Config camera la yeu to song con

Stop line va traffic light ROI sai thi pipeline sai du model co tot.

---

## 12. Checklist de bien du an thanh CV-ready

### 12.1. Bat buoc truoc khi dua vao CV

- [ ] Co video sample 30-60 giay.
- [ ] Config camera dung voi video sample.
- [ ] Chay end-to-end khong crash.
- [ ] Debug video co bbox, track_id, stop line.
- [ ] Co it nhat 1 event output neu video co vi pham.
- [ ] `event.json` co schema ro rang.
- [ ] Plate OCR co ket qua thuc, khong phai placeholder.
- [ ] README co demo GIF/video.
- [ ] Co metric thuc te.
- [ ] `python -m pytest` pass trong moi truong moi.

### 12.2. Nen co neu muon gay an tuong hon

- [ ] So sanh truoc/sau OCR fusion.
- [ ] Bang loi thuong gap: false positive, false negative.
- [ ] Anh minh hoa pipeline trong README.
- [ ] Script evaluate metrics.
- [ ] Huong dan annotate stop line/ROI.
- [ ] Dockerfile hoac environment setup on dinh.

---

## 13. Cach ke du an nay trong CV

Khi du an da co demo va metrics, co the viet:

```text
Traffic Violation Detection & License Plate Recognition
- Built an offline fixed-camera computer vision pipeline using YOLOv8, ByteTrack, OpenCV, and PaddleOCR to detect red-light crossing violations and recognize license plates.
- Implemented vehicle tracking, stop-line crossing geometry, traffic-light color estimation, OCR temporal fusion, and evidence generation with JSON metadata.
- Evaluated on N annotated traffic clips, achieving X% event precision, Y% recall, Z% plate accuracy, and K FPS processing speed.
```

Phien ban tieng Viet de tu giai thich khi phong van:

```text
Em xay dung mot pipeline xu ly video giao thong tu camera co dinh. Dau tien he thong dung YOLO de detect xe, sau do ByteTrack de gan ID cho xe qua nhieu frame. Em dung ROI va HSV threshold de xac dinh den giao thong, dung geometry de kiem tra xe co cat qua vach dung hay khong. Neu xe cat qua vach luc den do, he thong tao event vi pham, crop bien so, dung PaddleOCR de doc bien so va hop nhat ket qua OCR qua nhieu frame. Cuoi cung he thong luu anh, clip va JSON metadata lam bang chung.
```

---

## 14. Cau hoi phong van co the gap

### Cau 1: Vi sao can tracking, sao khong detect tung frame la du?

Detection chi cho biet trong mot frame co xe. Vi pham vuot den do la su kien theo thoi gian, can biet cung mot xe di tu dau den dau va co cat qua stop line hay khong. Tracking giup gan `track_id` va theo doi duong di cua xe.

### Cau 2: Vi sao dung HSV cho den giao thong thay vi train model?

Vi camera co dinh va ROI den giao thong co the danh dau thu cong. HSV threshold don gian, nhanh, khong can dataset. Voi MVP ban ngay, cach nay du de demo. Neu mo rong cho nhieu camera/dieu kien anh sang, co the thay bang model classification rieng.

### Cau 3: Diem yeu lon nhat cua pipeline la gi?

Plate OCR phu thuoc nhieu vao chat luong plate crop. Neu bien so nho, mo, nghieng hoac plate detector sai, OCR se sai. Ngoai ra tracking ID switch va ROI den giao thong sai cung co the lam sai event.

### Cau 4: Lam sao de giam false positive?

- Tang confidence threshold cho vehicle detector.
- Kiem tra track phai ton tai du so frame.
- Yeu cau den phai do voi confidence du cao.
- Kiem tra crossing direction.
- Them road ROI de bo qua xe ngoai vung duong.
- Review debug video va dieu chinh stop line.

### Cau 5: Lam sao de tang OCR accuracy?

- Dung plate detector dung domain bien so Viet Nam.
- Crop bien so ro hon va resize truoc OCR.
- OCR nhieu frame va temporal fusion.
- Loc format bien so Viet Nam.
- Chon frame co bien so lon/it mo nhat.

---

## 15. Lo trinh hoc neu ban la tay ngang

Nen hoc theo thu tu sau:

1. **Python + NumPy co ban**
   - List, dict, function, class.
   - Array va shape anh.

2. **OpenCV co ban**
   - Doc anh/video.
   - Ve bbox/text.
   - Crop anh.
   - BGR/HSV.

3. **Object Detection**
   - Bounding box la gi.
   - Confidence la gi.
   - YOLO dung nhu the nao.

4. **Tracking**
   - Track ID la gi.
   - Vi sao can tracking trong video.

5. **Geometry**
   - Diem, doan thang.
   - Kiem tra line crossing.

6. **OCR**
   - OCR la gi.
   - Vi sao bien so kho doc.
   - Fusion nhieu frame.

7. **Evaluation**
   - Precision, recall.
   - False positive, false negative.
   - Cach ghi metric that.

8. **Software engineering**
   - Tach module.
   - Config.
   - Tests.
   - README va demo.

---

## 16. Giai thich mot so thuat ngu

| Thuat ngu | Giai thich ngan |
|---|---|
| Frame | Mot anh trong video |
| FPS | So frame moi giay |
| Bounding box | Hinh chu nhat bao quanh vat the |
| Detection | Phat hien vat the trong anh |
| Tracking | Theo doi vat the qua nhieu frame |
| Track ID | Ma dinh danh cua mot vat the trong video |
| ROI | Vung anh can quan tam |
| HSV | Khong gian mau de loc mau de hon BGR/RGB |
| OCR | Nhan dien chu trong anh |
| Temporal fusion | Hop nhat ket qua qua nhieu frame |
| Event | Su kien co y nghia, o day la vi pham giao thong |
| Metadata | Du lieu mo ta event, luu trong JSON |
| Precision | Ty le du doan dung trong cac case he thong bao co |
| Recall | Ty le phat hien duoc trong cac case that su co |

---

## 17. Ket luan

Du an nay co nen lam main project CV khong? **Co.**

Nhung dieu kien la phai bien no tu scaffold thanh demo co bang chung that:

- Co video sample.
- Chay end-to-end.
- Luu duoc event.
- Doc duoc bien so o mot so case.
- Co metric that.
- README co demo ro rang.

Gia tri lon nhat cua project khong nam o viec "dung YOLO", ma nam o viec ban xay duoc mot pipeline co nhieu thanh phan: detection, tracking, traffic light estimation, geometry, OCR, evidence, storage va evaluation.

Neu ban hieu va giai thich duoc tung stage trong tai lieu nay, ban da co mot nen tang rat tot de trinh bay project trong phong van AI/CV Intern.

