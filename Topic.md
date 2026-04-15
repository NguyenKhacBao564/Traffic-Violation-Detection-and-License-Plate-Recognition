1) Tên dự án

Traffic Violation Detection and License Plate Recognition from Fixed-Camera Video

Tên ngắn gọn hơn cho repo:

traffic-violation-lpr

2) Mục tiêu MVP
Mục tiêu chính

Xây dựng hệ thống nhận video từ 1 camera giao thông cố định, phát hiện xe ô tô vượt đèn đỏ / vượt vạch dừng khi đèn đỏ, sau đó:

xác định đúng chiếc xe vi phạm
đọc biển số
lưu ảnh bằng chứng
lưu clip bằng chứng
lưu thông tin sự kiện vi phạm vào DB/JSON
Điều rất quan trọng

MVP này chỉ làm cho ô tô, chưa làm xe máy.

Lý do:

giảm độ khó tracking
giảm che khuất
giảm nhiễu OCR
dễ xây logic event hơn
3) Phạm vi chính thức của MVP
In-scope

MVP v1 chỉ hỗ trợ:

1 camera cố định
video ban ngày
thời tiết tương đối tốt
1 loại vi phạm: vượt đèn đỏ
1 loại phương tiện chính: ô tô
xử lý offline trên video trước
Out-of-scope

Chưa làm ngay:

real-time production
nhiều camera
xe máy
ban đêm
mưa lớn
vượt tốc độ
sai làn
nhận diện người lái
tích hợp hệ thống xử phạt thật

Phần này cực quan trọng vì nó bảo vệ dự án khỏi bị phình quá sớm.

4) Định nghĩa chính xác “vi phạm”

Ta cần định nghĩa bằng ngôn ngữ máy tính, không phải mô tả cảm tính.

Định nghĩa sự kiện vi phạm

Một xe bị xem là vi phạm nếu thỏa mãn cả 3 điều kiện:

Xe được track ổn định qua nhiều frame
Tại thời điểm xe vượt qua stop line, trạng thái đèn là red
Xe đó là car hoặc nhóm phương tiện bạn cho phép trong MVP

Viết logic gần đúng:

Violation
=
(
track_valid
)
∧
(
cross_stop_line
)
∧
(
light
=
𝑟
𝑒
𝑑
)
Violation=(track_valid)∧(cross_stop_line)∧(light=red)
Diễn giải kỹ hơn

Ta xét tâm đáy của bounding box xe hoặc một điểm đại diện gần bánh trước.

Nếu tại thời điểm 
𝑡
1
t
1
	​

, điểm đó ở trước stop line,
và tại thời điểm 
𝑡
2
t
2
	​

, điểm đó ở sau stop line,
trong khi đèn vẫn là đỏ,
thì sinh ra event vi phạm.

Điểm này quan trọng vì nó biến bài toán từ nhận diện ảnh sang suy luận trên chuỗi thời gian.

5) Input và Output chính thức
Input
video .mp4 hoặc .avi
camera cố định
góc nhìn thấy:
vùng đường xe đi
stop line
đèn giao thông hoặc trạng thái đèn được cung cấp
Output cho mỗi event

Mỗi vi phạm nên sinh ra một record như sau:

{
  "event_id": "EVT_0001",
  "timestamp": "2026-04-15T14:32:08",
  "camera_id": "CAM_01",
  "violation_type": "red_light_crossing",
  "vehicle_type": "car",
  "track_id": 17,
  "plate_text": "51H-12345",
  "plate_confidence": 0.93,
  "evidence_image_path": "outputs/events/EVT_0001/frame.jpg",
  "evidence_clip_path": "outputs/events/EVT_0001/clip.mp4"
}
Output hình ảnh/video
1 ảnh full frame có bbox xe vi phạm
1 ảnh crop biển số
1 clip ngắn 3–8 giây quanh thời điểm vi phạm
1 file JSON hoặc 1 row trong DB
6) Kiến trúc MVP v1

Pipeline của bạn nên là:

Input Video
   ↓
Frame Reader
   ↓
Vehicle Detector
   ↓
Vehicle Tracker
   ↓
Traffic Light State Estimation
   ↓
Stop Line Crossing Logic
   ↓
Violation Event Trigger
   ↓
Plate Detector
   ↓
Plate OCR
   ↓
Temporal OCR Fusion
   ↓
Evidence Builder
   ↓
JSON/DB Storage
Ý nghĩa từng khối
(1) Vehicle Detector

Mỗi frame tìm xe ô tô.

(2) Vehicle Tracker

Gán track_id để biết cùng một xe qua nhiều frame.

(3) Traffic Light State

Biết tại frame hiện tại đèn là đỏ, vàng hay xanh.

(4) Stop Line Logic

Kiểm tra track nào đi từ trước vạch sang sau vạch.

(5) Event Trigger

Nếu đang đỏ và vừa vượt vạch thì tạo event.

(6) Plate Detector + OCR

Chỉ chạy mạnh khi đã có xe vi phạm, giúp tiết kiệm compute và giảm OCR sai.

(7) Temporal OCR Fusion

Đọc biển số qua nhiều frame rồi bỏ phiếu kết quả tốt nhất.

(8) Evidence Builder

Cắt ảnh, clip, lưu metadata.

7) Chọn chiến lược kỹ thuật cho MVP

Tôi chốt luôn hướng kỹ thuật đầu tiên để bạn không bị phân tán.

Baseline stack
Detection: YOLO
Tracking: ByteTrack hoặc BoT-SORT
Plate detection: YOLO custom
OCR: PaddleOCR
Video processing: OpenCV
Storage: JSON trước, DB sau
UI: chưa cần ở MVP code đầu
Quyết định quan trọng

Bản đầu không cần:

training lớn ngay
microservice phức tạp
frontend đẹp
cloud deployment

Chỉ cần:

pipeline chạy end-to-end
cấu trúc repo sạch
có output kiểm chứng được
8) Dữ liệu cần cho giai đoạn kế tiếp

Giờ ta chuyển sang câu hỏi dữ liệu, nhưng theo cách có hệ thống.

Bạn cần 3 lớp dữ liệu.

Lớp A — Video thô

Đây là nguyên liệu gốc.

Bạn nên chuẩn bị:

5 đến 10 video ban ngày
mỗi video khoảng 1–5 phút
camera cố định
có stop line tương đối rõ
có đèn giao thông hoặc thấy được pha đèn
biển số không quá nhỏ
Lớp B — Annotation scene

Mỗi video hoặc mỗi camera cần file cấu hình scene:

{
  "camera_id": "CAM_01",
  "stop_line": [[x1, y1], [x2, y2]],
  "road_roi": [[...]],
  "traffic_light_roi": [[...]],
  "direction": "forward"
}

Tức là có một số thành phần bạn không cần AI học, mà đánh dấu hình học thủ công.

Lớp C — Annotation object/event

Về sau bạn sẽ cần:

bbox xe
bbox biển số
label event vi phạm / không vi phạm

Nhưng ở bước hiện tại, chưa cần annotate toàn bộ ngay.

9) Thành công của MVP được đo như thế nào?

Đây là phần rất nhiều người bỏ qua.

Không đo thành công bằng:
“model chạy được”
“có bbox”
“OCR đọc được vài biển số”
Phải đo bằng 4 câu hỏi
Câu 1

Hệ thống có bắt đúng xe vi phạm không?

Câu 2

Hệ thống có đọc đúng biển số của đúng xe đó không?

Câu 3

Hệ thống có lưu được bằng chứng đủ xem lại không?

Câu 4

False positive có chấp nhận được không?

Metric MVP ban đầu

Chưa cần quá học thuật, bạn có thể dùng:

số event đúng / tổng event hệ thống tạo ra
số biển số đúng / tổng event vi phạm đúng
số event bỏ sót
số event báo sai

Viết đơn giản:

Event Precision
=
𝑇
𝑃
𝑇
𝑃
+
𝐹
𝑃
Event Precision=
TP+FP
TP
	​

Event Recall
=
𝑇
𝑃
𝑇
𝑃
+
𝐹
𝑁
Event Recall=
TP+FN
TP
	​

Plate Accuracy on True Events
=
plate đ
u
ˊ
ng
true events
Plate Accuracy on True Events=
true events
plate đ
u
ˊ
ng
	​

10) Thiết kế repo ngay từ đầu

Đây là cấu trúc mà tôi khuyên bạn dùng để Claude CLI scaffold.

traffic-violation-lpr/
│
├── configs/
│   ├── cameras/
│   │   └── cam_01.json
│   └── default.yaml
│
├── data/
│   ├── raw_videos/
│   ├── frames/
│   ├── annotations/
│   └── samples/
│
├── models/
│   ├── detectors/
│   ├── plate_detector/
│   └── ocr/
│
├── src/
│   ├── io/
│   │   ├── video_reader.py
│   │   └── writer.py
│   │
│   ├── detection/
│   │   └── vehicle_detector.py
│   │
│   ├── tracking/
│   │   └── vehicle_tracker.py
│   │
│   ├── traffic_light/
│   │   └── light_state.py
│   │
│   ├── geometry/
│   │   └── line_crossing.py
│   │
│   ├── violation/
│   │   ├── event_state.py
│   │   └── violation_engine.py
│   │
│   ├── plate/
│   │   ├── plate_detector.py
│   │   ├── plate_ocr.py
│   │   └── fusion.py
│   │
│   ├── evidence/
│   │   └── evidence_builder.py
│   │
│   ├── storage/
│   │   └── event_store.py
│   │
│   └── main.py
│
├── outputs/
│   ├── debug_videos/
│   ├── events/
│   └── logs/
│
├── notebooks/
├── requirements.txt
├── README.md
└── .env

Cấu trúc này đủ tốt cho:

debug
scale dần
thay model sau này
dễ giao việc cho Claude CLI
11) Việc bạn nên làm ngay bây giờ

Đây là bước thực tế kế tiếp, không vòng vo.

Việc 1 — Chuẩn bị 5–10 video mẫu

Tiêu chí:

camera cố định
ban ngày
có stop line
có đèn đỏ rõ
có xe ô tô chạy qua
biển số không quá bé

Không cần dài.
Mỗi video 1–5 phút là đủ cho bước đầu.

Việc 2 — Chọn 1 camera scene duy nhất

Đừng gom nhiều scene ngay.

Ta cần 1 scene để:

dễ debug
dễ tune rule
dễ xác định stop line
Việc 3 — Tạo file scene config đầu tiên

Ví dụ configs/cameras/cam_01.json chứa:

stop line
traffic light ROI
road ROI
camera_id
Việc 4 — Yêu cầu Claude CLI scaffold repo

Chưa cần code model hoàn chỉnh.
Chỉ cần tạo skeleton project sạch, class rỗng, interface rõ.

12) Prompt đầu tiên bạn có thể đưa cho Claude CLI

Dưới đây là prompt tốt để bắt đầu code phần khung:

Create a clean Python project scaffold for an MVP traffic violation detection system.

Project goal:
- Input: fixed-camera traffic video
- Detect cars
- Track vehicles across frames
- Determine traffic light state
- Detect when a tracked car crosses a stop line during red light
- Trigger a violation event
- Run license plate detection and OCR for violating cars
- Save event metadata, evidence image, and evidence clip

Requirements:
- Use a modular src/ structure
- Add type hints
- Add dataclasses for event records and track state
- Create placeholder modules for detection, tracking, traffic light, geometry, violation engine, plate OCR, evidence builder, and storage
- Add a main pipeline entrypoint
- Add a camera config JSON example
- Add requirements.txt
- Add README with architecture overview and run instructions

Do not implement full model inference yet.
Focus on clean architecture, interfaces, and file organization.

Prompt này đúng vì nó buộc Claude CLI làm:

kiến trúc trước
interface trước
không lao vào code chắp vá
13) Phần kiến thức ML/DL bạn cần hiểu ngay ở bước này

Tôi muốn bạn nắm một ý rất quan trọng.

Tại sao chưa chọn dataset quá sớm?

Vì dữ liệu phải phục vụ cho state transition của hệ thống, không chỉ cho detector.

Bài toán của bạn thực chất là:

Video
→
Objects
→
Tracks
→
Events
Video→Objects→Tracks→Events

Nếu mới nhìn bằng tư duy CV cơ bản, bạn sẽ nghĩ:

Image
→
Bounding box
Image→Bounding box

Nhưng đó là chưa đủ.

Trong dự án này, phần giá trị nhất nằm ở:

Track dynamics
+
Scene geometry
+
Rule logic
Track dynamics+Scene geometry+Rule logic

tức là:

xe nào
đã đi như thế nào
trong ngữ cảnh giao thông nào
vào đúng thời điểm nào

Đó là lý do bước đặc tả này quan trọng hơn việc chọn ngay một dataset công khai.