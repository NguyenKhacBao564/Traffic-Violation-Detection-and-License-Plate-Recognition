# Dữ liệu cho project

## Cấu trúc thư mục

```
data/
├── raw_videos/        ← Video gốc (quay thực tế hoặc tải về)
│   └── README.md
├── frames/            ← Frame đã cắt ra (dùng cho debug)
├── annotations/       ← Annotation JSON cho từng video
└── samples/           ← Video ngắn mẫu (< 30s)
```

## Chuẩn bị video

### Nguồn video gợi ý

1. **Tự quay** — Dùng điện thoại quay giao thông thực tế tại TP.HCM
2. **YouTube** — Tìm video "giao thông TP HCM" có camera quan sát từ trên cao
3. **Dataset công khai:**
   - AI City Challenge (MSSD dataset)
   - MVI (Malaysian Vehicle dataset)
   - BoxCars116k (Czech Republic)

### Tiêu chí chọn video

- [ ] Camera **cố định** (không pan/zoom)
- [ ] Thời gian: ban ngày, thời tiết tương đối tốt
- [ ] Thấy rõ **stop line** (vạch dừng)
- [ ] Thấy rõ **đèn giao thông**
- [ ] Có xe ô tô chạy qua
- [ ] Biển số không quá nhỏ (nên ≥ 60px chiều cao)

### Công cụ cắt video

```bash
# Cắt 30 giây từ giây 60
ffmpeg -i input.mp4 -ss 60 -t 30 -c copy data/raw_videos/clip1.mp4
```

## Annotations

### Scene annotation

Mỗi camera cần 1 file `configs/cameras/{camera_id}.json` chứa:
- Stop line coordinates
- Traffic light ROI
- Road ROI

### Event annotation (Layer C — tùy chọn)

Dùng [CVAT](https://cvat.org) hoặc [Label Studio](https://labelstud.io)
để annotate:
- Bbox xe
- Bbox biển số
- Label vi phạm / không vi phạm

## Tải weights mẫu

```bash
# YOLOv8
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt -O models/detectors/

# Plate detector (pretrained)
# Tự train đơn giản trên ~100 ảnh hoặc dùng pretrained
```
