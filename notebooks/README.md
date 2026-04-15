# Jupyter Notebooks cho traffic-violation-lpr

## Các notebook có sẵn

| # | Notebook | Mục tiêu |
|---|----------|----------|
| 01 | `01_video_exploration.ipynb` | Xem video, định nghĩa stop line, test light ROI |
| 02 | `02_yolo_detection_test.ipynb` | Test YOLO detect xe trên video mẫu |
| 03 | `03_bytetrack_test.ipynb` | Test ByteTrack tracking |
| 04 | `04_traffic_light_test.ipynb` | Test HSV light detection |
| 05 | `05_stop_line_test.ipynb` | Test line crossing logic |
| 06 | `06_full_pipeline_test.ipynb` | Chạy full pipeline trên video nhỏ |
| 07 | `07_ocr_fusion_test.ipynb` | Test PaddleOCR và temporal fusion |

## Cách dùng

```bash
# Cài jupyter
pip install jupyter notebook
conda install ipykernel -y

# Thêm kernel cho môi trường lpr
python -m ipykernel install --user --name lpr --display-name "traffic-violation-lpr"

# Chạy
jupyter notebook
```
