import numpy as np
import cv2

from src.plate.plate_detector import PlateDetector
from src.plate.plate_ocr import PlateOCR


def test_plate_detector_heuristic_finds_blue_plate():
    frame = np.zeros((160, 240, 3), dtype=np.uint8)
    vehicle_bbox = (40, 40, 210, 140)
    cv2.rectangle(frame, (60, 55), (190, 135), (30, 30, 30), -1)
    cv2.rectangle(frame, (95, 100), (160, 118), (255, 80, 20), -1)  # blue in BGR

    detector = PlateDetector(model_path=None)
    detections = detector.detect_with_scores(frame, vehicle_bbox=vehicle_bbox)

    assert detections
    assert detections[0].source == "color_heuristic"
    x1, y1, x2, y2 = detections[0].bbox
    assert x1 <= 100 <= x2
    assert y1 <= 110 <= y2


def test_plate_detector_falls_back_to_lower_vehicle_crop():
    frame = np.zeros((160, 240, 3), dtype=np.uint8)
    detector = PlateDetector(model_path=None)

    detections = detector.detect_with_scores(frame, vehicle_bbox=(40, 40, 210, 140))

    assert detections
    assert detections[0].source == "vehicle_lower_crop"


def test_plate_ocr_none_backend_filters_without_crashing():
    ocr = PlateOCR(backend="none")
    text, conf = ocr.read_and_filter(np.zeros((20, 80, 3), dtype=np.uint8))

    assert text == ""
    assert conf == 0.0


def test_plate_ocr_rejects_unknown_backend():
    try:
        PlateOCR(backend="bad")
    except ValueError as exc:
        assert "hyperlpr" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
