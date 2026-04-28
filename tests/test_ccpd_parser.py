from src.datasets.ccpd import decode_plate, padded_bbox, parse_ccpd_filename


def test_parse_ccpd_filename_extracts_bbox_and_plate():
    ann = parse_ccpd_filename(
        "ccpd_base/025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg"
    )

    assert ann.source == "ccpd_base"
    assert ann.bbox == (154, 383, 386, 473)
    assert ann.vertices[0] == (386, 473)
    assert ann.plate_indices == (0, 0, 22, 27, 27, 33, 16)
    assert ann.plate_text == "\u7696AY339S"


def test_decode_plate_supports_chinese_ccpd_mapping():
    assert decode_plate((12, 0, 28, 29, 30, 31, 32)) == "\u4eacA45678"


def test_yolo_label_is_normalized():
    ann = parse_ccpd_filename(
        "ccpd_base/025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg"
    )

    assert ann.yolo_label(720, 1160).startswith("0 0.375000 0.368966 0.322222")


def test_padded_bbox_clips_to_image_bounds():
    assert padded_bbox((5, 5, 20, 20), 30, 30, pad_ratio=1.0) == (0, 0, 30, 30)

