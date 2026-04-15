"""License plate OCR using PaddleOCR."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from paddleocr import PaddleOCR


class PlateOCR:
    """PaddleOCR-based license plate text recognition."""

    def __init__(
        self,
        lang: str = "vi_en",
        use_angle_cls: bool = True,
    ) -> None:
        self.ocr = PaddleOCR(
            lang=lang,
            use_angle_cls=use_angle_cls,
            show_log=False,
        )

    def read(self, plate_image: np.ndarray) -> tuple[str, float]:
        """
        Read text from a plate image.

        Returns:
            (text, confidence)
        """
        result = self.ocr.ocr(plate_image, cls=True)
        if not result or not result[0]:
            return "", 0.0

        best_text = ""
        best_conf = 0.0
        for line in result[0]:
            text = line[1][0]
            conf = line[1][1]
            if conf > best_conf:
                best_text = text
                best_conf = conf

        return best_text, float(best_conf)

    def read_and_filter(self, plate_image: np.ndarray) -> tuple[str, float]:
        """
        Read plate text and filter to valid Vietnamese plate format.
        Format examples: 51H-12345, 43A-1234.56
        """
        text, conf = self.read(plate_image)
        # Basic filter: alphanumeric, dash, spaces, Vietnamese characters
        import re
        filtered = re.sub(r"[^A-Za-z0-9ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯư\s\-]", "", text)
        return filtered.strip(), conf
