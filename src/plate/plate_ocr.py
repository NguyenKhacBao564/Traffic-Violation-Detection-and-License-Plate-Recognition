"""License plate OCR with optional PaddleOCR/EasyOCR backends."""
from __future__ import annotations

import cv2
import numpy as np


class PlateOCR:
    """OCR wrapper for license plate text recognition.

    `backend="auto"` tries PaddleOCR first, then HyperLPR3, then EasyOCR. If no
    backend is installed, OCR calls return an empty result instead of crashing
    the pipeline.
    """

    def __init__(
        self,
        lang: str = "ch",
        use_angle_cls: bool = True,
        backend: str = "auto",
        easyocr_languages: list[str] | None = None,
        gpu: bool = False,
    ) -> None:
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.backend_name = "none"
        self.ocr = None
        self.easyocr_languages = easyocr_languages or ["ch_sim", "en"]

        if backend not in {"auto", "paddle", "hyperlpr", "easyocr", "none"}:
            raise ValueError("backend must be one of: auto, paddle, hyperlpr, easyocr, none")

        if backend in {"auto", "paddle"} and self._init_paddle():
            return
        if backend in {"auto", "hyperlpr"} and self._init_hyperlpr():
            return
        if backend in {"auto", "easyocr"} and self._init_easyocr(gpu=gpu):
            return

        if backend != "none":
            print("[WARN] No OCR backend available; plate_text will remain empty.")

    def _init_paddle(self) -> bool:
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except Exception:
            return False
        self.ocr = PaddleOCR(
            lang=self.lang,
            use_angle_cls=self.use_angle_cls,
            show_log=False,
        )
        self.backend_name = "paddle"
        return True

    def _init_hyperlpr(self) -> bool:
        try:
            import hyperlpr3  # type: ignore
        except Exception:
            return False
        self.ocr = hyperlpr3.LicensePlateCatcher()
        self.backend_name = "hyperlpr"
        return True

    def _init_easyocr(self, gpu: bool = False) -> bool:
        try:
            import easyocr  # type: ignore
        except Exception:
            return False
        self.ocr = easyocr.Reader(self.easyocr_languages, gpu=gpu, verbose=False)
        self.backend_name = "easyocr"
        return True

    @property
    def is_available(self) -> bool:
        """Return True when an OCR backend was initialized."""
        return self.ocr is not None

    def read(self, plate_image: np.ndarray) -> tuple[str, float]:
        """
        Read text from a plate image.

        Returns:
            (text, confidence)
        """
        if self.ocr is None or plate_image is None or plate_image.size == 0:
            return "", 0.0

        prepared = self._prepare_plate(plate_image)
        if self.backend_name == "paddle":
            return self._read_paddle(prepared)
        if self.backend_name == "hyperlpr":
            return self._read_hyperlpr(prepared)
        if self.backend_name == "easyocr":
            return self._read_easyocr(prepared)
        return "", 0.0

    def _read_paddle(self, plate_image: np.ndarray) -> tuple[str, float]:
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

    def _read_hyperlpr(self, plate_image: np.ndarray) -> tuple[str, float]:
        result = self.ocr.pipeline(plate_image)
        if not result:
            return "", 0.0
        best_text = ""
        best_conf = 0.0
        for item in result:
            if len(item) < 2:
                continue
            text = str(item[0])
            conf = float(item[1])
            if conf > best_conf:
                best_text = text
                best_conf = conf
        return best_text, best_conf

    def _read_easyocr(self, plate_image: np.ndarray) -> tuple[str, float]:
        result = self.ocr.readtext(plate_image, detail=1, paragraph=False)
        if not result:
            return "", 0.0

        best_text = ""
        best_conf = 0.0
        for item in result:
            if len(item) < 3:
                continue
            text = str(item[1])
            conf = float(item[2])
            if conf > best_conf:
                best_text = text
                best_conf = conf
        return best_text, best_conf

    def read_and_filter(self, plate_image: np.ndarray) -> tuple[str, float]:
        """
        Read plate text and keep license-plate-like characters.

        The project video currently contains Chinese plates, while the original
        portfolio idea can still be adapted to Vietnamese plates. The filter
        therefore keeps CJK characters, Latin letters, digits, and separators.
        """
        text, conf = self.read(plate_image)
        import re

        filtered = re.sub(
            r"[^A-Za-z0-9\u4e00-\u9fffÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯư\s\-.]",
            "",
            text,
        )
        filtered = re.sub(r"[\s.]+", "", filtered).upper()
        return filtered.strip("-"), conf

    @staticmethod
    def _prepare_plate(plate_image: np.ndarray) -> np.ndarray:
        """Resize and lightly pad a plate crop before OCR."""
        image = plate_image
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return image
        target_h = 96
        if h < target_h:
            scale = target_h / h
            image = cv2.resize(image, (int(w * scale), target_h), interpolation=cv2.INTER_CUBIC)
        image = cv2.copyMakeBorder(image, 8, 8, 12, 12, cv2.BORDER_REPLICATE)
        return image
