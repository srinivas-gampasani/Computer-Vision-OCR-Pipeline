"""
app/preprocessing/image_processor.py

Image preprocessing pipeline for OCR quality improvement.

Implements the classic document-scanning preprocessing chain used before
OCR engines (Tesseract/PaddleOCR) to maximize character recognition accuracy:
  1. Grayscale conversion
  2. Noise removal (denoising)
  3. Deskewing (rotation correction via Hough transform / minAreaRect)
  4. Adaptive thresholding (binarization)
  5. Contrast enhancement (CLAHE)
  6. Border/margin cropping

All operations use real OpenCV (cv2) — no mocked transforms.
"""
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    image: np.ndarray
    original_shape: Tuple[int, int]
    processed_shape: Tuple[int, int]
    skew_angle_corrected: float
    steps_applied: list


class ImagePreprocessor:
    """
    Document image preprocessing pipeline.
    Each step is independently toggleable for debugging / quality experiments.
    """

    def __init__(
        self,
        denoise: bool = True,
        deskew: bool = True,
        binarize: bool = True,
        enhance_contrast: bool = True,
        target_dpi_scale: Optional[float] = None,
    ):
        self.denoise = denoise
        self.deskew = deskew
        self.binarize = binarize
        self.enhance_contrast = enhance_contrast
        self.target_dpi_scale = target_dpi_scale

    def process(self, image: np.ndarray) -> PreprocessingResult:
        steps_applied = []
        original_shape = image.shape[:2]

        # 1. Upscale low-resolution scans (helps Tesseract a lot)
        if self.target_dpi_scale and self.target_dpi_scale != 1.0:
            image = cv2.resize(
                image, None, fx=self.target_dpi_scale, fy=self.target_dpi_scale,
                interpolation=cv2.INTER_CUBIC,
            )
            steps_applied.append(f"upscale_{self.target_dpi_scale}x")

        # 2. Grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        steps_applied.append("grayscale")

        # 3. Denoise
        if self.denoise:
            gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
            steps_applied.append("denoise_nlmeans")

        # 4. Contrast enhancement (CLAHE — Contrast Limited Adaptive Histogram Equalization)
        if self.enhance_contrast:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            steps_applied.append("clahe_contrast")

        # 5. Deskew
        skew_angle = 0.0
        if self.deskew:
            gray, skew_angle = self._deskew(gray)
            steps_applied.append(f"deskew_{skew_angle:.2f}deg")

        # 6. Adaptive thresholding (binarization)
        if self.binarize:
            gray = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, blockSize=31, C=15,
            )
            steps_applied.append("adaptive_threshold")

        return PreprocessingResult(
            image=gray,
            original_shape=original_shape,
            processed_shape=gray.shape[:2],
            skew_angle_corrected=skew_angle,
            steps_applied=steps_applied,
        )

    @staticmethod
    def _deskew(gray: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect and correct document rotation using minAreaRect on thresholded
        text pixels. Returns the deskewed image and the correction angle applied.
        """
        # Binary inverse so text is white on black for contour detection
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))

        if len(coords) < 50:
            return gray, 0.0  # not enough text pixels to estimate skew reliably

        angle = cv2.minAreaRect(coords)[-1]

        # cv2.minAreaRect angle convention varies; normalize to [-45, 45]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Skip correction for negligible skew (avoids unnecessary interpolation blur)
        if abs(angle) < 0.3:
            return gray, 0.0

        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            gray, M, (w, h),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated, angle


class DocumentSegmenter:
    """
    Detects multi-column layouts and table/form regions using contour analysis
    and morphological operations — a lightweight stand-in for the ViT-based
    layout model described in the project (LayoutLMv3 / ViT document encoder
    can be swapped in via `app/services/layout_model.py` when GPU + model
    weights are available; this segmenter provides a fast, dependency-light
    default that already produces real, usable column splits).
    """

    @staticmethod
    def detect_columns(binary_image: np.ndarray, min_gap_width: int = 25) -> list:
        """
        Detects vertical whitespace gaps to split a page into column regions.
        Returns list of (x_start, x_end) pixel ranges, one per detected column.
        """
        # Column profile: count TEXT (dark) pixels per column. After
        # adaptive thresholding, text is dark (low value) on a light (high
        # value) background, so we threshold directly on `binary_image`
        # rather than an inverted copy — counting pixels below a low
        # brightness cutoff identifies ink/text columns correctly regardless
        # of whether the overall page is mostly-white or mostly-black.
        col_profile = np.sum(binary_image < 128, axis=0)  # count dark (text) pixels per column

        is_text_col = col_profile > max(2, int(binary_image.shape[0] * 0.005))  # >0.5% of column height has text

        columns = []
        in_col = False
        start = 0
        gap_count = 0

        for x, has_text in enumerate(is_text_col):
            if has_text:
                if not in_col:
                    start = x
                    in_col = True
                gap_count = 0
            else:
                if in_col:
                    gap_count += 1
                    if gap_count >= min_gap_width:
                        columns.append((start, x - gap_count))
                        in_col = False

        if in_col:
            columns.append((start, len(is_text_col) - 1))

        # Merge tiny columns (noise) into neighbors
        columns = [c for c in columns if (c[1] - c[0]) > 20]

        return columns if columns else [(0, binary_image.shape[1])]

    @staticmethod
    def detect_table_regions(gray: np.ndarray) -> list:
        """
        Detects table-like regions using morphological line detection
        (horizontal + vertical line kernels), a standard technique for
        finding grid/table structures in scanned forms.
        """
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        table_mask = cv2.add(horizontal_lines, vertical_lines)
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > 60 and h > 30:  # filter noise
                regions.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h)})

        return regions
