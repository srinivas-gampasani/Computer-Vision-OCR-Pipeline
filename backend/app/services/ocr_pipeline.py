"""
app/services/ocr_pipeline.py

End-to-end orchestrator: raw image bytes → preprocessing → OCR → layout
structuring → final JSON output. This is the single entry point used by
both the REST API and the CLI tool.
"""
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from app.preprocessing.image_processor import ImagePreprocessor, DocumentSegmenter
from app.services.ocr_engine import OCREngineRouter, OCRResult
from app.services.structure_extractor import StructureExtractor, StructuredDocument

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    document: StructuredDocument
    preprocessing_steps: List[str]
    skew_corrected_degrees: float
    table_regions_detected: int
    columns_detected: int
    total_pipeline_time_ms: float
    ocr_engine_used: str

    def to_dict(self) -> Dict[str, Any]:
        d = self.document.to_dict()
        d["pipeline_info"] = {
            "preprocessing_steps": self.preprocessing_steps,
            "skew_corrected_degrees": round(self.skew_corrected_degrees, 2),
            "table_regions_detected": self.table_regions_detected,
            "columns_detected": self.columns_detected,
            "total_pipeline_time_ms": round(self.total_pipeline_time_ms, 2),
            "ocr_engine_used": self.ocr_engine_used,
        }
        return d


class OCRPipeline:
    """
    Full document digitization pipeline:
      bytes/path → cv2 decode → preprocess (denoise/deskew/binarize) →
      segment (columns + tables) → OCR (Tesseract/PaddleOCR) →
      structure (key-value + tables + classification) → JSON
    """

    def __init__(
        self,
        prefer_paddle: bool = False,
        ocr_lang: str = "eng",
        denoise: bool = True,
        deskew: bool = True,
        binarize: bool = True,
        upscale_factor: Optional[float] = None,
        min_word_confidence: float = 0.0,
    ):
        self.preprocessor = ImagePreprocessor(
            denoise=denoise, deskew=deskew, binarize=binarize,
            enhance_contrast=True, target_dpi_scale=upscale_factor,
        )
        self.segmenter = DocumentSegmenter()
        self.ocr_router = OCREngineRouter(prefer_paddle=prefer_paddle, lang=ocr_lang)
        self.structure_extractor = StructureExtractor()
        self.min_word_confidence = min_word_confidence

    @staticmethod
    def load_image(path: str) -> np.ndarray:
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Could not load image: {path}")
        return image

    @staticmethod
    def load_image_from_bytes(data: bytes) -> np.ndarray:
        arr = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image bytes — unsupported or corrupt format.")
        return image

    def process(self, image: np.ndarray, detect_tables: bool = True) -> PipelineResult:
        t0 = time.time()

        # 1. Preprocess
        prep_result = self.preprocessor.process(image)
        processed = prep_result.image

        # 2. Segment — columns + table regions (real OpenCV morphology)
        columns = self.segmenter.detect_columns(processed)
        table_regions = self.segmenter.detect_table_regions(processed) if detect_tables else []

        # 3. OCR (real Tesseract/PaddleOCR call)
        ocr_result = self.ocr_router.extract(processed, min_confidence=self.min_word_confidence)

        # 4. Structure extraction
        structured = self.structure_extractor.extract(ocr_result, table_regions=table_regions)

        elapsed = (time.time() - t0) * 1000

        return PipelineResult(
            document=structured,
            preprocessing_steps=prep_result.steps_applied,
            skew_corrected_degrees=prep_result.skew_angle_corrected,
            table_regions_detected=len(table_regions),
            columns_detected=len(columns),
            total_pipeline_time_ms=elapsed,
            ocr_engine_used=self.ocr_router.get_active_engine_name(),
        )

    def process_file(self, path: str, detect_tables: bool = True) -> PipelineResult:
        image = self.load_image(path)
        return self.process(image, detect_tables=detect_tables)

    def process_bytes(self, data: bytes, detect_tables: bool = True) -> PipelineResult:
        image = self.load_image_from_bytes(data)
        return self.process(image, detect_tables=detect_tables)
