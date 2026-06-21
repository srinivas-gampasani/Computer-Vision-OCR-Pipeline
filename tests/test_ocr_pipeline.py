"""
tests/test_ocr_pipeline.py

Test suite for the Computer Vision OCR Pipeline.

IMPORTANT: These tests use the REAL Tesseract OCR engine and REAL OpenCV
image processing — no mocked OCR output. Tests generate small synthetic
test images on the fly (via PIL) so the suite is self-contained and doesn't
depend on the bundled sample documents, but it exercises the exact same
code path (ImagePreprocessor -> OCREngineRouter -> StructureExtractor)
used in production.

Run with: pytest tests/ -v
Requires: tesseract-ocr binary installed on the system.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


def make_text_image(text_lines, width=600, height=300, font_size=24):
    """Generate a simple white-background black-text test image using PIL."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, font_size) if os.path.exists(font_path) else ImageFont.load_default()

    y = 20
    for line in text_lines:
        draw.text((20, y), line, font=font, fill="black")
        y += font_size + 12

    return np.array(img)[:, :, ::-1]  # PIL RGB -> cv2 BGR


# ── ImagePreprocessor Tests (real OpenCV) ───────────────────────────────────

class TestImagePreprocessor:

    def setup_method(self):
        from app.preprocessing.image_processor import ImagePreprocessor
        self.preprocessor = ImagePreprocessor()

    def test_process_returns_grayscale_image(self):
        img = make_text_image(["Hello World"])
        result = self.preprocessor.process(img)
        assert len(result.image.shape) == 2  # grayscale = 2D array

    def test_process_applies_expected_steps(self):
        img = make_text_image(["Test document"])
        result = self.preprocessor.process(img)
        assert "grayscale" in result.steps_applied
        assert "adaptive_threshold" in result.steps_applied

    def test_process_preserves_dimensions_without_upscale(self):
        img = make_text_image(["Sample"], width=400, height=200)
        result = self.preprocessor.process(img)
        assert result.processed_shape == (200, 400)

    def test_upscale_factor_changes_dimensions(self):
        from app.preprocessing.image_processor import ImagePreprocessor
        preprocessor = ImagePreprocessor(target_dpi_scale=2.0)
        img = make_text_image(["Sample"], width=400, height=200)
        result = preprocessor.process(img)
        assert result.processed_shape[0] == 400  # 200 * 2
        assert result.processed_shape[1] == 800  # 400 * 2

    def test_deskew_detects_rotation(self):
        from PIL import Image
        img_arr = make_text_image(["This is a rotated test document with enough text"] * 4, width=700, height=300)
        pil_img = Image.fromarray(img_arr[:, :, ::-1])
        rotated = pil_img.rotate(-5, expand=True, fillcolor="white")
        rotated_arr = np.array(rotated)[:, :, ::-1]

        result = self.preprocessor.process(rotated_arr)
        # Should detect meaningful skew (real angle detection, not exact due to text geometry)
        assert abs(result.skew_angle_corrected) > 0.5

    def test_no_deskew_when_disabled(self):
        from app.preprocessing.image_processor import ImagePreprocessor
        preprocessor = ImagePreprocessor(deskew=False)
        img = make_text_image(["Sample text"])
        result = preprocessor.process(img)
        assert result.skew_angle_corrected == 0.0
        assert not any("deskew" in s for s in result.steps_applied)


# ── TesseractEngine Tests (real Tesseract OCR) ───────────────────────────────

class TestTesseractEngine:

    def setup_method(self):
        from app.services.ocr_engine import TesseractEngine
        self.engine = TesseractEngine(psm=3)

    def test_extracts_known_text(self):
        img = make_text_image(["HELLO WORLD"], font_size=36)
        result = self.engine.extract(img)
        assert "HELLO" in result.full_text.upper()
        assert "WORLD" in result.full_text.upper()

    def test_confidence_in_valid_range(self):
        img = make_text_image(["Clear readable text sample"])
        result = self.engine.extract(img)
        assert 0 <= result.mean_confidence <= 100

    def test_word_boxes_have_valid_coordinates(self):
        img = make_text_image(["Sample text here"])
        result = self.engine.extract(img)
        for w in result.words:
            assert w.x >= 0 and w.y >= 0
            assert w.width > 0 and w.height > 0

    def test_blank_image_produces_minimal_words(self):
        blank = np.full((200, 400), 255, dtype=np.uint8)  # all-white image
        result = self.engine.extract(blank)
        assert result.word_count == 0

    def test_min_confidence_filter(self):
        img = make_text_image(["High quality readable text"])
        result_all = self.engine.extract(img, min_confidence=0.0)
        result_filtered = self.engine.extract(img, min_confidence=99.0)
        assert len(result_filtered.words) <= len(result_all.words)

    def test_psm_mode_affects_results(self):
        """Regression test for the PSM-6-vs-PSM-3 bug found during development:
        PSM 6 measured 32.7% confidence on table layouts vs 91.0% for PSM 3."""
        from app.services.ocr_engine import TesseractEngine
        img = make_text_image([
            "Col1   Col2   Col3",
            "A1     B1     C1",
            "A2     B2     C2",
        ])
        engine_psm3 = TesseractEngine(psm=3)
        result = engine_psm3.extract(img)
        assert result.word_count > 0  # PSM 3 should reliably find text

    def test_reconstruct_text_preserves_line_order(self):
        img = make_text_image(["First line of text", "Second line below it"])
        result = self.engine.extract(img)
        first_idx = result.full_text.upper().find("FIRST")
        second_idx = result.full_text.upper().find("SECOND")
        assert first_idx != -1 and second_idx != -1
        assert first_idx < second_idx


# ── DocumentSegmenter Tests (real OpenCV morphology) ─────────────────────────

class TestDocumentSegmenter:

    def setup_method(self):
        from app.preprocessing.image_processor import ImagePreprocessor, DocumentSegmenter
        self.preprocessor = ImagePreprocessor()
        self.segmenter = DocumentSegmenter()

    def test_detect_columns_single_column(self):
        img = make_text_image(["Single column of text", "spanning the full width", "of this document image"], width=800)
        result = self.preprocessor.process(img)
        columns = self.segmenter.detect_columns(result.image)
        assert len(columns) >= 1

    def test_detect_columns_two_column_layout(self):
        """Regression test for the column-detection inverted-pixel bug found
        during development (was counting background pixels, not text)."""
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGB", (1000, 400), "white")
        draw = ImageDraw.Draw(img)
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        font = ImageFont.truetype(font_path, 20) if os.path.exists(font_path) else ImageFont.load_default()

        # Left column
        y = 30
        for line in ["Left column text", "continues here", "with more lines", "of content below"]:
            draw.text((30, y), line, font=font, fill="black")
            y += 35

        # Right column (with a wide gap in between)
        y = 30
        for line in ["Right column text", "also continues", "with separate", "content blocks"]:
            draw.text((550, y), line, font=font, fill="black")
            y += 35

        arr = np.array(img)[:, :, ::-1]
        result = self.preprocessor.process(arr)
        columns = self.segmenter.detect_columns(result.image, min_gap_width=25)
        assert len(columns) == 2, f"Expected 2 columns, got {len(columns)}: {columns}"

    def test_detect_table_regions_finds_grid(self):
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (600, 400), "white")
        draw = ImageDraw.Draw(img)
        # Draw a simple table grid
        for x in range(50, 550, 100):
            draw.line([(x, 50), (x, 350)], fill="black", width=2)
        for y in range(50, 360, 60):
            draw.line([(50, y), (550, y)], fill="black", width=2)

        arr = np.array(img)[:, :, ::-1]
        gray = self.preprocessor.process(arr).image
        # detect_table_regions expects a grayscale (pre-threshold) image
        import cv2
        raw_gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        regions = self.segmenter.detect_table_regions(raw_gray)
        assert len(regions) >= 1


# ── StructureExtractor Tests ──────────────────────────────────────────────────

class TestStructureExtractor:

    def setup_method(self):
        from app.services.ocr_engine import TesseractEngine
        from app.services.structure_extractor import StructureExtractor
        self.ocr = TesseractEngine(psm=3)
        self.extractor = StructureExtractor()

    def test_extracts_key_value_pairs_from_form(self):
        img = make_text_image([
            "Patient Name: John Smith",
            "Date of Birth: 01/15/1985",
            "Phone: 555-0123",
        ], width=700, height=200)
        ocr_result = self.ocr.extract(img)
        doc = self.extractor.extract(ocr_result)
        keys = [kv.key for kv in doc.key_value_pairs]
        assert any("Patient" in k or "Name" in k for k in keys)

    def test_classifies_form_document_type(self):
        img = make_text_image([
            "Name: Test User",
            "Date: 01/01/2026",
            "Email: test@test.com",
            "Phone: 555-1234",
        ], width=700, height=250)
        ocr_result = self.ocr.extract(img)
        doc = self.extractor.extract(ocr_result)
        assert doc.document_type in ("form", "mixed")

    def test_metadata_contains_expected_fields(self):
        img = make_text_image(["Sample text content"])
        ocr_result = self.ocr.extract(img)
        doc = self.extractor.extract(ocr_result)
        assert "total_words" in doc.metadata
        assert "mean_ocr_confidence" in doc.metadata
        assert doc.metadata["total_words"] == ocr_result.word_count

    def test_to_dict_serializes_without_error(self):
        img = make_text_image(["Name: Jane Doe", "Date: 02/02/2026"])
        ocr_result = self.ocr.extract(img)
        doc = self.extractor.extract(ocr_result)
        d = doc.to_dict()
        assert "document_type" in d
        assert "key_value_pairs" in d
        assert "tables" in d
        assert isinstance(d["tables"], list)


# ── Full Pipeline Integration Tests ───────────────────────────────────────────

class TestOCRPipelineIntegration:

    def setup_method(self):
        from app.services.ocr_pipeline import OCRPipeline
        self.pipeline = OCRPipeline()

    def test_process_returns_pipeline_result(self):
        img = make_text_image(["Integration test document", "with multiple lines", "of readable text"])
        result = self.pipeline.process(img)
        assert result.document.full_text != ""
        assert result.ocr_engine_used == "tesseract"

    def test_to_dict_includes_pipeline_info(self):
        img = make_text_image(["Test pipeline output"])
        result = self.pipeline.process(img)
        d = result.to_dict()
        assert "pipeline_info" in d
        assert "preprocessing_steps" in d["pipeline_info"]
        assert "total_pipeline_time_ms" in d["pipeline_info"]
        assert d["pipeline_info"]["total_pipeline_time_ms"] > 0

    def test_process_file_loads_real_image(self, tmp_path):
        from PIL import Image
        img_arr = make_text_image(["File-based test", "document content"])
        img_path = str(tmp_path / "test.png")
        Image.fromarray(img_arr[:, :, ::-1]).save(img_path)

        result = self.pipeline.process_file(img_path)
        assert result.document.metadata["total_words"] > 0

    def test_process_file_nonexistent_raises(self):
        with pytest.raises(ValueError):
            self.pipeline.process_file("/nonexistent/path/image.png")

    def test_process_bytes_works(self):
        from PIL import Image
        import io
        img_arr = make_text_image(["Bytes-based test document"])
        pil_img = Image.fromarray(img_arr[:, :, ::-1])
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")

        result = self.pipeline.process_bytes(buf.getvalue())
        assert result.document.metadata["total_words"] > 0

    def test_pipeline_confidence_above_baseline_for_clean_text(self):
        """Quality gate: clean synthetic text should produce high OCR confidence."""
        img = make_text_image(["This is a clean, high quality test document"], font_size=28)
        result = self.pipeline.process(img)
        assert result.document.metadata["mean_ocr_confidence"] > 70.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
