"""
app/services/structure_extractor.py

Converts raw OCR word-boxes into structured, layout-aware JSON output:
  - Groups words into lines, paragraphs, and blocks using spatial clustering
  - Detects key-value pairs in form-style documents (e.g. "Patient Name: John Doe")
  - Detects table rows/columns using detected table regions from DocumentSegmenter
  - Classifies document type (form / table / paragraph / mixed) from layout signals

This is the lightweight production stand-in for the ViT/LayoutLMv3 document
layout model referenced in the project description. The interface
(`LayoutModel.predict_regions`) is designed so a real transformer-based
layout model (e.g. microsoft/layoutlmv3-base or a ViT document classifier)
can be dropped in without changing any downstream code — see
`app/services/vit_layout_model.py` for the swappable model wrapper.
"""
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.services.ocr_engine import OCRResult, WordBox

logger = logging.getLogger(__name__)


@dataclass
class KeyValuePair:
    key: str
    value: str
    confidence: float
    bbox: Dict[str, int]


@dataclass
class TableCell:
    row: int
    col: int
    text: str
    confidence: float


@dataclass
class StructuredDocument:
    document_type: str  # "form" | "table" | "paragraph" | "mixed"
    full_text: str
    key_value_pairs: List[KeyValuePair]
    tables: List[List[List[TableCell]]]  # list of tables; each table = list of rows; each row = list of cells
    paragraphs: List[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_type": self.document_type,
            "full_text": self.full_text,
            "key_value_pairs": [
                {"key": kv.key, "value": kv.value, "confidence": round(kv.confidence, 2), "bbox": kv.bbox}
                for kv in self.key_value_pairs
            ],
            "tables": [
                [
                    [{"row": c.row, "col": c.col, "text": c.text, "confidence": round(c.confidence, 2)} for c in row]
                    for row in table
                ]
                for table in self.tables
            ],
            "paragraphs": self.paragraphs,
            "metadata": self.metadata,
        }


class LineGrouper:
    """Groups OCR word boxes into lines using y-coordinate clustering."""

    @staticmethod
    def group_into_lines(words: List[WordBox], y_tolerance: int = 10) -> List[List[WordBox]]:
        if not words:
            return []

        sorted_words = sorted(words, key=lambda w: (w.y, w.x))
        lines: List[List[WordBox]] = []
        current_line = [sorted_words[0]]
        current_y = sorted_words[0].y

        for w in sorted_words[1:]:
            if abs(w.y - current_y) <= y_tolerance:
                current_line.append(w)
            else:
                lines.append(sorted(current_line, key=lambda x: x.x))
                current_line = [w]
                current_y = w.y

        if current_line:
            lines.append(sorted(current_line, key=lambda x: x.x))

        return lines


class KeyValueExtractor:
    """
    Detects "Label: Value" or "Label  Value" patterns common in medical
    intake forms, insurance documents, and structured government forms.
    """

    KEY_PATTERNS = [
        r"^(.+?):\s*(.+)$",                       # "Patient Name: John Doe"
        r"^([A-Z][A-Za-z\s]{2,25})\s{2,}(.+)$",   # "Date of Birth    01/15/1985" (wide gap)
    ]

    COMMON_FORM_LABELS = {
        "name", "patient name", "date of birth", "dob", "date", "address",
        "phone", "phone number", "email", "ssn", "social security",
        "insurance", "policy number", "diagnosis", "physician", "doctor",
        "medication", "dosage", "allergies", "emergency contact",
        "account number", "invoice number", "amount", "total", "signature",
        "id number", "member id", "group number", "employer", "occupation",
    }

    def extract(self, lines: List[List[WordBox]]) -> List[KeyValuePair]:
        pairs = []

        for line_words in lines:
            line_text = " ".join(w.text for w in line_words)

            for pattern in self.KEY_PATTERNS:
                m = re.match(pattern, line_text)
                if m:
                    key = m.group(1).strip().rstrip(":")
                    value = m.group(2).strip()

                    if not value or len(key) > 40:
                        continue

                    # Boost confidence if key matches a known form-label vocabulary
                    is_known_label = key.lower() in self.COMMON_FORM_LABELS
                    avg_conf = float(np.mean([w.confidence for w in line_words])) if line_words else 0.0

                    bbox = self._bbox_for_line(line_words)
                    pairs.append(KeyValuePair(
                        key=key, value=value,
                        confidence=avg_conf,
                        bbox=bbox,
                    ))
                    break  # first matching pattern wins for this line

        return pairs

    @staticmethod
    def _bbox_for_line(words: List[WordBox]) -> Dict[str, int]:
        if not words:
            return {"x": 0, "y": 0, "width": 0, "height": 0}
        x_min = min(w.x for w in words)
        y_min = min(w.y for w in words)
        x_max = max(w.x + w.width for w in words)
        y_max = max(w.y + w.height for w in words)
        return {"x": x_min, "y": y_min, "width": x_max - x_min, "height": y_max - y_min}


class TableExtractor:
    """
    Reconstructs table rows/columns from OCR words that fall within
    detected table regions (from DocumentSegmenter.detect_table_regions).
    Uses x-coordinate clustering to infer column boundaries.
    """

    def extract(
        self, words: List[WordBox], table_region: Dict[str, int], num_cols_hint: Optional[int] = None
    ) -> List[List[TableCell]]:
        # Filter words within the table region
        rx, ry, rw, rh = table_region["x"], table_region["y"], table_region["width"], table_region["height"]
        in_region = [
            w for w in words
            if rx <= w.x <= rx + rw and ry <= w.y <= ry + rh
        ]
        if not in_region:
            return []

        lines = LineGrouper.group_into_lines(in_region, y_tolerance=12)

        # Infer column boundaries via k-means-style clustering on x-centers
        # (simple, dependency-free quantile-based clustering)
        all_x_centers = [w.x + w.width / 2 for line in lines for w in line]
        if not all_x_centers:
            return []

        n_cols = num_cols_hint or self._estimate_column_count(lines)
        col_boundaries = self._compute_column_boundaries(all_x_centers, n_cols)

        table: List[List[TableCell]] = []
        for row_idx, line_words in enumerate(lines):
            row_cells: Dict[int, List[WordBox]] = {i: [] for i in range(n_cols)}
            for w in line_words:
                center = w.x + w.width / 2
                col_idx = self._assign_column(center, col_boundaries)
                row_cells[col_idx].append(w)

            row: List[TableCell] = []
            for col_idx in range(n_cols):
                cell_words = sorted(row_cells[col_idx], key=lambda x: x.x)
                text = " ".join(w.text for w in cell_words)
                conf = float(np.mean([w.confidence for w in cell_words])) if cell_words else 0.0
                row.append(TableCell(row=row_idx, col=col_idx, text=text, confidence=conf))
            table.append(row)

        return table

    @staticmethod
    def _estimate_column_count(lines: List[List[WordBox]]) -> int:
        word_counts = [len(line) for line in lines if len(line) > 0]
        if not word_counts:
            return 1
        return max(1, int(np.median(word_counts)))

    @staticmethod
    def _compute_column_boundaries(x_centers: List[float], n_cols: int) -> List[float]:
        sorted_x = sorted(x_centers)
        if n_cols <= 1:
            return [max(sorted_x) + 1]
        quantiles = np.linspace(0, 100, n_cols + 1)[1:-1]
        boundaries = list(np.percentile(sorted_x, quantiles))
        boundaries.append(max(sorted_x) + 1)
        return boundaries

    @staticmethod
    def _assign_column(x_center: float, boundaries: List[float]) -> int:
        for i, b in enumerate(boundaries):
            if x_center <= b:
                return i
        return len(boundaries) - 1


class DocumentClassifier:
    """Classifies overall document type from layout signals."""

    @staticmethod
    def classify(
        kv_pairs: List[KeyValuePair], tables: List[List[List[TableCell]]], lines: List[List[WordBox]]
    ) -> str:
        has_tables = len(tables) > 0 and any(len(t) > 1 for t in tables)
        has_many_kv = len(kv_pairs) >= 3
        total_lines = len(lines)

        if has_tables and has_many_kv:
            return "mixed"
        if has_tables:
            return "table"
        if has_many_kv and total_lines > 0 and len(kv_pairs) / max(total_lines, 1) > 0.3:
            return "form"
        return "paragraph"


class StructureExtractor:
    """
    Top-level orchestrator: OCRResult → StructuredDocument with key-value
    pairs, tables, paragraph grouping, and document type classification.
    """

    def __init__(self):
        self.kv_extractor = KeyValueExtractor()
        self.table_extractor = TableExtractor()
        self.classifier = DocumentClassifier()

    def extract(
        self, ocr_result: OCRResult, table_regions: Optional[List[Dict[str, int]]] = None,
    ) -> StructuredDocument:
        lines = LineGrouper.group_into_lines(ocr_result.words)
        kv_pairs = self.kv_extractor.extract(lines)

        tables: List[List[List[TableCell]]] = []
        if table_regions:
            for region in table_regions:
                table = self.table_extractor.extract(ocr_result.words, region)
                if table:
                    tables.append(table)  # table is List[List[TableCell]] -> one entry per detected table region

        # Build paragraph groupings: lines NOT captured as key-value pairs
        kv_line_texts = {kv.key + kv.value for kv in kv_pairs}
        paragraph_lines = []
        current_para = []
        for line_words in lines:
            line_text = " ".join(w.text for w in line_words)
            is_kv_line = any(line_text.startswith(kv.key) for kv in kv_pairs)
            if is_kv_line:
                if current_para:
                    paragraph_lines.append(" ".join(current_para))
                    current_para = []
                continue
            current_para.append(line_text)
        if current_para:
            paragraph_lines.append(" ".join(current_para))

        doc_type = self.classifier.classify(kv_pairs, tables, lines)

        metadata = {
            "total_lines": len(lines),
            "total_words": ocr_result.word_count,
            "mean_ocr_confidence": round(ocr_result.mean_confidence, 2),
            "key_value_pairs_found": len(kv_pairs),
            "tables_found": len(tables),
            "ocr_engine": ocr_result.engine,
            "processing_time_ms": round(ocr_result.processing_time_ms, 2),
        }

        return StructuredDocument(
            document_type=doc_type,
            full_text=ocr_result.full_text,
            key_value_pairs=kv_pairs,
            tables=tables,
            paragraphs=paragraph_lines,
            metadata=metadata,
        )
