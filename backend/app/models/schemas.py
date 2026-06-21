from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class WordResult(BaseModel):
    text: str
    confidence: float
    bbox: BoundingBox
    line: int
    block: int


class KeyValueResult(BaseModel):
    key: str
    value: str
    confidence: float
    bbox: BoundingBox


class TableCellResult(BaseModel):
    row: int
    col: int
    text: str
    confidence: float


class PipelineInfo(BaseModel):
    preprocessing_steps: List[str]
    skew_corrected_degrees: float
    table_regions_detected: int
    columns_detected: int
    total_pipeline_time_ms: float
    ocr_engine_used: str


class OCRDocumentResponse(BaseModel):
    document_type: str
    full_text: str
    key_value_pairs: List[KeyValueResult]
    tables: List[List[TableCellResult]]
    paragraphs: List[str]
    metadata: Dict[str, Any]
    pipeline_info: PipelineInfo
    status: str = "success"


class ProcessOptions(BaseModel):
    denoise: bool = True
    deskew: bool = True
    binarize: bool = True
    detect_tables: bool = True
    min_word_confidence: float = Field(default=0.0, ge=0.0, le=100.0)
    upscale_factor: Optional[float] = Field(default=None, ge=1.0, le=4.0)


class SampleDocumentInfo(BaseModel):
    filename: str
    description: str
    document_type_expected: str


class ErrorResponse(BaseModel):
    status: str = "error"
    message: str
    detail: Optional[str] = None
