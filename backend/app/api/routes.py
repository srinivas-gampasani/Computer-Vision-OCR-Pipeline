"""
app/api/routes.py

REST API for the Computer Vision OCR Pipeline.
"""
import logging
import os
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse

from app.models.schemas import OCRDocumentResponse, SampleDocumentInfo, ErrorResponse
from app.services.ocr_pipeline import OCRPipeline
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

SAMPLE_DOCS = [
    SampleDocumentInfo(
        filename="patient_intake_form.png",
        description="Clean medical intake form with 16 labeled fields",
        document_type_expected="form",
    ),
    SampleDocumentInfo(
        filename="discharge_summary_2col.png",
        description="Two-column cardiology discharge summary (dense paragraph text)",
        document_type_expected="paragraph",
    ),
    SampleDocumentInfo(
        filename="medication_table.png",
        description="6-row medication administration record table",
        document_type_expected="table",
    ),
    SampleDocumentInfo(
        filename="lab_results_low_quality_skewed.png",
        description="Lab results report — artificially degraded with 3.5° rotation, gaussian noise, and blur to test preprocessing robustness",
        document_type_expected="form",
    ),
]


def get_pipeline(options: dict = None) -> OCRPipeline:
    opts = options or {}
    return OCRPipeline(
        prefer_paddle=settings.PREFER_PADDLE,
        ocr_lang=settings.OCR_LANG,
        denoise=opts.get("denoise", settings.DENOISE),
        deskew=opts.get("deskew", settings.DESKEW),
        binarize=opts.get("binarize", settings.BINARIZE),
        upscale_factor=opts.get("upscale_factor", settings.UPSCALE_FACTOR),
        min_word_confidence=opts.get("min_word_confidence", settings.MIN_WORD_CONFIDENCE),
    )


@router.post(
    "/process",
    response_model=OCRDocumentResponse,
    summary="Process an uploaded document image through the full OCR pipeline",
)
async def process_document(
    file: UploadFile = File(...),
    denoise: bool = Form(default=True),
    deskew: bool = Form(default=True),
    binarize: bool = Form(default=True),
    detect_tables: bool = Form(default=True),
    min_word_confidence: float = Form(default=0.0),
):
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {settings.ALLOWED_EXTENSIONS}")

    data = await file.read()
    size_mb = len(data) / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(400, f"File too large: {size_mb:.1f}MB (max {settings.MAX_UPLOAD_SIZE_MB}MB)")

    try:
        pipeline = get_pipeline({
            "denoise": denoise, "deskew": deskew, "binarize": binarize,
            "min_word_confidence": min_word_confidence,
        })
        result = pipeline.process_bytes(data, detect_tables=detect_tables)
        return JSONResponse(content=result.to_dict())
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        raise HTTPException(500, f"Internal processing error: {str(e)}")


@router.get("/samples", response_model=list[SampleDocumentInfo], summary="List available sample documents")
async def list_samples():
    return SAMPLE_DOCS


@router.post(
    "/process-sample/{filename}",
    response_model=OCRDocumentResponse,
    summary="Process one of the bundled sample documents (for demo purposes)",
)
async def process_sample(filename: str):
    valid_names = {s.filename for s in SAMPLE_DOCS}
    if filename not in valid_names:
        raise HTTPException(404, f"Unknown sample: {filename}")

    path = os.path.join(settings.SAMPLE_DOCS_PATH, filename)
    if not os.path.exists(path):
        raise HTTPException(404, f"Sample file not found on disk: {path}")

    try:
        pipeline = get_pipeline()
        result = pipeline.process_file(path)
        return JSONResponse(content=result.to_dict())
    except Exception as e:
        logger.error(f"Sample processing error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@router.get("/engine-info", summary="Report which OCR engine and capabilities are active")
async def engine_info():
    pipeline = get_pipeline()
    return {
        "ocr_engine": pipeline.ocr_router.get_active_engine_name(),
        "tesseract_available": True,
        "paddle_available": pipeline.ocr_router.paddle.is_available if pipeline.ocr_router.paddle else False,
        "language": settings.OCR_LANG,
        "preprocessing_defaults": {
            "denoise": settings.DENOISE,
            "deskew": settings.DESKEW,
            "binarize": settings.BINARIZE,
        },
    }
