"""
app/main.py

Computer Vision OCR Pipeline — main FastAPI application.
"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Computer Vision OCR Pipeline",
    description=(
        "Automated document digitization system combining OpenCV preprocessing "
        "(deskew, denoise, adaptive thresholding), document layout segmentation "
        "(column + table detection), and Tesseract/PaddleOCR character "
        "recognition — extracting structured JSON output from forms, "
        "multi-column documents, and tables."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Computer Vision OCR Pipeline",
        "version": "1.0.0",
    }
