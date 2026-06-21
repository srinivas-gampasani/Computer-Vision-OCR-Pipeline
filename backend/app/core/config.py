from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    # OCR engine
    OCR_LANG: str = os.getenv("OCR_LANG", "eng")
    PREFER_PADDLE: bool = os.getenv("PREFER_PADDLE", "false").lower() == "true"
    MIN_WORD_CONFIDENCE: float = float(os.getenv("MIN_WORD_CONFIDENCE", "0"))

    # Preprocessing
    DENOISE: bool = os.getenv("DENOISE", "true").lower() == "true"
    DESKEW: bool = os.getenv("DESKEW", "true").lower() == "true"
    BINARIZE: bool = os.getenv("BINARIZE", "true").lower() == "true"
    UPSCALE_FACTOR: float = float(os.getenv("UPSCALE_FACTOR", "0")) or None

    # ViT layout model (optional)
    USE_VIT_LAYOUT: bool = os.getenv("USE_VIT_LAYOUT", "false").lower() == "true"

    # File handling
    MAX_UPLOAD_SIZE_MB: int = 25
    ALLOWED_EXTENSIONS: set = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".pdf"}
    SAMPLE_DOCS_PATH: str = "data/sample_documents"

    # App
    APP_NAME: str = "Computer Vision OCR Pipeline"
    DEBUG: bool = False

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
