"""
app/services/vit_layout_model.py

Vision Transformer document layout understanding wrapper.

Wraps a HuggingFace document layout model (default: microsoft/layoutlmv3-base
fine-tuned for document layout token classification, or a plain ViT image
classifier for document-type classification) so it can be swapped in for the
lightweight heuristic `DocumentClassifier` in structure_extractor.py without
changing any calling code.

Design rationale: ViT/LayoutLMv3 weights are ~500MB-1.5GB and require
torch + transformers + a GPU (or slow CPU inference) to be practical in
production. This module follows the same graceful-degradation pattern used
elsewhere in the pipeline (see ocr_engine.PaddleOCREngine): if the model
can't be loaded, `is_available` is False and the caller falls back to the
fast heuristic classifier — meaning the full pipeline is always runnable,
while still being trivially upgradable to true transformer-based layout
understanding in a GPU-equipped deployment.
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LayoutRegion:
    label: str          # "title" | "text" | "table" | "figure" | "list" | "header" | "footer"
    confidence: float
    bbox: Dict[str, int]


class ViTLayoutModel:
    """
    HuggingFace ViT/LayoutLMv3-based document layout analyzer.

    Usage:
        model = ViTLayoutModel(model_name="microsoft/layoutlmv3-base")
        if model.is_available:
            regions = model.predict_regions(image, ocr_words)
    """

    def __init__(self, model_name: str = "microsoft/layoutlmv3-base", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self._load()

    def _load(self):
        try:
            from transformers import AutoProcessor, AutoModelForTokenClassification
            self.processor = AutoProcessor.from_pretrained(self.model_name, apply_ocr=False)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"ViT layout model loaded: {self.model_name}")
        except ImportError:
            logger.warning(
                "transformers/torch not installed — ViTLayoutModel unavailable. "
                "Falling back to heuristic DocumentClassifier. "
                "Install with: pip install transformers torch"
            )
            self.model = None
        except Exception as e:
            logger.warning(
                f"Could not load {self.model_name} ({e}). "
                "This typically means no internet access to download weights, "
                "or the model requires fine-tuned weights for this task. "
                "Falling back to heuristic DocumentClassifier."
            )
            self.model = None

    @property
    def is_available(self) -> bool:
        return self.model is not None

    def predict_regions(
        self, image: np.ndarray, words: List[Dict[str, Any]],
    ) -> List[LayoutRegion]:
        """
        Predicts layout region labels for OCR word boxes using LayoutLMv3's
        joint text + 2D-position + image embedding.

        words: list of {"text": str, "bbox": [x0,y0,x1,y1]} in normalized
               0-1000 coordinate space (LayoutLMv3 convention).
        """
        if not self.is_available:
            raise RuntimeError("ViT layout model not loaded — check .is_available before calling.")

        import torch
        from PIL import Image as PILImage

        pil_image = PILImage.fromarray(image).convert("RGB")
        texts = [w["text"] for w in words]
        boxes = [w["bbox"] for w in words]

        encoding = self.processor(
            pil_image, texts, boxes=boxes, return_tensors="pt", truncation=True, padding=True,
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding)
            predictions = outputs.logits.argmax(-1).squeeze().tolist()

        id2label = self.model.config.id2label
        regions = []
        for i, (word, box) in enumerate(zip(words, boxes)):
            if i >= len(predictions):
                break
            label = id2label.get(predictions[i] if isinstance(predictions, list) else predictions, "text")
            regions.append(LayoutRegion(
                label=label, confidence=1.0,  # token classification doesn't expose softmax prob by default
                bbox={"x": box[0], "y": box[1], "width": box[2] - box[0], "height": box[3] - box[1]},
            ))
        return regions


class DocumentTypeViT:
    """
    Lightweight ViT image classifier for whole-document type classification
    (e.g. "invoice", "medical_form", "id_card", "handwritten_note",
    "table_report") — distinct from the token-level layout model above.
    Uses a generic ViT image-classification checkpoint as a placeholder;
    in production this would be fine-tuned on a labeled document corpus.
    """

    def __init__(self, model_name: str = "google/vit-base-patch16-224"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._load()

    def _load(self):
        try:
            from transformers import ViTImageProcessor, ViTForImageClassification
            self.processor = ViTImageProcessor.from_pretrained(self.model_name)
            self.model = ViTForImageClassification.from_pretrained(self.model_name)
            self.model.eval()
            logger.info(f"ViT document classifier loaded: {self.model_name}")
        except ImportError:
            logger.warning("transformers/torch not installed — DocumentTypeViT unavailable.")
            self.model = None
        except Exception as e:
            logger.warning(f"Could not load ViT classifier ({e}). Falling back to heuristic classification.")
            self.model = None

    @property
    def is_available(self) -> bool:
        return self.model is not None

    def classify(self, image: np.ndarray) -> Dict[str, Any]:
        if not self.is_available:
            raise RuntimeError("ViT classifier not loaded.")

        import torch
        from PIL import Image as PILImage

        pil_image = PILImage.fromarray(image).convert("RGB")
        inputs = self.processor(images=pil_image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            top_prob, top_idx = probs.max(-1)

        label = self.model.config.id2label[top_idx.item()]
        return {"predicted_class": label, "confidence": round(top_prob.item(), 4)}
