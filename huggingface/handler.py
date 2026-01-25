"""
Custom handler for Hugging Face Inference API
返回聚合后的 AI vs Human 概率
"""

import torch
import json
import base64
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor
from io import BytesIO


class EndpointHandler:
    def __init__(self, path=""):
        # Load model and processor
        self.model = AutoModelForImageClassification.from_pretrained(path)
        self.processor = AutoImageProcessor.from_pretrained(path)
        self.model.eval()

        # Load source metadata
        try:
            import os
            meta_path = os.path.join(path, "source_meta.json")
            with open(meta_path) as f:
                meta = json.load(f)
            self.source_names = meta["source_names"]
            self.source_is_real = meta["source_is_real"]
        except Exception:
            # Fallback - use model config
            self.source_names = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]
            self.source_is_real = {
                "afhq": True, "celebahq": True, "coco": True, "ffhq": True,
                "imagenet": True, "landscape": True, "lsun": True, "metfaces": True
            }

    def __call__(self, data):
        """Process inference request"""
        # Handle different input formats
        if isinstance(data, dict):
            image_data = data.get("inputs") or data.get("image") or data.get("data")
        else:
            image_data = data

        # Convert to PIL Image
        image = self._load_image(image_data)

        # Inference
        inputs = self.processor(image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        # Top-1 决定 + 置信度
        top_idx = probs.argmax().item()
        top_source = self.source_names[top_idx]
        top_confidence = probs[top_idx].item()
        is_real = self.source_is_real.get(top_source, False)

        if is_real:
            human_prob = top_confidence
            ai_prob = 1.0 - human_prob
        else:
            ai_prob = top_confidence
            human_prob = 1.0 - ai_prob

        # Get top 3 AI sources
        ai_sources = []
        for i, name in enumerate(self.source_names):
            if not self.source_is_real.get(name, False):
                ai_sources.append({"label": name, "score": round(probs[i].item(), 3)})

        ai_sources.sort(key=lambda x: x["score"], reverse=True)
        top3_sources = ai_sources[:3]

        return {
            "ai_probability": round(ai_prob, 3),
            "human_probability": round(human_prob, 3),
            "predicted_source": top_source,
            "top3_sources": top3_sources
        }

    def _load_image(self, image_data):
        """Load image from various formats"""
        # Already a PIL Image
        if isinstance(image_data, Image.Image):
            return image_data.convert("RGB")

        # Bytes
        if isinstance(image_data, bytes):
            return Image.open(BytesIO(image_data)).convert("RGB")

        # Base64 encoded string
        if isinstance(image_data, str):
            # Remove data URL prefix if present
            if "base64," in image_data:
                image_data = image_data.split("base64,")[1]

            # Decode base64
            image_bytes = base64.b64decode(image_data)
            return Image.open(BytesIO(image_bytes)).convert("RGB")

        # List (could be from JSON)
        if isinstance(image_data, list):
            # Assume it's a nested structure, try first element
            return self._load_image(image_data[0])

        raise ValueError(f"Unsupported image format: {type(image_data)}")
