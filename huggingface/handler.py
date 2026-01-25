"""
Custom handler for Hugging Face Inference API
返回聚合后的 AI vs Human 概率
"""

import torch
import json
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor
from huggingface_hub import hf_hub_download
from io import BytesIO


class EndpointHandler:
    def __init__(self, path=""):
        # Load model and processor
        self.model = AutoModelForImageClassification.from_pretrained(path)
        self.processor = AutoImageProcessor.from_pretrained(path)
        self.model.eval()

        # Load source metadata
        try:
            meta_path = hf_hub_download(repo_id=path, filename="source_meta.json", local_files_only=True)
        except:
            meta_path = f"{path}/source_meta.json"

        try:
            with open(meta_path) as f:
                meta = json.load(f)
            self.source_names = meta["source_names"]
            self.source_is_real = meta["source_is_real"]
        except Exception:
            # Fallback
            self.source_names = list(self.model.config.id2label.values())
            self.source_is_real = {
                "afhq": True, "celebahq": True, "coco": True, "ffhq": True,
                "imagenet": True, "landscape": True, "lsun": True, "metfaces": True
            }

    def __call__(self, data):
        """Process inference request"""
        # Handle input
        if isinstance(data, dict):
            image_data = data.get("inputs") or data.get("image")
        else:
            image_data = data

        # Load image
        if isinstance(image_data, bytes):
            image = Image.open(BytesIO(image_data)).convert("RGB")
        elif isinstance(image_data, Image.Image):
            image = image_data.convert("RGB")
        else:
            image = Image.open(BytesIO(image_data)).convert("RGB")

        # Inference
        inputs = self.processor(image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        # Calculate aggregate probabilities
        human_prob = 0.0
        for i, name in enumerate(self.source_names):
            if self.source_is_real.get(name, False):
                human_prob += probs[i].item()
        ai_prob = 1.0 - human_prob

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
            "top3_sources": top3_sources
        }
