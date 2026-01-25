#!/usr/bin/env python3
"""Test model on example images"""

import torch
import requests
from PIL import Image
from io import BytesIO
from transformers import AutoModelForImageClassification, AutoImageProcessor
import json
from huggingface_hub import hf_hub_download

MODEL_ID = "boluobobo/ItsNotAI-ai-detector-v1"

# Load model
print("Loading model...")
model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model.eval()

# Load metadata
meta_path = hf_hub_download(repo_id=MODEL_ID, filename="source_meta.json")
with open(meta_path) as f:
    meta = json.load(f)
source_names = meta["source_names"]
source_is_real = meta["source_is_real"]

def test_image(url, name):
    print(f"\n{'='*50}")
    print(f"Testing: {name}")
    print(f"URL: {url[:80]}...")

    try:
        response = requests.get(url, timeout=30)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        print(f"Image size: {img.size}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    inputs = processor(img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    # Top 5
    top5_idx = probs.argsort(descending=True)[:5]

    print("\nTop 5 Predictions:")
    for idx in top5_idx:
        name = source_names[idx]
        prob = probs[idx].item()
        is_real = source_is_real.get(name, False)
        marker = "Real" if is_real else "AI"
        print(f"  [{marker}] {name}: {prob:.2%}")

    # Aggregate
    real_prob = sum(probs[i].item() for i, n in enumerate(source_names) if source_is_real.get(n, False))
    fake_prob = 1.0 - real_prob

    print(f"\nAggregate: Real={real_prob:.2%}, AI={fake_prob:.2%}")

# Test images
ai_url = "https://substack-post-media.s3.amazonaws.com/public/images/c6e4cb48-6f63-426d-8e55-4f54870592a7_354x338.png"
real_url = "https://substack-post-media.s3.amazonaws.com/public/images/8aba102b-5739-4fd7-8d6c-d479c702f253_1510x983.png"

test_image(ai_url, "AI Generated Image")
test_image(real_url, "Real Image")
