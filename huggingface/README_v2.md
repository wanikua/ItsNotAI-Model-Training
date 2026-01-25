---
license: apache-2.0
language:
- en
- zh
library_name: transformers
pipeline_tag: image-classification
tags:
- ai-detection
- ai-image-detection
- deepfake-detection
- fake-image-detection
- ai-art-detection
- stable-diffusion-detection
- midjourney-detection
- dall-e-detection
- flux-detection
- image-classification
- image-forensics
- digital-art-verification
- vit
- beit
- pytorch
- computer-vision
- dual-head
datasets:
- artifact
metrics:
- accuracy
- f1
- precision
- recall
base_model: microsoft/beit-large-patch16-224
model-index:
- name: ItsNotAI-ai-detector-v2
  results:
  - task:
      type: image-classification
      name: AI Image Detection
    metrics:
    - type: accuracy
      value: 0.9347
      name: Accuracy
    - type: f1
      value: 0.9340
      name: F1 Score
    - type: precision
      value: 0.9339
      name: Precision
    - type: recall
      value: 0.9347
      name: Recall
  - task:
      type: image-classification
      name: Binary AI Detection
    metrics:
    - type: accuracy
      value: 0.9507
      name: Binary Accuracy
    - type: f1
      value: 0.9638
      name: Binary F1 Score
    - type: precision
      value: 0.9637
      name: Binary Precision
    - type: recall
      value: 0.9639
      name: Binary Recall
---

# ItsNotAI v2 - Dual-Head AI Image Detector

> **Detect AI-generated images | Identify the AI generator | Verify human-made artwork**

A Vision Transformer model with **dual-head architecture** that detects AI-generated images and identifies the specific AI generator used.

**Website**: [https://itsnotai.org](https://itsnotai.org)

> **Note**: This is one of the models used by ItsNotAI. For official verification at [itsnotai.org](https://itsnotai.org), we use an ensemble of multiple models combined with human expert review to ensure maximum accuracy.

---

## Model Versions

| Version | Architecture | Best For |
|---------|--------------|----------|
| **v2 (Latest)** | Dual-head | Real/AI detection + Source ID |
| [v1](https://huggingface.co/boluobobo/ItsNotAI-ai-detector-v1) | Single-head | Source identification only |

**What's New in v2**:
- Dual-head architecture: dedicated binary classifier for Real/AI detection
- FLUX image detection support
- Improved Midjourney detection
- Enhanced real photo recognition with class weights

---

## Performance

### Binary Classification (Real vs AI)

| Metric | Value |
|--------|-------|
| **Accuracy** | 95.07% |
| **Precision** | 96.37% |
| **Recall** | 96.39% |
| **F1 Score** | 96.38% |
| **AUC** | 0.990 |

### Multi-class Classification (Source Identification)

| Metric | Value |
|--------|-------|
| **Accuracy** | 93.47% |
| **Precision** | 93.39% |
| **Recall** | 93.47% |
| **F1 Score** | 93.40% |
| **AUC** | 0.994 |

---

## Quick Start

### Installation

```bash
pip install transformers torch pillow huggingface_hub
```

### Recommended Usage (Dual-Head)

```python
import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification, AutoImageProcessor
from huggingface_hub import hf_hub_download
from PIL import Image
import json

# Load model
model_id = "boluobobo/ItsNotAI-ai-detector-v2"
model = AutoModelForImageClassification.from_pretrained(model_id)
processor = AutoImageProcessor.from_pretrained(model_id)
model.eval()

# Load metadata and binary head
meta_path = hf_hub_download(repo_id=model_id, filename="source_meta.json")
with open(meta_path) as f:
    meta = json.load(f)

source_names = meta["source_names"]
source_is_real = meta["source_is_real"]
hidden_size = meta.get("hidden_size", model.config.hidden_size)

# Load binary classification head
binary_head_path = hf_hub_download(repo_id=model_id, filename="binary_head.pt")
binary_head = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden_size, 2))
binary_head.load_state_dict(torch.load(binary_head_path, map_location="cpu"))
binary_head.eval()

def get_backbone_features(pixel_values):
    """Extract CLS token features from backbone"""
    if hasattr(model, 'beit'):
        outputs = model.beit(pixel_values)
    elif hasattr(model, 'vit'):
        outputs = model.vit(pixel_values)
    return outputs.last_hidden_state[:, 0]

def detect_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        # Multi-class prediction (source identification)
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

        # Binary prediction (Real vs AI)
        features = get_backbone_features(inputs["pixel_values"])
        binary_logits = binary_head(features)
        binary_probs = torch.softmax(binary_logits, dim=-1)[0]

        human_prob = binary_probs[0].item()  # index 0 = Real
        ai_prob = binary_probs[1].item()     # index 1 = AI

    # Get predicted source
    pred_idx = probs.argmax().item()
    predicted_source = source_names[pred_idx]

    # Get top 3 AI sources
    ai_sources = []
    for i, name in enumerate(source_names):
        if not source_is_real.get(name, False):
            ai_sources.append({"label": name, "score": round(probs[i].item(), 3)})
    ai_sources.sort(key=lambda x: x["score"], reverse=True)

    return {
        "ai_probability": round(ai_prob, 3),
        "human_probability": round(human_prob, 3),
        "predicted_source": predicted_source,
        "is_real": human_prob > ai_prob,
        "top3_sources": ai_sources[:3]
    }

# Example
result = detect_image("test.jpg")
print(f"AI Probability: {result['ai_probability']:.1%}")
print(f"Human Probability: {result['human_probability']:.1%}")
print(f"Verdict: {'Real' if result['is_real'] else 'AI Generated'}")
print(f"Predicted Source: {result['predicted_source']}")
```

**Example Output:**
```json
{
  "ai_probability": 0.892,
  "human_probability": 0.108,
  "predicted_source": "midjourney",
  "is_real": false,
  "top3_sources": [
    {"label": "midjourney", "score": 0.756},
    {"label": "stable_diffusion", "score": 0.089},
    {"label": "flux", "score": 0.042}
  ]
}
```

### Basic Usage (Single-Head Fallback)

```python
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import torch

# Load model
model_id = "boluobobo/ItsNotAI-ai-detector-v2"
model = AutoModelForImageClassification.from_pretrained(model_id)
processor = AutoImageProcessor.from_pretrained(model_id)

# Real source labels
REAL_LABELS = {"afhq", "celebahq", "coco", "ffhq", "imagenet", "landscape", "lsun", "metfaces"}

# Load and predict
image = Image.open("your_image.jpg").convert("RGB")
inputs = processor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]

pred_idx = probs.argmax().item()
label = model.config.id2label[str(pred_idx)]
confidence = probs[pred_idx].item()
is_real = label in REAL_LABELS

print(f"Prediction: {label}")
print(f"Confidence: {confidence:.2%}")
print(f"Is Real: {is_real}")
```

---

## Model Description

### Architecture

- **Base Model**: microsoft/beit-large-patch16-224 (BEiT-Large)
- **Parameters**: ~304M
- **Input Size**: 224x224 pixels
- **Mode**: Dual-head (Multi-class + Binary classification)

### Dual-Head Design

```
Input Image
     │
     ▼
┌─────────────┐
│  BEiT-Large │ (Backbone)
│   Encoder   │
└──────┬──────┘
       │
   CLS Token
       │
       ├────────────────┬────────────────┐
       ▼                ▼                │
┌─────────────┐  ┌─────────────┐         │
│ Multi-class │  │   Binary    │         │
│    Head     │  │    Head     │         │
│ (33 classes)│  │ (Real/AI)   │         │
└─────────────┘  └─────────────┘         │
       │                │                │
       ▼                ▼                │
 Source ID         Real vs AI           │
 (e.g., SD,        Probability          │
  Midjourney)                           │
```

**Why Dual-Head?**
- Binary head provides more accurate Real/AI classification
- Multi-class head identifies the specific AI generator
- Best of both worlds: accurate detection + source identification

### Output Labels (33 Classes)

**Real Sources (8)**:
- afhq, celebahq, coco, ffhq, imagenet, landscape, lsun, metfaces

**AI Sources (25)**:

| Category | Models |
|----------|--------|
| **Diffusion** | stable_diffusion, latent_diffusion, ddpm, vq_diffusion, palette, diffusion_gan, denoising_diffusion_gan, flux |
| **GAN** | stylegan1, stylegan2, stylegan3, pro_gan, big_gan, cycle_gan, star_gan, gansformer, projected_gan |
| **Commercial** | midjourney, dalle, glide |
| **Other** | gau_gan, taming_transformer, generative_inpainting, lama, mat, cips, face_synthetics, sfhq |

---

## About ItsNotAI

Most AI detectors focus on catching AI usage. **ItsNotAI takes the opposite approach: helping artists prove their work is human-made.**

### Key Features

- **Verifiable Label**: Beyond just a percentage score, we provide artists with a verifiable "Not AI" label that can be embedded in their work.
- **Industry-Focused**: We specialize in digital painting, manga illustration, and texture design, developed in deep collaboration with 100+ professional artists.
- **Artist-First**: Our industry endorsements and artist partnerships create a trust network that goes beyond pure technical metrics.

### Use Cases

- **Artists & Creators**: Prove your artwork is human-made, protect your reputation
- **Stock Photo Platforms**: Filter AI-generated uploads, maintain content quality
- **Social Media Moderation**: Detect AI-generated profile pictures and fake content
- **News & Media**: Verify photo authenticity, combat misinformation
- **NFT Marketplaces**: Ensure digital art authenticity
- **Academic Research**: Study AI image generation patterns

---

## Training Details

- **Base**: Fine-tuned from [ItsNotAI-ai-detector-v1](https://huggingface.co/boluobobo/ItsNotAI-ai-detector-v1)
- **Dataset**: ArtiFact + FLUX + Midjourney + Real photos (~50K+ images)
- **Architecture**: Dual-head (multi-class + binary classification)
- **Epochs**: 10
- **Batch Size**: 64
- **Learning Rate**: 5e-6
- **Optimizer**: AdamW with cosine scheduler
- **Loss**: Focal Loss with label smoothing (0.1)
- **Binary Class Weights**: [1.5, 1.0] (boost real photo recognition)
- **Hardware**: NVIDIA A100 GPU

---

## v1 vs v2 Comparison

| Feature | v1 | v2 |
|---------|----|----|
| Architecture | Single-head | **Dual-head** |
| FLUX Detection | No | **Yes** |
| Midjourney Enhanced | Basic | **Improved** |
| Binary Classification | Derived from top-1 | **Dedicated head** |

---

## Files in This Repository

| File | Description |
|------|-------------|
| `config.json` | Model configuration |
| `model.safetensors` | Model weights |
| `preprocessor_config.json` | Image processor config |
| `source_meta.json` | Source names and metadata |
| `binary_head.pt` | Binary classification head weights |
| `handler.py` | Custom inference handler |

---

## API / Inference Endpoint

Deploy as a Hugging Face Inference Endpoint for production use:

```python
import requests
import base64

API_URL = "https://api-inference.huggingface.co/models/boluobobo/ItsNotAI-ai-detector-v2"
headers = {"Authorization": "Bearer YOUR_HF_TOKEN"}

with open("image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

response = requests.post(API_URL, headers=headers, json={"inputs": image_base64})
print(response.json())
```

---

## FAQ

**Q: What's the difference between v1 and v2?**
A: v2 adds a dedicated binary classification head for improved Real/AI detection, plus enhanced FLUX and Midjourney detection.

**Q: Can this detect FLUX images?**
A: Yes! v2 is specifically trained on FLUX-generated images.

**Q: Can this detect Midjourney images?**
A: Yes, with improved detection compared to v1.

**Q: Should I use the binary head or multi-class head?**
A: Use the binary head for Real/AI classification (more accurate), and multi-class head if you need to identify the specific AI generator.

**Q: What image formats are supported?**
A: PNG, JPG, WEBP, and other common formats. Images are automatically resized to 224x224 for processing.

---

## Limitations

- Best performance on 224x224 or larger images
- May have reduced accuracy on heavily compressed images
- Trained primarily on Western-style images
- New AI generators not in training data may not be correctly identified
- FLUX detection is based on limited training samples

---

## Citation

```bibtex
@misc{itsnotai2025v2,
  title={ItsNotAI v2: Dual-Head AI Image Detection},
  author={ItsNotAI Team},
  year={2025},
  url={https://huggingface.co/boluobobo/ItsNotAI-ai-detector-v2}
}
```

---

## License

Apache 2.0

---

## Links

- **v1 Model**: [boluobobo/ItsNotAI-ai-detector-v1](https://huggingface.co/boluobobo/ItsNotAI-ai-detector-v1)
- **Demo Space**: [ItsNotAI Demo](https://huggingface.co/spaces/boluobobo/ItsNotAI-Demo)
- **Website**: [https://itsnotai.org](https://itsnotai.org)
