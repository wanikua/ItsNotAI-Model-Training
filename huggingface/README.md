---
license: apache-2.0
language:
- en
- zh
library_name: transformers
pipeline_tag: image-classification
tags:
- ai-detection
- deepfake
- image-classification
- vit
- beit
- pytorch
datasets:
- artifact
metrics:
- accuracy
- f1
- precision
- recall
base_model: microsoft/beit-large-patch16-224
model-index:
- name: ItsNotAI-v1-multiclass
  results:
  - task:
      type: image-classification
      name: AI Image Detection
    metrics:
    - type: accuracy
      value: 0.9351
      name: Accuracy
    - type: f1
      value: 0.9411
      name: F1 Score
    - type: precision
      value: 0.9540
      name: Precision
    - type: recall
      value: 0.9351
      name: Recall
---

# ItsNotAI v1 - Multiclass AI Image Detector

A Vision Transformer model fine-tuned to detect AI-generated images and identify their source generator.

**Website**: [https://itsnotai.org](https://itsnotai.org)

## About ItsNotAI

Most AI detectors focus on catching AI usage. **ItsNotAI takes the opposite approach: helping artists prove their work is human-made.**

### Key Features

- **Verifiable Label**: Beyond just a percentage score, we provide artists with a verifiable "Not AI" label that can be embedded in their work.

- **Industry-Focused**: We specialize in digital painting, manga illustration, and texture design, developed in deep collaboration with 100+ professional artists.

- **Artist-First**: Our industry endorsements and artist partnerships create a trust network that goes beyond pure technical metrics.

## Model Description

This model can:
1. **Detect** whether an image is real or AI-generated
2. **Identify** the specific AI generator used (e.g., Stable Diffusion, DALL-E 3, Midjourney, StyleGAN2, etc.)

### Architecture
- **Base Model**: microsoft/beit-large-patch16-224 (BEiT-Large)
- **Parameters**: ~304M
- **Input Size**: 224x224 pixels
- **Mode**: Multi-class classification

### Supported Sources (33 Classes)

**Real (8 classes)**: afhq, celebahq, coco, ffhq, imagenet, landscape, lsun, metfaces

**AI (25 classes)**: big_gan, cips, cycle_gan, ddpm, denoising_diffusion_gan, diffusion_gan, face_synthetics, gansformer, gau_gan, generative_inpainting, glide, lama, latent_diffusion, mat, palette, pro_gan, projected_gan, sfhq, stable_diffusion, star_gan, stylegan1, stylegan2, stylegan3, taming_transformer, vq_diffusion

## Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 93.51% |
| **Precision** | 95.40% |
| **Recall** | 93.51% |
| **F1 Score** | 94.11% |

## Quick Start

### Installation

```bash
pip install transformers torch pillow
```

### Basic Usage

```python
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import torch

# Load model
model_id = "boluobobo/ItsNotAI-v1-multiclass"
model = AutoModelForImageClassification.from_pretrained(model_id)
processor = AutoImageProcessor.from_pretrained(model_id)

# Load image
image = Image.open("your_image.jpg").convert("RGB")

# Predict
inputs = processor(image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    pred_idx = outputs.logits.argmax(-1).item()

# Get label
label = model.config.id2label[pred_idx]
confidence = probs[0][pred_idx].item()

print(f"Prediction: {label}")
print(f"Confidence: {confidence:.2%}")
```

### With Source Metadata

```python
import json
from huggingface_hub import hf_hub_download

# Download source metadata
meta_path = hf_hub_download(repo_id=model_id, filename="source_meta.json")
with open(meta_path) as f:
    meta = json.load(f)

source_names = meta["source_names"]
source_is_real = meta["source_is_real"]

# Check if prediction is real or AI
predicted_source = source_names[pred_idx]
is_real = source_is_real.get(predicted_source, False)

print(f"Source: {predicted_source}")
print(f"Is Real: {'Yes' if is_real else 'No (AI Generated)'}")

# Get all probabilities
for i, (name, prob) in enumerate(zip(source_names, probs[0].tolist())):
    if prob > 0.01:  # Show only >1%
        marker = "Real" if source_is_real.get(name, False) else "AI"
        print(f"  [{marker}] {name}: {prob:.2%}")
```

### Calculate Real vs Fake Probability

```python
# Aggregate real vs fake probabilities
real_prob = sum(
    probs[0][i].item()
    for i, name in enumerate(source_names)
    if source_is_real.get(name, False)
)
fake_prob = 1.0 - real_prob

print(f"Real: {real_prob:.2%}")
print(f"AI Generated: {fake_prob:.2%}")
```

## Training Details

- **Dataset**: ArtiFact (50K+ images from multiple sources)
- **Epochs**: 10
- **Batch Size**: 64
- **Learning Rate**: 5e-6
- **Optimizer**: AdamW with cosine scheduler
- **Loss**: Focal Loss with label smoothing (0.1)
- **Hardware**: NVIDIA T4 / A100 GPU

## Limitations

- Best performance on 224x224 or larger images
- May have reduced accuracy on heavily compressed images
- Trained primarily on Western-style images
- New AI generators not in training data may not be correctly identified

## Citation

```bibtex
@misc{itsnotai2025,
  title={ItsNotAI: Multi-class AI Image Detection},
  author={ItsNotAI Team},
  year={2025},
  url={https://huggingface.co/boluobobo/ItsNotAI-v1-multiclass}
}
```

## License

Apache 2.0
