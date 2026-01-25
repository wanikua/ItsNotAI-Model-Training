"""
ItsNotAI v2 - AI Image Detector
Gradio app for Hugging Face Spaces

Dual-head model: Multi-class + Binary classification
+ FLUX specialist detector (ensemble)
"""

import gradio as gr
import torch
import torch.nn as nn
import json
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor, ViTForImageClassification, ViTImageProcessor
from huggingface_hub import hf_hub_download

# ==============================================
# Main Model (v2 multi-class + binary head)
# ==============================================
MODEL_ID = "boluobobo/ItsNotAI-ai-detector-v2"

print("Loading main model...")
model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model.eval()

# Load source metadata
try:
    meta_path = hf_hub_download(repo_id=MODEL_ID, filename="source_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    source_names = meta["source_names"]
    source_is_real = meta["source_is_real"]
    dual_head = meta.get("dual_head", False)
    hidden_size = meta.get("hidden_size", model.config.hidden_size)
except Exception:
    # Fallback
    source_names = list(model.config.id2label.values())
    source_is_real = {}
    dual_head = False
    hidden_size = model.config.hidden_size

# Load binary head if dual-head mode
binary_head = None
if dual_head:
    try:
        binary_head_path = hf_hub_download(repo_id=MODEL_ID, filename="binary_head.pt")
        binary_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 2),
        )
        binary_head.load_state_dict(torch.load(binary_head_path, map_location="cpu", weights_only=True))
        binary_head.eval()
        print("Binary head loaded - using dual-head mode")
    except Exception as e:
        print(f"Binary head not found, falling back to single-head mode: {e}")
        dual_head = False

print(f"Main model: {len(source_names)} classes, dual_head={dual_head}")

# ==============================================
# FLUX Specialist Detector (ensemble)
# ==============================================
FLUX_MODEL_ID = "ash12321/flux-detector-vit"
FLUX_THRESHOLD = 0.85  # High confidence threshold

flux_model = None
flux_processor = None

try:
    print("Loading FLUX specialist detector...")
    flux_model = ViTForImageClassification.from_pretrained(FLUX_MODEL_ID)
    flux_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    flux_model.eval()
    print(f"FLUX detector loaded (threshold={FLUX_THRESHOLD})")
except Exception as e:
    print(f"FLUX detector not available: {e}")


def get_backbone_features(pixel_values):
    """Extract backbone features (CLS token)"""
    if hasattr(model, 'beit'):
        outputs = model.beit(pixel_values)
        return outputs.last_hidden_state[:, 0]
    elif hasattr(model, 'vit'):
        outputs = model.vit(pixel_values)
        return outputs.last_hidden_state[:, 0]
    else:
        for name, module in model.named_children():
            if name not in ["classifier", "head", "fc"]:
                outputs = module(pixel_values)
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    return outputs.pooler_output
                else:
                    return outputs.last_hidden_state[:, 0]
        raise ValueError("Cannot extract backbone features from this model")


def detect_flux(image: Image.Image):
    """
    FLUX specialist detection.
    Returns (is_flux, confidence) or (None, None) if detector not available.
    """
    if flux_model is None or flux_processor is None:
        return None, None

    try:
        inputs = flux_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = flux_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            # index 0 = Real, index 1 = FLUX
            flux_prob = probs[1].item()
            return flux_prob >= FLUX_THRESHOLD, flux_prob
    except Exception:
        return None, None


def predict(image: Image.Image):
    """Predict if image is real or AI-generated"""
    if image is None:
        return None, None, "Please upload an image", None

    # Preprocess
    image = image.convert("RGB")

    # ==============================================
    # Step 1: FLUX specialist detection (priority)
    # ==============================================
    flux_detected, flux_confidence = detect_flux(image)
    used_flux_detector = False

    if flux_detected:
        # High confidence FLUX detection - use specialist result
        ai_prob = flux_confidence
        human_prob = 1.0 - ai_prob
        is_real = False
        predicted_source = "FLUX"
        used_flux_detector = True

    # ==============================================
    # Step 2: Main model inference
    # ==============================================
    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

        if not used_flux_detector:
            # Dual-head mode: use binary head
            if dual_head and binary_head is not None:
                features = get_backbone_features(inputs["pixel_values"])
                binary_logits = binary_head(features)
                binary_probs = torch.softmax(binary_logits, dim=-1)[0]
                # Binary head: index 0 = Real, index 1 = AI
                human_prob = binary_probs[0].item()
                ai_prob = binary_probs[1].item()
                is_real = human_prob > ai_prob
            else:
                # Fallback: Top-1 decision
                pred_idx = probs.argmax().item()
                predicted_source_top1 = source_names[pred_idx]
                confidence = probs[pred_idx].item()
                is_real = source_is_real.get(predicted_source_top1, False)

                if is_real:
                    human_prob = confidence
                    ai_prob = 1.0 - human_prob
                else:
                    ai_prob = confidence
                    human_prob = 1.0 - ai_prob

    # Top-1 multi-class prediction (for source display)
    pred_idx = probs.argmax().item()
    if not used_flux_detector:
        predicted_source = source_names[pred_idx]
    confidence = probs[pred_idx].item()

    # Get top 3 AI sources only (exclude real sources)
    ai_sources = []
    for i, (name, prob) in enumerate(zip(source_names, probs.tolist())):
        if not source_is_real.get(name, False):
            ai_sources.append({"label": name, "score": round(prob, 3)})

    ai_sources.sort(key=lambda x: x["score"], reverse=True)
    top3_sources = ai_sources[:3]

    # If FLUX detected, add it to top sources
    if used_flux_detector:
        top3_sources.insert(0, {"label": "FLUX (specialist)", "score": round(flux_confidence, 3)})
        top3_sources = top3_sources[:3]

    # API-style JSON output
    api_output = {
        "ai_probability": round(ai_prob, 3),
        "human_probability": round(human_prob, 3),
        "predicted_source": predicted_source,
        "top3_sources": top3_sources,
        "dual_head": dual_head,
        "flux_detected": used_flux_detector,
        "flux_confidence": round(flux_confidence, 3) if flux_confidence else None,
    }

    # Top predictions for bar chart
    top_preds = {}
    for i, (name, prob) in enumerate(zip(source_names, probs.tolist())):
        if prob > 0.01:
            marker = "[Real]" if source_is_real.get(name, False) else "[AI]"
            top_preds[f"{marker} {name}"] = prob

    # Add FLUX specialist result if detected
    if used_flux_detector:
        top_preds["[AI] FLUX (specialist)"] = flux_confidence

    top_preds = dict(sorted(top_preds.items(), key=lambda x: x[1], reverse=True)[:10])

    # Summary
    if used_flux_detector:
        mode_note = " *(FLUX Specialist)*"
    elif dual_head:
        mode_note = " *(Binary Head)*"
    else:
        mode_note = ""

    verdict_emoji = "Human" if is_real else "AI Generated"

    summary = f"""
### Verdict: **{verdict_emoji}**{mode_note}

| Category | Probability |
|----------|-------------|
| Human | {human_prob:.1%} |
| AI Generated | {ai_prob:.1%} |

---

**Predicted Source**: {predicted_source} ({confidence:.1%})
"""

    if used_flux_detector:
        summary += f"\n**FLUX Detector**: {flux_confidence:.1%} confidence"

    summary += "\n\n**Top AI Sources**:"
    for src in top3_sources[:3]:
        summary += f"\n- {src['label']}: {src['score']:.1%}"

    return (
        {"Human": human_prob, "AI Generated": ai_prob},
        top_preds,
        summary,
        api_output
    )


# Custom CSS
css = """
.main-title {
    text-align: center;
    margin-bottom: 1rem;
}
.gradio-container {
    max-width: 1200px !important;
}
"""

# Gradio interface
with gr.Blocks(css=css, title="ItsNotAI v2 - AI Image Detector") as demo:
    gr.Markdown(
        """
        # ItsNotAI v2 - AI Image Detector

        Upload an image to detect if it's **Human-made** or **AI-generated**, and identify the potential source.

        **New in v2**: Dual-head architecture with 95%+ binary accuracy. Supports FLUX, Midjourney, Stable Diffusion, DALL-E, and more.
        """,
        elem_classes="main-title"
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="pil",
                label="Upload Image",
                height=400
            )
            submit_btn = gr.Button("Analyze", variant="primary", size="lg")

        with gr.Column(scale=1):
            # Main result
            result_label = gr.Label(
                label="Human vs AI",
                num_top_classes=2
            )

            # Top predictions
            top_preds_label = gr.Label(
                label="Top Predictions by Source",
                num_top_classes=10
            )

            # Detailed summary
            summary_md = gr.Markdown(label="Details")

            # API-style JSON output
            with gr.Accordion("API Output (JSON)", open=False):
                json_output = gr.JSON(label="API Output")

    # Event handlers
    submit_btn.click(
        fn=predict,
        inputs=[input_image],
        outputs=[result_label, top_preds_label, summary_md, json_output]
    )

    input_image.change(
        fn=predict,
        inputs=[input_image],
        outputs=[result_label, top_preds_label, summary_md, json_output]
    )

    gr.Markdown(
        """
        ---

        ### About ItsNotAI

        Most AI detectors focus on catching AI. **We help artists prove their work is human-made.**

        - **Model**: [boluobobo/ItsNotAI-ai-detector-v2](https://huggingface.co/boluobobo/ItsNotAI-ai-detector-v2)
        - **Binary Accuracy**: 95.07%
        - **Multi-class Accuracy**: 93.47%
        - **Website**: [itsnotai.org](https://itsnotai.org)

        ### Disclaimer

        This tool is for educational and research purposes. Results should not be used as definitive proof of image authenticity.
        """
    )


if __name__ == "__main__":
    demo.launch()
