"""
ItsNotAI v1 - AI Image Detector
Gradio app for Hugging Face Spaces

Multi-class source detection (no binary head)
"""

import gradio as gr
import torch
import json
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor
from huggingface_hub import hf_hub_download

# Model configuration
MODEL_ID = "boluobobo/ItsNotAI-ai-detector-v1"

# Load model and processor
print("Loading model...")
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
except Exception:
    # Fallback
    source_names = list(model.config.id2label.values())
    source_is_real = {}

print(f"Loaded {len(source_names)} classes")


def predict(image: Image.Image):
    """Predict if image is real or AI-generated"""
    if image is None:
        return None, "Please upload an image", None

    # Preprocess
    image = image.convert("RGB")
    inputs = processor(image, return_tensors="pt")

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    # Top-1 多分类预测
    pred_idx = probs.argmax().item()
    predicted_source = source_names[pred_idx]
    confidence = probs[pred_idx].item()
    is_real = source_is_real.get(predicted_source, False)

    # Get top 3 AI sources only (exclude real sources)
    ai_sources = []
    for i, (name, prob) in enumerate(zip(source_names, probs.tolist())):
        if not source_is_real.get(name, False):
            ai_sources.append({"label": name, "score": round(prob, 3)})

    ai_sources.sort(key=lambda x: x["score"], reverse=True)
    top3_sources = ai_sources[:3]

    # API-style JSON output (v1: no binary probabilities)
    api_output = {
        "predicted_source": predicted_source,
        "is_real": is_real,
        "confidence": round(confidence, 3),
        "top3_sources": top3_sources,
    }

    # Top predictions for bar chart
    top_preds = {}
    for i, (name, prob) in enumerate(zip(source_names, probs.tolist())):
        if prob > 0.01:  # Only show >1%
            marker = "[Real]" if source_is_real.get(name, False) else "[AI]"
            top_preds[f"{marker} {name}"] = prob

    top_preds = dict(sorted(top_preds.items(), key=lambda x: x[1], reverse=True)[:10])

    # Summary
    summary = f"""
### Verdict: **{"Human" if is_real else "AI Generated"}**

**Predicted Source**: {predicted_source} ({confidence:.1%})

**Top AI Sources**:
"""
    for src in top3_sources[:3]:
        summary += f"\n- {src['label']}: {src['score']:.1%}"

    return (
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
.result-box {
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}
"""

# Gradio interface
with gr.Blocks(css=css, title="ItsNotAI - AI Image Detector") as demo:
    gr.Markdown(
        """
        # ItsNotAI - AI Image Detector

        Upload an image to detect if it's **real** or **AI-generated**, and identify the potential source.

        Supports: Stable Diffusion, DALL-E, Midjourney, StyleGAN, and more.
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
            # Top predictions by source
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
        outputs=[top_preds_label, summary_md, json_output]
    )

    input_image.change(
        fn=predict,
        inputs=[input_image],
        outputs=[top_preds_label, summary_md, json_output]
    )

    gr.Markdown(
        """
        ---

        ### About

        This model is based on **BEiT-Large** fine-tuned on the ArtiFact dataset.

        - **Accuracy**: 93.51%
        - **Model**: [boluobobo/ItsNotAI-ai-detector-v1](https://huggingface.co/boluobobo/ItsNotAI-ai-detector-v1)

        ### Disclaimer

        This tool is for educational and research purposes. Results should not be used as definitive proof of image authenticity.
        """
    )


if __name__ == "__main__":
    demo.launch()
