"""
ItsNotAI - AI Image Detector
Gradio app for Hugging Face Spaces
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
        return None, None, "Please upload an image", None

    # Preprocess
    image = image.convert("RGB")
    inputs = processor(image, return_tensors="pt")

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    # Top-1 决定 + 置信度
    pred_idx = probs.argmax().item()
    predicted_source = source_names[pred_idx]
    confidence = probs[pred_idx].item()
    is_real = source_is_real.get(predicted_source, False)

    # 根据 top-1 预测计算概率
    if is_real:
        human_prob = confidence
        ai_prob = 1.0 - human_prob
    else:
        ai_prob = confidence
        human_prob = 1.0 - ai_prob

    # Get top 3 AI sources only (exclude real sources)
    ai_sources = []
    for i, (name, prob) in enumerate(zip(source_names, probs.tolist())):
        if not source_is_real.get(name, False):
            ai_sources.append({"label": name, "score": round(prob, 3)})

    # Sort by score descending and take top 3
    ai_sources.sort(key=lambda x: x["score"], reverse=True)
    top3_sources = ai_sources[:3]

    # API-style JSON output
    api_output = {
        "ai_probability": round(ai_prob, 3),
        "human_probability": round(human_prob, 3),
        "predicted_source": predicted_source,
        "top3_sources": top3_sources
    }

    # Top predictions for bar chart (keep for UI)
    top_preds = {}
    for i, (name, prob) in enumerate(zip(source_names, probs.tolist())):
        if prob > 0.01:  # Only show >1%
            marker = "[Real]" if source_is_real.get(name, False) else "[AI]"
            top_preds[f"{marker} {name}"] = prob

    # Sort by probability
    top_preds = dict(sorted(top_preds.items(), key=lambda x: x[1], reverse=True)[:10])

    # Summary
    summary = f"""
**Verdict**: {"Real Image" if is_real else "AI Generated"}

**Predicted Source**: {predicted_source}

**Confidence**: {confidence:.2%}

---

### Aggregate Probabilities

| Category | Probability |
|----------|-------------|
| Real | {human_prob:.2%} |
| AI Generated | {ai_prob:.2%} |
"""

    return (
        {"Real": human_prob, "AI Generated": ai_prob},
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

            gr.Examples(
                examples=[],  # Add example images if available
                inputs=input_image,
            )

        with gr.Column(scale=1):
            # Main result
            result_label = gr.Label(
                label="Real vs AI",
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
