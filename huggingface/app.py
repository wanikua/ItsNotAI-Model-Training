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
MODEL_ID = "boluobobo/ItsNotAI-v1-multiclass"

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
        return None, None, "Please upload an image"

    # Preprocess
    image = image.convert("RGB")
    inputs = processor(image, return_tensors="pt")

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    # Get prediction
    pred_idx = probs.argmax().item()
    predicted_source = source_names[pred_idx]
    confidence = probs[pred_idx].item()
    is_real = source_is_real.get(predicted_source, False)

    # Calculate aggregate real vs fake
    real_prob = sum(
        probs[i].item()
        for i, name in enumerate(source_names)
        if source_is_real.get(name, False)
    )
    fake_prob = 1.0 - real_prob

    # Main result
    if is_real:
        main_result = f"Real Image ({confidence:.1%})"
        result_color = "green"
    else:
        main_result = f"AI Generated ({confidence:.1%})"
        result_color = "red"

    # Top predictions (bar chart data)
    top_preds = {}
    for i, (name, prob) in enumerate(zip(source_names, probs.tolist())):
        if prob > 0.01:  # Only show >1%
            marker = "[Real]" if source_is_real.get(name, False) else "[AI]"
            top_preds[f"{marker} {name}"] = prob

    # Sort by probability
    top_preds = dict(sorted(top_preds.items(), key=lambda x: x[1], reverse=True)[:10])

    # Summary
    summary = f"""
## Detection Result

**Verdict**: {"Real Image" if is_real else "AI Generated"}

**Predicted Source**: {predicted_source}

**Confidence**: {confidence:.2%}

---

### Aggregate Probabilities

| Category | Probability |
|----------|-------------|
| Real | {real_prob:.2%} |
| AI Generated | {fake_prob:.2%} |
"""

    return (
        {"Real": real_prob, "AI Generated": fake_prob},
        top_preds,
        summary
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

    # Event handlers
    submit_btn.click(
        fn=predict,
        inputs=[input_image],
        outputs=[result_label, top_preds_label, summary_md]
    )

    input_image.change(
        fn=predict,
        inputs=[input_image],
        outputs=[result_label, top_preds_label, summary_md]
    )

    gr.Markdown(
        """
        ---

        ### About

        This model is based on **BEiT-Large** fine-tuned on the ArtiFact dataset.

        - **Accuracy**: 93.51%
        - **Model**: [boluobobo/ItsNotAI-v1-multiclass](https://huggingface.co/boluobobo/ItsNotAI-v1-multiclass)

        ### Disclaimer

        This tool is for educational and research purposes. Results should not be used as definitive proof of image authenticity.
        """
    )


if __name__ == "__main__":
    demo.launch()
