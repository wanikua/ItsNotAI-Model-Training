#!/usr/bin/env python3
"""
Revert model config.json to original 33 specific labels
"""

import json
from huggingface_hub import hf_hub_download, HfApi

MODEL_ID = "boluobobo/ItsNotAI-ai-detector-v1"

# Original 33 labels in order
ORIGINAL_LABELS = [
    "afhq", "big_gan", "celebahq", "cips", "coco", "cycle_gan", "ddpm",
    "denoising_diffusion_gan", "diffusion_gan", "face_synthetics", "ffhq",
    "gansformer", "gau_gan", "generative_inpainting", "glide", "imagenet",
    "lama", "landscape", "latent_diffusion", "lsun", "mat", "metfaces",
    "palette", "pro_gan", "projected_gan", "sfhq", "stable_diffusion",
    "star_gan", "stylegan1", "stylegan2", "stylegan3", "taming_transformer",
    "vq_diffusion"
]


def revert_labels(token: str = None):
    # Download current config
    config_path = hf_hub_download(repo_id=MODEL_ID, filename="config.json", token=token)

    with open(config_path) as f:
        config = json.load(f)

    # Create original mappings
    id2label = {str(i): label for i, label in enumerate(ORIGINAL_LABELS)}
    label2id = {label: i for i, label in enumerate(ORIGINAL_LABELS)}

    # Update config
    config["id2label"] = id2label
    config["label2id"] = label2id
    config["num_labels"] = len(ORIGINAL_LABELS)

    # Save locally
    output_path = "/tmp/config.json"
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Reverted to {len(ORIGINAL_LABELS)} original labels:")
    for i, label in enumerate(ORIGINAL_LABELS):
        print(f"  {i}: {label}")

    # Upload
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=output_path,
        path_in_repo="config.json",
        repo_id=MODEL_ID,
        repo_type="model",
        token=token,
    )

    print(f"\nconfig.json updated on {MODEL_ID}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    revert_labels(args.token)
