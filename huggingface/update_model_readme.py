#!/usr/bin/env python3
"""
Update the README.md on the Hugging Face model repository

Usage:
    python update_model_readme.py

This will upload the README.md to boluobobo/ItsNotAI-ai-detector-v1
"""

from huggingface_hub import HfApi
from pathlib import Path


def update_readme(model_id: str = "boluobobo/ItsNotAI-ai-detector-v1", token: str = None):
    """Update the model README"""

    api = HfApi(token=token)

    script_dir = Path(__file__).parent
    readme_path = script_dir / "README.md"

    if not readme_path.exists():
        print(f"Error: README.md not found at {readme_path}")
        return

    # Upload README
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=model_id,
        repo_type="model",
        token=token,
    )

    print(f"README.md uploaded to: https://huggingface.co/{model_id}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="boluobobo/ItsNotAI-ai-detector-v1")
    parser.add_argument("--token", default=None, help="HF token")
    args = parser.parse_args()

    update_readme(args.model_id, args.token)
