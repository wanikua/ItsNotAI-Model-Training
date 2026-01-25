#!/usr/bin/env python3
"""
Deploy ItsNotAI to Hugging Face Spaces
一键部署脚本

Usage:
    python deploy_to_spaces.py --space-id your-username/your-space-name

Example:
    python deploy_to_spaces.py --space-id boluobobo/ItsNotAI-Demo
"""

import argparse
from huggingface_hub import HfApi, create_repo
from pathlib import Path


def deploy_space(space_id: str, token: str = None):
    """Deploy the Gradio app to Hugging Face Spaces"""

    api = HfApi(token=token)

    # Create or get the space
    try:
        create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True,
            token=token,
        )
        print(f"Created/verified space: {space_id}")
    except Exception as e:
        print(f"Space creation: {e}")

    # Get current directory
    script_dir = Path(__file__).parent

    # Files to upload
    files_to_upload = [
        ("app.py", "app.py"),
        ("requirements.txt", "requirements.txt"),
    ]

    # Upload files
    for local_file, repo_file in files_to_upload:
        local_path = script_dir / local_file
        if local_path.exists():
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_file,
                repo_id=space_id,
                repo_type="space",
                token=token,
            )
            print(f"Uploaded: {local_file} -> {repo_file}")
        else:
            print(f"Warning: {local_file} not found")

    print(f"\nDeployment complete!")
    print(f"View your Space: https://huggingface.co/spaces/{space_id}")


def main():
    parser = argparse.ArgumentParser(description="Deploy ItsNotAI to Hugging Face Spaces")
    parser.add_argument(
        "--space-id",
        type=str,
        default="boluobobo/ItsNotAI-Demo",
        help="Hugging Face Space ID (e.g., username/space-name)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (or set HF_TOKEN env var)"
    )

    args = parser.parse_args()
    deploy_space(args.space_id, args.token)


if __name__ == "__main__":
    main()
