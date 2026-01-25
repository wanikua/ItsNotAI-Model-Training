#!/usr/bin/env python3
"""
Fix model config.json to include id2label mapping
"""

import json
from huggingface_hub import hf_hub_download, HfApi

MODEL_ID = "boluobobo/ItsNotAI-ai-detector-v1"


def fix_config(token: str = None):
    # Download current files
    config_path = hf_hub_download(repo_id=MODEL_ID, filename="config.json", token=token)
    meta_path = hf_hub_download(repo_id=MODEL_ID, filename="source_meta.json", token=token)

    # Load
    with open(config_path) as f:
        config = json.load(f)

    with open(meta_path) as f:
        meta = json.load(f)

    source_names = meta["source_names"]

    # Create mappings
    id2label = {str(i): name for i, name in enumerate(source_names)}
    label2id = {name: i for i, name in enumerate(source_names)}

    # Update config
    config["id2label"] = id2label
    config["label2id"] = label2id
    config["num_labels"] = len(source_names)

    # Save locally
    output_path = "/tmp/config.json"
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print("Updated config with labels:")
    for i, name in enumerate(source_names):
        marker = "Real" if meta["source_is_real"].get(name, False) else "AI"
        print(f"  {i}: {name} [{marker}]")

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
    fix_config(args.token)
