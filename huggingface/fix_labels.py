#!/usr/bin/env python3
"""
Fix model config.json to use 'real' label for all real sources
"""

import json
from huggingface_hub import hf_hub_download, HfApi

MODEL_ID = "boluobobo/ItsNotAI-ai-detector-v1"

# Real sources that should be labeled as "real"
REAL_SOURCES = {"afhq", "celebahq", "coco", "ffhq", "imagenet", "landscape", "lsun", "metfaces"}


def fix_labels(token: str = None):
    # Download current config
    config_path = hf_hub_download(repo_id=MODEL_ID, filename="config.json", token=token)

    with open(config_path) as f:
        config = json.load(f)

    # Get current id2label
    old_id2label = config.get("id2label", {})

    # Create new mappings - real sources become "real"
    new_id2label = {}
    new_label2id = {}

    for id_str, label in old_id2label.items():
        if label in REAL_SOURCES:
            new_id2label[id_str] = "real"
        else:
            new_id2label[id_str] = label

    # Build label2id (note: multiple IDs map to "real")
    for id_str, label in new_id2label.items():
        if label not in new_label2id:
            new_label2id[label] = int(id_str)

    # Update config
    config["id2label"] = new_id2label
    config["label2id"] = new_label2id

    # Save locally
    output_path = "/tmp/config.json"
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print("Updated labels:")
    for id_str in sorted(new_id2label.keys(), key=int):
        old_label = old_id2label.get(id_str, "?")
        new_label = new_id2label[id_str]
        if old_label != new_label:
            print(f"  {id_str}: {old_label} -> {new_label}")
        else:
            print(f"  {id_str}: {new_label}")

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
    fix_labels(args.token)
