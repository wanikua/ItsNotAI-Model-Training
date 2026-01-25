#!/usr/bin/env python3
"""
上传 source_meta.json 到 Hugging Face 模型仓库
这个文件用于标识哪些来源是真实图片，哪些是AI生成的
"""

import json
from huggingface_hub import hf_hub_download, HfApi

MODEL_ID = "boluobobo/ItsNotAI-ai-detector-v1"

# 真实图片来源 (ArtiFact 数据集中的 8 个真实来源)
REAL_SOURCES = {"afhq", "celebahq", "coco", "ffhq", "imagenet", "landscape", "lsun", "metfaces"}


def upload_source_meta(token: str = None):
    """上传 source_meta.json"""

    # 下载 config.json 获取所有标签
    config_path = hf_hub_download(repo_id=MODEL_ID, filename="config.json", token=token)

    with open(config_path) as f:
        config = json.load(f)

    # 获取所有来源名称
    id2label = config.get("id2label", {})
    source_names = [id2label[str(i)] for i in range(len(id2label))]

    # 构建 source_is_real 映射
    source_is_real = {name: (name in REAL_SOURCES) for name in source_names}

    # 创建 source_meta.json
    source_meta = {
        "source_names": source_names,
        "source_is_real": source_is_real,
        "num_labels": len(source_names),
        "multiclass": True,
    }

    # 保存到临时文件
    output_path = "/tmp/source_meta.json"
    with open(output_path, "w") as f:
        json.dump(source_meta, f, indent=2)

    print("source_meta.json 内容:")
    print(json.dumps(source_meta, indent=2))

    # 统计
    real_count = sum(1 for v in source_is_real.values() if v)
    ai_count = len(source_is_real) - real_count
    print(f"\n共 {len(source_names)} 个来源: {real_count} 真实, {ai_count} AI")

    # 上传到 Hugging Face
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=output_path,
        path_in_repo="source_meta.json",
        repo_id=MODEL_ID,
        repo_type="model",
        token=token,
    )

    print(f"\n✅ source_meta.json 已上传到 {MODEL_ID}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="Hugging Face token")
    args = parser.parse_args()
    upload_source_meta(args.token)
