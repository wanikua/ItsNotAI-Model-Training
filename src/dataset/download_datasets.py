#!/usr/bin/env python3
"""
数据集下载与准备脚本
支持: ArtiFact + Flux Detector 数据集
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path
from typing import Optional
from tqdm import tqdm

# Optional: for HuggingFace Hub downloads
try:
    from huggingface_hub import hf_hub_download, snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not installed. Install with: pip install huggingface_hub")

# For Kaggle downloads
try:
    import kagglehub
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False


def get_default_data_root() -> Path:
    """Get default data directory, works for both local and Colab."""
    # Check if running in Colab
    if 'google.colab' in sys.modules:
        return Path('/content/data')
    return Path(__file__).parent.parent.parent / 'data'


def download_artifact_dataset(
    data_root: Optional[Path] = None,
    subset: str = "all"  # "all", "train", "test"
) -> Path:
    """
    下载 ArtiFact 数据集
    
    ArtiFact: Large-Scale Dataset of Artificial and Factual Images
    GitHub: https://github.com/awsaf49/artifact
    Kaggle: https://www.kaggle.com/datasets/awsaf49/artifact-dataset
    
    Args:
        data_root: 数据存储根目录
        subset: 下载子集 ("all", "train", "test")
    
    Returns:
        数据集路径
    """
    if data_root is None:
        data_root = get_default_data_root()
    
    artifact_dir = data_root / "artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("📦 Downloading ArtiFact Dataset")
    print("=" * 60)
    
    if KAGGLE_AVAILABLE:
        try:
            # Download via kagglehub
            print("Using kagglehub to download...")
            path = kagglehub.dataset_download("awsaf49/artifact-dataset")
            print(f"Downloaded to: {path}")
            
            # Move to our data directory
            src_path = Path(path)
            if src_path.exists():
                for item in src_path.iterdir():
                    dest = artifact_dir / item.name
                    if not dest.exists():
                        shutil.move(str(item), str(dest))
            
            print(f"✅ ArtiFact dataset ready at: {artifact_dir}")
            return artifact_dir
            
        except Exception as e:
            print(f"Kaggle download failed: {e}")
    
    # Fallback: Manual download instructions
    print("\n" + "=" * 60)
    print("⚠️  自动下载失败，请手动下载:")
    print("=" * 60)
    print("\n选项 1: Kaggle CLI")
    print("  kaggle datasets download -d awsaf49/artifact-dataset")
    print(f"  unzip artifact-dataset.zip -d {artifact_dir}")
    print("\n选项 2: 直接从 Kaggle 网页下载")
    print("  https://www.kaggle.com/datasets/awsaf49/artifact-dataset")
    print(f"  然后解压到: {artifact_dir}")
    print("=" * 60 + "\n")
    
    return artifact_dir


def download_flux_detector_data(
    data_root: Optional[Path] = None,
) -> Path:
    """
    下载/准备 Flux Detector 训练数据
    
    来源: ash12321/flux-detector-final
    该模型使用 18K 图片训练 (10K Flux + 8K 负样本)
    
    注意: HuggingFace 上主要是模型权重，我们需要自己准备类似数据
    这里提供下载公开 Flux 图片的方案
    """
    if data_root is None:
        data_root = get_default_data_root()
    
    flux_dir = data_root / "flux"
    flux_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("📦 Preparing Flux Detection Data")
    print("=" * 60)
    
    # Try to download from HuggingFace datasets that contain Flux images
    if HF_AVAILABLE:
        try:
            # Try to download Flux-related datasets
            datasets_to_try = [
                "saq1b/flux-generated-images",
                "Rapidata/Flux-1-pro_t2i",
            ]
            
            for dataset_name in datasets_to_try:
                try:
                    print(f"Trying to download: {dataset_name}")
                    path = snapshot_download(
                        repo_id=dataset_name,
                        repo_type="dataset",
                        local_dir=flux_dir / dataset_name.split("/")[-1],
                        ignore_patterns=["*.md", "*.txt"]
                    )
                    print(f"✅ Downloaded: {dataset_name}")
                except Exception as e:
                    print(f"  Skipped {dataset_name}: {e}")
                    continue
                    
        except Exception as e:
            print(f"HuggingFace download failed: {e}")
    
    # Provide instructions for generating Flux images
    print("\n" + "=" * 60)
    print("💡 Flux 图片数据准备建议:")
    print("=" * 60)
    print("""
选项 1: 使用 Flux API 生成图片
  - 访问 https://replicate.com/black-forest-labs/flux-schnell
  - 或使用 fal.ai 的 Flux API
  
选项 2: 下载现有 Flux 图片集
  - CivitAI 上有很多 Flux 生成的示例图片
  - 搜索 "Flux" 标签的图片
  
选项 3: 本地运行 Flux 模型
  - 使用 ComfyUI + Flux 模型
  - 批量生成训练图片
  
建议: 生成 5000-10000 张 Flux 图片用于训练
      配对等量真实图片 (可用 Unsplash API 下载)
""")
    print("=" * 60 + "\n")
    
    return flux_dir


def download_real_images(
    data_root: Optional[Path] = None,
    num_images: int = 5000,
) -> Path:
    """
    下载真实图片用于负样本
    
    来源: Unsplash, Pexels 等免费图片库
    """
    if data_root is None:
        data_root = get_default_data_root()
    
    real_dir = data_root / "real"
    real_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("📦 Downloading Real Images")
    print("=" * 60)
    
    # Try Unsplash dataset from HuggingFace
    if HF_AVAILABLE:
        try:
            print("Downloading real images from HuggingFace...")
            # Some open image datasets
            datasets_to_try = [
                "reach-vb/unsplash_small",
            ]
            
            for dataset_name in datasets_to_try:
                try:
                    path = snapshot_download(
                        repo_id=dataset_name,
                        repo_type="dataset",
                        local_dir=real_dir / "unsplash",
                        ignore_patterns=["*.md", "*.txt", "*.json"]
                    )
                    print(f"✅ Downloaded real images")
                    return real_dir
                except Exception:
                    continue
                    
        except Exception as e:
            print(f"Download failed: {e}")
    
    print("\n💡 手动下载真实图片:")
    print("  - Unsplash API: https://unsplash.com/developers")
    print("  - Pexels API: https://www.pexels.com/api/")
    print(f"  将图片放入: {real_dir}")
    
    return real_dir


def prepare_combined_dataset(data_root: Optional[Path] = None) -> dict:
    """
    准备完整的组合数据集
    
    Returns:
        包含各数据集路径的字典
    """
    if data_root is None:
        data_root = get_default_data_root()
    
    data_root.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("🚀 准备 AI Image Detection 训练数据集")
    print("=" * 60 + "\n")
    
    paths = {
        "artifact": download_artifact_dataset(data_root),
        "flux": download_flux_detector_data(data_root),
        "real": download_real_images(data_root),
    }
    
    print("\n" + "=" * 60)
    print("📊 数据集准备完成!")
    print("=" * 60)
    print(f"ArtiFact: {paths['artifact']}")
    print(f"Flux:     {paths['flux']}")
    print(f"Real:     {paths['real']}")
    print("=" * 60 + "\n")
    
    return paths


def verify_dataset(data_root: Optional[Path] = None) -> dict:
    """验证数据集完整性并返回统计信息"""
    if data_root is None:
        data_root = get_default_data_root()
    
    stats = {}
    img_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    
    for subdir in ['artifact', 'flux', 'real']:
        path = data_root / subdir
        if path.exists():
            count = sum(
                1 for f in path.rglob('*') 
                if f.suffix.lower() in img_extensions
            )
            stats[subdir] = count
        else:
            stats[subdir] = 0
    
    print("\n📊 数据集统计:")
    for name, count in stats.items():
        status = "✅" if count > 0 else "❌"
        print(f"  {status} {name}: {count} 张图片")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="下载 AI Image Detection 训练数据集")
    parser.add_argument("--data-root", type=str, default=None, help="数据存储目录")
    parser.add_argument("--verify-only", action="store_true", help="仅验证现有数据集")
    parser.add_argument("--artifact-only", action="store_true", help="仅下载 ArtiFact")
    parser.add_argument("--flux-only", action="store_true", help="仅下载 Flux 数据")
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root) if args.data_root else None
    
    if args.verify_only:
        verify_dataset(data_root)
    elif args.artifact_only:
        download_artifact_dataset(data_root)
    elif args.flux_only:
        download_flux_detector_data(data_root)
    else:
        prepare_combined_dataset(data_root)
        verify_dataset(data_root)
