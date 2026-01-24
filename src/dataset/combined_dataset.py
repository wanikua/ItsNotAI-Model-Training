"""
统一数据集加载器
支持 ArtiFact + Flux + 自定义数据的混合加载
"""

import os
import sys
import random
from pathlib import Path
from typing import Literal, Optional, List, Tuple, Callable
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision import transforms

# Support both local and Colab imports
try:
    from src.utils.data_utils import IMG_EXTENSIONS
except ImportError:
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def get_default_data_root() -> Path:
    """Get default data directory, works for both local and Colab."""
    if 'google.colab' in sys.modules:
        return Path('/content/data')
    # Try to find the project root
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / 'pyproject.toml').exists():
            return parent / 'data'
    return Path('./data')


def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    """训练用数据增强"""
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        # 模拟 JPEG 压缩伪影 (通过轻微模糊和噪声)
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transforms(img_size: int = 224) -> transforms.Compose:
    """评估用变换"""
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),  # 256 for 224
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class ImageFolderDataset(Dataset):
    """
    简单的图片文件夹数据集
    支持从单个文件夹加载图片并分配标签
    """
    
    def __init__(
        self,
        root_dir: Path,
        label: int,  # 0=real, 1=fake
        transform: Optional[Callable] = None,
        limit: Optional[int] = None,
    ):
        self.root_dir = Path(root_dir)
        self.label = label
        self.transform = transform
        
        # 收集所有图片
        self.image_paths = []
        if self.root_dir.exists():
            for ext in IMG_EXTENSIONS:
                self.image_paths.extend(self.root_dir.rglob(f'*{ext}'))
                self.image_paths.extend(self.root_dir.rglob(f'*{ext.upper()}'))
        
        # 去重并限制数量
        self.image_paths = list(set(self.image_paths))
        if limit:
            self.image_paths = self.image_paths[:limit]
        
        print(f"  Loaded {len(self.image_paths)} images from {root_dir.name} (label={label})")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            # 如果图片损坏，返回一个随机的其他图片
            print(f"Warning: Failed to load {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))
        
        if self.transform:
            img = self.transform(img)
        
        return img, self.label


class ArtiFact_Dataset(Dataset):
    """
    ArtiFact 数据集加载器
    
    目录结构:
    artifact/
    ├── real/
    │   ├── image1.jpg
    │   └── ...
    └── fake/
        ├── image1.jpg
        └── ...
    """
    
    def __init__(
        self,
        root_dir: Path,
        split: Literal["train", "val", "test", "all"] = "train",
        transform: Optional[Callable] = None,
        split_ratio: float = 0.8,
        val_ratio: float = 0.1,
        limit: Optional[int] = None,
        seed: int = 42,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.label_map = {"real": 0, "fake": 1}
        
        # 收集数据
        self.data: List[Tuple[Path, int]] = []
        
        for label_name, label_id in self.label_map.items():
            label_dir = self.root_dir / label_name
            if not label_dir.exists():
                # 尝试其他可能的目录名
                alt_names = {
                    "real": ["real", "Real", "REAL", "authentic", "genuine"],
                    "fake": ["fake", "Fake", "FAKE", "synthetic", "ai", "generated"],
                }
                for alt in alt_names.get(label_name, []):
                    label_dir = self.root_dir / alt
                    if label_dir.exists():
                        break
            
            if label_dir.exists():
                for ext in IMG_EXTENSIONS:
                    for img_path in label_dir.rglob(f'*{ext}'):
                        self.data.append((img_path, label_id))
                    for img_path in label_dir.rglob(f'*{ext.upper()}'):
                        self.data.append((img_path, label_id))
        
        # 去重
        self.data = list(set(self.data))
        
        # 打乱并分割
        random.Random(seed).shuffle(self.data)
        
        if limit:
            self.data = self.data[:limit * 2]  # 保证分割后有足够数据
        
        n = len(self.data)
        train_end = int(n * split_ratio)
        val_end = int(n * (split_ratio + val_ratio))
        
        if split == "train":
            self.data = self.data[:train_end]
        elif split == "val":
            self.data = self.data[train_end:val_end]
        elif split == "test":
            self.data = self.data[val_end:]
        # "all" keeps everything
        
        if limit:
            self.data = self.data[:limit]
        
        # 统计
        real_count = sum(1 for _, l in self.data if l == 0)
        fake_count = sum(1 for _, l in self.data if l == 1)
        print(f"  ArtiFact [{split}]: {len(self.data)} total (real={real_count}, fake={fake_count})")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.data[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Failed to load {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


class CombinedAIDataset(Dataset):
    """
    组合数据集：合并 ArtiFact + Flux + 自定义数据
    
    支持:
    - 自动平衡 real/fake 比例
    - 加权采样
    - 灵活的数据源配置
    """
    
    def __init__(
        self,
        data_root: Optional[Path] = None,
        split: Literal["train", "val", "test"] = "train",
        transform: Optional[Callable] = None,
        limit: Optional[int] = None,
        balance_classes: bool = True,
        include_artifact: bool = True,
        include_flux: bool = True,
        seed: int = 42,
    ):
        if data_root is None:
            data_root = get_default_data_root()
        
        self.data_root = Path(data_root)
        self.split = split
        self.balance_classes = balance_classes
        
        # 设置变换
        if transform is None:
            if split == "train":
                transform = get_train_transforms()
            else:
                transform = get_eval_transforms()
        self.transform = transform
        
        print(f"\n📦 Loading CombinedAIDataset [{split}]")
        print("=" * 50)
        
        self.data: List[Tuple[Path, int]] = []
        
        # 加载 ArtiFact
        if include_artifact:
            artifact_dir = self.data_root / "artifact"
            if artifact_dir.exists():
                artifact_ds = ArtiFact_Dataset(
                    artifact_dir, 
                    split=split, 
                    transform=None,  # 我们统一处理
                    limit=limit,
                    seed=seed,
                )
                self.data.extend(artifact_ds.data)
        
        # 加载 Flux 数据 (假设结构: flux/fake/, flux/real/)
        if include_flux:
            flux_dir = self.data_root / "flux"
            if flux_dir.exists():
                # Flux 生成的图片 (fake)
                for subdir in flux_dir.iterdir():
                    if subdir.is_dir():
                        for ext in IMG_EXTENSIONS:
                            for img_path in subdir.rglob(f'*{ext}'):
                                # 默认 Flux 目录下都是生成图片
                                self.data.append((img_path, 1))
        
        # 加载额外的真实图片
        real_dir = self.data_root / "real"
        if real_dir.exists():
            for ext in IMG_EXTENSIONS:
                for img_path in real_dir.rglob(f'*{ext}'):
                    self.data.append((img_path, 0))
        
        # 去重并打乱
        self.data = list(set(self.data))
        random.Random(seed).shuffle(self.data)
        
        # 平衡类别
        if balance_classes and len(self.data) > 0:
            real_data = [(p, l) for p, l in self.data if l == 0]
            fake_data = [(p, l) for p, l in self.data if l == 1]
            
            min_count = min(len(real_data), len(fake_data))
            if min_count > 0:
                self.data = real_data[:min_count] + fake_data[:min_count]
                random.Random(seed).shuffle(self.data)
        
        # 应用限制
        if limit:
            self.data = self.data[:limit]
        
        # 统计
        real_count = sum(1 for _, l in self.data if l == 0)
        fake_count = sum(1 for _, l in self.data if l == 1)
        print(f"  Total: {len(self.data)} (real={real_count}, fake={fake_count})")
        print("=" * 50 + "\n")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.data[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Failed to load {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))
        
        if self.transform:
            img = self.transform(img)
        
        return img, label
    
    def get_sample_weights(self) -> torch.Tensor:
        """获取用于 WeightedRandomSampler 的权重"""
        labels = [l for _, l in self.data]
        class_counts = [labels.count(0), labels.count(1)]
        weights = [1.0 / class_counts[l] for l in labels]
        return torch.tensor(weights, dtype=torch.float)


def create_dataloaders(
    data_root: Optional[Path] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    limit: Optional[int] = None,
    **dataset_kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建 train/val/test DataLoader
    
    Args:
        data_root: 数据目录
        batch_size: 批次大小
        num_workers: 数据加载线程数
        limit: 限制每个分割的样本数
        **dataset_kwargs: 传递给 CombinedAIDataset 的额外参数
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds = CombinedAIDataset(
        data_root=data_root,
        split="train",
        limit=limit,
        **dataset_kwargs,
    )
    
    val_ds = CombinedAIDataset(
        data_root=data_root,
        split="val",
        limit=limit // 5 if limit else None,
        **dataset_kwargs,
    )
    
    test_ds = CombinedAIDataset(
        data_root=data_root,
        split="test",
        limit=limit // 5 if limit else None,
        **dataset_kwargs,
    )
    
    # 使用加权采样平衡训练数据
    train_weights = train_ds.get_sample_weights()
    train_sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_ds),
        replacement=True,
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试数据集加载
    print("Testing CombinedAIDataset...")
    
    ds = CombinedAIDataset(split="train", limit=100)
    print(f"Dataset size: {len(ds)}")
    
    if len(ds) > 0:
        img, label = ds[0]
        print(f"Sample shape: {img.shape}, label: {label}")
    else:
        print("No data found. Please run download_datasets.py first.")
