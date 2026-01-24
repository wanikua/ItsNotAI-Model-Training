"""
统一数据集加载器
支持 ArtiFact + Flux + 自定义数据的混合加载
"""

import os
import sys
import random
from pathlib import Path
from typing import Literal, Optional, List, Tuple, Callable, Dict
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

    支持两种目录结构:

    1. ArtiFact 官方结构 (通过 metadata.csv):
    artifact/
    ├── ImageNet/
    │   ├── metadata.csv  (含 image_path, target, category)
    │   └── images...
    ├── StyleGAN2/
    │   ├── metadata.csv
    │   └── images...
    └── ...

    2. 简单结构 (real/fake 子目录):
    artifact/
    ├── real/
    └── fake/

    多分类模式 (multiclass=True):
    - 每个来源/生成器作为一个类别
    - 返回 (图片, 来源ID) 而非 (图片, 0/1)
    """

    # 类级别的来源映射 (所有实例共享)
    SOURCE_TO_ID: Dict[str, int] = {}
    ID_TO_SOURCE: Dict[int, str] = {}
    SOURCE_IS_REAL: Dict[str, bool] = {}  # 记录每个来源是否为真实图片

    @classmethod
    def get_num_classes(cls) -> int:
        """获取类别数量"""
        return len(cls.SOURCE_TO_ID)

    @classmethod
    def get_source_names(cls) -> List[str]:
        """获取所有来源名称"""
        return [cls.ID_TO_SOURCE[i] for i in range(len(cls.ID_TO_SOURCE))]

    @classmethod
    def is_real_source(cls, source_id: int) -> bool:
        """判断来源是否为真实图片"""
        source_name = cls.ID_TO_SOURCE.get(source_id, "")
        return cls.SOURCE_IS_REAL.get(source_name, False)

    def __init__(
        self,
        root_dir: Path,
        split: Literal["train", "val", "test", "all"] = "train",
        transform: Optional[Callable] = None,
        split_ratio: float = 0.8,
        val_ratio: float = 0.1,
        limit: Optional[int] = None,
        seed: int = 42,
        multiclass: bool = False,  # 是否使用多分类模式
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.multiclass = multiclass
        self.label_map = {"real": 0, "fake": 1}

        # 收集数据: (图片路径, 来源名称, 是否真实)
        raw_data: List[Tuple[Path, str, bool]] = []

        # 首先尝试查找 metadata.csv 文件 (ArtiFact 官方格式)
        metadata_files = list(self.root_dir.rglob('metadata.csv'))

        # 定义真实图片来源 (ArtiFact 数据集中的 8 个真实来源)
        REAL_SOURCES = {
            'imagenet', 'coco', 'lsun', 'afhq', 'ffhq',
            'metfaces', 'celebahq', 'landscape'
        }

        if metadata_files:
            # 使用 metadata.csv 加载
            import csv
            print(f"  Found {len(metadata_files)} metadata.csv files")
            for meta_path in metadata_files:
                source_name = meta_path.parent.name  # 使用文件夹名作为来源
                is_real = source_name.lower() in REAL_SOURCES
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            # image_path 是相对于 artifact 根目录的路径
                            img_path = self.root_dir / row['image_path']
                            if img_path.exists():
                                raw_data.append((img_path, source_name, is_real))
                except Exception as e:
                    print(f"  Warning: Failed to read {meta_path}: {e}")
        else:
            # 回退到简单的 real/fake 目录结构
            for label_name in ["real", "fake"]:
                label_dir = self.root_dir / label_name
                if not label_dir.exists():
                    alt_names = {
                        "real": ["real", "Real", "REAL", "authentic", "genuine"],
                        "fake": ["fake", "Fake", "FAKE", "synthetic", "ai", "generated"],
                    }
                    for alt in alt_names.get(label_name, []):
                        label_dir = self.root_dir / alt
                        if label_dir.exists():
                            break

                if label_dir.exists():
                    is_real = (label_name == "real")
                    for ext in IMG_EXTENSIONS:
                        for img_path in label_dir.rglob(f'*{ext}'):
                            raw_data.append((img_path, label_name, is_real))
                        for img_path in label_dir.rglob(f'*{ext.upper()}'):
                            raw_data.append((img_path, label_name, is_real))

        # 构建来源映射 (第一次加载时)
        if not ArtiFact_Dataset.SOURCE_TO_ID:
            all_sources = sorted(set(src for _, src, _ in raw_data))
            for idx, src in enumerate(all_sources):
                ArtiFact_Dataset.SOURCE_TO_ID[src] = idx
                ArtiFact_Dataset.ID_TO_SOURCE[idx] = src
            # 记录每个来源是否为真实图片
            for _, src, is_real in raw_data:
                ArtiFact_Dataset.SOURCE_IS_REAL[src] = is_real
            print(f"  Found {len(all_sources)} sources: {all_sources}")

        # 转换为 (图片路径, 标签ID)
        self.data: List[Tuple[Path, int]] = []
        for img_path, source_name, is_real in raw_data:
            if self.multiclass:
                # 多分类: 使用来源ID作为标签
                label = ArtiFact_Dataset.SOURCE_TO_ID.get(source_name, 0)
            else:
                # 二分类: 0=real, 1=fake
                label = 0 if is_real else 1
            self.data.append((img_path, label))
        
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
    - 多分类模式 (识别具体生成器)
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
        multiclass: bool = False,  # 多分类模式
    ):
        if data_root is None:
            data_root = get_default_data_root()

        self.data_root = Path(data_root)
        self.split = split
        self.balance_classes = balance_classes
        self.multiclass = multiclass

        # 设置变换
        if transform is None:
            if split == "train":
                transform = get_train_transforms()
            else:
                transform = get_eval_transforms()
        self.transform = transform

        print(f"\n📦 Loading CombinedAIDataset [{split}]")
        print(f"  Mode: {'Multi-class' if multiclass else 'Binary'}")
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
                    multiclass=multiclass,
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

        # 平衡类别 (仅二分类模式)
        if balance_classes and not multiclass and len(self.data) > 0:
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
        if multiclass:
            from collections import Counter
            label_counts = Counter(l for _, l in self.data)
            print(f"  Total: {len(self.data)} images, {len(label_counts)} classes")
            # 显示前 5 个类别
            for source_id, count in label_counts.most_common(5):
                source_name = ArtiFact_Dataset.ID_TO_SOURCE.get(source_id, f"class_{source_id}")
                print(f"    - {source_name}: {count}")
            if len(label_counts) > 5:
                print(f"    ... and {len(label_counts) - 5} more classes")
        else:
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

    def get_num_classes(self) -> int:
        """获取类别数量"""
        if self.multiclass:
            return ArtiFact_Dataset.get_num_classes()
        return 2

    def get_source_names(self) -> List[str]:
        """获取来源名称列表"""
        if self.multiclass:
            return ArtiFact_Dataset.get_source_names()
        return ["real", "fake"]

    def get_source_is_real(self) -> Dict[str, bool]:
        """获取每个来源是否为真实图片"""
        if self.multiclass:
            return ArtiFact_Dataset.SOURCE_IS_REAL.copy()
        return {"real": True, "fake": False}

    def get_sample_weights(self) -> torch.Tensor:
        """获取用于 WeightedRandomSampler 的权重"""
        from collections import Counter
        labels = [l for _, l in self.data]
        label_counts = Counter(labels)
        # 为每个类别计算权重 (类别越少权重越高)
        weights = [1.0 / label_counts[l] for l in labels]
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
