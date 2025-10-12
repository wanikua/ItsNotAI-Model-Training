import os
from pathlib import Path
from typing import Literal
from PIL import Image
from torch.utils.data import Dataset
import random

from src.utils.data_utils import IMG_EXTENSIONS

class AIRecognitionDataset(Dataset):
    """
    Dataset for binary classification of real vs fake images.
    
    Args:
        root_dir (str | Path): Root folder containing 'real' and 'fake' subfolders.
        split (Literal["train", "val", "all"]): Which subset to load.
        transform (callable, optional): Image transform function.
        shuffle (bool): Whether to shuffle before splitting.
        split_ratio (float): Train/Val ratio (default 0.8).
    """

    def __init__(
        self,
        root_dir: str | Path = "data/test/archive",
        split: Literal["train", "val", "all"] = "train",
        transform=None,
        shuffle: bool = True,
        split_ratio: float = 0.8,
        subset_limit: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.root_dir = Path(root_dir)
        self.fake_dir = self.root_dir / "fake-v2"
        self.real_dir = self.root_dir / "real"
        self.transform = transform
        self.str_label_to_int = {"real": 0, "fake": 1}

        # gather all image paths
        data = self._load_all_data(limit=subset_limit)
        if shuffle:
            random.Random(seed).shuffle(data)

        # split train/val
        split_idx = int(split_ratio * len(data))
        if split == "train":
            self.data = data[:split_idx]
        elif split == "val":
            self.data = data[split_idx:]
        elif split == "all":
            self.data = data
        else:
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'all'.")

    def _load_all_data(self, limit: int | None = None) -> list[tuple[Path, str]]:
        """Return list of (image_path, label) tuples."""
        def valid_ext(name: str) -> bool:
            return any(name.lower().endswith(ext) for ext in IMG_EXTENSIONS)

        real_paths = [(path, "real") for path in self.real_dir.iterdir() if valid_ext(path.name)][:limit]
        fake_paths = [(path, "fake") for path in self.fake_dir.iterdir() if valid_ext(path.name)][:limit]
        return real_paths + fake_paths

    def __getitem__(self, index: int):
        img_path, label = self.data[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.str_label_to_int[label]
        return img, label

    def __len__(self):
        return len(self.data)