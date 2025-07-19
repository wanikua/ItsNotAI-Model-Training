import os
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image
from torch.utils.data import Dataset
import random

from src.utils.data_utils import IMG_EXTENSIONS

class AIRecognitionDataset(Dataset):
    """
        binary classification fo real/fake
    """
    def __init__(self, *, shuffle: bool = False) -> None:
        super().__init__()
        self.root_dir = Path("data/test/archive")
        self.real_dir = self.root_dir / Path("fake-v2")
        self.fake_dir = self.root_dir / Path("real")

        paths = self.get_paths()
        self.data = paths["real"] + paths["fake"]

        if shuffle:
            random.shuffle(self.data)
    
    def get_paths(self) -> dict[str, list[tuple]]:
        """
            returns a dict that contains the paths to every single image along with the label
        """
        real_paths = [ (fname, "real") for fname in self.real_dir.iterdir() if fname.name.lower().endswith(IMG_EXTENSIONS)]
        fake_paths = [ (fname, "fake") for fname in self.fake_dir.iterdir() if fname.name.lower().endswith(IMG_EXTENSIONS)]

        return {
            "real": real_paths,
            "fake": fake_paths
        }

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path, label = self.data[index]
        return img_path, label
