from typing import Tuple
import torch


IMG_EXTENSIONS: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def collate_pil(batch):
    """
    DataLoader default collate will try to make a tensor and fail on PIL Images,
    so we keep the images as a *list* and tensorise only the labels.
    """
    images, labels = zip(*batch)               # tuples → two tuples
    return list(images), list(labels)