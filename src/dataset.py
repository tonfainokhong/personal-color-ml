"""
dataset.py — Unified dataset for both Roboflow and Deep Armocromia.

Supports two layouts automatically:
  1. SUBFOLDER layout (Deep Armocromia):
       root/train/autumn/deep/img.jpg
       root/train/autumn/soft/img.jpg
       ...

  2. CSV layout (Roboflow):
       root/train/_classes.csv
       root/train/img001.jpg
       ...

The layout is detected automatically — if _classes.csv exists, use CSV mode,
otherwise use subfolder mode.
"""

import os
import csv
from pathlib import Path
from typing import Callable, Optional, Tuple
from collections import Counter

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms(split: str, image_size: int = 300):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size + 20, image_size + 20)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.15, hue=0.05),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PersonalColorDataset(Dataset):
    """
    Auto-detects layout and loads samples accordingly.
    Returns (image_tensor, season_idx, sub_idx)
    sub_idx is -1 when sub-class labels are unavailable.
    """

    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(
        self,
        root: str,
        season2idx: dict,
        sub2idx: dict,
        transform: Optional[Callable] = None,
    ):
        self.root       = Path(root)
        self.season2idx = season2idx
        self.sub2idx    = sub2idx
        self.transform  = transform

        # Auto-detect layout
        if (self.root / "_classes.csv").exists():
            self.samples = self._load_csv()
        else:
            self.samples = self._load_subfolders()

        if not self.samples:
            raise RuntimeError(
                f"No images found under {self.root}.\n"
                f"Check that the path is correct and folder/CSV names match config.yaml."
            )

    # ── Subfolder layout ────────────────────────────────────────────────
    def _load_subfolders(self):
        """
        Walks: root/season/sub_class/img.jpg  (or root/season/img.jpg)
        """
        samples = []
        for season_dir in sorted(self.root.iterdir()):
            if not season_dir.is_dir():
                continue
            season = season_dir.name.lower()
            if season not in self.season2idx:
                continue
            s_idx = self.season2idx[season]

            children = list(season_dir.iterdir())
            has_sub_dirs = any(c.is_dir() for c in children)

            if has_sub_dirs:
                for sub_dir in sorted(season_dir.iterdir()):
                    if not sub_dir.is_dir():
                        continue
                    sub_key = f"{season}_{sub_dir.name.lower()}"
                    sub_idx = self.sub2idx.get(sub_key, -1)
                    for img_path in sub_dir.iterdir():
                        if img_path.suffix.lower() in self.IMG_EXTENSIONS:
                            samples.append((img_path, s_idx, sub_idx))
            else:
                for img_path in season_dir.iterdir():
                    if img_path.suffix.lower() in self.IMG_EXTENSIONS:
                        samples.append((img_path, s_idx, -1))

        return samples

    # ── CSV layout ──────────────────────────────────────────────────────
    def _load_csv(self):
        """
        Reads _classes.csv with one-hot columns: filename, fall, spring, summer, winter
        """
        samples = []
        csv_path = self.root / "_classes.csv"
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = [k.strip() for k in reader.fieldnames]
            for row in reader:
                row = {k.strip(): v.strip() for k, v in row.items()}
                img_path = self.root / row["filename"]
                if not img_path.exists():
                    continue
                season_name = None
                for col in fieldnames:
                    if col == "filename":
                        continue
                    if row.get(col) == "1":
                        season_name = col.lower()
                        break
                if season_name is None:
                    continue
                s_idx = self.season2idx.get(season_name)
                if s_idx is None:
                    continue
                samples.append((img_path, s_idx, -1))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int, int]:
        img_path, season_idx, sub_idx = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, season_idx, sub_idx

    def season_counts(self) -> dict:
        return dict(Counter(s for _, s, _ in self.samples))

    def sub_counts(self) -> dict:
        return dict(Counter(sub for _, _, sub in self.samples if sub != -1))


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(cfg: dict, season2idx: dict, sub2idx: dict):
    from torch.utils.data import random_split

    root     = cfg["data"]["root"]
    img_size = cfg["data"]["image_size"]
    workers  = cfg["data"]["num_workers"]
    bs       = cfg["training"]["batch_size"]

    loaders = {}

    # ── Train + carve out 10% as validation ─────────────────────────────
    train_root = os.path.join(root, "train")
    if os.path.isdir(train_root):
        full_ds = PersonalColorDataset(
            root=train_root,
            season2idx=season2idx,
            sub2idx=sub2idx,
            transform=get_transforms("train", img_size),
        )
        val_size   = int(0.1 * len(full_ds))
        train_size = len(full_ds) - val_size
        train_ds, val_ds = random_split(
            full_ds, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Give val split clean transforms (no augmentation)
        val_ds.dataset = PersonalColorDataset(
            root=train_root,
            season2idx=season2idx,
            sub2idx=sub2idx,
            transform=get_transforms("valid", img_size),
        )

        pin = torch.cuda.is_available()
        loaders["train"] = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                      num_workers=workers, pin_memory=pin, drop_last=True)
        loaders["valid"] = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                                      num_workers=workers, pin_memory=pin)
        print(f"  train : {train_size} images | season counts: {full_ds.season_counts()}")
        print(f"  valid : {val_size} images")

    # ── Test split ───────────────────────────────────────────────────────
    test_root = os.path.join(root, "test")
    if os.path.isdir(test_root):
        test_ds = PersonalColorDataset(
            root=test_root,
            season2idx=season2idx,
            sub2idx=sub2idx,
            transform=get_transforms("test", img_size),
        )
        pin = torch.cuda.is_available()
        loaders["test"] = DataLoader(test_ds, batch_size=bs, shuffle=False,
                                     num_workers=workers, pin_memory=pin)
        print(f"   test : {len(test_ds)} images")

    # ── Fallback: valid/ folder if it exists ─────────────────────────────
    valid_root = os.path.join(root, "valid")
    if os.path.isdir(valid_root) and "valid" not in loaders:
        valid_ds = PersonalColorDataset(
            root=valid_root,
            season2idx=season2idx,
            sub2idx=sub2idx,
            transform=get_transforms("valid", img_size),
        )
        pin = torch.cuda.is_available()
        loaders["valid"] = DataLoader(valid_ds, batch_size=bs, shuffle=False,
                                      num_workers=workers, pin_memory=pin)
        print(f"  valid : {len(valid_ds)} images")

    return loaders