"""
utils.py — Label maps, checkpointing, logging helpers.
"""

import os
import json
import torch
import logging
from pathlib import Path


# ---------------------------------------------------------------------------
# Label encoding helpers
# ---------------------------------------------------------------------------

def build_label_maps(cfg_labels: dict):
    """
    Given the labels section of config, return:
      - season2idx   : {"autumn": 0, ...}
      - idx2season   : {0: "autumn", ...}
      - sub2idx      : {"autumn_deep": 0, "autumn_soft": 1, ...}  (global)
      - idx2sub      : reverse of sub2idx
      - season_sub_indices : {"autumn": [0,1,2], ...}  slices into sub2idx
    """
    seasons = cfg_labels["seasons"]
    season2idx = {s: i for i, s in enumerate(seasons)}
    idx2season = {i: s for s, i in season2idx.items()}

    sub2idx = {}
    season_sub_indices = {}
    counter = 0
    for season in seasons:
        subs = cfg_labels["sub_classes"][season]
        indices = []
        for sub in subs:
            key = f"{season}_{sub}"
            sub2idx[key] = counter
            indices.append(counter)
            counter += 1
        season_sub_indices[season] = indices

    idx2sub = {v: k for k, v in sub2idx.items()}
    return season2idx, idx2season, sub2idx, idx2sub, season_sub_indices


def encode_labels(season: str, sub: str, season2idx, sub2idx):
    """Return (season_idx, sub_idx) integer tensors."""
    s_idx = season2idx[season.lower()]
    sub_key = f"{season.lower()}_{sub.lower()}"
    sub_idx = sub2idx.get(sub_key, -1)   # -1 → unknown sub-class
    return s_idx, sub_idx


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    logging.info(f"Checkpoint saved → {path}")


def load_checkpoint(path: str, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    logging.info(f"Loaded checkpoint from {path} (epoch {ckpt.get('epoch', '?')})")
    return ckpt.get("epoch", 0), ckpt.get("best_val_acc", 0.0)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logger(log_dir: str, name: str = "train") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_label_maps(season2idx, sub2idx, output_dir: str):
    """Persist label maps to JSON so inference doesn't need config."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "label_maps.json"), "w") as f:
        json.dump({"season2idx": season2idx, "sub2idx": sub2idx}, f, indent=2)
