"""
evaluate.py — Evaluation, confusion matrix, and per-class metrics.

Usage:
    python src/evaluate.py --checkpoint outputs/checkpoints/best_model.pth
    python src/evaluate.py --checkpoint outputs/checkpoints/best_model.pth --split test
"""

import argparse
import os

import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

from dataset import build_dataloaders, get_transforms, PersonalColorDataset
from model   import build_model
from utils   import build_label_maps, load_checkpoint


# ---------------------------------------------------------------------------

def evaluate(model, loader, device, num_seasons, num_sub):
    model.eval()

    all_season_true, all_season_pred = [], []
    all_sub_true,    all_sub_pred    = [], []

    with torch.no_grad():
        for imgs, season_labels, sub_labels in loader:
            imgs          = imgs.to(device)
            season_logits, sub_logits = model(imgs)

            all_season_true.extend(season_labels.tolist())
            all_season_pred.extend(season_logits.argmax(1).cpu().tolist())

            valid_mask = sub_labels >= 0
            if valid_mask.any():
                all_sub_true.extend(sub_labels[valid_mask].tolist())
                all_sub_pred.extend(
                    sub_logits[valid_mask].argmax(1).cpu().tolist()
                )

    return (
        np.array(all_season_true), np.array(all_season_pred),
        np.array(all_sub_true),    np.array(all_sub_pred),
    )


def plot_confusion_matrix(y_true, y_pred, labels, title, save_path):
    cm  = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix → {save_path}")


# ---------------------------------------------------------------------------

def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    season2idx, idx2season, sub2idx, idx2sub, season_sub_indices = \
        build_label_maps(cfg["labels"])

    season_names = [idx2season[i] for i in range(len(idx2season))]
    sub_names    = [idx2sub[i]    for i in range(len(idx2sub))]

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")

    model = build_model(cfg, num_sub_classes=len(sub2idx)).to(device)
    load_checkpoint(args.checkpoint, model)

    loaders = build_dataloaders(cfg, season2idx, sub2idx)
    loader  = loaders.get(args.split)
    if loader is None:
        raise ValueError(f"Split '{args.split}' not found. Available: {list(loaders.keys())}")

    print(f"\nEvaluating on '{args.split}' split…")
    s_true, s_pred, sub_true, sub_pred = evaluate(
        model, loader, device, len(season2idx), len(sub2idx)
    )

    # ── Season metrics ───────────────────────────────────────────────────
    print("\n── Season Classification ──────────────────────────────────────")
    print(classification_report(s_true, s_pred, target_names=season_names, digits=4))

    # ── Sub-class metrics ─────────────────────────────────────────────────
    if len(sub_true) > 0:
        print("── Sub-class Classification ───────────────────────────────────")
        print(classification_report(sub_true, sub_pred, target_names=sub_names, digits=4))

    # ── Confusion matrices ────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    plot_confusion_matrix(
        s_true, s_pred, season_names,
        title="Season Confusion Matrix",
        save_path=os.path.join(args.output_dir, f"cm_season_{args.split}.png"),
    )

    if len(sub_true) > 0:
        plot_confusion_matrix(
            sub_true, sub_pred, sub_names,
            title="Sub-class Confusion Matrix",
            save_path=os.path.join(args.output_dir, f"cm_sub_{args.split}.png"),
        )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--split",      default="test", choices=["train", "valid", "test"])
    parser.add_argument("--output_dir", default="outputs/plots")
    args = parser.parse_args()
    main(args)
