"""
train.py — Training loop with early stopping.
"""

import argparse
import os
import time

import torch
import torch.optim as optim
import yaml

from dataset import build_dataloaders
from model   import build_model, HierarchicalLoss
from utils   import (
    build_label_maps, setup_logger, save_checkpoint,
    load_checkpoint, count_parameters, save_label_maps,
)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = season_correct = sub_correct = sub_total = n = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, season_labels, sub_labels in loader:
            imgs          = imgs.to(device)
            season_labels = season_labels.to(device)
            sub_labels    = sub_labels.to(device)

            season_logits, sub_logits = model(imgs)
            loss, s_loss, sub_loss    = criterion(
                season_logits, sub_logits, season_labels, sub_labels
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            preds_season   = season_logits.argmax(dim=1)
            season_correct += (preds_season == season_labels).sum().item()

            valid_mask = sub_labels >= 0
            if valid_mask.any():
                preds_sub   = sub_logits[valid_mask].argmax(dim=1)
                sub_correct += (preds_sub == sub_labels[valid_mask]).sum().item()
                sub_total   += valid_mask.sum().item()

            total_loss += loss.item() * imgs.size(0)
            n          += imgs.size(0)

    return total_loss / n, season_correct / n, (sub_correct / sub_total if sub_total > 0 else 0.0)


def main(args):
    # MPS fallback for Apple Silicon (fixes batch norm hang)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    t_cfg = cfg["training"]
    o_cfg = cfg["output"]

    os.makedirs(o_cfg["checkpoint_dir"], exist_ok=True)
    logger = setup_logger(o_cfg["log_dir"])

    season2idx, idx2season, sub2idx, idx2sub, season_sub_indices = \
        build_label_maps(cfg["labels"])
    save_label_maps(season2idx, sub2idx, o_cfg["checkpoint_dir"])
    logger.info(f"Seasons : {season2idx}")
    logger.info(f"Sub-classes ({len(sub2idx)}): {list(sub2idx.keys())}")

    logger.info("Building dataloaders…")
    loaders = build_dataloaders(cfg, season2idx, sub2idx)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model = build_model(cfg, num_sub_classes=len(sub2idx)).to(device)
    logger.info(f"Trainable params: {count_parameters(model):,}")

    if t_cfg["freeze_epochs"] > 0:
        model.freeze_backbone()
        logger.info(f"Backbone frozen for first {t_cfg['freeze_epochs']} epochs.")

    criterion = HierarchicalLoss(lambda_sub=t_cfg["lambda_sub"])
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=t_cfg["learning_rate"],
        weight_decay=t_cfg["weight_decay"],
    )

    if t_cfg["scheduler"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_cfg["epochs"])
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=t_cfg["scheduler_step_size"], gamma=t_cfg["scheduler_gamma"]
        )

    start_epoch  = 0
    best_val_acc = 0.0
    patience     = t_cfg.get("early_stopping_patience", 10)
    no_improve   = 0

    if args.resume:
        start_epoch, best_val_acc = load_checkpoint(args.resume, model, optimizer, scheduler)
        logger.info(f"Resuming from epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")

    for epoch in range(start_epoch, t_cfg["epochs"]):
        if epoch == t_cfg["freeze_epochs"] and t_cfg["freeze_epochs"] > 0:
            model.unfreeze_backbone()
            optimizer = optim.AdamW(
                model.parameters(),
                lr=t_cfg["learning_rate"] * 0.1,
                weight_decay=t_cfg["weight_decay"],
            )
            logger.info("Backbone unfrozen — fine-tuning all layers.")

        t0 = time.time()
        train_loss, train_s_acc, train_sub_acc = run_epoch(
            model, loaders["train"], criterion, optimizer, device, train=True
        )
        val_loss, val_s_acc, val_sub_acc = run_epoch(
            model, loaders.get("valid", loaders["train"]),
            criterion, optimizer, device, train=False
        )
        scheduler.step()
        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch+1:03d}/{t_cfg['epochs']} | "
            f"Train loss {train_loss:.4f}  season {train_s_acc:.3f}  sub {train_sub_acc:.3f} | "
            f"Val loss {val_loss:.4f}  season {val_s_acc:.3f}  sub {val_sub_acc:.3f} | "
            f"{elapsed:.1f}s"
        )

        if val_s_acc > best_val_acc:
            best_val_acc = val_s_acc
            no_improve   = 0
            save_checkpoint(
                {
                    "epoch":           epoch + 1,
                    "model_state":     model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_val_acc":    best_val_acc,
                    "season2idx":      season2idx,
                    "sub2idx":         sub2idx,
                },
                os.path.join(o_cfg["checkpoint_dir"], o_cfg["best_model_name"]),
            )
            logger.info(f"  ✓ New best val season acc: {best_val_acc:.4f}")
        else:
            no_improve += 1
            logger.info(f"  No improvement ({no_improve}/{patience})")
            if no_improve >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}.")
                break

    logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    main(args)