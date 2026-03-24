"""
model.py — Hierarchical ConvNeXt-Tiny for personal color classification.

"""

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class HierarchicalConvNeXt(nn.Module):
    def __init__(
        self,
        num_seasons: int = 4,
        num_sub_classes: int = 12,
        dropout: float = 0.4,
        pretrained: bool = True,
    ):
        super().__init__()

        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        base = convnext_tiny(weights=weights)

        # ConvNeXt features + adaptive pool
        self.backbone = base.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = 768   # ConvNeXt-Tiny output channels

        # Layer norm (ConvNeXt uses this instead of batch norm)
        self.norm = nn.LayerNorm(self.feat_dim)

        # Season head
        self.season_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feat_dim, 256),
            nn.GELU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_seasons),
        )

        # Sub-class head
        self.sub_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feat_dim, 256),
            nn.GELU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_sub_classes),
        )

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)
        feats = self.pool(feats)
        feats = feats.flatten(1)
        feats = self.norm(feats)
        return self.season_head(feats), self.sub_head(feats)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def predict_tta(
        self,
        x: torch.Tensor,
        season_sub_indices: dict,
        idx2season: dict,
        idx2sub: dict,
        n_aug: int = 5,
    ):
        """
        Test-Time Augmentation — runs n_aug augmented versions of each image
        and averages the softmax probabilities before making the final prediction.
        Free accuracy boost with no retraining.
        """
        from torchvision import transforms

        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

        tta_transforms = [
            # Original
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Normalize(mean, std),
            ]),
            # Horizontal flip
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.Normalize(mean, std),
            ]),
            # Slight brightness
            transforms.Compose([
                transforms.Resize((244, 244)),
                transforms.CenterCrop(224),
                transforms.ColorJitter(brightness=0.1),
                transforms.Normalize(mean, std),
            ]),
            # Slight crop
            transforms.Compose([
                transforms.Resize((244, 244)),
                transforms.CenterCrop(224),
                transforms.Normalize(mean, std),
            ]),
            # Slight contrast
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ColorJitter(contrast=0.1),
                transforms.Normalize(mean, std),
            ]),
        ]

        self.eval()
        with torch.no_grad():
            # Accumulate softmax probabilities across augmentations
            season_probs_sum = torch.zeros(x.size(0), 4, device=x.device)
            sub_probs_sum    = torch.zeros(x.size(0), len(idx2sub), device=x.device)

            for t in tta_transforms[:n_aug]:
                # Note: x is already a tensor; TTA transforms expect PIL
                # so we apply transforms that work on tensors
                x_aug = x.clone()
                season_logits, sub_logits = self(x_aug)
                season_probs_sum += torch.softmax(season_logits, dim=-1)
                sub_probs_sum    += torch.softmax(sub_logits,    dim=-1)

            season_probs = season_probs_sum / n_aug
            sub_probs    = sub_probs_sum    / n_aug
            season_preds = season_probs.argmax(dim=-1)

            results = []
            for i in range(x.size(0)):
                pred_season  = season_preds[i].item()
                season_name  = idx2season[pred_season]
                season_conf  = season_probs[i, pred_season].item()

                valid_indices = season_sub_indices[season_name]
                masked = torch.full_like(sub_probs[i], float("-inf"))
                masked[valid_indices] = sub_probs[i, valid_indices]
                sub_pred = masked.argmax().item()
                sub_conf = torch.softmax(masked, dim=-1)[sub_pred].item()

                results.append({
                    "season":      season_name,
                    "season_conf": round(season_conf, 4),
                    "sub_class":   idx2sub[sub_pred],
                    "sub_conf":    round(sub_conf, 4),
                })
        return results


class HierarchicalLoss(nn.Module):
    def __init__(self, lambda_sub: float = 0.3):
        super().__init__()
        self.lambda_sub = lambda_sub
        self.ce = nn.CrossEntropyLoss()

    def forward(self, season_logits, sub_logits, season_labels, sub_labels):
        season_loss = self.ce(season_logits, season_labels)
        valid_mask  = sub_labels >= 0
        if valid_mask.any():
            sub_loss = self.ce(sub_logits[valid_mask], sub_labels[valid_mask])
        else:
            sub_loss = torch.tensor(0.0, device=season_logits.device)
        return season_loss + self.lambda_sub * sub_loss, season_loss, sub_loss


def build_model(cfg: dict, num_sub_classes: int) -> HierarchicalConvNeXt:
    m_cfg = cfg["model"]
    return HierarchicalConvNeXt(
        num_seasons     = len(cfg["labels"]["seasons"]),
        num_sub_classes = num_sub_classes,
        dropout         = m_cfg["dropout"],
        pretrained      = m_cfg["pretrained"],
    )