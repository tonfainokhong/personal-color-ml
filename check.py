import sys
sys.path.insert(0, 'src')
import yaml
from utils import build_label_maps
from dataset import build_dataloaders

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

season2idx, idx2season, sub2idx, idx2sub, season_sub_indices = build_label_maps(cfg["labels"])
loaders = build_dataloaders(cfg, season2idx, sub2idx)

for split, loader in loaders.items():
    imgs, s, sub = next(iter(loader))
    print(f"{split}: batch shape {imgs.shape}, sample labels {s[:4].tolist()}")

print("\nAll good — ready to train!")