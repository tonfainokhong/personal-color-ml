# Personal Color ML

Hierarchical EfficientNet-B3 fine-tuned for personal color (Armocromia) season classification.

Outputs both a **season** prediction (Spring / Summer / Autumn / Winter) and a **sub-class** prediction (e.g. Deep Autumn, Soft Summer).

---

## File Structure

```
personal-color-ml/
├── data/
│   └── roboflow/
│       ├── train/
│       │   ├── autumn/        ← one folder per season
│       │   │   ├── deep/      ← one folder per sub-class (optional)
│       │   │   └── soft/
│       │   ├── spring/
│       │   ├── summer/
│       │   └── winter/
│       ├── valid/
│       └── test/
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── configs/
│   └── config.yaml
├── outputs/               ← created automatically (git-ignored)
│   ├── checkpoints/
│   └── logs/
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Data

Place your Roboflow export under `data/roboflow/` so that the structure matches above.
Folder names must match the season names in `configs/config.yaml` (case-insensitive).

If your Roboflow export uses flat season folders (no sub-class subdirectories), that's fine —
sub-class labels will be skipped automatically and only the season head will be trained.

---

## Training

```bash
# From the project root
python src/train.py

# Custom config
python src/train.py --config configs/config.yaml

# Resume from checkpoint
python src/train.py --resume outputs/checkpoints/best_model.pth
```

Key settings in `configs/config.yaml`:
- `training.freeze_epochs` — number of epochs to keep the backbone frozen (gradual fine-tuning)
- `training.lambda_sub` — weight of sub-class loss relative to season loss
- `training.learning_rate` — initial LR (backbone gets 10× lower LR after unfreeze)

---

## Evaluation

```bash
python src/evaluate.py --checkpoint outputs/checkpoints/best_model.pth --split test
```

Outputs:
- Per-class precision / recall / F1 printed to console
- Confusion matrix PNGs saved to `outputs/plots/`

---

## Inference (single image)

```python
import torch
from torchvision import transforms
from PIL import Image
from src.model import build_model
from src.utils import build_label_maps, load_checkpoint
import yaml

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

season2idx, idx2season, sub2idx, idx2sub, season_sub_indices = build_label_maps(cfg["labels"])

model = build_model(cfg, num_sub_classes=len(sub2idx))
load_checkpoint("outputs/checkpoints/best_model.pth", model)

img = Image.open("path/to/face.jpg").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
x = transform(img).unsqueeze(0)

results = model.predict(x, season_sub_indices, idx2season)
r = results[0]
print(f"Season   : {r['season']}  ({r['season_conf']*100:.1f}%)")
print(f"Sub-class: {idx2sub[r['sub_idx']]}  ({r['sub_conf']*100:.1f}%)")
```

---

## Notes

- **531 images is small** — the freeze-then-unfreeze strategy and ColorJitter augmentation help a lot here.
- Once you have the Deep Armocromia dataset, add it under `data/deeparmocromia/` with the same folder structure and update `data.root` in the config (or create a second config file).
- Sub-class folder names in your Roboflow export must match the keys in `configs/config.yaml → labels.sub_classes`. Update them if needed.
