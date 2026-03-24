"""
main.py — FastAPI backend for personal color analysis.

Endpoints:
  POST /predict   — accepts image file, returns season + sub-class + confidence
  GET  /health    — health check

Usage:
  cd personal-color-ml
  uvicorn backend.main:app --reload --port 8000
"""

import io
import os
import sys
import json
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model import HierarchicalConvNeXt

app = FastAPI(title="Personal Color API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model at startup ──────────────────────────────────────────────────
HF_REPO  = os.environ.get("HF_REPO_ID", "tonfai-n/chromia-model")
HF_TOKEN = os.environ.get("HF_TOKEN")

CHECKPOINT = hf_hub_download(repo_id=HF_REPO, filename="best_model_deep.pth", token=HF_TOKEN)
LABEL_MAPS = hf_hub_download(repo_id=HF_REPO, filename="label_maps.json", token=HF_TOKEN)

device = torch.device("cpu")  # Render doesn't have MPS — always use CPU

with open(LABEL_MAPS) as f:
    maps = json.load(f)

season2idx = maps["season2idx"]
sub2idx    = maps["sub2idx"]
idx2season = {int(v): k for k, v in season2idx.items()}
idx2sub    = {int(v): k for k, v in sub2idx.items()}

season_sub_indices = {}
for sub_key, sub_idx_val in sub2idx.items():
    parts  = sub_key.split("_")
    season = parts[0]
    season_sub_indices.setdefault(season, []).append(sub_idx_val)

model = HierarchicalConvNeXt(
    num_seasons=len(season2idx),
    num_sub_classes=len(sub2idx),
    dropout=0.4,
    pretrained=False,
).to(device)

ckpt = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Model loaded | device={device} | seasons={list(idx2season.values())}")

# ── TTA transforms ─────────────────────────────────────────────────────────
_mean = [0.485, 0.456, 0.406]
_std  = [0.229, 0.224, 0.225]

tta_transforms = [
    transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(_mean, _std)]),
    transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), transforms.Normalize(_mean, _std)]),
    transforms.Compose([transforms.Resize((244, 244)), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(_mean, _std)]),
    transforms.Compose([transforms.Resize((224, 224)), transforms.ColorJitter(brightness=0.1), transforms.ToTensor(), transforms.Normalize(_mean, _std)]),
    transforms.Compose([transforms.Resize((224, 224)), transforms.ColorJitter(contrast=0.1), transforms.ToTensor(), transforms.Normalize(_mean, _std)]),
]


class PredictionResponse(BaseModel):
    season:      str
    season_conf: float
    sub_class:   str
    sub_conf:    float
    all_seasons: dict


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image.")
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read image.")

    season_probs_sum = torch.zeros(1, len(season2idx), device=device)
    sub_probs_sum    = torch.zeros(1, len(sub2idx),    device=device)

    with torch.no_grad():
        for t in tta_transforms:
            x = t(img).unsqueeze(0).to(device)
            s_logits, sub_logits = model(x)
            season_probs_sum += torch.softmax(s_logits,   dim=-1)
            sub_probs_sum    += torch.softmax(sub_logits, dim=-1)

    season_probs     = season_probs_sum / len(tta_transforms)
    sub_probs        = sub_probs_sum    / len(tta_transforms)
    pred_season_idx  = season_probs.argmax(dim=-1).item()
    pred_season_name = idx2season[pred_season_idx]
    season_conf      = season_probs[0, pred_season_idx].item()

    valid_indices = season_sub_indices[pred_season_name]
    masked = torch.full_like(sub_probs[0], float("-inf"))
    masked[valid_indices] = sub_probs[0, valid_indices]
    pred_sub_idx  = masked.argmax().item()
    pred_sub_name = idx2sub[pred_sub_idx]
    sub_conf      = torch.softmax(masked, dim=-1)[pred_sub_idx].item()

    all_seasons = {idx2season[i]: round(season_probs[0, i].item(), 4)
                   for i in range(len(season2idx))}

    return PredictionResponse(
        season      = pred_season_name,
        season_conf = round(season_conf, 4),
        sub_class   = pred_sub_name,
        sub_conf    = round(sub_conf, 4),
        all_seasons = all_seasons,
    )
