# Chromia — Personal Color ML

A computer-vision project for **personal color analysis** (Armocromia). It predicts one of four color seasons—Spring, Summer, Autumn, or Winter—and a compatible subtype within a 12-subtype taxonomy.

**Model:** ImageNet-pretrained **ConvNeXt-Tiny** (~28M parameters) with separate season and subtype classification heads. The configured training dataset is Deep Armocromia, not the original small Roboflow dataset described in an earlier version of this README.

## Project results

| Measure | Result |
| --- | --- |
| Labeled image dataset | **4,920 images** |
| Classification taxonomy | **4 seasons / 12 subtypes** |
| Four-season held-out test accuracy | **55.3%** |
| Four-class chance-level reference | 25% (55.3% is ~2.2× this reference) |
| Test-time augmentation | Reported **2–3% accuracy improvement** |

These are the project's reported final evaluation figures. The dataset, trained weights, and complete experiment logs are not included in this public repository, so these numbers cannot be independently reproduced from the repository alone. The code below documents the training, evaluation, and inference pipelines; it should not be taken as a published benchmark on a downloadable dataset.

## Architecture and approach

1. **Data preparation:** Load images and season/subtype labels using either a directory structure or a Roboflow-style one-hot CSV. Augment training images through crops, flips, mild color jitter, and rotation.
2. **Hierarchical model:** Fine-tune ConvNeXt-Tiny with a shared feature extractor and two supervised classification heads (four seasons and 12 subtypes). The training loss combines season and subtype cross-entropy with a configurable weight.
3. **Training:** Freeze the pretrained backbone for the initial epochs, then unfreeze it for fine-tuning. Use AdamW, learning-rate scheduling, validation-based early stopping, and checkpointing.
4. **Evaluation:** Produce per-class precision, recall, F1, and confusion matrices for season and subtype predictions on an available test split.
5. **Inference:** The FastAPI service retrieves the published checkpoint from Hugging Face, applies five input transformations for test-time augmentation, averages predictions, and constrains subtype selection to the predicted season.

## Repository structure

```text
personal-color-ml/
├── backend/
│   └── main.py            # FastAPI /predict and /health
├── configs/
│   └── config.yaml        # Dataset, labels, model and training settings
├── src/
│   ├── dataset.py         # Dataset loading and transforms
│   ├── model.py           # Hierarchical ConvNeXt-Tiny
│   ├── train.py           # Training, validation and checkpointing
│   ├── evaluate.py        # Per-class metrics and confusion matrices
│   └── utils.py
├── requirements.txt
└── README.md
```

The source dataset and training outputs are intentionally Git-ignored; this repository contains code, not the 4,920 training images or the checkpoint files.

## Setup

Use Python 3.11+ and install the project dependencies:

```bash
pip install -r requirements.txt
```

### Dataset

The current configuration in `configs/config.yaml` uses `data/deeparmocromia` and 224 × 224 image inputs.

Example hierarchical directory structure:

```text
data/deeparmocromia/
├── train/
│   ├── autumn/deep/...
│   ├── autumn/soft/...
│   ├── spring/bright/...
│   ├── summer/cool/...
│   └── winter/deep/...
└── test/
    ├── autumn/deep/...
    ├── spring/bright/...
    ├── summer/cool/...
    └── winter/deep/...
```

The loader also supports a season-only folder layout and Roboflow-style CSV labels. Directory names must match the season and subtype names in `configs/config.yaml`. If a `train/` directory is supplied, `src/dataset.py` creates a seeded 10% internal validation split. A separate `test/` directory is used for held-out evaluation.

### Training

```bash
python src/train.py --config configs/config.yaml
```

With the current configuration, the best checkpoint is written to `outputs/checkpoints_deep/best_model_deep.pth`. To resume from an existing checkpoint:

```bash
python src/train.py --config configs/config.yaml --resume outputs/checkpoints_deep/best_model_deep.pth
```

### Evaluation

```bash
python src/evaluate.py --config configs/config.yaml --checkpoint outputs/checkpoints_deep/best_model_deep.pth --split test
```

Run this only after providing the dataset and a matching checkpoint. The evaluator prints class-level precision/recall/F1 and writes confusion-matrix plots under `outputs/plots/`.

### Local inference API

The FastAPI backend downloads the model checkpoint and label mapping from the Hugging Face repository specified by `HF_REPO_ID` (default: `tonfai-n/chromia-model`). If needed, set `HF_TOKEN` for access.

```bash
uvicorn backend.main:app --reload --port 8000
```

- `GET /health` — API health check
- `POST /predict` — multipart image upload returning season, compatible subtype, confidence scores, and the four-season distribution

The deployment frontend is maintained separately from this model repository.

## Notes and limitations

- The model is a **probabilistic classification system**, not a scientifically validated determination of someone's personal color season.
- Lighting, color balance, camera processing, makeup, and differences in label definitions can affect predictions.
- Reported performance should be interpreted in the context of the project's held-out test set, not as a claim of performance across all users or lighting conditions.
- The five-pass test-time augmentation path used by the inference service is in `backend/main.py`. The standalone `predict_tta` helper in `src/model.py` is not the inference-service implementation.
