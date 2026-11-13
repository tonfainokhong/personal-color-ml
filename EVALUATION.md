# Chromia — Evaluation Summary

This page records the final-project metrics reported by the developer. The public repository contains the evaluation code but **not** the source images, trained checkpoint, saved predictions, or complete run logs; therefore the figures below are **reported results, not independently reproduced from this repository**.

| Item | Reported result |
| --- | --- |
| Dataset | 4,920 labeled images |
| Classification task | 4 seasons; 12 compatible subtypes |
| Four-season test accuracy | 55.3% |
| Uniform four-class chance reference | 25% (55.3% is ~2.2× this reference) |
| Test-time augmentation | Reported 2–3% accuracy gain (unit not specified in the original result) |

**How evaluation works:** `src/evaluate.py` reads the test split and a trained checkpoint, then prints per-class precision, recall, and F1 scores and saves confusion matrices. The current deployed inference route uses five image transforms, averages the model's predicted probabilities, and masks incompatible subtypes.

To reproduce evaluation with the corresponding private dataset and checkpoint available locally:

```bash
python src/evaluate.py --config configs/config.yaml --checkpoint outputs/checkpoints_deep/best_model_deep.pth --split test
```

The model is exploratory; performance can vary with lighting, camera characteristics, and labeling conventions. These results should not be interpreted as universal color-analysis accuracy.
