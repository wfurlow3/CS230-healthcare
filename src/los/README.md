# LOS preprocessing and baselines

## Preprocess LOS data
From repo root (expects `data/raw` and writes to `data/processed`):
```bash
python -m src.los.preprocess_los
```
This builds LOS sequences, extends/uses vocab, and writes:
- `data/processed/los_instances.jsonl`
- `data/processed/los_features.npz` and `los_labels.npz`

## Baseline models
Run from repo root, using the processed LOS features:
- Logistic regression: `python src/los/baselines/train_los_logreg.py`
- GRU baseline: `python src/los/baselines/train_los_gru.py`
- Transformer baseline: `python src/los/baselines/train_los_transformer.py`

Ensure preprocessing is completed first so the feature/label files exist.
