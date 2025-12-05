# LOS preprocessing and baselines

## Preprocess LOS data
AFTER running main.py from the repo root:
```bash
python -m src.los.preprocess_los
```
This builds LOS sequences, extends/uses vocab, and writes:
- `data/processed/los_instances.jsonl`
- `data/processed/los_features.npz` and `los_labels.npz`

It expects `data/raw` and writes to `data/processed`, so if you haven't run main.py yet, it will have no vocab to draw from.

## Baseline models
Run from repo root, using the processed LOS features:
- Logistic regression: `python src/los/baselines/train_los_logreg.py`
- GRU baseline: `python src/los/baselines/train_los_gru.py`
- Transformer baseline: `python src/los/baselines/train_los_transformer.py`

All baseline models use sequences of one-hot encoded tokens and predict whether an inpatient encounter meets a median threshold for length of stay (around 2 hours).
Ensure preprocessing is completed first so the feature/label files exist.
