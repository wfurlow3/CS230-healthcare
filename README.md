# cs230 project

This repository contains the data processing and model training code used for EHR pretraining and LOS downstream tasks.

## Layout
- `src/`: Python package with data loading, preprocessing, vocab building, and model code.
- `data/`: Expected location for raw and processed data (`data/raw`, `data/processed`).
- `model/`: Serialized model artifacts (config, weights, vocab).
- `main.py`: Entry point to run preprocessing and training end-to-end.
- `requirements.txt`: Python dependencies.

## How to run
From the repo root:
```bash
python main.py
```
This will kick off the preprocessing and training pipeline using the configured settings and data under `data/`.
