# CS230 Final Project

This repository contains the data processing and model training code used for EHR pretraining and LOS (length of stay) downstream tasks.

## Layout
- `src/`: Folder containing the bulk of the scripts - data loading, preprocessing, vocab building, and model code.
- `data/`: Expected location for raw and processed data (`data/raw`, `data/processed`). The user has to load this data in theirselves into data/raw. We grabbed the 1K Sample Synthetic Patient Records, CSV from this link: https://synthea.mitre.org/downloads. The data needs to be in an identical CSV format for preprocessing to function.
- `model/`: Model configuration and hyperparameters.
- `main.py`: Main entry script - runs preprocessing and training.
- `requirements.txt`: Python dependencies.

## How to run
From the repo root:
```bash
python main.py
```
This will kick off the preprocessing and training pipeline using the configured settings and data under `data/`.

The most important creations in `data/processed` are:
- `vocab.json`, which contains all unique tokens from the raw data
- `sequences.jsonl`, which gives all sequences of tokens for each patient encounter
- `los_instances.jsonl`, which gives all length-of-stay inputs (sequences of tokens for the first hour of an inpatient encounter) as well as its 0/1 encoding which represents whether the stay went past the median length of stay (slightly above 2 hours).
