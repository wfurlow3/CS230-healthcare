# src

Core Python package for data preprocessing, vocabulary building, datasets, and model training.

Key modules:
- `build_sequences.py`: Pretraining sequence and vocab builder.
- `los/`: Length-of-stay preprocessing and baselines (see `los/README.md`).
- `data_io.py`: CSV loaders with column validation and date parsing.
- `dataset.py`, `model.py`, `train_mlm.py`: Dataset and modeling code for pretraining.
- `utils.py`: Token and observation helpers.
- `vocab.py`: Vocab construction and I/O.

Typical entry point for full pipeline: run `python main.py` from the repo root.
