cs230 EHR Pipeline
==================

Overview
--------
This project builds tokenized EHR sequences, trains a masked language model (MLM), and provides downstream length-of-stay (LOS) baselines.

High-level steps
----------------
1) Run `main.py` to build sequences/vocab and train the MLM.
   - This writes `data/processed/sequences.jsonl` and `data/processed/vocab.json`.
2) Run `src/los/preprocess_los.py` to construct LOS instances using the same vocab/tokens.
   - This writes `data/processed/los_instances.jsonl`, `los_features.npz`, and `los_labels.npz`.
3) Train LOS models (logreg, GRU, transformer) using either baseline one-hot or the provided pretrained embeddings.

Quickstart commands
-------------------
- Build sequences + train MLM:
  - `python main.py`
- Prepare LOS data (must follow main.py to share vocab/tokens):
  - `python src/los/preprocess_los.py`
- Train LOS logistic regression (one-hot):
  - `python src/los/baselines/train_los_logreg.py --processed_dir data/processed`
- Train LOS logistic regression with pretrained embeddings:
  - `python src/los/baselines/train_los_logreg.py --processed_dir data/processed --embeddings_path word_embeddings.pt`
- Train LOS GRU with pretrained embeddings (truncates to last 512 tokens by default):
  - `python src/los/baselines/train_los_gru.py --processed_dir data/processed --embeddings_path word_embeddings.pt --max_len 512`
- Train LOS transformer with pretrained embeddings:
  - `python src/los/baselines/train_los_transformer.py --processed_dir data/processed --embeddings_path word_embeddings.pt --max_len 512`

Toggling pretrained embeddings
------------------------------
- Logistic regression: include `--embeddings_path word_embeddings.pt` (uses dense embedding features). Omit to fall back to sparse one-hot.
- GRU: include `--embeddings_path word_embeddings.pt`; add `--freeze_emb` to keep embeddings fixed. Uses truncated sequences (default `--max_len 512`, keeps last tokens).
- Transformer: include `--embeddings_path word_embeddings.pt`; add `--freeze_emb` if desired. Uses truncated sequences (default `--max_len 512`, keeps last tokens).

Key artifacts
-------------
- Input: `data/raw/*.csv` (encounters, conditions, observations, medications, patients).
- Processed: `data/processed/sequences.jsonl`, `vocab.json`, `los_instances.jsonl`, `los_features.npz`, `los_labels.npz`.
- Models: `data/processed/model/` (MLM) and `los_logreg*.npz`, `los_gru.pt`, `los_transformer.pt`.

Notes
-----
- Always run `main.py` before `src/los/preprocess_los.py` so LOS tokens align with the vocab used by pretrained embeddings.
- LOS scripts default to balanced class weights and fixed seeds; adjust CLI flags as needed. 
