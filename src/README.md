src Overview
============

Key modules
-----------
- `build_sequences.py`: builds chronological EHR token sequences and vocabulary; run via `python src/build_sequences.py` (usually invoked inside `main.py`).
- `train_mlm.py`: trains the masked language model on `data/processed/sequences.jsonl`.
- `los/`: LOS-specific preprocessing and baselines.
- `config.py`: shared hyperparameters and special tokens.
- `utils.py`: token cleaning and observation binning helpers.

Standard workflow
-----------------
1) `python main.py`
   - Calls `build_sequences.preprocess` to generate `data/processed/sequences.jsonl` and `vocab.json`.
   - Trains MLM (`train_mlm.py`) using those sequences.
2) `python src/los/preprocess_los.py`
   - Uses the same vocab/tokens to produce LOS instances and labels.
3) Train LOS baselines from `src/los/baselines/`:
   - Logistic regression (one-hot or embeddings)
   - GRU (pretrained embeddings; optional freeze)
   - Transformer (pretrained embeddings; optional freeze)

Script entry points (common flags)
----------------------------------
- `python main.py`
  - No required args; uses `data/raw` and writes to `data/processed`.

- `python src/train_mlm.py --sequences_file data/processed/sequences.jsonl --vocab_file data/processed/vocab.json`
  - Optional: `--batch_size`, `--epochs`, `--lr`, `--mask_prob`, etc.

- `python src/los/preprocess_los.py`
  - Optional: `--allowed_classes` (defaults to inpatient), `--raw_data_dir`, `--processed_dir`.
  - Produces `los_instances.jsonl`, `los_features.npz`, `los_labels.npz`.

LOS baselines
-------------
- Logistic regression (one-hot):
  - `python src/los/baselines/train_los_logreg.py --processed_dir data/processed`
- Logistic regression (pretrained embeddings):
  - `python src/los/baselines/train_los_logreg.py --processed_dir data/processed --embeddings_path word_embeddings.pt`
- GRU (pretrained embeddings, truncates to last `--max_len`, default 512):
  - `python src/los/baselines/train_los_gru.py --processed_dir data/processed --embeddings_path word_embeddings.pt --max_len 512`
  - Add `--freeze_emb` to keep embeddings fixed.
- Transformer (pretrained embeddings, truncates to last `--max_len`, default 512):
  - `python src/los/baselines/train_los_transformer.py --processed_dir data/processed --embeddings_path word_embeddings.pt --max_len 512`
  - Add `--freeze_emb` to keep embeddings fixed.

Toggling embeddings
-------------------
- Logreg: include `--embeddings_path` to switch from one-hot to dense embedding features.
- GRU/Transformer: include `--embeddings_path` to load pretrained embeddings; add `--freeze_emb` to stop finetuning.

Reminder
--------
Always run `main.py` before `src/los/preprocess_los.py` so LOS vocab/tokens match the pretrained embeddings and MLM preprocessing. 
