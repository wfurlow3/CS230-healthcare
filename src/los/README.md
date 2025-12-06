LOS Pipeline and Baselines
==========================

Setup order
-----------
1) Run `python main.py` (root) to build `data/processed/sequences.jsonl` and `vocab.json`.
2) Run `python src/los/preprocess_los.py` to produce LOS data aligned to that vocab:
   - `los_instances.jsonl`, `los_features.npz`, `los_labels.npz`

Preprocessing details
---------------------
- Uses the same tokens/vocab as the main pipeline (loaded from `data/processed/sequences.jsonl` and `vocab.json`).
- Builds first-hour tokens plus prior-encounter history tokens.
- Outputs balanced labels via median LOS split.

LOS models (baselines)
----------------------
- Logistic regression (one-hot):
  - `python src/los/baselines/train_los_logreg.py --processed_dir data/processed`
- Logistic regression (pretrained embeddings; includes per-token counts + embedding sum):
  - `python src/los/baselines/train_los_logreg.py --processed_dir data/processed --embeddings_path word_embeddings.pt`
- GRU (pretrained embeddings, truncates to last `--max_len`, default 512):
  - `python src/los/baselines/train_los_gru.py --processed_dir data/processed --embeddings_path word_embeddings.pt --max_len 512`
  - Add `--freeze_emb` to stop embedding finetuning.
- Transformer (pretrained embeddings, truncates to last `--max_len`, default 512):
  - `python src/los/baselines/train_los_transformer.py --processed_dir data/processed --embeddings_path word_embeddings.pt --max_len 512`
  - Add `--freeze_emb` to stop embedding finetuning.

Key arguments
-------------
- `--processed_dir`: root for processed artifacts (defaults to `data/processed`).
- `--embeddings_path`: path to `word_embeddings.pt` (vocab_size x dim) aligned with `vocab.json`.
- `--max_len` (GRU/Transformer): keep the last N tokens for each sequence; also sizes positional embeddings for the transformer.
- `--freeze_emb` (GRU/Transformer): keep the pretrained embedding weights fixed.

Notes
-----
- Always run `main.py` before `preprocess_los.py` to guarantee vocab alignment for pretrained embeddings.
- GRU/Transformer training updates embeddings unless `--freeze_emb` is set; the source `word_embeddings.pt` file is never overwritten. 
