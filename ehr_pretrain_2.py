import os
from pathlib import Path

from src.build_sequences import preprocess
from src.train_mlm import main as train_main


def main():
    base_dir = Path(os.getcwd())
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"

    preprocess(raw_dir, processed_dir)

    seq_path = processed_dir / "sequences.jsonl"
    vocab_path = processed_dir / "vocab.json"
    if not seq_path.exists() or not vocab_path.exists():
        print("preprocessing did not produce sequences or vocab; skipping training.")
        return
    train_main()


if __name__ == "__main__":
    main()
