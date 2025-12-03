import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Tuple

from .config import SPECIAL_TOKENS, VOCAB_LIMIT


def build_vocab(sequences: Iterable[dict], max_size: int = VOCAB_LIMIT) -> Tuple[Dict[str, int], Dict[int, str]]:
    counter = Counter()
    for record in sequences:
        for token in record["tokens"]:
            if token in SPECIAL_TOKENS:
                continue
            counter[token] += 1
    vocab = {}
    idx = 0
    for token in SPECIAL_TOKENS:
        vocab[token] = idx
        idx += 1
    remaining = max(0, max_size - len(SPECIAL_TOKENS))
    for token, _ in counter.most_common(remaining):
        if token in vocab:
            continue
        vocab[token] = idx
        idx += 1
    idx_to_token = {idx: tok for tok, idx in vocab.items()}
    return vocab, idx_to_token


def save_vocab(vocab: Dict[str, int], path: Path) -> None:
    with open(path, "w") as f:
        json.dump(vocab, f, indent=2)


def load_vocab(path: Path) -> Dict[str, int]:
    with open(path, "r") as f:
        return json.load(f)
