import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Tuple

from .config import SPECIAL_TOKENS, VOCAB_LIMIT

MIN_FREQ_DX = 5  # legacy, no longer used for filtering
MIN_FREQ_MED = 10  # legacy, no longer used for filtering


def build_vocab(sequences: Iterable[dict], max_size: int = VOCAB_LIMIT) -> Tuple[Dict[str, int], Dict[int, str]]:
    token_counts = Counter()
    for record in sequences:
        for token in record["tokens"]:
            if token in SPECIAL_TOKENS:
                continue
            token_counts[token] += 1

    # Keep only OBS_, DX_, and MED_ tokens observed in the data (no filtering)
    candidates_sorted = sorted(
        [(tok, count) for tok, count in token_counts.items() if tok.startswith(("OBS_", "DX_", "MED_"))],
        key=lambda x: (-x[1], x[0]),
    )

    vocab_list = list(SPECIAL_TOKENS)
    remaining = max(0, max_size - len(vocab_list))

    # Fill remaining slots with highest-frequency candidates
    for tok, _ in candidates_sorted:
        if len(vocab_list) >= max_size:
            break
        if tok not in vocab_list:
            vocab_list.append(tok)

    vocab = {tok: idx for idx, tok in enumerate(vocab_list)}
    idx_to_token = {idx: tok for tok, idx in vocab.items()}
    return vocab, idx_to_token


def save_vocab(vocab: Dict[str, int], path: Path) -> None:
    with open(path, "w") as f:
        json.dump(vocab, f, indent=2)


def load_vocab(path: Path) -> Dict[str, int]:
    with open(path, "r") as f:
        return json.load(f)
