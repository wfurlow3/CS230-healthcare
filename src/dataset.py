import json
import math
import random
from collections import Counter
from typing import Dict, Iterable, Optional, Set, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .config import BATCH_SIZE, MASK_PROB, MAX_LEN, SEED, SPECIAL_TOKENS


def split_encounters(sequences: Iterable[dict], val_ratio: float = 0.1, seed: int = SEED) -> Tuple[Set[str], Set[str]]:
    encounters = [record["encounter"] for record in sequences]
    rng = random.Random(seed)
    rng.shuffle(encounters)
    if len(encounters) <= 1:
        return set(encounters), set()
    val_size = max(1, int(len(encounters) * val_ratio))
    if val_size >= len(encounters):
        val_size = max(1, len(encounters) // 5)
    val_ids = set(encounters[:val_size])
    train_ids = set(encounters[val_size:])
    if not train_ids:
        train_ids = val_ids
        val_ids = set()
    return train_ids, val_ids


class EHRDataset(Dataset):
    """Simple masked language modeling dataset for encounter token sequences."""

    def __init__(
        self,
        sequences_path: str,
        vocab: Dict[str, int],
        allowed_encounters: Optional[Iterable[str]] = None,
        max_len: int = MAX_LEN,
        mask_prob: float = MASK_PROB,
        seed: int = SEED,
    ):
        self.records = []
        self.token_freq = Counter()
        allowed = set(allowed_encounters) if allowed_encounters else None
        with open(sequences_path, "r") as f:
            for line in f:
                obj = json.loads(line)
                if allowed and obj["encounter"] not in allowed:
                    continue
                ids = [vocab.get(tok, vocab["[MASK]"]) for tok in obj["tokens"]]
                self.records.append(ids)
                self.token_freq.update(ids)
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.pad_id = vocab["[PAD]"]
        self.mask_id = vocab["[MASK]"]
        self.vocab_size = len(vocab)
        self.special_ids = {vocab[token] for token in SPECIAL_TOKENS if token in vocab}
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = list(self.records[idx])
        ids = ids[: self.max_len]
        attn = [1] * len(ids)
        if len(ids) < self.max_len:
            pad_len = self.max_len - len(ids)
            ids.extend([self.pad_id] * pad_len)
            attn.extend([0] * pad_len)
        labels = [-100] * len(ids)
        candidates = [i for i, token_id in enumerate(ids) if token_id not in self.special_ids and token_id != self.pad_id]
        if candidates:
            num_to_mask = max(1, int(len(candidates) * self.mask_prob))
            num_to_mask = min(num_to_mask, len(candidates))
            weights = [1.0 / math.sqrt(self.token_freq[ids[i]]) for i in candidates]
            mask_positions = []
            available_positions = list(candidates)
            available_weights = list(weights)
            for _ in range(num_to_mask):
                total = sum(available_weights)
                r = self.rng.random() * total
                cum = 0.0
                for idx, w in enumerate(available_weights):
                    cum += w
                    if cum >= r:
                        chosen_pos = available_positions.pop(idx)
                        available_weights.pop(idx)
                        mask_positions.append(chosen_pos)
                        break
            for pos in mask_positions:
                labels[pos] = ids[pos]
                rand = self.rng.random()
                if rand < 0.8:
                    ids[pos] = self.mask_id
                elif rand < 0.9:
                    ids[pos] = self.rng.randrange(self.vocab_size)
                # else keep original token
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def make_dataloaders(seq_path: str, vocab: Dict[str, int], train_ids: Set[str], val_ids: Set[str], batch_size: int = BATCH_SIZE, max_len: int = MAX_LEN):
    train_dataset = EHRDataset(seq_path, vocab, allowed_encounters=train_ids, max_len=max_len)
    val_dataset = EHRDataset(seq_path, vocab, allowed_encounters=val_ids, max_len=max_len) if val_ids else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
    return train_loader, val_loader
