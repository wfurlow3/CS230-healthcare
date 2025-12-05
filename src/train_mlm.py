import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import (
    BATCH_SIZE,
    GRAD_CLIP_NORM,
    LEARNING_RATE,
    MAX_LEN,
    NUM_EPOCHS,
    PRINT_EVERY,
    SEED,
    SPECIAL_TOKENS,
    WEIGHT_DECAY,
)
from .dataset import EHRDataset, make_dataloaders, split_encounters
from .model import make_mlm_model
from .vocab import load_vocab


def train_epoch(model: torch.nn.Module, dataloader: Iterable[Dict[str, torch.Tensor]], optimizer: torch.optim.Optimizer, device: torch.device, print_every: int = PRINT_EVERY):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_masked = 0
    progress = tqdm(dataloader, desc="train", leave=False)
    for step, batch in enumerate(progress):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        with torch.no_grad():
            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
            mask = labels != -100
            total_correct += ((predictions == labels) & mask).sum().item()
            total_masked += mask.sum().item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        if (step + 1) % print_every == 0:
            progress.set_postfix(loss=f"{loss.item():.4f}")
    progress.close()
    avg_loss = total_loss / max(1, len(dataloader))
    accuracy = total_correct / max(1, total_masked)
    return avg_loss, accuracy


def eval_epoch(model: torch.nn.Module, dataloader: Iterable[Dict[str, torch.Tensor]], device: torch.device):
    if dataloader is None or len(dataloader.dataset) == 0:
        return float("nan"), float("nan")
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_masked = 0
    with torch.no_grad():
        progress = tqdm(dataloader, desc="eval", leave=False)
        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
            mask = labels != -100
            total_correct += ((predictions == labels) & mask).sum().item()
            total_masked += mask.sum().item()
        progress.close()
    avg_loss = total_loss / max(1, len(dataloader))
    accuracy = total_correct / max(1, total_masked)
    return avg_loss, accuracy


def demo_mask_fill(model: torch.nn.Module, sequences: Iterable[dict], vocab: Dict[str, int], idx_to_token: Dict[int, str], device: torch.device, max_examples: int = 3):
    print("sample mask-fill predictions:")
    specials = set(SPECIAL_TOKENS)
    mask_token_id = vocab["[MASK]"]
    pad_id = vocab["[PAD]"]
    examples_shown = 0
    rng = random.Random(SEED)
    model.eval()
    for record in sequences:
        tokens = record["tokens"]
        candidate_positions = [i for i, tok in enumerate(tokens) if tok not in specials]
        if not candidate_positions:
            continue
        pos = rng.choice(candidate_positions)
        masked_tokens = list(tokens)
        original_token = masked_tokens[pos]
        masked_tokens[pos] = "[MASK]"
        ids = [vocab.get(tok, mask_token_id) for tok in masked_tokens][:MAX_LEN]
        attn = [1] * len(ids)
        if len(ids) < MAX_LEN:
            pad_len = MAX_LEN - len(ids)
            ids.extend([pad_id] * pad_len)
            attn.extend([0] * pad_len)
        input_ids = torch.tensor([ids], dtype=torch.long).to(device)
        attention_mask = torch.tensor([attn], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        masked_index = pos if pos < MAX_LEN else MAX_LEN - 1
        topk = torch.topk(logits[0, masked_index], k=5)
        predictions = [idx_to_token.get(idx.item(), "[UNK]") for idx in topk.indices]
        print(f"  encounter {record['encounter']} token '{original_token}' -> top5 {predictions}")
        examples_shown += 1
        if examples_shown >= max_examples:
            break
    if examples_shown == 0:
        print("warning: no suitable sequences found for mask-fill demo.")


def load_sequences(seq_path: Path):
    with open(seq_path, "r") as f:
        return [json.loads(line) for line in f]


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    base_dir = Path(os.getcwd())
    processed_dir = base_dir / "data" / "processed"
    seq_path = processed_dir / "sequences.jsonl"
    vocab_path = processed_dir / "vocab.json"
    output_dir = base_dir / "model"
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences = load_sequences(seq_path)
    vocab = load_vocab(vocab_path)
    idx_to_token = {idx: tok for tok, idx in vocab.items()}

    train_ids, val_ids = split_encounters(sequences)
    print(f"train encounters: {len(train_ids)} | val encounters: {len(val_ids)}")
    train_loader, val_loader = make_dataloaders(str(seq_path), vocab, train_ids, val_ids, batch_size=BATCH_SIZE, max_len=MAX_LEN)

    model = make_mlm_model(len(vocab))
    device = torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    num_epochs = NUM_EPOCHS
    for epoch in range(1, num_epochs + 1):
        print(f"epoch {epoch}/{num_epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, device)
        if np.isnan(val_loss):
            print(f"  train loss: {train_loss:.4f} | train acc: {train_acc:.4f} | val loss: n/a (no val set) | val acc: n/a")
        else:
            print(f"  train loss: {train_loss:.4f} | train acc: {train_acc:.4f} | val loss: {val_loss:.4f} | val acc: {val_acc:.4f}")

    model.save_pretrained(output_dir)
    print(f"saved model and artifacts to {output_dir}")

    demo_mask_fill(model, sequences, vocab, idx_to_token, device)


if __name__ == "__main__":
    main()
