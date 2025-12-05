import argparse
import csv
import json
import math
import os
import random
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import (
    ATTN_DROPOUT_PROB,
    BATCH_SIZE,
    GRAD_CLIP_NORM,
    HIDDEN_DROPOUT_PROB,
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


def train_epoch(
    model: torch.nn.Module,
    dataloader: Iterable[Dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    print_every: int = PRINT_EVERY,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> tuple[float, float, float, Counter]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_correct_wo_kept = 0
    total_masked = 0
    total_masked_wo_kept = 0
    label_counter: Counter = Counter()
    progress = tqdm(dataloader, desc="train", leave=False)
    for step, batch in enumerate(progress):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        with torch.no_grad():
            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
            mask = labels != -100
            kept_mask = mask & (batch["input_ids"] == labels)
            total_correct += ((predictions == labels) & mask).sum().item()
            total_masked += mask.sum().item()
            total_correct_wo_kept += ((predictions == labels) & mask & ~kept_mask).sum().item()
            total_masked_wo_kept += (mask & ~kept_mask).sum().item()
            label_counter.update(labels[mask].detach().cpu().tolist())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        if (step + 1) % print_every == 0:
            progress.set_postfix(loss=f"{loss.item():.4f}")
    progress.close()
    avg_loss = total_loss / max(1, len(dataloader))
    accuracy = total_correct / max(1, total_masked)
    accuracy_wo_kept = total_correct_wo_kept / max(1, total_masked_wo_kept)
    return avg_loss, accuracy, accuracy_wo_kept, label_counter


def eval_epoch(model: torch.nn.Module, dataloader: Iterable[Dict[str, torch.Tensor]], device: torch.device):
    if dataloader is None or len(dataloader.dataset) == 0:
        return float("nan"), float("nan"), float("nan")
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_correct_wo_kept = 0
    total_masked = 0
    total_masked_wo_kept = 0
    with torch.no_grad():
        progress = tqdm(dataloader, desc="eval", leave=False)
        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
            mask = labels != -100
            kept_mask = mask & (batch["input_ids"] == labels)
            total_correct += ((predictions == labels) & mask).sum().item()
            total_masked += mask.sum().item()
            total_correct_wo_kept += ((predictions == labels) & mask & ~kept_mask).sum().item()
            total_masked_wo_kept += (mask & ~kept_mask).sum().item()
        progress.close()
    avg_loss = total_loss / max(1, len(dataloader))
    accuracy = total_correct / max(1, total_masked)
    accuracy_wo_kept = total_correct_wo_kept / max(1, total_masked_wo_kept)
    return avg_loss, accuracy, accuracy_wo_kept


def demo_mask_fill(
    model: torch.nn.Module,
    sequences: Iterable[dict],
    vocab: Dict[str, int],
    idx_to_token: Dict[int, str],
    device: torch.device,
    max_examples: int = 3,
):
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


def print_label_distribution(label_counter: Counter, idx_to_token: Dict[int, str], top_k: int = 10):
    total = sum(label_counter.values())
    if total == 0:
        print("no masked labels observed during training.")
        return
    print("top masked labels (token_id: token -> count | pct):")
    for idx, count in label_counter.most_common(top_k):
        tok = idx_to_token.get(idx, f"<id={idx}>")
        pct = (count / total) * 100.0
        print(f"  {idx}: {tok} -> {count} ({pct:.2f}%)")


def run_experiment(config: Dict, seed: int = SEED) -> Dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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
    train_loader, val_loader = make_dataloaders(
        str(seq_path), vocab, train_ids, val_ids, batch_size=BATCH_SIZE, max_len=MAX_LEN
    )

    model = make_mlm_model(
        len(vocab),
        hidden_dropout_prob=config.get("dropout", HIDDEN_DROPOUT_PROB),
        attn_dropout_prob=config.get("dropout", ATTN_DROPOUT_PROB),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", LEARNING_RATE),
        weight_decay=config.get("weight_decay", WEIGHT_DECAY),
    )
    total_steps = max(1, len(train_loader) * NUM_EPOCHS)
    warmup_steps = max(1, int(0.1 * total_steps))

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    num_epochs = NUM_EPOCHS
    best_val_loss = float("inf")
    patience = 0
    last_train_loss = float("nan")
    last_train_acc = float("nan")
    last_val_loss = float("nan")
    last_val_acc = float("nan")
    last_train_acc_wo_kept = float("nan")
    last_val_acc_wo_kept = float("nan")
    for epoch in range(1, num_epochs + 1):
        print(f"epoch {epoch}/{num_epochs}")
        last_train_loss, last_train_acc, last_train_acc_wo_kept, label_counter = train_epoch(
            model, train_loader, optimizer, device, scheduler=scheduler
        )
        last_val_loss, last_val_acc, last_val_acc_wo_kept = eval_epoch(model, val_loader, device)
        if np.isnan(last_val_loss):
            print(
                f"  train loss: {last_train_loss:.4f} | train acc: {last_train_acc:.4f} | "
                f"train acc (excl kept): {last_train_acc_wo_kept:.4f} | val loss: n/a (no val set) | val acc: n/a | val acc (excl kept): n/a"
            )
        else:
            print(
                f"  train loss: {last_train_loss:.4f} | train acc: {last_train_acc:.4f} | "
                f"train acc (excl kept): {last_train_acc_wo_kept:.4f} | "
                f"val loss: {last_val_loss:.4f} | val acc: {last_val_acc:.4f} | val acc (excl kept): {last_val_acc_wo_kept:.4f}"
            )
            if last_val_loss < best_val_loss:
                best_val_loss = last_val_loss
                patience = 0
            else:
                patience += 1
                if patience >= 3:
                    print("Early stopping: validation loss did not improve for 3 checks.")
                    break
        print_label_distribution(label_counter, idx_to_token)

    model.save_pretrained(output_dir)
    print(f"saved model and artifacts to {output_dir}")

    demo_mask_fill(model, sequences, vocab, idx_to_token, device)
    return {
        "train_loss": last_train_loss,
        "train_accuracy": last_train_acc,
        "train_accuracy_excl_kept": last_train_acc_wo_kept,
        "val_loss": last_val_loss,
        "val_accuracy": last_val_acc,
        "val_accuracy_excl_kept": last_val_acc_wo_kept,
        "config": config,
    }


def run_hparam_search():
    base_config = {
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "dropout": HIDDEN_DROPOUT_PROB,
    }
    lrs = [
        0.5 * base_config["learning_rate"],
        base_config["learning_rate"],
        2.0 * base_config["learning_rate"],
    ]
    dropouts = [0.1, 0.2]
    weight_decays = [0.01, 0.05]

    results: List[Dict] = []
    for lr, dropout, weight_decay in product(lrs, dropouts, weight_decays):
        cfg = dict(base_config)
        cfg.update({"learning_rate": lr, "dropout": dropout, "weight_decay": weight_decay})
        print(
            f"\n=== Running config: lr={lr:.6f}, dropout={dropout}, weight_decay={weight_decay} ==="
        )
        res = run_experiment(cfg, seed=SEED)
        res.update({"learning_rate": lr, "dropout": dropout, "weight_decay": weight_decay})
        results.append(res)

    if not results:
        print("No results collected during hyperparameter search.")
        return

    base_dir = Path(os.getcwd())
    output_dir = base_dir / "model"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "hparam_search_results.csv"
    fieldnames = [
        "learning_rate",
        "dropout",
        "weight_decay",
        "train_loss",
        "val_loss",
        "val_accuracy",
        "val_accuracy_excl_kept",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"\nHyperparameter search results written to {csv_path}")

    def _best_key(item):
        return (round(item["val_loss"], 4), -item.get("val_accuracy", 0.0))

    results_sorted = sorted(results, key=_best_key)
    best = results_sorted[0]
    print("\nResults (sorted by val_loss, tie-breaker val_accuracy):")
    print(
        f"{'lr':>10} {'dropout':>10} {'w_decay':>10} {'val_loss':>10} {'val_acc':>10} {'val_acc_no_keep':>15} {'train_loss':>12}"
    )
    for r in results_sorted:
        marker = "*" if r is best else " "
        print(
            f"{marker} {r['learning_rate']:>9.6f} {r['dropout']:>10.2f} {r['weight_decay']:>10.3f} {r['val_loss']:>10.4f} {r['val_accuracy']:>10.4f} {r.get('val_accuracy_excl_kept', float('nan')):>15.4f} {r['train_loss']:>12.4f}"
        )
    print(
        f"\nBest config: lr={best['learning_rate']}, dropout={best['dropout']}, weight_decay={best['weight_decay']} (val_loss={best['val_loss']:.4f}, val_acc={best['val_accuracy']:.4f})"
    )

    best_config = {
        "learning_rate": best["learning_rate"],
        "dropout": best["dropout"],
        "weight_decay": best["weight_decay"],
    }
    best_metrics = {
        "train_loss": best.get("train_loss"),
        "train_accuracy": best.get("train_accuracy"),
        "val_loss": best.get("val_loss"),
        "val_accuracy": best.get("val_accuracy"),
        "val_accuracy_excl_kept": best.get("val_accuracy_excl_kept"),
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "seed": SEED,
    }
    config_path = output_dir / "best_hparam_config.json"
    summary_path = output_dir / "best_hparam_summary.json"
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=2)
    with open(summary_path, "w") as f:
        json.dump({"config": best_config, "metrics": best_metrics}, f, indent=2)
    print(f"Best configuration written to {config_path}")
    print(f"Best run summary (config + metrics) written to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Train masked LM or run hyperparameter search.")
    parser.add_argument(
        "--hparam_search",
        action="store_true",
        help="Run a small grid search over LR, dropout, and weight decay.",
    )
    args = parser.parse_args()

    if args.hparam_search:
        run_hparam_search()
    else:
        run_experiment(
            {
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "dropout": HIDDEN_DROPOUT_PROB,
            },
            seed=SEED,
        )


if __name__ == "__main__":
    main()
