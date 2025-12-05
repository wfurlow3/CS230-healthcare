import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


def load_instances(instances_path: Path) -> List[dict]:
    records = []
    with open(instances_path, "r") as f:
        for line in f:
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"no instances found at {instances_path}")
    return records


class LOSSequenceDataset(Dataset):
    def __init__(self, records: List[dict], token_to_idx: dict):
        self.token_to_idx = token_to_idx
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        tokens = record["tokens"]
        ids = [self.token_to_idx.get(tok, -1) for tok in tokens]
        ids = [i for i in ids if i >= 0]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(record["label"], dtype=torch.float32),
        }


class GRUBaseline(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_size: int = 128, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        packed_out, _ = self.gru(x)
        mask = attention_mask.unsqueeze(-1)
        masked_out = packed_out * mask
        summed = masked_out.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / lengths
        logits = self.classifier(pooled).squeeze(-1)
        return logits


def collate(batch):
    # Pad to the longest sequence in the batch (no truncation).
    input_lists = [item["input_ids"] for item in batch]
    labels = torch.stack([item["label"] for item in batch], dim=0)
    max_len = max(x.size(0) for x in input_lists) if input_lists else 0
    padded_inputs = []
    attn_masks = []
    for seq in input_lists:
        pad_len = max_len - seq.size(0)
        if pad_len > 0:
            seq = torch.cat([seq, torch.zeros(pad_len, dtype=torch.long)])
        padded_inputs.append(seq)
        attn = torch.cat([torch.ones(seq.size(0) - pad_len), torch.zeros(pad_len)]) if pad_len > 0 else torch.ones(seq.size(0))
        attn_masks.append(attn)
    return {
        "input_ids": torch.stack(padded_inputs, dim=0) if padded_inputs else torch.empty(0, dtype=torch.long),
        "attention_mask": torch.stack(attn_masks, dim=0) if attn_masks else torch.empty(0, dtype=torch.float32),
        "label": labels,
    }


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    bce = nn.BCEWithLogitsLoss()
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = bce(logits, batch["label"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(dataloader))


def eval_epoch(model, dataloader, device):
    model.eval()
    bce = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    preds, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = bce(logits, batch["label"])
            total_loss += loss.item()
            probs = torch.sigmoid(logits)
            preds.append(probs.cpu())
            labels.append(batch["label"].cpu())
    if preds:
        preds = torch.cat(preds).numpy()
        labels = torch.cat(labels).numpy()
        binary = (preds >= 0.5).astype(np.float32)
        acc = accuracy_score(labels, binary)
        f1 = f1_score(labels, binary, zero_division=0)
        try:
            auc = roc_auc_score(labels, preds)
        except ValueError:
            auc = float("nan")
    else:
        acc = f1 = auc = float("nan")
    return total_loss / max(1, len(dataloader)), acc, f1, auc


def main():
    parser = argparse.ArgumentParser(description="GRU baseline for LOS classification on token sequences.")
    parser.add_argument("--processed_dir", type=Path, default=Path("data") / "processed")
    parser.add_argument("--instances_file", type=str, default="los_instances.jsonl")
    parser.add_argument("--labels_file", type=str, default="los_labels.npz")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    processed_dir = args.processed_dir
    instances_path = processed_dir / args.instances_file
    labels_path = processed_dir / args.labels_file
    label_bundle = np.load(labels_path, allow_pickle=True)
    classes = label_bundle["classes"].tolist()
    token_to_idx = {tok: i + 1 for i, tok in enumerate(classes)}  # reserve 0 for pad

    records = load_instances(instances_path)
    train_recs, val_recs = train_test_split(records, test_size=args.test_size, random_state=args.seed, stratify=[r["label"] for r in records])

    train_ds = LOSSequenceDataset(train_recs, token_to_idx)
    val_ds = LOSSequenceDataset(val_recs, token_to_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUBaseline(vocab_size=len(token_to_idx) + 1, embed_dim=args.embed_dim, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, val_f1, val_auc = eval_epoch(model, val_loader, device)
        print(
            f"epoch {epoch}/{args.epochs} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | val acc {val_acc:.3f} | val f1 {val_f1:.3f} | val auc {val_auc:.3f}"
        )

    model_path = processed_dir / "los_gru.pt"
    torch.save({"model_state": model.state_dict(), "token_to_idx": token_to_idx, "config": vars(args)}, model_path)
    print(f"saved GRU baseline to {model_path}")


if __name__ == "__main__":
    main()

