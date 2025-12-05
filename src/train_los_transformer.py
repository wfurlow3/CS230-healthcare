import argparse
import json
from pathlib import Path
from typing import List

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


def collate(batch):
    # Pad to longest sequence in batch; build attention mask.
    input_lists = [item["input_ids"] for item in batch]
    labels = torch.stack([item["label"] for item in batch], dim=0) if batch else torch.empty(0)
    max_len = max((x.size(0) for x in input_lists), default=0)
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


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, embed_dim: int):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return self.pos_emb(positions)


class TransformerBaseline(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, num_heads: int = 4, num_layers: int = 2, dim_ff: int = 128, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_emb = PositionalEmbedding(max_len=max_len, embed_dim=embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # attention_mask: 1 for real tokens, 0 for pad
        token_embed = self.token_emb(input_ids)
        pos_embed = self.pos_emb(input_ids)
        x = token_embed + pos_embed
        src_key_padding_mask = attention_mask == 0  # True where pad
        encoded = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        mask = attention_mask.unsqueeze(-1)
        summed = (encoded * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / lengths
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled).squeeze(-1)
        return logits


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
    parser = argparse.ArgumentParser(description="Transformer baseline for LOS classification on token sequences.")
    parser.add_argument("--processed_dir", type=Path, default=Path("data") / "processed")
    parser.add_argument("--instances_file", type=str, default="los_instances.jsonl")
    parser.add_argument("--labels_file", type=str, default="los_labels.npz")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_ff", type=int, default=128)
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
    model = TransformerBaseline(
        vocab_size=len(token_to_idx) + 1,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
        max_len=512,
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, val_f1, val_auc = eval_epoch(model, val_loader, device)
        print(
            f"epoch {epoch}/{args.epochs} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | val acc {val_acc:.3f} | val f1 {val_f1:.3f} | val auc {val_auc:.3f}"
        )

    model_path = processed_dir / "los_transformer.pt"
    torch.save({"model_state": model.state_dict(), "token_to_idx": token_to_idx, "config": vars(args)}, model_path)
    print(f"saved transformer baseline to {model_path}")


if __name__ == "__main__":
    main()

