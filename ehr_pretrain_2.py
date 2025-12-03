import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    from transformers import BertConfig, BertForMaskedLM
except ImportError:
    print("error: transformers is required but not installed. Try `pip install transformers`.")
    sys.exit(1)


SEED = 13
WINDOW_HOURS = 24
BIN_SIZE_HOURS = 6
MAX_LEN = 256
VOCAB_LIMIT = 8000
MASK_PROB = 0.15
SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[ADMIT]", "[DISCH]"]


def clean_fragment(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    text = re.sub(r"[^A-Za-z0-9]+", "_", text.upper())
    text = text.strip("_")
    return text or None


def load_csv(path, date_cols=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"required file missing: {path}")
    df = pd.read_csv(path)
    df.columns = [c.upper() for c in df.columns]
    missing = [col for col in (date_cols or []) if col not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(path)} missing required date columns: {missing}")
    for col in date_cols or []:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_convert(None)
    return df


def load_conditions(data_dir):
    path = os.path.join(data_dir, "conditions.csv")
    df = load_csv(path, date_cols=["START", "STOP"])
    required = ["ENCOUNTER", "START", "CODE", "DESCRIPTION"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"conditions.csv missing required columns: {missing}")
    return df


def load_observations(data_dir):
    path = os.path.join(data_dir, "observations.csv")
    df = load_csv(path, date_cols=["DATE"])
    required = ["ENCOUNTER", "DATE", "CODE", "DESCRIPTION", "CATEGORY"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"observations.csv missing required columns: {missing}")
    return df


def load_encounters(data_dir):
    path = os.path.join(data_dir, "encounters.csv")
    df = load_csv(path, date_cols=["START", "STOP"])
    print(df.columns)
    required = ["ID", "START", "STOP"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"encounters.csv missing required columns: {missing}")
    return df


def extract_encounter_windows(enc_df):
    windows = {}
    for _, row in enc_df.iterrows():
        enc_id = row["ID"]
        if pd.isna(enc_id):
            raise ValueError("encounters.csv contains rows without Id column.")
        start = row["START"]
        if pd.isna(start):
            raise ValueError(f"encounter {enc_id} missing START column.")
        stop = row["STOP"]
        if pd.isna(stop):
            raise ValueError(f"encounter {enc_id} missing STOP column.")
        windows[str(enc_id)] = (start, stop)
    if not windows:
        raise ValueError("encounters.csv did not yield any encounter windows.")
    return windows


def build_sequences(cond_df, obs_df, windows):
    events = defaultdict(list)

    for _, row in cond_df.iterrows():
        enc_id = str(row["ENCOUNTER"])
        if enc_id not in windows:
            continue
        token = None
        if not pd.isna(row["CODE"]):
            code = clean_fragment(row["CODE"])
            if code:
                token = f"DX_{code}"
        if not token and not pd.isna(row["DESCRIPTION"]):
            desc = clean_fragment(row["DESCRIPTION"])
            if desc:
                token = f"DX_{desc}"
        if not token:
            continue
        event_time = row["START"] if not pd.isna(row["START"]) else None
        events[enc_id].append((event_time, token))

    for _, row in obs_df.iterrows():
        enc_id = str(row["ENCOUNTER"])
        if enc_id not in windows:
            continue
        fragment = None
        if not pd.isna(row["CODE"]):
            fragment = clean_fragment(row["CODE"])
        if not fragment and not pd.isna(row["DESCRIPTION"]):
            fragment = clean_fragment(row["DESCRIPTION"])
        if not fragment:
            continue
        category = str(row["CATEGORY"]).upper() if not pd.isna(row["CATEGORY"]) else ""
        prefix = "OBS"
        if "VITAL" in category:
            prefix = "OBS_VITAL"
        elif "LAB" in category:
            prefix = "OBS_LAB"
        token = f"{prefix}_{fragment}_SEEN"
        event_time = row["DATE"] if not pd.isna(row["DATE"]) else None
        events[enc_id].append((event_time, token))

    sequences = []
    for enc_id, (start_time, _) in windows.items():
        if pd.isna(start_time):
            continue
        bin_sets = [set() for _ in range(WINDOW_HOURS // BIN_SIZE_HOURS)]
        for event_time, token in events.get(enc_id, []):
            ts = event_time if event_time is not None and not pd.isna(event_time) else start_time
            if pd.isna(ts):
                continue
            delta_hours = (ts - start_time).total_seconds() / 3600.0
            if delta_hours < 0 or delta_hours >= WINDOW_HOURS:
                continue
            bin_idx = min(int(delta_hours // BIN_SIZE_HOURS), len(bin_sets) - 1)
            bin_sets[bin_idx].add(token)
        seq_tokens = ["[ADMIT]"]
        total_event_tokens = 0
        for bin_set in bin_sets:
            bin_tokens = sorted(bin_set)
            trimmed = bin_tokens[:20]
            total_event_tokens += len(trimmed)
            seq_tokens.extend(trimmed)
        seq_tokens.append("[DISCH]")
        if total_event_tokens == 0:
            continue
        sequences.append({"encounter": enc_id, "tokens": seq_tokens})
    return sequences


def save_sequences(sequences, path):
    with open(path, "w") as f:
        for record in sequences:
            f.write(json.dumps(record) + "\n")


def build_vocab(sequences, max_size=VOCAB_LIMIT):
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


def split_encounters(sequences, val_ratio=0.1, seed=SEED):
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

    def __init__(self, sequences_path, vocab, allowed_encounters=None, max_len=MAX_LEN, mask_prob=MASK_PROB, seed=SEED):
        self.records = []
        allowed = set(allowed_encounters) if allowed_encounters else None
        with open(sequences_path, "r") as f:
            for line in f:
                obj = json.loads(line)
                if allowed and obj["encounter"] not in allowed:
                    continue
                ids = [vocab.get(tok, vocab["[MASK]"]) for tok in obj["tokens"]]
                self.records.append(ids)
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.pad_id = vocab["[PAD]"]
        self.mask_id = vocab["[MASK]"]
        self.vocab_size = len(vocab)
        self.special_ids = {vocab[token] for token in SPECIAL_TOKENS if token in vocab}
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
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
            mask_positions = self.rng.sample(candidates, num_to_mask)
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


def make_dataloaders(seq_path, vocab, train_ids, val_ids, batch_size=64, max_len=MAX_LEN):
    train_dataset = EHRDataset(seq_path, vocab, allowed_encounters=train_ids, max_len=max_len)
    val_dataset = EHRDataset(seq_path, vocab, allowed_encounters=val_ids, max_len=max_len) if val_ids else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
    return train_loader, val_loader


def train_epoch(model, dataloader, optimizer, device, print_every=50):
    model.train()
    total_loss = 0.0
    progress = tqdm(dataloader, desc="train", leave=False)
    for step, batch in enumerate(progress):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        if (step + 1) % print_every == 0:
            progress.set_postfix(loss=f"{loss.item():.4f}")
    progress.close()
    return total_loss / max(1, len(dataloader))


def eval_epoch(model, dataloader, device):
    if dataloader is None or len(dataloader.dataset) == 0:
        return float("nan")
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        progress = tqdm(dataloader, desc="eval", leave=False)
        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
        progress.close()
    return total_loss / max(1, len(dataloader))


def demo_mask_fill(model, sequences, vocab, idx_to_token, device, max_examples=3):
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


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    data_dir = os.path.join(os.getcwd(), "data")
    output_dir = os.path.join(os.getcwd(), "model")
    os.makedirs(output_dir, exist_ok=True)

    conditions = load_conditions(data_dir)
    observations = load_observations(data_dir)
    encounters = load_encounters(data_dir)

    windows = extract_encounter_windows(encounters)

    sequences = build_sequences(conditions, observations, windows)
    if not sequences:
        print("no sequences found in first 24h; exiting.")
        return

    seq_path = os.path.join(output_dir, "sequences.jsonl")
    save_sequences(sequences, seq_path)

    vocab, idx_to_token = build_vocab(sequences)
    vocab_path = os.path.join(output_dir, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2)

    avg_len = float(np.mean([len(record["tokens"]) for record in sequences]))
    print(f"encounters: {len(sequences)} | avg tokens: {avg_len:.1f} | vocab size: {len(vocab)}")
    if len(sequences) < 500:
        print("warning: fewer than 500 encounters after filtering; model quality may suffer.")

    train_ids, val_ids = split_encounters(sequences)
    print(f"train encounters: {len(train_ids)} | val encounters: {len(val_ids)}")
    train_loader, val_loader = make_dataloaders(seq_path, vocab, train_ids, val_ids)

    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=256,
        num_hidden_layers=3,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=512,
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2,
    )
    model = BertForMaskedLM(config)
    device = torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    num_epochs = 3
    for epoch in range(1, num_epochs + 1):
        print(f"epoch {epoch}/{num_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = eval_epoch(model, val_loader, device)
        if np.isnan(val_loss):
            print(f"  train loss: {train_loss:.4f} | val loss: n/a (no val set)")
        else:
            print(f"  train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")

    model.save_pretrained(output_dir)
    print(f"saved model and artifacts to {output_dir}")

    demo_mask_fill(model, sequences, vocab, idx_to_token, device)


if __name__ == "__main__":
    main()
