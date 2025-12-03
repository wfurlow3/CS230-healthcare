import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .config import BIN_SIZE_HOURS, WINDOW_HOURS
from .data_io import clean_fragment, load_conditions, load_encounters, load_observations
from .vocab import build_vocab, save_vocab


def extract_encounter_windows(enc_df: pd.DataFrame) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
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


def build_sequences(cond_df: pd.DataFrame, obs_df: pd.DataFrame, windows: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]) -> List[Dict[str, Iterable[str]]]:
    events: Dict[str, List[Tuple[pd.Timestamp, str]]] = defaultdict(list)

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


def save_sequences(sequences: Iterable[dict], path: Path) -> None:
    with open(path, "w") as f:
        for record in sequences:
            f.write(json.dumps(record) + "\n")


def preprocess(raw_data_dir: Path, processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)

    conditions = load_conditions(str(raw_data_dir))
    observations = load_observations(str(raw_data_dir))
    encounters = load_encounters(str(raw_data_dir))

    windows = extract_encounter_windows(encounters)

    sequences = build_sequences(conditions, observations, windows)
    if not sequences:
        print("no sequences found in first 24h; exiting.")
        return

    seq_path = processed_dir / "sequences.jsonl"
    save_sequences(sequences, seq_path)

    vocab, _ = build_vocab(sequences)
    vocab_path = processed_dir / "vocab.json"
    save_vocab(vocab, vocab_path)

    avg_len = float(np.mean([len(record["tokens"]) for record in sequences]))
    print(f"encounters: {len(sequences)} | avg tokens: {avg_len:.1f} | vocab size: {len(vocab)}")
    if len(sequences) < 500:
        print("warning: fewer than 500 encounters after filtering; model quality may suffer.")


if __name__ == "__main__":
    base_dir = Path(os.getcwd())
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    preprocess(raw_dir, processed_dir)
