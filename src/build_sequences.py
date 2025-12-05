import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

# No longer using BIN_SIZE_HOURS or WINDOW_HOURS - sequences are chronological
from .data_io import load_conditions, load_encounters, load_observations, load_medications, load_patients
from .vocab import build_vocab, save_vocab
from .utils import clean_fragment, bucket_age, OBS_WHITELIST, bin_observation

def extract_encounter_windows(enc_df: pd.DataFrame) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    windows = {}
    for _, row in enc_df.iterrows():
        enc_id = row["ID"]
        start = row["START"]
        stop = row["STOP"]
        windows[str(enc_id)] = (start, stop)
    return windows

def build_demo_by_encounter(patients_df, encounters_df):
    patients = patients_df.set_index("ID")
    demo_by_enc = {}

    for _, row in encounters_df.iterrows():
        enc_id = str(row["ID"])
        pat_id = row["PATIENT"]
        pat = patients.loc[pat_id]

        start = row["START"]
        age_years = (start - pat["BIRTHDATE"]).days / 365.25

        sex_token = f"SEX_{str(pat['GENDER']).upper()}"
        race_token = f"RACE_{clean_fragment(pat['RACE'])}"
        age_token = bucket_age(age_years)

        demo_by_enc[enc_id] = {
            "sex": sex_token,
            "race": race_token,
            "age": age_token,
        }

    return demo_by_enc

def build_sequences(cond_df: pd.DataFrame, obs_df: pd.DataFrame, windows: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]], meds_df: pd.DataFrame, demo_by_enc) -> List[Dict[str, Iterable[str]]]:
    events: Dict[str, List[Tuple[pd.Timestamp, str]]] = defaultdict(list)
    
    # Conditions
    for _, row in cond_df.iterrows():
        enc_id = str(row["ENCOUNTER"])
        if enc_id not in windows:
            continue
        token = None
        if not pd.isna(row["CODE"]):
            code = clean_fragment(row["CODE"])
            if code:
                token = f"DX_{code}"
        if not token:
            continue
        event_time = row["START"] if not pd.isna(row["START"]) else None
        events[enc_id].append((event_time, token))

    # Observations (whitelisted + binned)
    for _, row in obs_df.iterrows():
        enc_id = str(row["ENCOUNTER"])
        if enc_id not in windows:
            continue

        code_clean = clean_fragment(row["CODE"])
        if code_clean not in OBS_WHITELIST:
            continue

        value = float(row["VALUE"])
        obs_type, bucket = bin_observation(code_clean, value)

        token = f"OBS_{obs_type}_{bucket}"
        event_time = row["DATE"]
        events[enc_id].append((event_time, token))

    # Medications
    for _, row in meds_df.iterrows():
        enc_id = str(row["ENCOUNTER"])
        if enc_id not in windows:
            continue
        code = clean_fragment(row["CODE"])
        if not code:
            continue
        token = f"MED_{code}"
        event_time = row["START"] if not pd.isna(row["START"]) else None
        events[enc_id].append((event_time, token))

    sequences = []
    for enc_id, (start_time, stop_time) in windows.items():
        if pd.isna(start_time):
            continue
        # Get all events for this encounter and sort chronologically
        encounter_events = events.get(enc_id, [])
        if not encounter_events:
            continue
        
        # Sort events by timestamp (events without timestamps go to the end)
        def sort_key(event_tuple):
            event_time, _ = event_tuple
            if event_time is None or pd.isna(event_time):
                # Put events without timestamps at the end
                return pd.Timestamp.max
            return event_time
        
        sorted_events = sorted(encounter_events, key=sort_key)
        

        if len(sorted_events) <= 2: # Only include encounters that have at least 3 tokens
            continue

        if len(sorted_events) > 251:
            sorted_events = sorted_events[-256:]
        # Build sequence chronologically
        demo = demo_by_enc[enc_id]
        seq_tokens = [
            "[ADMIT]",
            demo["sex"],
            demo["race"],
            demo["age"],
        ]
        for event_time, token in sorted_events:
            seq_tokens.append(token)
        seq_tokens.append("[DISCH]")
        
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
    meds = load_medications(str(raw_data_dir))
    patients = load_patients(str(raw_data_dir))

    demo_by_enc = build_demo_by_encounter(patients, encounters)
    windows = extract_encounter_windows(encounters)

    sequences = build_sequences(conditions, observations, windows, meds, demo_by_enc)
    if not sequences:
        print("no sequences found; exiting.")
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
