import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import SPECIAL_TOKENS
from .utils import OBS_WHITELIST, bucket_age, clean_fragment, bin_observation
from .vocab import load_vocab
from .data_io import load_conditions, load_encounters, load_observations, load_medications, load_patients

ALLOWED_L0S_CLASSES = ("INPATIENT",)

def filter_encounters_by_class(encounters, allowed):
    if "ENCOUNTERCLASS" not in encounters.columns:
        raise ValueError("encounters.csv missing ENCOUNTERCLASS column required for LOS filtering.")
    allowed_upper = {c.upper() for c in allowed}
    mask = encounters["ENCOUNTERCLASS"].str.upper().isin(allowed_upper)
    filtered = encounters[mask].copy()
    if filtered.empty:
        raise ValueError(f"no encounters matched LOS classes: {allowed_upper}")
    return filtered


def build_demo_by_encounter(patients_df, encounters_df):
    patients = patients_df.set_index("ID")
    demo_by_enc = {}
    for _, row in encounters_df.iterrows():
        enc_id = str(row["ID"])
        pat_id = row["PATIENT"]
        if pat_id not in patients.index:
            continue
        pat = patients.loc[pat_id]
        start = row["START"]
        if pd.isna(start) or pd.isna(pat["BIRTHDATE"]):
            continue
        age_years = (start - pat["BIRTHDATE"]).days / 365.25
        sex_token = f"SEX_{str(pat['GENDER']).upper()}"
        race_token = f"RACE_{clean_fragment(pat['RACE'])}"
        age_token = bucket_age(age_years)
        demo_by_enc[enc_id] = {"sex": sex_token, "race": race_token, "age": age_token}
    return demo_by_enc


def collect_event_tokens(cond_df, obs_df, meds_df, valid_encounters):
    valid = set(valid_encounters)
    events = defaultdict(list)

    for _, row in cond_df.iterrows():
        enc_id = str(row["ENCOUNTER"])
        if enc_id not in valid:
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

    for _, row in obs_df.iterrows():
        enc_id = str(row["ENCOUNTER"])
        if enc_id not in valid:
            continue
        code_clean = clean_fragment(row["CODE"])
        if code_clean not in OBS_WHITELIST:
            continue
        try:
            value = float(row["VALUE"])
        except (TypeError, ValueError):
            continue
        if pd.isna(value):
            continue
        obs_type, bucket = bin_observation(code_clean, value)
        token = f"OBS_{obs_type}_{bucket}"
        event_time = row["DATE"]
        events[enc_id].append((event_time, token))

    for _, row in meds_df.iterrows():
        enc_id = str(row["ENCOUNTER"])
        if enc_id not in valid:
            continue
        code = clean_fragment(row["CODE"])
        if not code:
            continue
        token = f"MED_{code}"
        event_time = row["START"] if not pd.isna(row["START"]) else None
        events[enc_id].append((event_time, token))

    return events


def sort_events(events):
    def sort_key(item):
        ts, _ = item
        if ts is None or pd.isna(ts):
            return pd.Timestamp.max
        return ts

    return sorted(events, key=sort_key)


def build_patient_event_index(events_by_encounter, encounter_meta):
    patient_events = defaultdict(list)
    for enc_id, events in events_by_encounter.items():
        meta = encounter_meta.get(enc_id)
        if not meta:
            continue
        patient = str(meta["patient"])
        for ts, token in events:
            if ts is None or pd.isna(ts):
                continue
            patient_events[patient].append((ts, token))
    for patient in list(patient_events):
        patient_events[patient].sort(key=lambda item: item[0])
    return patient_events


def generate_los_sequences(
    encounters_df,
    patients_df,
    cond_df,
    obs_df,
    meds_df,
    vocab,
    allowed_classes,
    one_hour,
):
    encounter_meta = {}
    for _, row in encounters_df.iterrows():
        encounter_meta[str(row["ID"])] = {
            "patient": str(row["PATIENT"]),
            "start": row["START"],
            "stop": row["STOP"],
        }

    demo_by_enc = build_demo_by_encounter(patients_df, encounters_df)
    events_by_enc = collect_event_tokens(cond_df, obs_df, meds_df, encounter_meta.keys())
    for enc_id, events in list(events_by_enc.items()):
        events_by_enc[enc_id] = sort_events(events)

    patient_events = build_patient_event_index(events_by_enc, encounter_meta)
    target_encounters = filter_encounters_by_class(encounters_df, allowed_classes)
    vocab_tokens = set(vocab.keys())

    sequences = []
    lengths = []
    for _, row in target_encounters.iterrows():
        enc_id = str(row["ID"])
        meta = encounter_meta.get(enc_id)
        demo = demo_by_enc.get(enc_id)
        if not meta or not demo:
            continue
        start = meta["start"]
        stop = meta["stop"]
        if pd.isna(start) or pd.isna(stop):
            continue
        los_hours = (stop - start).total_seconds() / 3600.0
        if not np.isfinite(los_hours) or los_hours <= 0:
            continue

        cutoff = start + one_hour
        patient_id = str(meta["patient"])
        history_tokens = []
        for ts, token in patient_events.get(patient_id, []):
            if ts < start:
                history_tokens.append(token)
            else:
                break

        curr_tokens = []
        included_times = []
        for ts, token in events_by_enc.get(enc_id, []):
            if ts is None or pd.isna(ts):
                continue
            if ts < start:
                continue
            if ts > cutoff:
                continue
            curr_tokens.append(token)
            included_times.append(ts)
        if included_times:
            assert all(ts <= cutoff for ts in included_times), "found event beyond cutoff in LOS sequence"

        seq_tokens = [
            "[ADMIT]",
            demo["sex"],
            demo["race"],
            demo["age"],
            *history_tokens,
            *curr_tokens,
            "[DISCH]",
        ]
        filtered_tokens = [tok for tok in seq_tokens if tok in vocab_tokens]
        if len(filtered_tokens) < 2:
            continue

        sequences.append(
            {
                "patient_id": patient_id,
                "encounter_id": enc_id,
                "tokens": filtered_tokens,
                "los_hours": float(los_hours),
            }
        )
        lengths.append(los_hours)

    if not sequences:
        raise ValueError("no LOS sequences generated; check filtering and input data.")

    median_hours = float(np.median(lengths))
    for record in sequences:
        record["los_label"] = int(record["los_hours"] > median_hours)

    return sequences


def save_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LOS sequences (history + first-hour events).")
    parser.add_argument("--encounters_csv", type=Path, required=True, help="Path to encounters.csv")
    parser.add_argument("--conditions_csv", type=Path, required=True, help="Path to conditions.csv")
    parser.add_argument("--observations_csv", type=Path, required=True, help="Path to observations.csv")
    parser.add_argument("--medications_csv", type=Path, required=True, help="Path to medications.csv")
    parser.add_argument("--patients_csv", type=Path, required=True, help="Path to patients.csv")
    parser.add_argument("--vocab_path", type=Path, required=True, help="Path to pretrained vocab JSON")
    parser.add_argument("--output_path", type=Path, required=True, help="Output JSONL path for LOS sequences")
    parser.add_argument(
        "--allowed_classes",
        type=str,
        nargs="+",
        default=list(ALLOWED_L0S_CLASSES),
        help="Encounter classes eligible for LOS (default: INPATIENT)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    encounters = load_encounters(args.encounters_csv)
    conditions = load_conditions(args.conditions_csv)
    observations = load_observations(args.observations_csv)
    medications = load_medications(args.medications_csv)
    patients = load_patients(args.patients_csv)
    vocab = load_vocab(args.vocab_path) 

    one_hour = pd.Timedelta(hours=1)
    sequences = generate_los_sequences(
        encounters,
        patients,
        conditions,
        observations,
        medications,
        vocab,
        allowed_classes=args.allowed_classes,
        one_hour=one_hour,
    )
    save_jsonl(sequences, args.output_path)

    lengths = [len(rec["tokens"]) for rec in sequences]
    avg_len = float(np.mean(lengths)) if lengths else 0.0
    max_len = max(lengths) if lengths else 0
    print(
        f"Wrote {len(sequences)} LOS sequences to {args.output_path} | avg tokens: {avg_len:.1f} | max tokens: {max_len}"
    )


if __name__ == "__main__":
    main()
