import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer

from .build_sequences import build_demo_by_encounter, build_sequences, extract_encounter_windows
from .config import LOS_WINDOW_HOURS, SPECIAL_TOKENS
from .data_io import load_conditions, load_encounters, load_observations, load_patients
from .vocab import load_vocab


def build_patient_event_index(events_by_encounter, encounters):
    patient_events = defaultdict(list)
    patient_by_encounter = {}
    for _, row in encounters.iterrows():
        patient_by_encounter[str(row["ID"])] = str(row.get("PATIENT"))
    for enc_id, events in events_by_encounter.items():
        patient = patient_by_encounter.get(str(enc_id))
        if not patient:
            continue
        for ts, token in events:
            if ts is None or pd.isna(ts):
                continue
            patient_events[patient].append((ts, token))
    for patient in list(patient_events):
        patient_events[patient].sort(key=lambda item: item[0])
    return patient_events


def build_los_sequences(events_by_encounter, encounters, vocab, demo_by_enc=None, cutoff_hours=LOS_WINDOW_HOURS):
    patient_events = build_patient_event_index(events_by_encounter, encounters)
    cutoff_delta = pd.Timedelta(hours=cutoff_hours)
    specials = set(SPECIAL_TOKENS)
    vocab_tokens = {tok for tok in vocab if tok not in specials}
    inpatient = encounters[encounters["ENCOUNTERCLASS"] == "Inpatient"]
    records = []
    lengths = []
    for _, row in inpatient.iterrows():
        enc_id = str(row["ID"])
        patient = str(row.get("PATIENT"))
        start = row.get("START")
        stop = row.get("STOP")
        if pd.isna(start) or pd.isna(stop):
            continue
        los_hours = (stop - start).total_seconds() / 3600.0
        if not np.isfinite(los_hours) or los_hours <= 0:
            continue
        cutoff_time = start + cutoff_delta

        tokens = []
        if demo_by_enc and enc_id in demo_by_enc:
            demo = demo_by_enc[enc_id]
            for tok in (demo.get("sex"), demo.get("race"), demo.get("age")):
                if tok and tok in vocab_tokens:
                    tokens.append(tok)
        for ts, token in patient_events.get(patient, []):
            if pd.isna(ts) or ts > cutoff_time:
                continue
            if token in vocab_tokens:
                tokens.append(token)

        if not tokens:
            continue

        lengths.append(los_hours)
        records.append(
            {
                "encounter": enc_id,
                "patient": patient,
                "start": start.isoformat(),
                "stop": stop.isoformat(),
                "los_hours": float(los_hours),
                "tokens": tokens,
            }
        )

    if not records:
        raise ValueError("no LOS sequences generated; check cutoff and data validity.")

    median_hours = float(np.median(lengths))
    for record in records:
        record["label"] = int(record["los_hours"] > median_hours)

    return records, median_hours


def save_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def vectorize_instances(records, vocab):
    specials = set(SPECIAL_TOKENS)
    classes = [tok for tok, idx in sorted(vocab.items(), key=lambda item: item[1]) if tok not in specials]
    mlb = MultiLabelBinarizer(classes=classes, sparse_output=True)
    token_lists = [rec["tokens"] for rec in records]
    X = mlb.fit_transform(token_lists)
    y = np.array([rec["label"] for rec in records], dtype=np.int64)
    encounters = [rec["encounter"] for rec in records]
    patients = [rec["patient"] for rec in records]
    return sparse.csr_matrix(X), y, encounters, patients, classes

def main():
    base_dir = Path.cwd()
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"

    encounters = load_encounters(str(raw_dir))
    conditions = load_conditions(str(raw_dir))
    observations = load_observations(str(raw_dir))
    patients = load_patients(str(raw_dir))
    windows = extract_encounter_windows(encounters)

    sequences, events_by_encounter = build_sequences(conditions, observations, windows, return_events=True)
    demo_by_enc = build_demo_by_encounter(patients, encounters)

    vocab_path = processed_dir / "vocab.json"
    vocab = load_vocab(vocab_path)

    records, median_hours = build_los_sequences(
        events_by_encounter,
        encounters,
        vocab,
        demo_by_enc,
        cutoff_hours=1,
    )

    inst_path = processed_dir / "los_instances.jsonl"
    save_jsonl(records, inst_path)

    X, y, encounter_ids, patient_ids, classes = vectorize_instances(records, vocab)
    features_path = processed_dir / "los_features.npz"
    labels_path = processed_dir / "los_labels.npz"
    sparse.save_npz(features_path, X)
    np.savez_compressed(
        labels_path,
        y=y,
        encounter=np.array(encounter_ids),
        patient=np.array(patient_ids),
        median_hours=median_hours,
        classes=np.array(classes),
    )

    pos_rate = float(np.mean(y))
    print(
        f"LOS sequences: {len(records)} | median stay hours: {median_hours:.2f} | positive rate: {pos_rate:.3f}"
    )
    print(f"wrote instances to {inst_path}")
    print(f"wrote features to {features_path} and labels/meta to {labels_path}")


if __name__ == "__main__":
    main()
