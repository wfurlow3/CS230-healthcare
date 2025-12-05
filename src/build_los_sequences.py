import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer

from .build_sequences import build_sequences, extract_encounter_windows
from .config import LOS_WINDOW_HOURS, SPECIAL_TOKENS
from .data_io import load_conditions, load_encounters, load_observations
from .vocab import load_vocab


def filter_encounters_by_class(encounters: pd.DataFrame, allowed_classes: Sequence[str]) -> pd.DataFrame:
    if "ENCOUNTERCLASS" not in encounters.columns:
        raise ValueError("encounters.csv missing ENCOUNTERCLASS column needed for LOS filtering.")
    allowed = {c.upper() for c in allowed_classes}
    filtered = encounters[encounters["ENCOUNTERCLASS"].str.upper().isin(allowed)]
    if filtered.empty:
        raise ValueError(f"no encounters matched LOS classes: {sorted(allowed)}")
    return filtered


def build_patient_event_index(events_by_encounter: Dict[str, Iterable[Tuple[pd.Timestamp, str]]], encounters: pd.DataFrame) -> Dict[str, List[Tuple[pd.Timestamp, str]]]:
    patient_events: Dict[str, List[Tuple[pd.Timestamp, str]]] = defaultdict(list)
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


def build_los_sequences(
    events_by_encounter: Dict[str, Iterable[Tuple[pd.Timestamp, str]]],
    encounters: pd.DataFrame,
    vocab: Dict[str, int],
    cutoff_hours: float = LOS_WINDOW_HOURS,
    allowed_classes: Sequence[str] = ("INPATIENT",),
) -> Tuple[List[dict], float]:
    allowed = filter_encounters_by_class(encounters, allowed_classes)
    patient_events = build_patient_event_index(events_by_encounter, encounters)
    cutoff_delta = pd.Timedelta(hours=cutoff_hours)
    specials = set(SPECIAL_TOKENS)
    vocab_tokens = {tok for tok in vocab if tok not in specials}

    records: List[dict] = []
    lengths = []
    for _, row in allowed.iterrows():
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
        seen = set()
        for ts, token in patient_events.get(patient, []):
            if pd.isna(ts) or ts > cutoff_time:
                continue
            if token in vocab_tokens and token not in seen:
                tokens.append(token)
                seen.add(token)

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
        raise ValueError("no LOS sequences generated; check encounter classes and cutoff.")

    median_hours = float(np.median(lengths))
    for record in records:
        record["label"] = int(record["los_hours"] > median_hours)

    return records, median_hours


def save_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def vectorize_instances(records: List[dict], vocab: Dict[str, int]) -> Tuple[sparse.csr_matrix, np.ndarray, List[str], List[str], List[str]]:
    specials = set(SPECIAL_TOKENS)
    classes = [tok for tok, idx in sorted(vocab.items(), key=lambda item: item[1]) if tok not in specials]
    mlb = MultiLabelBinarizer(classes=classes, sparse_output=True)
    token_lists = [rec["tokens"] for rec in records]
    X = mlb.fit_transform(token_lists)
    y = np.array([rec["label"] for rec in records], dtype=np.int64)
    encounters = [rec["encounter"] for rec in records]
    patients = [rec["patient"] for rec in records]
    return sparse.csr_matrix(X), y, encounters, patients, classes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LOS sequences using token events from build_sequences.")
    parser.add_argument("--raw_dir", type=Path, default=Path("data") / "raw", help="Directory containing raw CSVs.")
    parser.add_argument("--processed_dir", type=Path, default=Path("data") / "processed", help="Directory containing vocab.json and where LOS outputs are written.")
    parser.add_argument("--allowed_classes", type=str, nargs="+", default=["INPATIENT"], help="Encounter classes to include.")
    parser.add_argument("--cutoff_hours", type=float, default=LOS_WINDOW_HOURS, help="Include patient events up to this many hours after encounter start.")
    return parser.parse_args()


def main():
    args = parse_args()

    encounters = load_encounters(str(args.raw_dir))
    conditions = load_conditions(str(args.raw_dir))
    observations = load_observations(str(args.raw_dir))
    windows = extract_encounter_windows(encounters)

    try:
        sequences, events_by_encounter = build_sequences(conditions, observations, windows, return_events=True)
    except TypeError as e:
        raise RuntimeError("build_sequences must support return_events=True to supply events for LOS.") from e
    if not events_by_encounter:
        raise ValueError("no events returned from build_sequences; cannot build LOS sequences.")

    vocab_path = args.processed_dir / "vocab.json"
    vocab = load_vocab(vocab_path)

    records, median_hours = build_los_sequences(
        events_by_encounter,
        encounters,
        vocab,
        cutoff_hours=args.cutoff_hours,
        allowed_classes=args.allowed_classes,
    )

    inst_path = args.processed_dir / "los_instances.jsonl"
    save_jsonl(records, inst_path)

    X, y, encounter_ids, patient_ids, classes = vectorize_instances(records, vocab)
    features_path = args.processed_dir / "los_features.npz"
    labels_path = args.processed_dir / "los_labels.npz"
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
