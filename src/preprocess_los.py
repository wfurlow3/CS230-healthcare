import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer

from .config import SPECIAL_TOKENS
from .data_io import load_conditions, load_encounters, load_observations
from .vocab import load_vocab


def load_sequences(path: Path) -> List[dict]:
    records = []
    with open(path, "r") as f:
        for line in f:
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"no sequences found in {path}")
    return records


def filter_encounters_by_class(encounters: pd.DataFrame, allowed_classes: Sequence[str]) -> pd.DataFrame:
    if "ENCOUNTERCLASS" not in encounters.columns:
        raise ValueError("encounters.csv missing ENCOUNTERCLASS for filtering.")
    allowed = {c.upper() for c in allowed_classes}
    filtered = encounters[encounters["ENCOUNTERCLASS"].str.upper().isin(allowed)]
    if filtered.empty:
        raise ValueError(f"no encounters left after filtering for classes {allowed_classes}.")
    return filtered


def compute_last_event_times(cond_df: pd.DataFrame, obs_df: pd.DataFrame) -> Dict[str, pd.Timestamp]:
    last_ts: Dict[str, pd.Timestamp] = {}

    def update(enc_id: str, ts: object):
        if pd.isna(ts):
            return
        current = last_ts.get(enc_id)
        if current is None or ts > current:
            last_ts[enc_id] = ts

    for _, row in cond_df.iterrows():
        enc_id = str(row["ENCOUNTER"])
        update(enc_id, row.get("START"))
        update(enc_id, row.get("STOP"))

    for _, row in obs_df.iterrows():
        enc_id = str(row["ENCOUNTER"])
        update(enc_id, row.get("DATE"))

    return last_ts


def attach_metadata(sequences: Iterable[dict], encounters: pd.DataFrame, last_event_times: Dict[str, pd.Timestamp]) -> List[dict]:
    meta = {}
    for _, row in encounters.iterrows():
        enc_id = str(row["ID"])
        meta[enc_id] = {"patient": str(row["PATIENT"]), "start": row["START"], "stop": row["STOP"]}
    records = []
    for record in sequences:
        enc_id = record["encounter"]
        info = meta.get(enc_id)
        if not info:
            continue
        start, stop = info["start"], info["stop"]
        if pd.isna(start):
            continue
        last_event = last_event_times.get(enc_id)
        adjusted_stop = stop
        if last_event is not None and not pd.isna(last_event):
            adjusted_stop = min(last_event, stop) if not pd.isna(stop) else last_event
        if pd.isna(adjusted_stop) or adjusted_stop < start:
            adjusted_stop = stop
        if pd.isna(adjusted_stop) or adjusted_stop < start:
            continue
        records.append(
            {
                "encounter": enc_id,
                "patient": info["patient"],
                "start": start,
                "stop": adjusted_stop,
                "tokens": record["tokens"],
            }
        )
    if not records:
        raise ValueError("no encounters with valid start/stop matched the sequences file.")
    return records


def compute_los_hours(records: List[dict]) -> Tuple[float, List[dict]]:
    lengths = []
    for record in records:
        los_hours = (record["stop"] - record["start"]).total_seconds() / 3600.0
        record["los_hours"] = los_hours
        lengths.append(los_hours)
    if not lengths:
        raise ValueError("no valid lengths of stay found.")
    median_hours = float(np.median(lengths))
    return median_hours, records


def build_los_instances(records: List[dict], vocab: Dict[str, int]) -> Tuple[List[dict], float]:
    specials = set(SPECIAL_TOKENS)
    vocab_tokens = {tok for tok in vocab if tok not in specials}
    records = sorted(records, key=lambda r: (r["patient"], r["start"]))
    median_hours, records = compute_los_hours(records)

    history: Dict[str, set] = defaultdict(set)
    instances: List[dict] = []
    for record in records:
        patient = record["patient"]
        prev_tokens = history[patient]
        curr_tokens = {tok for tok in record["tokens"] if tok not in specials and tok in vocab_tokens}
        combined_tokens = sorted(prev_tokens | curr_tokens)
        if not combined_tokens:
            history[patient].update(curr_tokens)
            continue
        label = int(record["los_hours"] > median_hours)
        instances.append(
            {
                "encounter": record["encounter"],
                "patient": patient,
                "label": label,
                "los_hours": record["los_hours"],
                "tokens": combined_tokens,
                "start": record["start"].isoformat(),
                "stop": record["stop"].isoformat(),
                "prev_token_count": len(prev_tokens),
                "curr_token_count": len(curr_tokens),
            }
        )
        history[patient].update(curr_tokens)
    if not instances:
        raise ValueError("no labelable encounters were produced.")
    return instances, median_hours


def save_instances(instances: Iterable[dict], path: Path) -> None:
    with open(path, "w") as f:
        for record in instances:
            f.write(json.dumps(record) + "\n")


def vectorize_instances(instances: List[dict], vocab: Dict[str, int]) -> Tuple[sparse.csr_matrix, np.ndarray, List[str], List[str], List[str]]:
    specials = set(SPECIAL_TOKENS)
    classes = [tok for tok, idx in sorted(vocab.items(), key=lambda item: item[1]) if tok not in specials]
    mlb = MultiLabelBinarizer(classes=classes, sparse_output=True)
    token_lists = [inst["tokens"] for inst in instances]
    X = mlb.fit_transform(token_lists)
    y = np.array([inst["label"] for inst in instances], dtype=np.int64)
    encounters = [inst["encounter"] for inst in instances]
    patients = [inst["patient"] for inst in instances]
    return sparse.csr_matrix(X), y, encounters, patients, classes


def preprocess_los(raw_data_dir: Path, processed_dir: Path, allowed_classes: Sequence[str] = ("INPATIENT",)) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    seq_path = processed_dir / "sequences.jsonl"
    vocab_path = processed_dir / "vocab.json"

    sequences = load_sequences(seq_path)
    vocab = load_vocab(vocab_path)
    encounters = load_encounters(str(raw_data_dir))
    conditions = load_conditions(str(raw_data_dir))
    observations = load_observations(str(raw_data_dir))
    last_event_times = compute_last_event_times(conditions, observations)
    encounters = filter_encounters_by_class(encounters, allowed_classes)
    records = attach_metadata(sequences, encounters, last_event_times)

    instances, median_hours = build_los_instances(records, vocab)
    inst_path = processed_dir / "los_instances.jsonl"
    save_instances(instances, inst_path)

    X, y, encounter_ids, patient_ids, classes = vectorize_instances(instances, vocab)
    feature_path = processed_dir / "los_features.npz"
    label_path = processed_dir / "los_labels.npz"
    sparse.save_npz(feature_path, X)
    np.savez_compressed(
        label_path,
        y=y,
        encounter=np.array(encounter_ids),
        patient=np.array(patient_ids),
        median_hours=median_hours,
        classes=np.array(classes),
    )

    pos_rate = float(np.mean(y))
    print(
        f"encounters (filtered {allowed_classes}): {len(instances)} | median stay hours: {median_hours:.2f} | positive rate: {pos_rate:.3f}"
    )
    print(f"wrote {inst_path}")
    print(f"wrote sparse features to {feature_path} and labels/meta to {label_path}")


if __name__ == "__main__":
    base_dir = Path.cwd()
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    preprocess_los(raw_dir, processed_dir)
