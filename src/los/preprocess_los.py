import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer

from ..config import LOS_WINDOW_HOURS, SPECIAL_TOKENS
from ..data_io import load_conditions, load_encounters, load_observations, load_medications
from ..utils import clean_fragment, OBS_WHITELIST, bin_observation
from ..vocab import load_vocab


def filter_encounters_by_class(encounters: pd.DataFrame, allowed_classes: Sequence[str]) -> pd.DataFrame:
    if "ENCOUNTERCLASS" not in encounters.columns:
        raise ValueError("encounters.csv missing ENCOUNTERCLASS for filtering.")
    allowed = {c.upper() for c in allowed_classes}
    filtered = encounters[encounters["ENCOUNTERCLASS"].str.upper().isin(allowed)]
    if filtered.empty:
        raise ValueError(f"no encounters left after filtering for classes {allowed_classes}.")
    return filtered


def extract_encounter_windows(enc_df: pd.DataFrame) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    windows = {}
    for _, row in enc_df.iterrows():
        enc_id = row["ID"]
        start = row["START"]
        stop = row["STOP"]
        windows[str(enc_id)] = (start, stop)
    return windows


def build_los_sequences(
    cond_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    windows: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]],
    window_hours: int,
    bin_size_hours: int,
) -> List[Dict[str, Iterable[str]]]:
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
    num_bins = max(1, window_hours // bin_size_hours)
    for enc_id, (start_time, _) in windows.items():
        if pd.isna(start_time):
            continue
        bin_sets = [set() for _ in range(num_bins)]
        for event_time, token in events.get(enc_id, []):
            ts = event_time if event_time is not None and not pd.isna(event_time) else start_time
            if pd.isna(ts):
                continue
            delta_hours = (ts - start_time).total_seconds() / 3600.0
            if delta_hours < 0 or delta_hours >= window_hours:
                continue
            bin_idx = min(int(delta_hours // bin_size_hours), len(bin_sets) - 1)
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


def load_prebuilt_sequences(seq_path: Path) -> List[dict]:
    """
    Load sequences produced by the main preprocessing step so LOS tokens
    match the vocabulary generated by `src.build_sequences.preprocess`.
    """
    if not seq_path.exists():
        raise FileNotFoundError(
            f"missing {seq_path}; run main preprocessing to build sequences and vocab first."
        )
    sequences: List[dict] = []
    with open(seq_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Ensure encounter ids are strings for consistent downstream matching
            sequences.append({"encounter": str(obj["encounter"]), "tokens": obj["tokens"]})
    if not sequences:
        raise ValueError(f"no sequences found in {seq_path}.")
    return sequences


def build_first_hour_sequences(
    cond_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    meds_df: pd.DataFrame,
    windows: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]],
    window_hours: int,
) -> Dict[str, List[str]]:
    """
    Build per-encounter tokens for the first `window_hours` of the visit using
    the same tokenization as the main sequence builder (DX_/OBS_/MED_).
    """
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
        if not token:
            continue
        event_time = row["START"] if not pd.isna(row["START"]) else None
        events[enc_id].append((event_time, token))

    for _, row in obs_df.iterrows():
        enc_id = str(row["ENCOUNTER"])
        if enc_id not in windows:
            continue
        code_clean = clean_fragment(row["CODE"])
        if code_clean not in OBS_WHITELIST:
            continue
        try:
            value = float(row["VALUE"])
        except (TypeError, ValueError):
            continue
        obs_type, bucket = bin_observation(code_clean, value)
        token = f"OBS_{obs_type}_{bucket}"
        event_time = row["DATE"] if not pd.isna(row["DATE"]) else None
        events[enc_id].append((event_time, token))

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

    sequences: Dict[str, List[str]] = {}
    for enc_id, (start_time, _) in windows.items():
        if pd.isna(start_time):
            continue
        encounter_events = []
        for event_time, token in events.get(enc_id, []):
            ts = event_time if event_time is not None and not pd.isna(event_time) else start_time
            if pd.isna(ts):
                continue
            delta_hours = (ts - start_time).total_seconds() / 3600.0
            if delta_hours < 0 or delta_hours >= window_hours:
                continue
            encounter_events.append((ts, token))
        if not encounter_events:
            continue
        encounter_events.sort(key=lambda t: t[0])
        ordered_tokens = ["[ADMIT]"] + [tok for _, tok in encounter_events]
        sequences[enc_id] = ordered_tokens
    return sequences


def index_sequences_by_encounter(sequences: Iterable[dict]) -> Dict[str, List[str]]:
    return {str(rec["encounter"]): rec["tokens"] for rec in sequences}


def compute_history_tokens(encounters: pd.DataFrame, full_sequences: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    For each encounter, accumulate all prior tokens for that patient based on
    chronological order of encounters (any class).
    """
    specials = set(SPECIAL_TOKENS)
    history: Dict[str, List[str]] = {}
    patient_history: Dict[str, List[str]] = defaultdict(list)

    encounters_sorted = encounters.sort_values(["PATIENT", "START"])
    for _, row in encounters_sorted.iterrows():
        enc_id = str(row["ID"])
        patient = str(row["PATIENT"])
        history[enc_id] = list(patient_history[patient])
        current_tokens = [tok for tok in full_sequences.get(enc_id, []) if tok not in specials]
        patient_history[patient].extend(current_tokens)
    return history


def attach_metadata_with_history(
    sequences: Dict[str, List[str]],
    encounters: pd.DataFrame,
    last_event_times: Dict[str, pd.Timestamp],
    history_tokens: Dict[str, List[str]],
) -> List[dict]:
    meta = {}
    for _, row in encounters.iterrows():
        enc_id = str(row["ID"])
        meta[enc_id] = {"patient": str(row["PATIENT"]), "start": row["START"], "stop": row["STOP"]}

    records = []
    for enc_id, seq_tokens in sequences.items():
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
                "tokens": seq_tokens,
                "history_tokens": history_tokens.get(enc_id, []),
            }
        )
    if not records:
        raise ValueError("no encounters with valid start/stop matched the sequences file.")
    return records


def build_los_instances(records: List[dict], vocab: Dict[str, int]) -> Tuple[List[dict], float]:
    specials = set(SPECIAL_TOKENS)
    vocab_tokens = {tok for tok in vocab if tok not in specials}
    records = sorted(records, key=lambda r: (r["patient"], r["start"]))
    median_hours, records = compute_los_hours(records)

    instances: List[dict] = []
    for record in records:
        prev_tokens = [tok for tok in record.get("history_tokens", []) if tok in vocab_tokens]
        curr_tokens = [tok for tok in record["tokens"] if tok not in specials and tok in vocab_tokens]
        combined_tokens = prev_tokens + curr_tokens
        if not combined_tokens:
            continue
        label = int(record["los_hours"] > median_hours)
        instances.append(
            {
                "encounter": record["encounter"],
                "patient": record["patient"],
                "label": label,
                "los_hours": record["los_hours"],
                "tokens": combined_tokens,
                "start": record["start"].isoformat(),
                "stop": record["stop"].isoformat(),
                "prev_token_count": len(prev_tokens),
                "curr_token_count": len(curr_tokens),
            }
        )
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
    vocab_path = processed_dir / "vocab.json"
    seq_path = processed_dir / "sequences.jsonl"

    vocab = load_vocab(vocab_path)
    encounters = load_encounters(str(raw_data_dir))
    conditions = load_conditions(str(raw_data_dir))
    observations = load_observations(str(raw_data_dir))
    medications = load_medications(str(raw_data_dir))
    sequences = load_prebuilt_sequences(seq_path)
    seq_by_encounter = index_sequences_by_encounter(sequences)
    windows = extract_encounter_windows(encounters)
    first_hour_sequences = build_first_hour_sequences(
        conditions, observations, medications, windows, LOS_WINDOW_HOURS
    )
    history_tokens = compute_history_tokens(encounters, seq_by_encounter)
    last_event_times = compute_last_event_times(conditions, observations)
    encounters = filter_encounters_by_class(encounters, allowed_classes)
    filtered_first_hour_sequences = {
        enc: toks for enc, toks in first_hour_sequences.items() if enc in set(encounters["ID"].astype(str))
    }
    records = attach_metadata_with_history(filtered_first_hour_sequences, encounters, last_event_times, history_tokens)

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
