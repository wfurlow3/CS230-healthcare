import os
import re
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd





def load_csv(path: str, date_cols: Optional[Iterable[str]] = None) -> pd.DataFrame:
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

def load_medications(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "medications.csv")
    df = load_csv(path, date_cols=["START", "STOP"])
    return df

def load_patients(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "medications.csv")
    df = load_csv(path, date_cols=["BIRTHDATE"])
    return df

def load_conditions(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "conditions.csv")
    df = load_csv(path, date_cols=["START", "STOP"])
    required = ["ENCOUNTER", "START", "CODE", "DESCRIPTION"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"conditions.csv missing required columns: {missing}")
    return df


def load_observations(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "observations.csv")
    df = load_csv(path, date_cols=["DATE"])
    required = ["ENCOUNTER", "DATE", "CODE", "DESCRIPTION", "CATEGORY"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"observations.csv missing required columns: {missing}")
    return df


def load_encounters(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "encounters.csv")
    df = load_csv(path, date_cols=["START", "STOP"])
    print(df.columns)
    required = ["ID", "START", "STOP"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"encounters.csv missing required columns: {missing}")
    return df
