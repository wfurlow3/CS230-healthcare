import re
from typing import Optional

import numpy as np
import pandas as pd


OBS_WHITELIST = {
    "8867_4": "HR",      # heart rate
    "8480_6": "SBP",     # systolic BP
    "8462_4": "DBP",     # diastolic BP
    "9279_1": "RR",      # respiratory rate
    "8310_5": "TEMP",    # body temp
    "2708_6": "O2SAT",   # oxygen saturation
    "39156_5": "BMI",    # body mass index
    "2947_0": "NA",      # sodium
    "6298_4": "K",       # potassium
    "38483_4": "CREAT",  # creatinine
    "718_7": "HGB",      # hemoglobin
}


def bucket_age(age_years: float) -> str:
    if age_years < 18:
        return "AGE_0_17"
    elif age_years < 35:
        return "AGE_18_34"
    elif age_years < 50:
        return "AGE_35_49"
    elif age_years < 65:
        return "AGE_50_64"
    elif age_years < 80:
        return "AGE_65_79"
    else:
        return "AGE_80_PLUS"


def clean_fragment(value: Optional[object]) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    text = re.sub(r"[^A-Za-z0-9]+", "_", text.upper())
    text = text.strip("_")
    return text or None


def bin_hr(v: float) -> str:
    if v < 50:
        return "LT_50"
    elif v < 70:
        return "50_69"
    elif v < 90:
        return "70_89"
    elif v < 110:
        return "90_109"
    elif v < 130:
        return "110_129"
    else:
        return "GE_130"


def bin_sbp(v: float) -> str:
    if v < 90:
        return "LT_90"
    elif v < 120:
        return "90_119"
    elif v < 140:
        return "120_139"
    elif v < 160:
        return "140_159"
    else:
        return "GE_160"


def bin_dbp(v: float) -> str:
    if v < 60:
        return "LT_60"
    elif v < 80:
        return "60_79"
    elif v < 90:
        return "80_89"
    else:
        return "GE_90"


def bin_rr(v: float) -> str:
    if v < 12:
        return "LT_12"
    elif v < 20:
        return "12_19"
    elif v < 30:
        return "20_29"
    else:
        return "GE_30"


def bin_temp_c(v: float) -> str:
    if v < 36.0:
        return "LT_36_0"
    elif v < 37.5:
        return "36_0_37_4"
    elif v < 39.0:
        return "37_5_38_9"
    else:
        return "GE_39_0"


def bin_o2sat(v: float) -> str:
    if v < 90:
        return "LT_90"
    elif v < 95:
        return "90_94"
    else:
        return "GE_95"


def bin_bmi(v: float) -> str:
    if v < 18.5:
        return "LT_18_5"
    elif v < 25:
        return "18_5_24_9"
    elif v < 30:
        return "25_0_29_9"
    elif v < 35:
        return "30_0_34_9"
    elif v < 40:
        return "35_0_39_9"
    else:
        return "GE_40_0"


def bin_na(v: float) -> str:
    if v < 130:
        return "LT_130"
    elif v < 135:
        return "130_134"
    elif v < 146:
        return "135_145"
    else:
        return "GE_146"


def bin_k(v: float) -> str:
    if v < 3.0:
        return "LT_3_0"
    elif v < 4.0:
        return "3_0_3_9"
    elif v < 5.5:
        return "4_0_5_4"
    else:
        return "GE_5_5"


def bin_creat(v: float) -> str:
    if v < 0.7:
        return "LT_0_7"
    elif v < 1.3:
        return "0_7_1_2"
    elif v < 2.0:
        return "1_3_1_9"
    else:
        return "GE_2_0"


def bin_wbc(v: float) -> str:
    if v < 4.0:
        return "LT_4_0"
    elif v < 11.0:
        return "4_0_10_9"
    else:
        return "GE_11_0"


def bin_hgb(v: float) -> str:
    if v < 10.0:
        return "LT_10_0"
    elif v < 13.0:
        return "10_0_12_9"
    elif v < 17.0:
        return "13_0_16_9"
    else:
        return "GE_17_0"


def bin_observation(code_clean: str, value: float) -> tuple[str, str]:
    obs_type = OBS_WHITELIST[code_clean]

    if obs_type == "HR":
        bucket = bin_hr(value)
    elif obs_type == "SBP":
        bucket = bin_sbp(value)
    elif obs_type == "DBP":
        bucket = bin_dbp(value)
    elif obs_type == "RR":
        bucket = bin_rr(value)
    elif obs_type == "TEMP":
        bucket = bin_temp_c(value)
    elif obs_type == "O2SAT":
        bucket = bin_o2sat(value)
    elif obs_type == "BMI":
        bucket = bin_bmi(value)
    elif obs_type == "NA":
        bucket = bin_na(value)
    elif obs_type == "K":
        bucket = bin_k(value)
    elif obs_type == "CREAT":
        bucket = bin_creat(value)
    elif obs_type == "WBC":
        bucket = bin_wbc(value)
    elif obs_type == "HGB":
        bucket = bin_hgb(value)
    else:
        raise ValueError(f"Unsupported obs_type: {obs_type}")

    return obs_type, bucket
