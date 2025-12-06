import json
from pathlib import Path
from typing import Dict, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_curve,
)
import matplotlib.pyplot as plt

SliceFn = Callable[[pd.DataFrame], pd.Series]


def compute_global_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Optional[float]]:
    """
    Compute basic classification and ranking metrics.
    Returns a dict with keys:
      - tn, fp, fn, tp
      - precision
      - recall
      - f1
      - specificity
      - accuracy
      - roc_auc
      - pr_auc
      - brier
    If a metric cannot be computed (for example only one class present),
    set it to None instead of raising.
    """
    y_pred = (y_score >= threshold).astype(int)

    tn = fp = fn = tp = None
    try:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
    except Exception:
        pass

    def safe_metric(fn_metric):
        try:
            return fn_metric()
        except Exception:
            return None

    precision = safe_metric(lambda: precision_score(y_true, y_pred, zero_division=0))
    recall = safe_metric(lambda: recall_score(y_true, y_pred, zero_division=0))
    f1 = safe_metric(lambda: f1_score(y_true, y_pred, zero_division=0))
    accuracy = safe_metric(lambda: accuracy_score(y_true, y_pred))
    specificity = None
    if tn is not None and fp is not None and (tn + fp) > 0:
        specificity = tn / (tn + fp)

    roc_auc = safe_metric(lambda: roc_auc_score(y_true, y_score))
    pr_auc = safe_metric(lambda: average_precision_score(y_true, y_score))
    brier = safe_metric(lambda: brier_score_loss(y_true, y_score))

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": brier,
    }


def print_global_metrics(metrics: Dict[str, Optional[float]]) -> None:
    """
    Print all metrics in a readable way with 3 decimal places where possible.
    """
    for key, val in metrics.items():
        if val is None:
            formatted = "n/a"
        elif isinstance(val, (float, int)):
            formatted = f"{val:.3f}"
        else:
            formatted = str(val)
        print(f"{key}: {formatted}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    outpath: str,
) -> None:
    """
    Plot ROC curve and save it to outpath (PNG).
    Use sklearn.metrics.roc_curve and roc_auc_score.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = roc_auc_score(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc_val:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    outpath: str,
) -> None:
    """
    Plot precision recall curve and save it to outpath (PNG).
    Use sklearn.metrics.precision_recall_curve and average_precision_score.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure()
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def compute_reliability_bins(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Bin predictions into n_bins equal width bins between 0 and 1.
    Returns a DataFrame with columns:
      - bin_lower
      - bin_upper
      - n
      - avg_pred
      - avg_true
    Skip bins with n == 0.
    """
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper == 1.0:
            mask = (y_score >= lower) & (y_score <= upper)
        else:
            mask = (y_score >= lower) & (y_score < upper)
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bin_lower": lower,
                "bin_upper": upper,
                "n": int(mask.sum()),
                "avg_pred": float(y_score[mask].mean()),
                "avg_true": float(y_true[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_reliability_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    outpath: str,
    n_bins: int = 10,
) -> None:
    """
    Use compute_reliability_bins and plot:
      - x axis: avg_pred
      - y axis: avg_true
      - a y = x reference line
    Save plot to outpath (PNG).
    """
    df = compute_reliability_bins(y_true, y_score, n_bins=n_bins)
    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    if not df.empty:
        plt.plot(df["avg_pred"], df["avg_true"], marker="o", label="Empirical")
    plt.xlabel("Avg Predicted Probability")
    plt.ylabel("Avg Observed Frequency")
    plt.title("Reliability Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def default_slices(metadata: pd.DataFrame) -> Dict[str, SliceFn]:
    """
    Build a set of simple default slice functions based on common columns.
    Only include slices when the needed column exists.

    Use these conventions:
      - If "AGE" exists:
          age_lt_40: AGE < 40
          age_40_65: 40 <= AGE <= 65
          age_gt_65: AGE > 65
      - If "SEX" exists:
          sex_male: SEX == "M" or "Male"
          sex_female: SEX == "F" or "Female"
      - If "SEQ_LEN" exists:
          short_history: SEQ_LEN <= 2
          medium_history: 3 <= SEQ_LEN <= 5
          long_history: SEQ_LEN >= 6
    Each slice function should accept the metadata DataFrame and return a boolean Series.
    """
    slices: Dict[str, SliceFn] = {}
    cols = set(metadata.columns)
    if "AGE" in cols:
        slices["age_lt_40"] = lambda df: df["AGE"] < 40
        slices["age_40_65"] = lambda df: (df["AGE"] >= 40) & (df["AGE"] <= 65)
        slices["age_gt_65"] = lambda df: df["AGE"] > 65
    if "SEX" in cols:
        slices["sex_male"] = lambda df: df["SEX"].astype(str).str.upper().isin(["M", "MALE"])
        slices["sex_female"] = lambda df: df["SEX"].astype(str).str.upper().isin(["F", "FEMALE"])
    if "SEQ_LEN" in cols:
        slices["short_history"] = lambda df: df["SEQ_LEN"] <= 2
        slices["medium_history"] = lambda df: (df["SEQ_LEN"] >= 3) & (df["SEQ_LEN"] <= 5)
        slices["long_history"] = lambda df: df["SEQ_LEN"] >= 6
    return slices


def evaluate_slices(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metadata: Optional[pd.DataFrame],
    slice_fns: Dict[str, SliceFn],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    For each slice, compute the same summary metrics as compute_global_metrics,
    but restricted to the rows where the slice mask is True.

    Returns a DataFrame where each row is a slice with columns:
      - slice_name
      - n
      - precision
      - recall
      - f1
      - specificity
      - accuracy
      - roc_auc
      - pr_auc
      - brier
    Skip slices with n == 0 or 1.
    If metadata is None, return an empty DataFrame.
    """
    if metadata is None:
        return pd.DataFrame()
    rows = []
    for name, fn_slice in slice_fns.items():
        try:
            mask = fn_slice(metadata)
        except Exception:
            continue
        if mask is None or len(mask) == 0:
            continue
        idx = np.where(mask.values if isinstance(mask, pd.Series) else mask)[0]
        if len(idx) <= 1:
            continue
        yt = y_true[idx]
        ys = y_score[idx]
        metrics = compute_global_metrics(yt, ys, threshold=threshold)
        rows.append(
            {
                "slice_name": name,
                "n": len(idx),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "specificity": metrics.get("specificity"),
                "accuracy": metrics.get("accuracy"),
                "roc_auc": metrics.get("roc_auc"),
                "pr_auc": metrics.get("pr_auc"),
                "brier": metrics.get("brier"),
            }
        )
    return pd.DataFrame(rows)


def top_k_errors(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metadata: Optional[pd.DataFrame] = None,
    k: int = 20,
    threshold: float = 0.5,
    kind: str = "fp",
) -> pd.DataFrame:
    """
    Return a DataFrame with up to k most confident errors of the given kind.

    For kind = "fp": y_true == 0 and y_pred == 1, sorted by y_score descending.
    For kind = "fn": y_true == 1 and y_pred == 0, sorted by y_score ascending.

    The DataFrame should include:
      - index: original index in the input arrays
      - y_true
      - y_score
      - y_pred
      - error_type
    If metadata is provided, include its columns too for the selected rows.
    """
    y_pred = (y_score >= threshold).astype(int)
    if kind == "fp":
        mask = (y_true == 0) & (y_pred == 1)
        order = np.argsort(-y_score)
        error_type = "fp"
    else:
        mask = (y_true == 1) & (y_pred == 0)
        order = np.argsort(y_score)
        error_type = "fn"
    idx_errors = np.where(mask)[0]
    ordered = [i for i in order if i in idx_errors][:k]
    data = {
        "index": ordered,
        "y_true": y_true[ordered],
        "y_score": y_score[ordered],
        "y_pred": y_pred[ordered],
        "error_type": [error_type] * len(ordered),
    }
    df_errors = pd.DataFrame(data)
    if metadata is not None and len(df_errors) > 0:
        meta_subset = metadata.iloc[ordered]
        df_errors = pd.concat([df_errors.reset_index(drop=True), meta_subset.reset_index(drop=True)], axis=1)
    return df_errors


def compare_models_global(
    y_true_a: np.ndarray,
    y_score_a: np.ndarray,
    y_true_b: np.ndarray,
    y_score_b: np.ndarray,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Compare two models (A and B) using compute_global_metrics.
    Assume y_true_a and y_true_b are identical, but do not rely on that too much.

    Return a DataFrame with two rows:
      - model: "A" or "B"
      - the metrics from compute_global_metrics as columns
    """
    metrics_a = compute_global_metrics(y_true_a, y_score_a, threshold=threshold)
    metrics_b = compute_global_metrics(y_true_b, y_score_b, threshold=threshold)
    rows = []
    for name, metrics in (("A", metrics_a), ("B", metrics_b)):
        row = {"model": name}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def compare_models_slices(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
    metadata: Optional[pd.DataFrame],
    slice_fns: Dict[str, SliceFn],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    For each slice, compute metrics for model A and B and return a DataFrame with:
      - slice_name
      - n
      - roc_auc_a, roc_auc_b, delta_roc_auc
      - f1_a, f1_b, delta_f1
      - accuracy_a, accuracy_b, delta_accuracy
    Skip slices with too few samples.
    """
    if metadata is None:
        return pd.DataFrame()
    rows = []
    for name, fn_slice in slice_fns.items():
        try:
            mask = fn_slice(metadata)
        except Exception:
            continue
        if mask is None or len(mask) == 0:
            continue
        idx = np.where(mask.values if isinstance(mask, pd.Series) else mask)[0]
        if len(idx) <= 1:
            continue
        yt = y_true[idx]
        ys_a = y_score_a[idx]
        ys_b = y_score_b[idx]
        m_a = compute_global_metrics(yt, ys_a, threshold=threshold)
        m_b = compute_global_metrics(yt, ys_b, threshold=threshold)

        def delta(a, b):
            if a is None or b is None:
                return None
            return b - a

        rows.append(
            {
                "slice_name": name,
                "n": len(idx),
                "roc_auc_a": m_a.get("roc_auc"),
                "roc_auc_b": m_b.get("roc_auc"),
                "delta_roc_auc": delta(m_a.get("roc_auc"), m_b.get("roc_auc")),
                "f1_a": m_a.get("f1"),
                "f1_b": m_b.get("f1"),
                "delta_f1": delta(m_a.get("f1"), m_b.get("f1")),
                "accuracy_a": m_a.get("accuracy"),
                "accuracy_b": m_b.get("accuracy"),
                "delta_accuracy": delta(m_a.get("accuracy"), m_b.get("accuracy")),
            }
        )
    return pd.DataFrame(rows)


def run_full_analysis(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metadata: Optional[pd.DataFrame],
    outdir: str,
    threshold: float = 0.5,
) -> None:
    """
    High level helper that:
      1. Creates outdir if it does not exist.
      2. Computes global metrics and saves them to global_metrics.json.
      3. Saves ROC curve to roc.png.
      4. Saves PR curve to pr.png.
      5. Computes reliability bins and saves to reliability_bins.csv.
      6. Saves reliability plot to reliability.png.
      7. If metadata is not None:
            - builds default slices
            - evaluates them and saves to slices.csv
            - saves top_k_errors for FP and FN to top_fp.csv and top_fn.csv.
    """
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    metrics = compute_global_metrics(y_true, y_score, threshold=threshold)
    with open(out_path / "global_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_roc_curve(y_true, y_score, out_path / "roc.png")
    plot_pr_curve(y_true, y_score, out_path / "pr.png")

    bins_df = compute_reliability_bins(y_true, y_score)
    bins_df.to_csv(out_path / "reliability_bins.csv", index=False)
    plot_reliability_curve(y_true, y_score, out_path / "reliability.png")

    if metadata is not None:
        slices = default_slices(metadata)
        slice_results = evaluate_slices(y_true, y_score, metadata, slices, threshold=threshold)
        slice_results.to_csv(out_path / "slices.csv", index=False)
        top_k_errors(y_true, y_score, metadata, k=20, threshold=threshold, kind="fp").to_csv(
            out_path / "top_fp.csv", index=False
        )
        top_k_errors(y_true, y_score, metadata, k=20, threshold=threshold, kind="fn").to_csv(
            out_path / "top_fn.csv", index=False
        )
