import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


def load_data(features_path: Path, labels_path: Path) -> Tuple[sparse.csr_matrix, np.ndarray, dict]:
    X = sparse.load_npz(features_path)
    label_bundle = np.load(labels_path, allow_pickle=True)
    y = label_bundle["y"]
    meta = {
        "encounter": label_bundle["encounter"],
        "patient": label_bundle["patient"],
        "median_hours": float(label_bundle["median_hours"]),
        "classes": label_bundle["classes"].tolist(),
    }
    return X, y, meta


def train_baseline(
    X: sparse.csr_matrix,
    y: np.ndarray,
    seed: int = 13,
    test_size: float = 0.2,
):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    model = LogisticRegression(
        penalty="l2",
        solver="saga",
        max_iter=10000,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    def evaluate(split_name: str, X_split, y_split):
        probs = model.predict_proba(X_split)[:, 1]
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(y_split, preds)
        f1 = f1_score(y_split, preds, zero_division=0)
        try:
            auc = roc_auc_score(y_split, probs)
        except ValueError:
            auc = float("nan")
        print(f"{split_name}: acc={acc:.3f} | f1={f1:.3f} | auc={auc:.3f}")

    evaluate("train", X_train, y_train)
    evaluate("val", X_val, y_val)
    return model


def main():
    parser = argparse.ArgumentParser(description="Baseline logistic regression for LOS classification.")
    parser.add_argument(
        "--processed_dir",
        type=Path,
        default=Path("data") / "processed",
        help="Directory containing los_features.npz and los_labels.npz",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed for split and model.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Validation split fraction.")
    args = parser.parse_args()

    features_path = args.processed_dir / "los_features.npz"
    labels_path = args.processed_dir / "los_labels.npz"

    X, y, meta = load_data(features_path, labels_path)
    print(f"loaded features {X.shape} | positives={int(y.sum())}/{len(y)}")
    print(f"median stay hours used for label: {meta['median_hours']:.2f}")

    model = train_baseline(X, y, seed=args.seed, test_size=args.test_size)

    model_path = args.processed_dir / "los_logreg.npz"
    sparse.save_npz(model_path, sparse.csr_matrix(model.coef_))
    with open(args.processed_dir / "los_logreg_meta.json", "w") as f:
        json.dump(
            {
                "intercept": model.intercept_.tolist(),
                "classes": meta["classes"],
                "seed": args.seed,
                "test_size": args.test_size,
            },
            f,
            indent=2,
        )
    print(f"saved logistic regression weights to {model_path} and metadata to los_logreg_meta.json")


if __name__ == "__main__":
    main()

