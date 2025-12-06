import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


def load_instances(instances_path: Path) -> List[dict]:
    records = []
    with open(instances_path, "r") as f:
        for line in f:
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"no instances found at {instances_path}")
    return records


def load_labels(labels_path: Path) -> Tuple[np.ndarray, dict]:
    label_bundle = np.load(labels_path, allow_pickle=True)
    y = label_bundle["y"]
    meta = {
        "encounter": label_bundle["encounter"],
        "patient": label_bundle["patient"],
        "median_hours": float(label_bundle["median_hours"]),
        "classes": label_bundle["classes"].tolist(),
    }
    return y, meta


def load_vocab(vocab_path: Path) -> dict:
    with open(vocab_path, "r") as f:
        return json.load(f)


def build_dense_features(records: List[dict], vocab: dict, embeddings: torch.Tensor) -> np.ndarray:
    """
    Build features that keep per-token counts (no truncation) AND include
    a pooled embedding summary. For each encounter:
      - counts: vocab_size vector with raw token counts (preserves identity)
      - emb_sum: weighted sum of embeddings by count (dim)
    Final feature = concat[counts, emb_sum] -> shape vocab_size + dim.
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings tensor must be 2D (vocab_size x dim).")
    vocab_size, dim = embeddings.shape
    if len(vocab) != vocab_size:
        raise ValueError(f"vocab size {len(vocab)} does not match embeddings {vocab_size}.")

    vectors = []
    for rec in records:
        counts = torch.zeros(vocab_size, dtype=torch.float32)
        for tok in rec["tokens"]:
            idx = vocab.get(tok)
            if idx is not None:
                counts[idx] += 1.0
        if counts.sum() == 0:
            raise ValueError(f"encounter {rec.get('encounter')} has no tokens in vocab.")
        emb_sum = counts @ embeddings  # (dim,)
        feat = torch.cat([counts, emb_sum], dim=0)
        vectors.append(feat)
    return torch.stack(vectors).numpy()


def train_baseline(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 13,
    test_size: float = 0.2,
):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    model = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=5000,
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
    parser.add_argument(
        "--embeddings_path",
        type=Path,
        default=Path("word_embeddings.pt"),
        help="Path to pretrained embeddings tensor (vocab_size x dim).",
    )
    args = parser.parse_args()

    labels_path = args.processed_dir / "los_labels.npz"
    instances_path = args.processed_dir / "los_instances.jsonl"
    vocab_path = args.processed_dir / "vocab.json"

    records = load_instances(instances_path)
    y, meta = load_labels(labels_path)
    vocab = load_vocab(vocab_path)
    embeddings = torch.load(args.embeddings_path, map_location="cpu")
    X = build_dense_features(records, vocab, embeddings)

    if X.shape[0] != len(y):
        raise ValueError(f"feature rows ({X.shape[0]}) and labels ({len(y)}) mismatch.")

    print(f"loaded dense features {X.shape} | positives={int(y.sum())}/{len(y)}")
    print(f"median stay hours used for label: {meta['median_hours']:.2f}")

    model = train_baseline(X, y, seed=args.seed, test_size=args.test_size)

    model_path = args.processed_dir / "los_logreg_dense.npz"
    np.savez_compressed(model_path, coef_=model.coef_, intercept_=model.intercept_)
    with open(args.processed_dir / "los_logreg_meta.json", "w") as f:
        json.dump(
            {
                "classes": meta["classes"],
                "seed": args.seed,
                "test_size": args.test_size,
                "embeddings_path": str(args.embeddings_path),
            },
            f,
            indent=2,
        )
    print(f"saved logistic regression weights to {model_path} and metadata to los_logreg_meta.json")

    # Save predictions for downstream error analysis (full dataset, probability of class 1).
    try:
        y_score = model.predict_proba(X)[:, 1]
        preds_path = args.processed_dir / "los_logreg_preds.npz"
        np.savez_compressed(
            preds_path,
            y_true=y,
            y_score=y_score,
            encounter=meta.get("encounter"),
            patient=meta.get("patient"),
        )
        print(f"saved prediction bundle for error analysis to {preds_path}")
    except Exception as e:
        print(f"warning: could not save prediction bundle: {e}")


if __name__ == "__main__":
    main()
