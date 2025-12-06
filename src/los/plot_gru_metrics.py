import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(path: Path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["metrics"], data.get("config", {}), data.get("uses_pretrained", False)


def plot_metric(metric_name: str, baseline_vals, strong_vals, out_path: Path):
    plt.figure()
    plt.plot(range(1, len(baseline_vals) + 1), baseline_vals, label="baseline (one-hot/random emb)", marker="o")
    plt.plot(range(1, len(strong_vals) + 1), strong_vals, label="pretrained embeddings", marker="o")
    plt.xlabel("epoch")
    plt.ylabel(metric_name)
    plt.title(f"GRU {metric_name}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot GRU metrics comparing baseline and pretrained-embedding runs.")
    parser.add_argument("--baseline_metrics", type=Path, required=True, help="Metrics JSON from baseline GRU run (one-hot/random embeddings).")
    parser.add_argument("--pretrained_metrics", type=Path, required=True, help="Metrics JSON from pretrained GRU run.")
    parser.add_argument("--out_dir", type=Path, default=Path("data") / "processed", help="Directory to write plots.")
    args = parser.parse_args()

    base_metrics, _, _ = load_metrics(args.baseline_metrics)
    strong_metrics, _, _ = load_metrics(args.pretrained_metrics)

    metrics_to_plot = ["train_loss", "val_loss", "val_acc", "val_f1", "val_auc"]
    for name in metrics_to_plot:
        if name not in base_metrics or name not in strong_metrics:
            continue
        out_path = args.out_dir / f"gru_{name}.png"
        plot_metric(name, base_metrics[name], strong_metrics[name], out_path)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
