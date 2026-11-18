import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


GROUPS = ["context", "context_surface", "context_history", "context_history_surface"]
LABELS = ["C", "C+S", "C+H", "C+H+S"]
STYLES = ["-", "--", "-.", ":"]


def save(fig, output, name):
    fig.savefig(output / (name + ".png"), dpi=360, facecolor="white")
    fig.savefig(output / (name + ".pdf"), facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = json.loads((args.run_dir / "data_audit.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.labelsize": 11,
                         "axes.titlesize": 11, "legend.fontsize": 10.5, "axes.edgecolor": "black",
                         "text.color": "black", "axes.labelcolor": "black", "savefig.bbox": "tight"})
    curves = []
    for group in GROUPS:
        files = sorted((args.run_dir / "predictions").glob("repeat_*_logistic_" + group + ".csv"))
        if len(files) != 5:
            raise RuntimeError("Expected five logistic prediction files for " + group)
        combined = pd.concat([pd.read_csv(path) for path in files])
        if combined.groupby("row_id").injury.nunique().max() != 1:
            raise RuntimeError("Inconsistent outcome labels")
        average = combined.groupby("row_id", sort=True).agg(injury=("injury", "first"), probability=("probability", "mean"))
        if len(average) != audit["development_rows"]:
            raise RuntimeError("Prediction row count does not match audit")
        curves.append(average)
    curve_scores = []
    for kind in ["roc", "precision_recall"]:
        fig, ax = plt.subplots(figsize=(3.55, 3.35), layout="constrained")
        for label, style, frame in zip(LABELS, STYLES, curves):
            y, score = frame.injury, frame.probability
            ap = average_precision_score(y, score)
            auc = roc_auc_score(y, score)
            if kind == "roc":
                x, z, _ = roc_curve(y, score)
                ax.plot(x, z, style, color="black", lw=1.4, label=f"{label} ({auc:.3f})")
                curve_scores.append({"feature_set": label, "average_precision": ap, "roc_auc": auc})
            else:
                z, x, _ = precision_recall_curve(y, score)
                ax.step(x, z, where="post", linestyle=style, color="black", lw=1.2, label=f"{label} ({ap:.4f})")
        if kind == "roc":
            ax.plot([0, 1], [0, 1], color="black", lw=0.7, linestyle=(0, (1, 5)))
            ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1))
            ax.legend(title="Inputs (ROC-AUC)", loc="lower right", frameon=False, title_fontsize=10.5)
        else:
            ax.axhline(audit["prevalence"], color="black", lw=0.7, linestyle=(0, (1, 5)))
            ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 0.22))
            ax.legend(title="Inputs (AP)", loc="upper right", frameon=False, title_fontsize=10.5)
        save(fig, args.output_dir, kind)
    pd.DataFrame(curve_scores).to_csv(args.output_dir / "curve_metrics.csv", index=False)
    boot = pd.read_csv(args.run_dir / "player_bootstrap_differences.csv")
    rows = boot[boot.metric.eq("average_precision")].copy()
    comparisons = {"surface_without_history": "S / no H", "surface_with_history": "S / with H",
                   "history_without_surface": "H / no S", "history_with_surface": "H / with S"}
    models = {"logistic": "LR", "random_forest": "RF", "xgboost": "XGB"}
    fig, ax = plt.subplots(figsize=(3.55, 3.3), layout="constrained")
    for index, row in enumerate(rows.itertuples()):
        ax.plot([row.lower_95, row.upper_95], [index, index], color="black", lw=1)
        ax.plot(row.observed_difference, index, "o", color="black", ms=3)
    ax.axvline(0, color="black", linestyle="--", lw=0.8)
    ax.set_yticks(np.arange(len(rows)), [models[r.model] + ": " + comparisons[r.comparison] for r in rows.itertuples()])
    ax.invert_yaxis()
    ax.set_xlabel("AP change with added features")
    ax.tick_params(axis="y", length=0)
    save(fig, args.output_dir, "paired_ap")
    fig, ax = plt.subplots(figsize=(3.55, 3.5), layout="constrained")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    boxes = [(0.90, "Competition data\n250 players | 5,712 player-games\n104 injury games"),
             (0.63, "Development only\n225 players | 5,192 player-games\n93 injury games"),
             (0.29, "Primary timing-audited cohort\n209 players | 4,776 player-games\n90 injury games")]
    for y, text in boxes:
        ax.text(0.5, y, text, ha="center", va="center", fontsize=11,
                bbox={"boxstyle": "square,pad=0.35", "facecolor": "white", "edgecolor": "black"})
    ax.plot([0.5, 0.5], [0.80, 0.74], color="black", lw=1)
    ax.plot([0.5, 0.5], [0.52, 0.40], color="black", lw=1)
    ax.text(0.5, 0.04, "Reserved validation: 25 players, 520 games\nTiming exclusions: 16 players, 416 games", ha="center", va="center", fontsize=10.5)
    save(fig, args.output_dir, "cohort_flow")
    print("saved=" + str(args.output_dir))


if __name__ == "__main__":
    main()
