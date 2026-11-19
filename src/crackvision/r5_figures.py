"""Render R5 paired group differences and factorial seed-level results."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def paired_plot(path, out):
    data = json.loads(Path(path).read_text(encoding="utf-8"))["paired_contrasts"]
    fig, ax = plt.subplots(figsize=(8, 2.85), layout="constrained")
    labels = ["Skip gate − ungated Swin", "Skip gate − CNN U-Net"]
    ax.axvline(0, color="#444444", lw=1, ls="--")
    for i, (label, color) in enumerate((("f1", "#126b79"), ("miou", "#a94826"))):
        for j, item in enumerate(data.values()):
            lo, hi = item["intervals"][label]
            center = item["observed"][label]
            y = j + (i - .5) * .22
            ax.hlines(y, lo, hi, lw=2, color=color)
            ax.scatter(center, y, s=28, color=color,
                       label=("F1" if label == "f1" else "mIoU") if j == 0 else None)
    ax.set(yticks=[0, 1], yticklabels=labels, ylim=(1.45, -.45),
           xlabel="Paired metric difference", title="Archived DeepCrack group-bootstrap intervals")
    ax.legend(loc="lower left", bbox_to_anchor=(1.0, -.03), frameon=False)
    fig.savefig(out, dpi=240)
    plt.close(fig)


def factorial_plot(csv_path, summary_path, out):
    with Path(csv_path).open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    colors = ["#35598c", "#ad6231"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), layout="constrained")
    for ax, metric, title in zip(axes, ("f1", "miou"), ("Foreground F1", "Two-class mean IoU")):
        for i, condition in enumerate("ABCDEF"):
            values = [float(row[metric]) for row in rows if row["condition"] == condition]
            result = summary["conditions"][condition][metric]
            lo, hi = result["seed_t_95ci"]
            color = colors[i // 3]
            ax.scatter(i + np.linspace(-.13, .13, len(values)), values,
                       s=17, color=color, alpha=.62, zorder=2)
            ax.vlines(i, lo, hi, color=color, lw=1.5, zorder=1)
            ax.scatter(i, result["mean"], marker="D", s=24, color="#151f2d", zorder=3)
        ax.axvline(2.5, color="#8b8b8b", lw=.75)
        ax.set_xticks(range(6), ["BCE", "Dice", "Hybrid"] * 2)
        ax.set(title=title, xlabel="Ungated Swin                 Decoder gate")
        ax.grid(axis="y", color="#dddddd", lw=.65)
        ax.set_axisbelow(True)
    fig.savefig(out, dpi=240)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archived", required=True)
    p.add_argument("--runs", type=Path)
    p.add_argument("--out", type=Path, required=True)
    options = p.parse_args()
    options.out.mkdir(parents=True, exist_ok=True)
    paired_plot(options.archived, options.out / "r5_archived_paired.png")
    if options.runs:
        factorial_plot(options.runs / "runs.csv", options.runs / "summary.json",
                       options.out / "r5_factorial.png")


if __name__ == "__main__":
    main()
