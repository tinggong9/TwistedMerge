"""Plotting and table helpers for generated experiment CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_latex_table(df: pd.DataFrame, path: Path, float_format: str = "%.3f") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(df.to_latex(index=False, float_format=float_format), encoding="utf-8")


def plot_accuracy_vs_obstruction(csv_path: Path, plot_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.scatter(df["obstruction_score"], df["naive_accuracy"], label="descended merge", alpha=0.75)
    ax.scatter(df["obstruction_score"], df["rank_lift_accuracy"], label="rank-lift merge", alpha=0.75)
    ax.set_xlabel("cocycle obstruction score")
    ax.set_ylabel("mean test accuracy")
    ax.set_title(title)
    ax.set_ylim(0.45, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)


def plot_rank_ablation(csv_path: Path, plot_path: Path) -> None:
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    summary = df.groupby(["experiment", "rank"])["accuracy"].mean().reset_index()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for experiment, part in summary.groupby("experiment"):
        ax.plot(part["rank"], part["accuracy"], marker="o", label=experiment)
    ax.set_xlabel("rank / number of branches")
    ax.set_ylabel("mean test accuracy")
    ax.set_title("Rank-lift ablation")
    ax.set_ylim(0.45, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
