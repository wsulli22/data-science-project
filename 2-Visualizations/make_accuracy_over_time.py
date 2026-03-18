#!/usr/bin/env python3
"""
make_accuracy_over_time.py

Implements README TODO [1]:
  Bar chart of |empirical_win_rate_pct - avg_kalshi_quoted_prob_pct|
  across 1-minute game-time buckets.
"""

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


def _resolve_input_path(input_file: str) -> str:
    """Resolve relative paths relative to this script file."""
    script_dir = os.path.dirname(__file__)
    if os.path.isabs(input_file):
        return input_file
    return os.path.normpath(os.path.join(script_dir, input_file))


def generate_accuracy_over_time(
    input_file: str = "../GatheringPreprocessingTransformation/GeneratedDataFiles/all_games_merged_clean.csv",
    num_time_bins: int = 40,
    min_obs_per_bucket: int = 5,
    output_filename: str = "accuracy_over_time.png",
):
    """
    Args:
        input_file: Path to `all_games_merged_clean.csv`.
        num_time_bins: Number of 1-minute buckets (40 for 0-40 min regulation).
        min_obs_per_bucket: Buckets with fewer observations are left blank (NaN).
        output_filename: Output PNG filename.
    """
    print("\nGENERATING ACCURACY OVER TIME\n")
    FILE = _resolve_input_path(input_file)

    script_dir = os.path.dirname(__file__)
    out_dir = os.path.join(script_dir, "GeneratedDataAndVisualizations")
    os.makedirs(out_dir, exist_ok=True)

    # ── configuration / binning ───────────────────────────────────────────
    bin_size_seconds = 2400 / num_time_bins
    time_edges = np.arange(0, 2401, bin_size_seconds)  # inclusive end due to arange behaviour
    time_labels = [
        f"{int(lo / 60)}-{int(hi / 60)} min" for lo, hi in zip(time_edges[:-1], time_edges[1:])
    ]

    # ── load ──────────────────────────────────────────────────────────────
    df = pd.read_csv(FILE)
    df = df.dropna(subset=["game_elapsed_seconds", "win_prob_pct", "team_won"])

    # Bin game time
    df["time_bin"] = pd.cut(
        df["game_elapsed_seconds"],
        bins=time_edges,
        labels=time_labels,
        right=False,
        include_lowest=True,
    )
    df = df.dropna(subset=["time_bin"])

    # Round quoted prob to integer percentage points (matches heatmap y-rows)
    df["prob_int"] = df["win_prob_pct"].round(0).astype(int)
    df = df[df["prob_int"].between(1, 99)]

    # ── aggregate ─────────────────────────────────────────────────────────
    grouped = (
        df.groupby("time_bin", observed=False)
        .agg(
            empirical_win_rate=("team_won", "mean"),
            kalshi_avg_prob_pct=("prob_int", "mean"),
            n=("team_won", "count"),
        )
        .reset_index()
    )

    grouped["empirical_win_rate_pct"] = grouped["empirical_win_rate"] * 100.0
    grouped["accuracy_error_pct_points"] = (
        grouped["empirical_win_rate_pct"] - grouped["kalshi_avg_prob_pct"]
    ).abs()

    grouped.loc[grouped["n"] < min_obs_per_bucket, "accuracy_error_pct_points"] = np.nan

    # Preserve the time bucket order from `time_labels`
    grouped = (
        grouped.set_index("time_bin")
        .reindex(time_labels)
        .reset_index()
        .rename(columns={"index": "time_bin"})
    )

    # ── plot ──────────────────────────────────────────────────────────────
    x = np.arange(len(time_labels))
    y = grouped["accuracy_error_pct_points"].to_numpy()

    fig, ax = plt.subplots(figsize=(16, 6))

    for i, val in enumerate(y):
        if np.isnan(val):
            continue
        ax.bar(i, float(val), color="steelblue", edgecolor="black", linewidth=0.3)

    label_step = max(1, int(round(len(time_labels) / 10)))
    xticks = x[::label_step]
    xticklabels = [time_labels[i] for i in range(0, len(time_labels), label_step)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=9, rotation=0)

    ax.set_xlabel("Game Time (minutes elapsed)")
    ax.set_ylabel("|Empirical win rate - avg Kalshi quoted prob| (percentage points)")

    n_games = df["kalshi_event"].nunique() if "kalshi_event" in df.columns else "?"
    total_obs = len(df)
    ax.set_title(
        "Calibration Accuracy Over Time (Absolute Error)\n"
        f"({n_games} games · {total_obs:,} observations · {num_time_bins} time bins · masked n<{min_obs_per_bucket})"
    )

    ax.set_ylim(bottom=0)
    plt.tight_layout()

    out_path = os.path.join(out_dir, output_filename)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved {output_filename} -> {out_path}")
    return grouped


if __name__ == "__main__":
    generate_accuracy_over_time()

