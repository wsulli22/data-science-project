#!/usr/bin/env python3
"""
make_games_per_day.py

Plots number of games per calendar day.

Definition:
  For each `kalshi_event` (game), compute its earliest `wallclock_ts`.
  That earliest timestamp is bucketed to a calendar day (UTC-ish; CSV is treated as-is).
  Count unique games per day.
"""

import os

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


def _resolve_input_path(input_file: str) -> str:
    """Resolve relative paths relative to this script file."""
    script_dir = os.path.dirname(__file__)
    if os.path.isabs(input_file):
        return input_file
    return os.path.normpath(os.path.join(script_dir, input_file))


def generate_games_per_day(
    input_file: str = "../1-GatheringPreprocessingTransformation/GeneratedDataFiles/all_games_merged_clean.csv",
    output_filename: str = "games_per_day.png",
):
    """
    Args:
        input_file: Path to `all_games_merged_clean.csv`.
        output_filename: Output PNG filename.
    """
    print("\nGENERATING GAMES PER DAY\n")
    FILE = _resolve_input_path(input_file)

    script_dir = os.path.dirname(__file__)
    out_dir = os.path.join(script_dir, "GeneratedDataAndVisualizations")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(FILE)
    df = df.dropna(subset=["kalshi_event", "wallclock_ts"])

    df["wallclock_ts"] = pd.to_datetime(df["wallclock_ts"], errors="coerce")
    df = df.dropna(subset=["wallclock_ts"])

    # Per game: earliest timestamp -> day
    game_start = df.groupby("kalshi_event", observed=False)["wallclock_ts"].min()
    game_day = game_start.dt.floor("D")

    counts = game_day.value_counts().sort_index()
    if counts.empty:
        raise ValueError("No games found after filtering; cannot plot games per day.")

    start_day = counts.index.min()
    end_day = counts.index.max()

    full_days = pd.date_range(start=start_day, end=end_day, freq="D")
    counts_full = counts.reindex(full_days, fill_value=0)

    # ── plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 6))

    # Use integer x positions so we always draw one "slot" per calendar day.
    # (Matplotlib bar patches with height==0 can be visually ambiguous, so we
    # also mark y=0 days with small x's for clarity.)
    x_idx = np.arange(len(counts_full), dtype=int)
    y = counts_full.to_numpy(dtype=float)

    colors = np.where(y == 0, "#d9d9d9", "steelblue")
    ax.bar(x_idx, y, width=0.9, color=colors, edgecolor="black", linewidth=0.2)
    zero_mask = y == 0
    if np.any(zero_mask):
        ax.scatter(
            x_idx[zero_mask],
            y[zero_mask],
            marker="x",
            color="#555555",
            s=35,
            zorder=3,
        )
    ax.set_ylabel("Number of Games")
    ax.set_xlabel("Day")

    ax.axhline(0, color="black", linewidth=1, alpha=0.7)

    # Show a subset of date labels for readability.
    tick_step = max(1, int(round(len(counts_full) / 10)))
    tick_positions = x_idx[::tick_step]
    tick_labels = [d.strftime("%Y-%m-%d") for d in counts_full.index[::tick_step]]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    ax.set_title(f"Games per Day ({start_day.date()} to {end_day.date()})")

    plt.tight_layout()
    out_path = os.path.join(out_dir, output_filename)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved {output_filename} -> {out_path}")
    return counts_full


if __name__ == "__main__":
    generate_games_per_day()

