#!/usr/bin/env python3
"""
game_durations.py

Graphs the duration of each game using:
  duration = end_wallclock_ts - start_wallclock_ts

How the game boundaries are computed:
  - For each `kalshi_event` (one game), take the min/max of `wallclock_ts`.
  - This file likely contains multiple rows per game (one row per timestamp),
    so we aggregate down to one row per `kalshi_event` for plotting.
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


def generate_game_durations(
    input_file: str = "../1-GatheringPreprocessingTransformation/GeneratedDataFiles/all_games_merged_clean.csv",
    output_filename: str = "game_durations.png",
    duration_unit: str = "minutes",
) -> pd.DataFrame:
    """
    Args:
        input_file: Path to `all_games_merged_clean.csv`.
        output_filename: Output PNG filename.
        duration_unit: 'minutes' or 'seconds' (controls plot y-axis units).

    Returns:
        DataFrame with one row per `kalshi_event` and computed duration.
    """
    if duration_unit not in {"minutes", "seconds"}:
        raise ValueError("duration_unit must be one of: {'minutes', 'seconds'}")

    print("\nGENERATING GAME DURATIONS\n")
    FILE = _resolve_input_path(input_file)

    script_dir = os.path.dirname(__file__)
    out_dir = os.path.join(script_dir, "GeneratedDataAndVisualizations")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(FILE)
    required_cols = {"kalshi_event", "wallclock_ts"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    df = df.dropna(subset=["kalshi_event", "wallclock_ts"]).copy()
    df["wallclock_ts"] = pd.to_datetime(df["wallclock_ts"], errors="coerce")
    df = df.dropna(subset=["wallclock_ts"])

    # Aggregate to one row per game.
    games = (
        df.groupby("kalshi_event", observed=False)
        .agg(start_ts=("wallclock_ts", "min"), end_ts=("wallclock_ts", "max"))
        .reset_index()
    )
    games["duration_seconds"] = (games["end_ts"] - games["start_ts"]).dt.total_seconds()
    games = games.dropna(subset=["duration_seconds"])

    # Filter out any unexpected bad rows (shouldn't happen, but keeps plots sane).
    games = games[games["duration_seconds"] > 0].copy()

    games = games.sort_values("start_ts").reset_index(drop=True)
    games["game_idx"] = np.arange(1, len(games) + 1, dtype=int)

    if duration_unit == "minutes":
        games["duration"] = games["duration_seconds"] / 60.0
        y_label = "Duration (minutes)"
    else:
        games["duration"] = games["duration_seconds"]
        y_label = "Duration (seconds)"

    # Stats for annotations.
    dur = games["duration"].to_numpy(dtype=float)
    dur_min = np.min(dur)
    dur_med = float(np.median(dur))
    dur_mean = float(np.mean(dur))
    dur_max = np.max(dur)

    # ── plot ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Each point = one game.
    ax1.scatter(
        games["game_idx"],
        games["duration"],
        s=10,
        alpha=0.55,
        color="#1f77b4",
        linewidths=0,
    )
    ax1.axhline(dur_mean, color="orange", linestyle="--", linewidth=1.2, label=f"Mean: {dur_mean:.1f}")
    ax1.axhline(dur_med, color="green", linestyle="--", linewidth=1.2, label=f"Median: {dur_med:.1f}")
    ax1.set_xlabel("Game rank (by start time)")
    ax1.set_ylabel(y_label)
    ax1.set_title(f"Duration of Each Game ({len(games)} games)")
    ax1.grid(True, alpha=0.25, linewidth=0.8)
    ax1.legend(loc="best", fontsize=9, framealpha=0.9)

    stats_text = (
        f"Min/Median/Mean/Max: {dur_min:.1f} / {dur_med:.1f} / {dur_mean:.1f} / {dur_max:.1f} ({duration_unit})"
    )
    ax1.text(
        0.01,
        0.98,
        stats_text,
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="black", linewidth=0.2),
    )

    # Distribution view.
    n_bins = 40 if duration_unit == "minutes" else 60
    ax2.hist(games["duration"], bins=n_bins, color="steelblue", edgecolor="black", linewidth=0.2)
    ax2.set_xlabel(y_label)
    ax2.set_ylabel("Number of Games")
    ax2.set_title("Duration Distribution")
    ax2.grid(True, alpha=0.2, linewidth=0.8)

    out_path = os.path.join(out_dir, output_filename)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved {output_filename} -> {out_path}")
    return games


if __name__ == "__main__":
    generate_game_durations()

