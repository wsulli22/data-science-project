#!/usr/bin/env python3
"""
make_heat_map.py

Reads all_games_merged_clean.csv and produces an aggregated calibration
heat map over (game time, Kalshi win probability).

Grid: Configurable time buckets (NUM_TIME_BINS) x 99 probability rows (1 per pct point)

Each cell's colour shows the empirical win rate -- the fraction of
observations in that cell where the team actually won.  If Kalshi is
perfectly calibrated the colour should track the y-axis value everywhere.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

def generateHeatMap(input_file="GeneratedDataFiles/all_games_merged_clean.csv", num_time_bins=16):
    """
    Generate a calibration heat map from merged game data.

    Args:
        input_file: Path to the CSV file containing merged game data.
                    Defaults to "GeneratedDataFiles/all_games_merged_clean.csv".
        num_time_bins: Number of time bins (0-2400 seconds regulation).
                       Defaults to 16.
    
    Saves heatmap.png to GeneratedDataFiles.
    """
    print("\nGENERATING HEATMAP (make_heat_map.py)\n")
    print(f"  Using input file: {input_file}")
    
    # ── configuration ────────────────────────────────────────────────────
    NUM_TIME_BINS = num_time_bins  # Number of time bins (0-2400 seconds regulation)
    
    FILE = input_file
    generated_data_dir = "GeneratedVisualizations"
    os.makedirs(generated_data_dir, exist_ok=True)

    # Calculate time bin edges based on NUM_TIME_BINS
    bin_size_seconds = 2400 / NUM_TIME_BINS  # Size of each bin in seconds
    TIME_EDGES = np.arange(0, 2401, bin_size_seconds)  # 0 to 2400 seconds
    TIME_LABELS = [f"{int(lo/60)}-{int(hi/60)} min"
                   for lo, hi in zip(TIME_EDGES[:-1], TIME_EDGES[1:])]

    # 1 row per integer win-probability percentage (1 % - 99 %)
    PROB_VALUES = np.arange(1, 100)                             # 1,2,...,99

    # Minimum observations for a cell to be coloured (otherwise shown as grey)
    MIN_OBS = 5

    # ── load ──────────────────────────────────────────────────────────────
    df = pd.read_csv(FILE)

    # ── bin the data ─────────────────────────────────────────────────────
    df["time_bin"] = pd.cut(
        df["game_elapsed_seconds"],
        bins=TIME_EDGES,
        labels=TIME_LABELS,
        right=False,
        include_lowest=True,
    )

    # Round win_prob_pct to nearest integer for the 1-per-pct rows
    # Drop rows with NaN win_prob_pct before converting to int
    df = df.dropna(subset=["win_prob_pct"])
    df["prob_int"] = df["win_prob_pct"].round(0).astype(int)
    df = df[df["prob_int"].between(1, 99)]                      # safety clip

    # ── aggregate ────────────────────────────────────────────────────────
    grouped = (
        df.groupby(["prob_int", "time_bin"], observed=False)["team_won"]
        .agg(["mean", "count"])
        .reset_index()
    )
    grouped.columns = ["prob_int", "time_bin", "empirical_win_rate", "n"]

    # Pivot into matrices
    win_rate_matrix = grouped.pivot(
        index="prob_int", columns="time_bin", values="empirical_win_rate"
    )
    count_matrix = grouped.pivot(
        index="prob_int", columns="time_bin", values="n"
    )

    # Mask cells with too few observations so they show as grey
    mask = count_matrix < MIN_OBS

    # ── y-axis tick labels (show every 5th %) ────────────────────────────
    ytick_positions = []
    ytick_labels = []
    for i, prob in enumerate(win_rate_matrix.index):
        if prob % 5 == 0:
            ytick_positions.append(i)
            ytick_labels.append(f"{prob}%")

    # ── build annotation matrix (empirical win rate as "XX%") ────────────
    annot_matrix = win_rate_matrix.copy().astype(object)
    for r in win_rate_matrix.index:
        for c in win_rate_matrix.columns:
            wr = win_rate_matrix.loc[r, c]
            n  = count_matrix.loc[r, c]
            if pd.isna(wr) or n < MIN_OBS:
                annot_matrix.loc[r, c] = ""
            else:
                annot_matrix.loc[r, c] = f"{wr*100:.0f}%"

    # ── plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 28))

    # Flip so high probability is on top
    data_plot = win_rate_matrix.iloc[::-1]
    mask_plot = mask.iloc[::-1]
    annot_plot = annot_matrix.iloc[::-1]

    # Count unique games and calculate dimensions for the title
    n_games = df["kalshi_event"].nunique() if "kalshi_event" in df.columns else "?"
    n_prob_rows = len(win_rate_matrix)
    n_time_bins = len(win_rate_matrix.columns)

    sns.heatmap(
        data_plot,
        mask=mask_plot,
        annot=annot_plot,
        fmt="",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Empirical Win Rate", "shrink": 0.6, "pad": 0.02},
        ax=ax,
        xticklabels=True,
        annot_kws={"fontsize": 5.5, "fontweight": "bold"},
    )

    # Grey fill for masked (low-data) cells
    ax.set_facecolor("#d9d9d9")

    # Fix y-axis: show every 5 %
    flipped_index = list(data_plot.index)
    ytick_positions_flipped = []
    ytick_labels_flipped = []
    for i, prob in enumerate(flipped_index):
        if prob % 5 == 0:
            ytick_positions_flipped.append(i + 0.5)      # center of cell
            ytick_labels_flipped.append(f"{prob}%")

    ax.set_yticks(ytick_positions_flipped)
    ax.set_yticklabels(ytick_labels_flipped, fontsize=10)

    ax.set_xlabel("Game Time (minutes elapsed)", fontsize=14, labelpad=12)
    ax.set_ylabel("Kalshi Quoted Win Probability", fontsize=14, labelpad=12)
    ax.set_title(
        "Calibration Heat Map — Kalshi Win Probability vs. Empirical Win Rate\n"
        f"({n_games} games · {len(df):,} observations · {n_prob_rows} prob rows × {n_time_bins} time bins · regulation time only)",
        fontsize=15,
        pad=16,
    )

    ax.tick_params(axis="x", labelsize=11, rotation=0)
    ax.tick_params(axis="y", labelsize=10)

    plt.tight_layout()
    output_file = os.path.join(generated_data_dir, "rawdata_heatmap.png")
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    print(f"\n  Saved rawdata_heatmap.png to GeneratedVisualizations ({len(win_rate_matrix)} prob rows × {len(win_rate_matrix.columns)} time bins)\n")
    #print(f"  Cells masked (< {MIN_OBS} obs): {mask.sum().sum()} / {mask.size}")
    plt.close()  # Close the figure instead of showing it


if __name__ == "__main__":
    generateHeatMap(input_file="GeneratedDataFiles/all_games_merged_clean_GOOD.csv", num_time_bins=16)
