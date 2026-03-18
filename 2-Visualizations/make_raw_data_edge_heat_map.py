#!/usr/bin/env python3
"""
make_raw_data_edge_heat_map.py

Implements README TODO [2]:
  Heat map of raw-data edge magnitude over the (Kalshi prob, time) grid.

For each cell:
  empirical_win_rate = mean(team_won)
  edge_pct_points = abs(empirical_win_rate_pct - kalshi_prob_pct)
where both quantities are in percentage units.
"""

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


def _resolve_input_path(input_file: str) -> str:
    """Resolve relative paths relative to this script file."""
    script_dir = os.path.dirname(__file__)
    if os.path.isabs(input_file):
        return input_file
    return os.path.normpath(os.path.join(script_dir, input_file))


def generate_raw_data_edge_heat_map(
    input_file: str = "../GatheringPreprocessingTransformation/GeneratedDataFiles/all_games_merged_clean.csv",
    num_time_bins: int = 40,
    min_obs_per_cell: int = 5,
    output_filename: str = "rawdata_edge_heatmap.png",
):
    """
    Args:
        input_file: Path to `all_games_merged_clean.csv`.
        num_time_bins: Number of 1-minute buckets (40 for 0-40 min regulation).
        min_obs_per_cell: Cells with fewer than this count are masked (grey).
        output_filename: Output PNG filename.
    """
    print("\nGENERATING RAW DATA EDGE HEATMAP\n")
    FILE = _resolve_input_path(input_file)

    script_dir = os.path.dirname(__file__)
    out_dir = os.path.join(script_dir, "GeneratedDataAndVisualizations")
    os.makedirs(out_dir, exist_ok=True)

    # ── binning (match make_raw_data_heat_map.py) ───────────────────────────
    bin_size_seconds = 2400 / num_time_bins
    time_edges = np.arange(0, 2401, bin_size_seconds)
    time_labels = [
        f"{int(lo / 60)}-{int(hi / 60)} min" for lo, hi in zip(time_edges[:-1], time_edges[1:])
    ]

    # 1 row per integer win-probability percentage (1..99)
    probs = np.arange(1, 100)

    # ── load ──────────────────────────────────────────────────────────────
    df = pd.read_csv(FILE)
    df = df.dropna(subset=["game_elapsed_seconds", "win_prob_pct", "team_won"])

    # Time bin
    df["time_bin"] = pd.cut(
        df["game_elapsed_seconds"],
        bins=time_edges,
        labels=time_labels,
        right=False,
        include_lowest=True,
    )
    df = df.dropna(subset=["time_bin"])

    # Probability rows
    df["prob_int"] = df["win_prob_pct"].round(0).astype(int)
    df = df[df["prob_int"].between(1, 99)]

    # ── aggregate to cell statistics ──────────────────────────────────────
    grouped = (
        df.groupby(["prob_int", "time_bin"], observed=False)
        .agg(
            empirical_win_rate=("team_won", "mean"),
            n=("team_won", "count"),
        )
        .reset_index()
    )

    grouped["edge_pct_points"] = (grouped["empirical_win_rate"] * 100.0 - grouped["prob_int"]).abs()

    edge_matrix = grouped.pivot(
        index="prob_int", columns="time_bin", values="edge_pct_points"
    )
    count_matrix = grouped.pivot(index="prob_int", columns="time_bin", values="n")

    # Ensure full grid shape
    edge_matrix = edge_matrix.reindex(index=probs, columns=time_labels)
    count_matrix = count_matrix.reindex(index=probs, columns=time_labels, fill_value=0)

    mask = count_matrix < min_obs_per_cell

    # ── build annotation matrix ──────────────────────────────────────────
    annot_matrix = edge_matrix.copy().astype(object)
    for r in probs:
        for c in time_labels:
            n = int(count_matrix.loc[r, c])
            val = edge_matrix.loc[r, c]
            if pd.isna(val) or n < min_obs_per_cell:
                annot_matrix.loc[r, c] = ""
            else:
                annot_matrix.loc[r, c] = f"{float(val):.2f}% ({n})"

    vmax = float(np.nanmax(edge_matrix.to_numpy())) if np.isfinite(np.nanmax(edge_matrix.to_numpy())) else 1.0

    # ── plot ──────────────────────────────────────────────────────────────
    fig_width = 16 * (num_time_bins / 20)
    fig, ax = plt.subplots(figsize=(fig_width, 28))

    data_plot = edge_matrix.iloc[::-1]
    mask_plot = mask.iloc[::-1]
    annot_plot = annot_matrix.iloc[::-1]

    sns.heatmap(
        data_plot,
        mask=mask_plot,
        annot=annot_plot,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=vmax,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Absolute Edge (percentage points)", "shrink": 0.6, "pad": 0.02},
        ax=ax,
        xticklabels=True,
        annot_kws={"fontsize": 5.5, "fontweight": "bold", "color": "white"},
    )

    ax.set_facecolor("#d9d9d9")  # grey for masked cells

    # Y-axis tick labels (show every 5%)
    flipped_index = list(data_plot.index)
    ytick_positions = []
    ytick_labels = []
    for i, prob in enumerate(flipped_index):
        if prob % 5 == 0:
            ytick_positions.append(i + 0.5)
            ytick_labels.append(f"{prob}%")
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels, fontsize=10)

    # Show every 5th x-axis label for readability with 40 bins
    x_tick_positions = ax.get_xticks()
    x_tick_labels = [label.get_text() for label in ax.get_xticklabels()]
    x_tick_positions_filtered = x_tick_positions[::5]
    x_tick_labels_filtered = x_tick_labels[::5]
    ax.set_xticks(x_tick_positions_filtered)
    ax.set_xticklabels(x_tick_labels_filtered, fontsize=11, rotation=0)

    n_games = df["kalshi_event"].nunique() if "kalshi_event" in df.columns else "?"
    n_obs = len(df)
    ax.set_xlabel("Game Time (minutes elapsed)", fontsize=14, labelpad=12)
    ax.set_ylabel("Kalshi Quoted Win Probability", fontsize=14, labelpad=12)
    ax.set_title(
        "Raw-Data Edge Heat Map — |Empirical - Kalshi|\n"
        f"({n_games} games · {n_obs:,} observations · masked cells with n<{min_obs_per_cell})",
        fontsize=15,
        pad=16,
    )

    ax.tick_params(axis="x", labelsize=11, rotation=0)
    ax.tick_params(axis="y", labelsize=10)

    plt.tight_layout()
    out_path = os.path.join(out_dir, output_filename)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved {output_filename} -> {out_path}")
    return edge_matrix


if __name__ == "__main__":
    generate_raw_data_edge_heat_map()

