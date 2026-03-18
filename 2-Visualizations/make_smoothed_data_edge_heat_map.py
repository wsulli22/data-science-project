#!/usr/bin/env python3
"""
make_smoothed_data_edge_heat_map.py

Implements README TODO [3]:
  Heat map of edge magnitude using the smoothed calibration surface.

edge_pct_points = abs(win_rate_pct - kalshi_prob_pct)
where:
  win_rate_pct = (smoothed empirical win probability) * 100
  kalshi_prob_pct = row label in the smoothed heatmap CSV
"""

import os
import re
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


def _time_bin_sort_key(time_bin_label: str) -> int:
    """
    Parse labels like '12-13 min' and sort by the start minute.
    """
    match = re.match(r"^(\d+)-(\d+)\s+min$", time_bin_label)
    if not match:
        return 0
    return int(match.group(1))


def generate_smoothed_data_edge_heat_map(
    input_file: str = "GeneratedDataAndVisualizations/smoothed_heatmap_data.csv",
    min_obs_per_cell: int = 5,
    output_filename: str = "smoothed_edge_heatmap.png",
):
    """
    Args:
        input_file: Path to `smoothed_heatmap_data.csv`.
        min_obs_per_cell: Cells with fewer than this count are masked (grey).
        output_filename: Output PNG filename.
    """
    print("\nGENERATING SMOOTHED DATA EDGE HEATMAP\n")
    FILE = _resolve_input_path(input_file)

    script_dir = os.path.dirname(__file__)
    out_dir = os.path.join(script_dir, "GeneratedDataAndVisualizations")
    os.makedirs(out_dir, exist_ok=True)

    # ── load CSV ─────────────────────────────────────────────────────────
    df = pd.read_csv(FILE, index_col=0)
    df.index = df.index.astype(int)  # kalshi_prob_pct

    # Identify time-bin probability columns (exclude *_count and *_distribution_pct)
    prob_cols = [
        c
        for c in df.columns
        if not c.endswith("_count") and not c.endswith("_distribution_pct")
    ]
    time_bins = sorted(prob_cols, key=_time_bin_sort_key)

    win_rate_matrix = df[time_bins].astype(float)  # values in [0,1]

    # Count columns match probability columns with a '_count' suffix
    count_cols = [f"{tb}_count" for tb in time_bins]
    missing = [c for c in count_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing count columns in smoothed CSV: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    count_matrix = df[count_cols].astype(float)
    count_matrix.columns = time_bins

    kalshi_probs = win_rate_matrix.index.to_numpy(dtype=float)  # 1..99

    # edge_pct_points = abs(win_rate_pct - kalshi_prob_pct)
    edge_matrix = (win_rate_matrix.mul(100.0).sub(kalshi_probs, axis=0)).abs()

    mask = count_matrix < min_obs_per_cell

    # ── annotation matrix ───────────────────────────────────────────────
    annot_matrix = edge_matrix.copy().astype(object)
    for prob in edge_matrix.index:
        for tb in time_bins:
            n = int(count_matrix.loc[prob, tb])
            val = edge_matrix.loc[prob, tb]
            if pd.isna(val) or n < min_obs_per_cell:
                annot_matrix.loc[prob, tb] = ""
            else:
                annot_matrix.loc[prob, tb] = f"{float(val):.2f}% ({n})"

    vmax = float(np.nanmax(edge_matrix.to_numpy())) if np.isfinite(np.nanmax(edge_matrix.to_numpy())) else 1.0

    # ── plot ──────────────────────────────────────────────────────────────
    num_time_bins = len(time_bins)
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

    # Show a proportional subset of x labels
    x_tick_positions = ax.get_xticks()
    x_tick_labels = [label.get_text() for label in ax.get_xticklabels()]
    label_step = max(1, int(round(num_time_bins / 10)))
    x_tick_positions_filtered = x_tick_positions[::label_step]
    x_tick_labels_filtered = x_tick_labels[::label_step]
    ax.set_xticks(x_tick_positions_filtered)
    ax.set_xticklabels(x_tick_labels_filtered, fontsize=11, rotation=0)

    ax.set_xlabel("Game Time (minutes elapsed)", fontsize=14, labelpad=12)
    ax.set_ylabel("Kalshi Quoted Win Probability", fontsize=14, labelpad=12)
    ax.set_title(
        "Smoothed-Data Edge Heat Map — |Empirical(GAM) - Kalshi|\n"
        f"(mask cells with count<{min_obs_per_cell})",
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
    generate_smoothed_data_edge_heat_map()

