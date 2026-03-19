#!/usr/bin/env python3
"""
smoothed_edge_heatmap.py

Signed calibration edge heatmap:
  1. Aggregate **raw** observations into (Kalshi prob × time) cells.
  2. For each cell: signed_edge = empirical_win_rate_pct - kalshi_prob_pct
     (NOT absolute value).
  3. Smooth that signed edge surface with a LinearGAM (same spline structure
     idea as the calibration GAM, but continuous response).

Previously this script loaded pre-smoothed empirical probabilities and took
|empirical - Kalshi|; that order is reversed here per project requirements.

Visualization: diverging colormap — orange = negative edge, blue = positive.
"""

import os
import re
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import seaborn as sns

from pygam import LinearGAM, s, te

warnings.filterwarnings("ignore", category=FutureWarning)


def _resolve_input_path(input_file: str) -> str:
    script_dir = os.path.dirname(__file__)
    if os.path.isabs(input_file):
        return input_file
    return os.path.normpath(os.path.join(script_dir, input_file))


def _time_bin_sort_key(time_bin_label: str) -> int:
    match = re.match(r"^(\d+)-(\d+)\s+min$", time_bin_label)
    if not match:
        return 0
    return int(match.group(1))


# Match smoothed_heatmap.py / rawdata_edge_heatmap defaults
NUM_TIME_BINS_DEFAULT = 40
N_SPLINES_TIME = 20
N_SPLINES_PROB = 20


def _orange_white_blue_cmap():
    """Negative (orange) -> white (0) -> positive (blue)."""
    return LinearSegmentedColormap.from_list(
        "orange_white_blue",
        [
            "#e66100",  # orange (negative)
            "#fde0c5",  # light orange
            "#f7f7f7",  # near white (center)
            "#c9e4f5",  # light blue
            "#2171b5",  # blue (positive)
        ],
    )


def generate_smoothed_data_edge_heat_map(
    input_file: str = "../1-GatheringPreprocessingTransformation/GeneratedDataFiles/all_games_merged_clean.csv",
    num_time_bins: int = NUM_TIME_BINS_DEFAULT,
    min_obs_per_cell: int = 5,
    output_filename: str = "smoothed_edge_heatmap.png",
):
    """
    Args:
        input_file: Raw merged games CSV (same source as raw edge heatmap).
        num_time_bins: Time buckets (40 => 1-minute bins over regulation).
        min_obs_per_cell: Cells with fewer observations are masked (grey).
        output_filename: Output PNG filename.
    """
    print("\nGENERATING SIGNED EDGE HEATMAP (raw cells → smooth)\n")
    FILE = _resolve_input_path(input_file)

    script_dir = os.path.dirname(__file__)
    out_dir = os.path.join(script_dir, "GeneratedDataAndVisualizations")
    os.makedirs(out_dir, exist_ok=True)

    # ── binning (match smoothed_heatmap.py) ─────────────────────
    time_edges = np.linspace(0, 2400, num_time_bins + 1)
    time_labels = [
        f"{int(lo / 60)}-{int(hi / 60)} min"
        for lo, hi in zip(time_edges[:-1], time_edges[1:])
    ]
    time_bin_to_frac = {
        lbl: ((time_edges[i] + time_edges[i + 1]) / 2.0) / 2400.0
        for i, lbl in enumerate(time_labels)
    }
    bin_centres_frac = (time_edges[:-1] + time_edges[1:]) / 2.0 / 2400.0

    probs = np.arange(1, 100)

    # ── load raw data ───────────────────────────────────────────────────
    df = pd.read_csv(FILE)
    df = df.dropna(subset=["kalshi_event", "team", "game_elapsed_seconds", "win_prob_pct", "team_won"])

    # If multiple Kalshi quotes share the same game clock (game_elapsed_seconds),
    # average the quoted probability and treat that as the quote for that clock.
    # Grouping is per game + team because the quote/ outcome are team-specific.
    df["clock_seconds"] = df["game_elapsed_seconds"].round(0)
    df = (
        df.groupby(["kalshi_event", "team", "clock_seconds"], observed=False, as_index=False)
        .agg(win_prob_pct=("win_prob_pct", "mean"), team_won=("team_won", "first"))
    )
    df["game_elapsed_seconds"] = df["clock_seconds"].astype(float)
    df = df.drop(columns=["clock_seconds"])

    df["time_bin"] = pd.cut(
        df["game_elapsed_seconds"],
        bins=time_edges,
        labels=time_labels,
        right=False,
        include_lowest=True,
    )
    df = df.dropna(subset=["time_bin"])

    df["prob_int"] = df["win_prob_pct"].round(0).astype(int)
    df = df[df["prob_int"].between(1, 99)]

    # ── per-cell signed edge (raw, not smoothed) ─────────────────────────
    grouped = (
        df.groupby(["prob_int", "time_bin"], observed=False)
        .agg(
            empirical_win_rate=("team_won", "mean"),
            n=("team_won", "count"),
        )
        .reset_index()
    )
    grouped["signed_edge_pct"] = (
        grouped["empirical_win_rate"] * 100.0 - grouped["prob_int"]
    )

    signed_raw = grouped.pivot(
        index="prob_int", columns="time_bin", values="signed_edge_pct"
    )
    count_matrix = grouped.pivot(index="prob_int", columns="time_bin", values="n")

    signed_raw = signed_raw.reindex(index=probs, columns=time_labels)
    count_matrix = count_matrix.reindex(
        index=probs, columns=time_labels, fill_value=0
    ).astype(float)

    # ── fit LinearGAM on cells with ≥1 obs (weighted by count) ──────────
    X_list, y_list, w_list = [], [], []
    for p in probs:
        for tb in time_labels:
            n = int(count_matrix.loc[p, tb])
            if n < 1:
                continue
            v = signed_raw.loc[p, tb]
            if pd.isna(v):
                continue
            X_list.append([time_bin_to_frac[tb], p / 100.0])
            y_list.append(float(v))
            w_list.append(float(n))

    X_train = np.asarray(X_list, dtype=float)
    y_train = np.asarray(y_list, dtype=float)
    w_train = np.asarray(w_list, dtype=float)

    if len(y_train) < 30:
        raise ValueError(
            f"Too few non-empty cells for GAM ({len(y_train)}). Check input data."
        )

    print("  Fitting LinearGAM on raw signed edge (weighted by cell counts)...")
    gam = LinearGAM(
        s(0, n_splines=N_SPLINES_TIME, spline_order=3)
        + s(1, n_splines=N_SPLINES_PROB, spline_order=3)
        + te(0, 1, n_splines=[8, 8])
    )
    gam.gridsearch(
        X_train,
        y_train,
        weights=w_train,
        lam=np.logspace(-3, 3, 11),
        progress=False,
    )
    print(f"    Best λ: {gam.lam}")

    # ── predict smoothed signed edge on full grid ────────────────────────
    prob_frac = probs / 100.0
    T, P = np.meshgrid(bin_centres_frac, prob_frac)
    grid = np.column_stack([T.ravel(), P.ravel()])
    Z = gam.predict(grid).reshape(len(probs), num_time_bins)
    edge_smoothed = pd.DataFrame(Z, index=probs, columns=time_labels)

    mask = count_matrix < min_obs_per_cell

    # ── annotations: smoothed value + n ─────────────────────────────────
    annot_matrix = edge_smoothed.copy().astype(object)
    for prob in edge_smoothed.index:
        for tb in time_labels:
            n = int(count_matrix.loc[prob, tb])
            val = edge_smoothed.loc[prob, tb]
            if pd.isna(val) or n < min_obs_per_cell:
                annot_matrix.loc[prob, tb] = ""
            else:
                annot_matrix.loc[prob, tb] = f"{float(val):+.2f}% ({n})"

    # Symmetric color scale around 0 (based on visible / unmasked cells)
    vals = edge_smoothed.to_numpy()
    m = mask.to_numpy()
    vis = vals[~m & np.isfinite(vals)]
    vlim = float(np.nanmax(np.abs(vis))) if vis.size else 1.0
    if vlim < 1e-6:
        vlim = 1.0

    cmap = _orange_white_blue_cmap()
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)

    num_time_bins = len(time_labels)
    fig_width = 16 * (num_time_bins / 20)
    fig, ax = plt.subplots(figsize=(fig_width, 28))

    data_plot = edge_smoothed.iloc[::-1]
    mask_plot = mask.iloc[::-1]
    annot_plot = annot_matrix.iloc[::-1]

    sns.heatmap(
        data_plot,
        mask=mask_plot,
        annot=annot_plot,
        fmt="",
        cmap=cmap,
        norm=norm,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={
            "label": "Signed edge (Empirical − Kalshi), percentage points",
            "shrink": 0.6,
            "pad": 0.02,
        },
        ax=ax,
        xticklabels=True,
        annot_kws={"fontsize": 5.5, "fontweight": "bold", "color": "black"},
    )

    ax.set_facecolor("#d9d9d9")

    flipped_index = list(data_plot.index)
    ytick_positions = []
    ytick_labels = []
    for i, prob in enumerate(flipped_index):
        if prob % 5 == 0:
            ytick_positions.append(i + 0.5)
            ytick_labels.append(f"{prob}%")
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels, fontsize=10)

    x_tick_positions = ax.get_xticks()
    x_tick_labels = [label.get_text() for label in ax.get_xticklabels()]
    label_step = max(1, int(round(num_time_bins / 10)))
    ax.set_xticks(x_tick_positions[::label_step])
    ax.set_xticklabels(x_tick_labels[::label_step], fontsize=11, rotation=0)

    ax.set_xlabel("Game Time (minutes elapsed)", fontsize=14, labelpad=12)
    ax.set_ylabel("Kalshi Quoted Win Probability", fontsize=14, labelpad=12)
    ax.set_title(
        "Smoothed Signed Edge — Empirical(raw cells) − Kalshi, then LinearGAM\n"
        f"(orange = Kalshi high vs outcomes, blue = Kalshi low · mask cells with count<{min_obs_per_cell})",
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
    return edge_smoothed


if __name__ == "__main__":
    generate_smoothed_data_edge_heat_map()
