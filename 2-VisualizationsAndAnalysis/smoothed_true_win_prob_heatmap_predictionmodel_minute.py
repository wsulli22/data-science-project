#!/usr/bin/env python3
"""
smoothed_true_win_prob_heatmap_predictionmodel_minute.py

Smoothed empirical (true) win-rate heatmap from all weekly CSV files in:
    ../3-PredictionModel/Data/week_*_games.csv

Same pipeline as smoothed_edge_heatmap_predictionmodel_minute.py, but the GAM
smooths empirical win probability (0–100%) instead of signed edge; diverging
RdYlGn colormap (red = low realized win rate, green = high).
"""

import os
from glob import glob

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
from pygam import LinearGAM, s, te


REGULATION_SECONDS = 40 * 60
OT_SECONDS = 5 * 60
MAX_OT_PERIODS = 3
TOTAL_SECONDS_TO_PLOT = REGULATION_SECONDS + (MAX_OT_PERIODS * OT_SECONDS)
TOTAL_MINUTES_TO_PLOT = TOTAL_SECONDS_TO_PLOT // 60

NUM_TIME_BINS_DEFAULT = TOTAL_MINUTES_TO_PLOT
N_SPLINES_TIME = 20
N_SPLINES_PROB = 20


def _long_team_view(df: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["kalshi_event", "game_elapsed_seconds", "winning_team"]

    team_1 = df[base_cols + ["team_1", "team_1_win_prob_pct", "team_1_volume"]].copy()
    team_1 = team_1.rename(
        columns={
            "team_1": "team",
            "team_1_win_prob_pct": "win_prob_pct",
            "team_1_volume": "volume",
        }
    )

    team_2 = df[base_cols + ["team_2", "team_2_win_prob_pct", "team_2_volume"]].copy()
    team_2 = team_2.rename(
        columns={
            "team_2": "team",
            "team_2_win_prob_pct": "win_prob_pct",
            "team_2_volume": "volume",
        }
    )

    long_df = pd.concat([team_1, team_2], ignore_index=True)
    long_df["team_won"] = (long_df["team"] == long_df["winning_team"]).astype(int)
    return long_df


def _load_all_predictionmodel_data(data_dir: str) -> pd.DataFrame:
    pattern = os.path.join(data_dir, "week_*_games.csv")
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No weekly CSV files found at: {pattern}")

    frames = []
    for path in files:
        frames.append(pd.read_csv(path))
    wide_df = pd.concat(frames, ignore_index=True)

    long_df = _long_team_view(wide_df)
    long_df = long_df.dropna(
        subset=["kalshi_event", "team", "game_elapsed_seconds", "win_prob_pct", "team_won"]
    )
    return long_df


def generate_smoothed_true_win_prob_heatmap_predictionmodel_minute(
    input_data_dir: str = "../3-PredictionModel/Data",
    num_time_bins: int = NUM_TIME_BINS_DEFAULT,
    min_obs_per_cell: int = 2,
    output_filename: str = "smoothed_true_win_prob_heatmap_predictionmodel_minute.png",
):
    print("\nGENERATING SMOOTHED TRUE WIN PROB HEATMAP FROM 3-PredictionModel/Data (MINUTE GROUPED)\n")

    script_dir = os.path.dirname(__file__)
    data_dir = (
        input_data_dir
        if os.path.isabs(input_data_dir)
        else os.path.normpath(os.path.join(script_dir, input_data_dir))
    )
    out_dir = os.path.join(script_dir, "GeneratedDataAndVisualizations")
    os.makedirs(out_dir, exist_ok=True)

    time_edges = np.linspace(0, TOTAL_SECONDS_TO_PLOT, num_time_bins + 1)
    time_labels = [f"{int(lo / 60)}-{int(hi / 60)} min" for lo, hi in zip(time_edges[:-1], time_edges[1:])]
    time_bin_to_frac = {
        lbl: ((time_edges[i] + time_edges[i + 1]) / 2.0) / float(TOTAL_SECONDS_TO_PLOT)
        for i, lbl in enumerate(time_labels)
    }
    bin_centres_frac = (time_edges[:-1] + time_edges[1:]) / 2.0 / float(TOTAL_SECONDS_TO_PLOT)
    probs = np.arange(1, 100)

    df = _load_all_predictionmodel_data(data_dir)

    df["minute_bucket"] = (df["game_elapsed_seconds"].astype(float) // 60).astype(int)
    df = df[df["minute_bucket"].between(0, TOTAL_MINUTES_TO_PLOT - 1)]
    df = (
        df.groupby(["kalshi_event", "team", "minute_bucket"], observed=False, as_index=False)
        .agg(
            win_prob_pct=("win_prob_pct", "mean"),
            team_won=("team_won", "first"),
            volume=("volume", "mean"),
        )
    )
    df["game_elapsed_seconds"] = (df["minute_bucket"] * 60).astype(float)

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

    grouped = (
        df.groupby(["prob_int", "time_bin"], observed=False)
        .agg(empirical_win_rate=("team_won", "mean"), n=("team_won", "count"))
        .reset_index()
    )
    grouped["empirical_win_pct"] = grouped["empirical_win_rate"] * 100.0

    win_raw = grouped.pivot(index="prob_int", columns="time_bin", values="empirical_win_pct")
    count_matrix = grouped.pivot(index="prob_int", columns="time_bin", values="n")
    win_raw = win_raw.reindex(index=probs, columns=time_labels)
    count_matrix = count_matrix.reindex(index=probs, columns=time_labels, fill_value=0).astype(float)

    x_list, y_list, w_list = [], [], []
    for p in probs:
        for tb in time_labels:
            n = int(count_matrix.loc[p, tb])
            if n < 1:
                continue
            v = win_raw.loc[p, tb]
            if pd.isna(v):
                continue
            x_list.append([time_bin_to_frac[tb], p / 100.0])
            y_list.append(float(v))
            w_list.append(float(n))

    x_train = np.asarray(x_list, dtype=float)
    y_train = np.asarray(y_list, dtype=float)
    w_train = np.asarray(w_list, dtype=float)

    if len(y_train) < 30:
        raise ValueError(f"Too few non-empty cells for GAM ({len(y_train)}). Check input data.")

    print("  Fitting LinearGAM on minute-grouped empirical win % (weighted by cell counts)...")
    gam = LinearGAM(
        s(0, n_splines=N_SPLINES_TIME, spline_order=3)
        + s(1, n_splines=N_SPLINES_PROB, spline_order=3)
        + te(0, 1, n_splines=[8, 8])
    )
    gam.gridsearch(
        x_train,
        y_train,
        weights=w_train,
        lam=np.logspace(-3, 3, 11),
        progress=False,
    )

    prob_frac = probs / 100.0
    t, p = np.meshgrid(bin_centres_frac, prob_frac)
    grid = np.column_stack([t.ravel(), p.ravel()])
    z = np.clip(gam.predict(grid), 0.0, 100.0).reshape(len(probs), num_time_bins)
    win_smoothed = pd.DataFrame(z, index=probs, columns=time_labels)

    mask = count_matrix < min_obs_per_cell
    annot_matrix = win_smoothed.copy().astype(object)
    for prob in win_smoothed.index:
        for tb in time_labels:
            n = int(count_matrix.loc[prob, tb])
            val = win_smoothed.loc[prob, tb]
            if pd.isna(val) or n < min_obs_per_cell:
                annot_matrix.loc[prob, tb] = ""
            else:
                annot_matrix.loc[prob, tb] = f"{float(val):.2f}% ({n})"

    cmap = colormaps["RdYlGn"]
    norm = TwoSlopeNorm(vmin=0.0, vcenter=50.0, vmax=100.0)

    fig_width = max(24, 18 * (num_time_bins / 20))
    fig, ax = plt.subplots(figsize=(fig_width, 28))

    data_plot = win_smoothed.iloc[::-1]
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
            "label": "Smoothed true win probability (empirical), %",
            "shrink": 0.6,
            "pad": 0.02,
        },
        ax=ax,
        xticklabels=True,
        annot_kws={"fontsize": 5.5, "fontweight": "bold", "color": "black"},
    )

    ax.set_facecolor("#d9d9d9")
    divider_seconds = [
        REGULATION_SECONDS,
        REGULATION_SECONDS + OT_SECONDS,
        REGULATION_SECONDS + (2 * OT_SECONDS),
    ]
    for boundary_seconds in divider_seconds:
        divider_x = (boundary_seconds / float(TOTAL_SECONDS_TO_PLOT)) * num_time_bins
        ax.axvline(x=divider_x, color="black", linewidth=3.5)

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

    n_games = df["kalshi_event"].nunique()
    n_obs = len(df)
    ax.set_xlabel("Game Time (minutes elapsed)", fontsize=14, labelpad=12)
    ax.set_ylabel("Kalshi Quoted Win Probability", fontsize=14, labelpad=12)
    ax.set_title(
        "Smoothed True Win Probability (All 3-PredictionModel/Data files, minute-grouped through OT3)\n"
        f"(green = high realized win rate, red = low — {n_games} games — "
        f"{n_obs:,} minute-grouped observations — mask cells with n<{min_obs_per_cell})",
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
    return win_smoothed


if __name__ == "__main__":
    generate_smoothed_true_win_prob_heatmap_predictionmodel_minute()
