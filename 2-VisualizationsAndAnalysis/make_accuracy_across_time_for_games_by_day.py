#!/usr/bin/env python3
"""
make_accuracy_across_time_for_games_by_day.py

Bar chart of calibration accuracy (0–100, 100 = perfect) bucketed by week.

For each `kalshi_event` (game):
  error_pp = |empirical_win_rate% − mean(round(win_prob_pct))|
  accuracy_score_0_100 = max(0, 100 − error_pp)

Games are bucketed into calendar weeks using the game's earliest `wallclock_ts`
(week starts Monday). The chart plots one bar per week from oldest to newest.
"""

import os

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def _resolve_input_path(input_file: str) -> str:
    """Resolve relative paths relative to this script file."""
    script_dir = os.path.dirname(__file__)
    if os.path.isabs(input_file):
        return input_file
    return os.path.normpath(os.path.join(script_dir, input_file))


def generate_accuracy_across_time_for_games_by_day(
    input_file: str = "../1-GatheringPreprocessingTransformation/GeneratedDataFiles/all_games_merged_clean.csv",
    min_games_per_week: int = 1,
    output_filename: str = "accuracy_across_time_for_games_by_day.png",
):
    """
    Plot weekly accuracy (aggregating per-game scores) ordered by week start.
    """
    print("\nGENERATING WEEKLY ACCURACY (0–100, bar chart)\n")
    FILE = _resolve_input_path(input_file)

    script_dir = os.path.dirname(__file__)
    out_dir = os.path.join(script_dir, "GeneratedDataAndVisualizations")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(FILE)
    df = df.dropna(subset=["kalshi_event", "wallclock_ts", "win_prob_pct", "team_won"])

    df["wallclock_ts"] = pd.to_datetime(df["wallclock_ts"], errors="coerce")
    df = df.dropna(subset=["wallclock_ts"])

    df["prob_int"] = df["win_prob_pct"].round(0).astype(int)
    df = df[df["prob_int"].between(1, 99)]

    per_game = (
        df.groupby("kalshi_event", observed=False)
        .agg(
            game_start_ts=("wallclock_ts", "min"),
            empirical_win_rate=("team_won", "mean"),
            kalshi_avg_prob_pct=("prob_int", "mean"),
        )
        .reset_index()
    )
    per_game["error_pp"] = (per_game["empirical_win_rate"] * 100.0 - per_game["kalshi_avg_prob_pct"]).abs()
    per_game["accuracy_0_100"] = np.maximum(0.0, 100.0 - per_game["error_pp"])
    per_game = per_game.sort_values("game_start_ts")

    # ── weekly aggregation ──────────────────────────────────────────────
    # Use week periods with Monday as week start.
    per_game["week_start"] = per_game["game_start_ts"].dt.to_period("W-MON").dt.start_time

    weekly = (
        per_game.groupby("week_start", observed=False)
        .agg(
            mean_accuracy_0_100=("accuracy_0_100", "mean"),
            n_games=("kalshi_event", "count"),
        )
        .reset_index()
    )
    weekly = weekly.sort_values("week_start")

    start_week = weekly["week_start"].min()
    end_week = weekly["week_start"].max()
    full_weeks = pd.date_range(start=start_week, end=end_week, freq="7D")

    plot_df = pd.DataFrame({"week_start": full_weeks}).merge(weekly, on="week_start", how="left")
    plot_df["n_games"] = plot_df["n_games"].fillna(0).astype(int)
    plot_df["mean_accuracy_0_100"] = plot_df["mean_accuracy_0_100"].fillna(np.nan)

    enough = plot_df["n_games"] >= min_games_per_week
    plot_df["accuracy_score_0_100"] = np.where(
        enough.to_numpy(),
        plot_df["mean_accuracy_0_100"].to_numpy(dtype=float),
        np.nan,
    )

    # ── plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 7))
    x_idx = np.arange(len(plot_df))

    y_min, y_max = 97.0, 100.5
    y_vals = plot_df["accuracy_score_0_100"].to_numpy(dtype=float)

    for i in range(len(plot_df)):
        if enough.iloc[i]:
            ax.bar(i, float(y_vals[i]), width=0.85, color="steelblue", edgecolor="black", linewidth=0.25)

    masked_idx = x_idx[~enough.to_numpy()]
    if len(masked_idx):
        ax.scatter(
            masked_idx,
            np.full(len(masked_idx), y_min + 0.08),
            marker="x",
            color="#555555",
            s=28,
            zorder=5,
            label=f"n < {min_games_per_week} games",
        )

    # OLS regression trend line (x = sequential week index)
    ok = np.isfinite(y_vals) & enough.to_numpy()
    has_trend = ok.sum() >= 2
    if has_trend:
        slope, intercept = np.polyfit(x_idx[ok].astype(float), y_vals[ok].astype(float), 1)
        line_y = slope * x_idx.astype(float) + intercept
        ax.plot(
            x_idx,
            line_y,
            color="crimson",
            linewidth=2.2,
            linestyle="--",
            zorder=4,
        )

    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("Accuracy score (100 = perfect; axis 97–100)")
    ax.set_xlabel("Week (start date)")

    tick_step = max(1, int(round(len(plot_df) / 15)))
    ax.set_xticks(x_idx[::tick_step])
    ax.set_xticklabels(
        [d.strftime("%Y-%m-%d") for d in plot_df["week_start"].iloc[::tick_step]],
        rotation=45,
        ha="right",
        fontsize=8,
    )

    ax.set_title(
        "Accuracy by Week (week start Monday; y-axis zoomed 97–100)\n"
        "Score per week = mean over games of (max(0, 100 − |empirical win rate% − avg Kalshi prob%|))"
    )

    legend_handles = [Patch(facecolor="steelblue", edgecolor="black", label="Weekly accuracy score")]
    if has_trend:
        delta_100 = slope * 100.0
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                color="crimson",
                linestyle="--",
                linewidth=2.2,
                label=f"Linear trend (Δ ≈ {delta_100:+.3f} pts per 100 weeks index)",
            )
        )
    if len(masked_idx):
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="x",
                color="w",
                markerfacecolor="#555555",
                markersize=8,
                linestyle="None",
                label=f"n < {min_games_per_week} games",
            )
        )
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(out_dir, output_filename)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved {output_filename} -> {out_path}")
    return plot_df


if __name__ == "__main__":
    generate_accuracy_across_time_for_games_by_day()
