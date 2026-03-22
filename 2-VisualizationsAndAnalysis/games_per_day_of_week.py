#!/usr/bin/env python3
"""
games_per_day_of_week.py

Plots average number of unique games by day of week.

Definition:
  1) For each `kalshi_event` (game), compute earliest `realworld_timestamp`.
  2) Count games for each calendar date.
  3) Take the mean games/day for each weekday across all dates in the range.
"""

import os

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def _resolve_input_path(input_file: str) -> str:
    script_dir = os.path.dirname(__file__)
    if os.path.isabs(input_file):
        return input_file
    return os.path.normpath(os.path.join(script_dir, input_file))


def generate_games_per_day_of_week(
    input_file: str = "../1-GatheringPreprocessingTransformation/GeneratedDataFiles/all_games_merged_clean.csv",
    output_filename: str = "games_per_day_of_week.png",
):
    print("\nGENERATING AVERAGE GAMES PER DAY OF WEEK\n")
    file_path = _resolve_input_path(input_file)

    script_dir = os.path.dirname(__file__)
    out_dir = os.path.join(script_dir, "GeneratedDataAndVisualizations")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(file_path)
    df = df.dropna(subset=["kalshi_event", "realworld_timestamp"])
    df["realworld_timestamp"] = pd.to_datetime(df["realworld_timestamp"], errors="coerce")
    df = df.dropna(subset=["realworld_timestamp"])

    game_start = df.groupby("kalshi_event", observed=False)["realworld_timestamp"].min()
    game_day = game_start.dt.floor("D")

    # Build a full date range so low/no-game weekdays are represented fairly.
    start_day = game_day.min()
    end_day = game_day.max()
    full_days = pd.date_range(start=start_day, end=end_day, freq="D")

    games_per_date = game_day.value_counts().reindex(full_days, fill_value=0).sort_index()
    daily_df = pd.DataFrame(
        {
            "date": games_per_date.index,
            "games": games_per_date.values.astype(float),
        }
    )
    daily_df["weekday"] = daily_df["date"].dt.day_name()

    avg_counts = (
        daily_df.groupby("weekday", observed=False)["games"]
        .mean()
        .reindex(DAY_ORDER, fill_value=0.0)
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(
        avg_counts.index,
        avg_counts.values,
        color="steelblue",
        edgecolor="black",
        linewidth=0.4,
    )

    for bar, value in zip(bars, avg_counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.05,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("Average Number of Games")
    ax.set_xlabel("Day of Week")
    ax.set_title("Average Games per Day of Week")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    out_path = os.path.join(out_dir, output_filename)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved {output_filename} -> {out_path}")
    return avg_counts


if __name__ == "__main__":
    generate_games_per_day_of_week()
