#!/usr/bin/env python3
"""
daily_max_concurrent_live_games.py

Plot the maximum number of concurrently "live" games per calendar day.

"Live" definition (matches your max-concurrency request):
  - start = min(wallclock_ts) for that kalshi_event/game
  - end   = max(wallclock_ts) for that kalshi_event/game
  - buffered window is [start, end + BUFFER_IN_MINS] (inclusive)
"""

from __future__ import annotations

import os
from datetime import timedelta

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


# Default single-buffer plot value.
BUFFER_IN_MINS_DEFAULT = 60 * 2


def _resolve_input_path(input_file: str) -> str:
    """Resolve relative paths relative to this script file."""
    script_dir = os.path.dirname(__file__)
    if os.path.isabs(input_file):
        return input_file
    return os.path.normpath(os.path.join(script_dir, input_file))


def load_start_end_by_game(
    input_file: str,
    chunksize: int = 200_000,
) -> tuple[dict[str, pd.Timestamp], dict[str, pd.Timestamp]]:
    """
    Load per-game earliest and latest timestamps (wallclock_ts).

    Returns:
      (start_by_game, end_by_game)
    """
    FILE = _resolve_input_path(input_file)
    if not os.path.exists(FILE):
        raise FileNotFoundError(f"Input CSV not found: {FILE}")

    usecols = ["kalshi_event", "wallclock_ts"]
    reader = pd.read_csv(
        FILE,
        usecols=usecols,
        chunksize=chunksize,
        dtype={"kalshi_event": "string", "wallclock_ts": "string"},
    )

    start_by_game: dict[str, pd.Timestamp] = {}
    end_by_game: dict[str, pd.Timestamp] = {}

    for chunk in reader:
        chunk = chunk.dropna(subset=["kalshi_event", "wallclock_ts"])
        if chunk.empty:
            continue

        chunk["wallclock_ts"] = pd.to_datetime(chunk["wallclock_ts"], errors="coerce")
        chunk = chunk.dropna(subset=["wallclock_ts", "kalshi_event"])
        if chunk.empty:
            continue

        grouped = chunk.groupby("kalshi_event", observed=False)["wallclock_ts"].agg(["min", "max"])
        for game_id, row in grouped.iterrows():
            ts_min: pd.Timestamp = row["min"]
            ts_max: pd.Timestamp = row["max"]

            prev_start = start_by_game.get(game_id)
            if prev_start is None or ts_min < prev_start:
                start_by_game[game_id] = ts_min

            prev_end = end_by_game.get(game_id)
            if prev_end is None or ts_max > prev_end:
                end_by_game[game_id] = ts_max

    if not start_by_game:
        raise ValueError("No games found after filtering/parse errors.")

    return start_by_game, end_by_game


def compute_daily_max_concurrent_live_games_from_game_bounds(
    start_by_game: dict[str, pd.Timestamp],
    end_by_game: dict[str, pd.Timestamp],
    buffer_in_mins: int,
) -> pd.Series:
    buffer = timedelta(minutes=int(buffer_in_mins))

    # Build sweep-line events:
    #   +1 at each game start
    #   -1 at each (game end + buffer)
    events: list[tuple[pd.Timestamp, int]] = []
    for game_id, start_ts in start_by_game.items():
        end_ts = end_by_game[game_id]
        events.append((start_ts, +1))
        events.append((end_ts + pd.Timedelta(buffer), -1))

    events.sort(key=lambda x: x[0])

    min_day = min(start_by_game.values()).floor("D")
    max_day = max((end_by_game[g] + pd.Timedelta(buffer)).to_pydatetime() for g in end_by_game)
    max_day = pd.to_datetime(max_day).floor("D")

    days = pd.date_range(start=min_day, end=max_day, freq="D")
    day_max = pd.Series(0, index=days, dtype=int)

    # Sweep, tracking current live count and recording per-day maximums.
    current_live = 0
    p = 0
    n = len(events)

    while p < n:
        t = events[p][0]

        start_delta = 0
        end_delta = 0
        while p < n and events[p][0] == t:
            delta = events[p][1]
            if delta == 1:
                start_delta += 1
            else:
                end_delta += 1
            p += 1

        # At time `t`, starts have happened, ends should still count (inclusive).
        current_live += start_delta
        day_t = t.floor("D")
        if day_t in day_max.index:
            if current_live > int(day_max.loc[day_t]):
                day_max.loc[day_t] = current_live

        # For times right after t, apply the ends.
        # `current_interval` is constant across [t, next_t).
        if p < n:
            next_t = events[p][0]
            current_interval = current_live - end_delta

            day_cursor = day_t
            while day_cursor < next_t:
                day_start = day_cursor
                day_end = day_start + pd.Timedelta(days=1)
                seg_end = min(next_t, day_end)
                if seg_end > day_start:
                    if current_interval > int(day_max.loc[day_start]):
                        day_max.loc[day_start] = current_interval
                day_cursor = day_end

        current_live -= end_delta

    return day_max


def compute_time_weighted_avg_concurrent_live_games_from_game_bounds(
    start_by_game: dict[str, pd.Timestamp],
    end_by_game: dict[str, pd.Timestamp],
    buffer_in_mins: int,
) -> float:
    """
    Compute time-weighted average concurrent live games over the full timeline
    covered by the buffered [start, end] windows.
    """
    buffer = timedelta(minutes=int(buffer_in_mins))

    # Build sweep-line events:
    #   +1 at each game start
    #   -1 at each (game end + buffer)
    events: list[tuple[pd.Timestamp, int]] = []
    for game_id, start_ts in start_by_game.items():
        end_ts = end_by_game[game_id]
        events.append((start_ts, +1))
        events.append((end_ts + pd.Timedelta(buffer), -1))

    events.sort(key=lambda x: x[0])

    # Sweep line: between event times, live count is constant.
    current_live = 0
    p = 0
    n = len(events)
    total_live_seconds = 0.0
    total_seconds = 0.0

    while p < n:
        t = events[p][0]

        start_delta = 0
        end_delta = 0
        while p < n and events[p][0] == t:
            delta = events[p][1]
            if delta == 1:
                start_delta += 1
            else:
                end_delta += 1
            p += 1

        # Apply starts at time t.
        current_live += start_delta

        # Interval is (t, next_t); the ends at t should not count after t.
        if p < n:
            next_t = events[p][0]
            duration_seconds = (next_t - t).total_seconds()
            if duration_seconds > 0:
                current_interval = current_live - end_delta
                total_live_seconds += float(current_interval) * duration_seconds
                total_seconds += duration_seconds

        # Now remove ends occurring at t.
        current_live -= end_delta

    if total_seconds <= 0:
        raise ValueError("Timeline duration is zero; cannot compute time-weighted average.")

    return total_live_seconds / total_seconds


def compute_daily_max_concurrent_live_games(
    input_file: str,
    buffer_in_mins: int = 60 * 2,
    chunksize: int = 200_000,
) -> pd.Series:
    start_by_game, end_by_game = load_start_end_by_game(input_file=input_file, chunksize=chunksize)
    return compute_daily_max_concurrent_live_games_from_game_bounds(
        start_by_game=start_by_game,
        end_by_game=end_by_game,
        buffer_in_mins=buffer_in_mins,
    )


def compute_time_weighted_avg_concurrent_live_games(
    input_file: str,
    buffer_in_mins: int = 60 * 2,
    chunksize: int = 200_000,
) -> float:
    """
    Compute time-weighted average concurrent live games.

    This answers: on average, how many games are live at "any point in time"
    over the full timeline covered by the buffered [start, end] windows.
    """
    start_by_game, end_by_game = load_start_end_by_game(input_file=input_file, chunksize=chunksize)
    return compute_time_weighted_avg_concurrent_live_games_from_game_bounds(
        start_by_game=start_by_game,
        end_by_game=end_by_game,
        buffer_in_mins=buffer_in_mins,
    )


def plot_daily_max_concurrent_live_games(
    input_file: str = "../1-GatheringPreprocessingTransformation/GeneratedDataFiles/all_games_merged_clean.csv",
    buffer_in_mins: int = 60 * 2,
    output_filename: str = "daily_max_concurrent_live_games.png",
) -> pd.Series:
    script_dir = os.path.dirname(__file__)
    out_dir = os.path.join(script_dir, "GeneratedDataAndVisualizations")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, output_filename)

    print("\nPLOTTING DAILY MAX CONCURRENT LIVE GAMES\n")
    day_max = compute_daily_max_concurrent_live_games(
        input_file=input_file,
        buffer_in_mins=buffer_in_mins,
    )

    # ── plot: 7 weekday bars ────────────────────────────────────────────
    # We interpret "concurrent live averages" as the average of the daily-max
    # concurrent live games for each weekday.
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday_means: list[float] = []
    for dow in range(7):
        mask = day_max.index.dayofweek == dow  # Monday=0
        weekday_means.append(float(day_max.loc[mask].mean()))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(day_names, weekday_means, color="#1f77b4", edgecolor="black", linewidth=0.3)
    ax.set_title(
        f"Avg Daily Max Concurrent Live Games by Weekday (buffer={buffer_in_mins} mins)"
    )
    ax.set_ylabel("Average max live games")
    ax.set_xlabel("Weekday")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")
    return day_max


if __name__ == "__main__":
    # Also keep the single-buffer plot available for quick inspection.
    plot_daily_max_concurrent_live_games(
        input_file="../1-GatheringPreprocessingTransformation/GeneratedDataFiles/all_games_merged_clean.csv",
        buffer_in_mins=BUFFER_IN_MINS_DEFAULT,
        output_filename="daily_max_concurrent_live_games.png",
    )

