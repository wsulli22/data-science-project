#!/usr/bin/env python3
"""
max_concurrent_live_games.py

Compute the maximum number of games that are "live" at the same time.

Definition:
  - Each game's live window is:
      start = min(wallclock_ts) over rows for that game
      end   = max(wallclock_ts) over rows for that game + BUFFER_IN_MINS
  - A game is counted as live at timestamp t if:
      start <= t <= end
    (i.e., the buffer after the end is still counted as live)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import pandas as pd


def _resolve_input_path(input_file: str) -> str:
    """Resolve relative paths relative to this script file."""
    script_dir = os.path.dirname(__file__)
    if os.path.isabs(input_file):
        return input_file
    return os.path.normpath(os.path.join(script_dir, input_file))


@dataclass(frozen=True)
class MaxLiveResult:
    max_live_games: int
    timestamp_of_max_live: str  # ISO-8601 string (best-effort)


def compute_max_concurrent_live_games(
    input_file: str = "../1-GatheringPreprocessingTransformation/GeneratedDataFiles/all_games_merged_clean.csv",
    buffer_in_mins: int = 60 * 2,
    chunksize: int = 200_000,
) -> MaxLiveResult:
    FILE = _resolve_input_path(input_file)

    if not os.path.exists(FILE):
        raise FileNotFoundError(f"Input CSV not found: {FILE}")

    buffer = timedelta(minutes=int(buffer_in_mins))

    # Global bounds per game: {kalshi_event: min_ts/max_ts}
    start_by_game: dict[str, pd.Timestamp] = {}
    end_by_game: dict[str, pd.Timestamp] = {}

    # Stream read in case the CSV is large.
    usecols = ["kalshi_event", "wallclock_ts"]
    reader = pd.read_csv(
        FILE,
        usecols=usecols,
        chunksize=chunksize,
        dtype={"kalshi_event": "string", "wallclock_ts": "string"},
    )

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

    # Build sweep-line events: at start => +1, at end+buffer => -1
    # We'll count at each timestamp *before* applying the -1 deltas so that
    # end+buffer is inclusive.
    events: list[tuple[pd.Timestamp, int]] = []
    for game_id, start_ts in start_by_game.items():
        end_ts = end_by_game[game_id]
        events.append((start_ts, +1))
        events.append((end_ts + pd.Timedelta(buffer), -1))

    events.sort(key=lambda x: x[0])

    current_live = 0
    max_live = 0
    timestamp_of_max: pd.Timestamp | None = None

    i = 0
    while i < len(events):
        t = events[i][0]

        start_delta = 0
        end_delta = 0
        while i < len(events) and events[i][0] == t:
            delta = events[i][1]
            if delta == 1:
                start_delta += 1
            else:
                end_delta += 1  # will negate later
            i += 1

        # Apply starts, check max at time t (inclusive).
        current_live += start_delta
        if current_live > max_live:
            max_live = current_live
            timestamp_of_max = t

        # Apply ends after counting at time t.
        current_live -= end_delta

    timestamp_str = timestamp_of_max.isoformat(sep=" ") if timestamp_of_max is not None else ""
    return MaxLiveResult(
        max_live_games=int(max_live),
        timestamp_of_max_live=timestamp_str,
    )


def main() -> None:
    result = compute_max_concurrent_live_games()
    print(
        f"Max live games: {result.max_live_games}\n"
        f"Timestamp of max live: {result.timestamp_of_max_live}"
    )


if __name__ == "__main__":
    main()

