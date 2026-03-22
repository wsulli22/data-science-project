#!/usr/bin/env python3
"""
create_test_data.py

Read `all_games_merged_clean.csv` and expand each game (`kalshi_event`) to one row per
calendar second from the earliest to the latest `realworld_timestamp` seen for that game.

For seconds with no observed row for a team, interpolate per-team values first.
Then pivot both teams into the same output row for each timestamp.

Interpolation details:
  - `win_prob_pct`, `volume`, and `game_elapsed_seconds` are linearly interpolated in
    time between the bracketing observations; `win_prob_pct` and `volume` are rounded
    to whole numbers.
  - `period` is not interpolated: use the period from the latest observation whose
    timestamp is still on or before the current second (last-known / forward-filled
    from the left).

Games are emitted in order of each game's start time (minimum `realworld_timestamp`),
with stable ordering among games that share the same start.

Output columns:
  kalshi_event, realworld_timestamp, game_elapsed_seconds, period,
  team_1, team_2, team_1_win_prob_pct, team_2_win_prob_pct,
  team_1_volume, team_2_volume, winning_team
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List, Optional

import numpy as np
import pandas as pd


def _default_input_path(script_dir: str) -> str:
    return os.path.normpath(
        os.path.join(
            script_dir,
            "..",
            "1-GatheringPreprocessingTransformation",
            "GeneratedDataFiles",
            "all_games_merged_clean.csv",
        )
    )


def _default_output_path(script_dir: str) -> str:
    return os.path.normpath(
        os.path.join(
            script_dir,
            "GeneratedDataFiles",
            "all_games_merged_clean_per_second.csv",
        )
    )


def _week_file_name(base_output_path: str, week_number: int) -> str:
    base_dir = os.path.dirname(base_output_path)
    return os.path.join(base_dir, f"week_{week_number}_games.csv")


def _floor_to_second(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.floor("s")


def _target_timestamps(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """One timestamp per calendar second from start through end (floored to seconds), inclusive."""
    start_s = _floor_to_second(start)
    end_s = _floor_to_second(end)
    if end_s < start_s:
        return pd.DatetimeIndex([])
    return pd.date_range(start_s, end_s, freq="s")


def _format_eta(seconds: float) -> str:
    seconds_i = max(0, int(round(seconds)))
    mins, secs = divmod(seconds_i, 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours}h {mins}m {secs}s"
    if mins > 0:
        return f"{mins}m {secs}s"
    return f"{secs}s"


def _short_label(label: str, max_len: int = 28) -> str:
    if len(label) <= max_len:
        return label
    return f"{label[: max_len - 3]}..."


def _expand_team_series(
    team_df: pd.DataFrame,
    target_idx: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    team_df: rows for one (kalshi_event, team), sorted by realworld_timestamp, deduped.
    target_idx: per-second timeline for this game (shared across teams).
    """
    if len(target_idx) == 0:
        return pd.DataFrame()

    ts_seconds = (target_idx.asi8 // 10**9).astype(np.int64)

    ts_obs = team_df["realworld_timestamp"].dt.floor("s")
    x = (ts_obs.astype("int64") // 10**9).to_numpy(dtype=np.int64)

    # Last value wins at duplicate seconds.
    nu = team_df.copy()
    nu["_x"] = x
    nu = nu.drop_duplicates(subset=["_x"], keep="last")
    xu = nu["_x"].to_numpy(dtype=np.int64)

    out: dict = {}
    for col in ("win_prob_pct", "volume", "game_elapsed_seconds"):
        y = nu[col].astype(float).to_numpy()
        yi = np.interp(ts_seconds.astype(float), xu.astype(float), y)
        out[col] = np.rint(yi).astype(int)

    # Period: last observation with time <= target (in seconds).
    periods = nu["period"].to_numpy()
    idx = np.searchsorted(xu, ts_seconds, side="right") - 1
    idx = np.clip(idx, 0, len(xu) - 1)
    out["period"] = periods[idx]

    row = team_df.iloc[0]
    return pd.DataFrame(
        {
            "kalshi_event": row["kalshi_event"],
            "team": row["team"],
            "realworld_timestamp": target_idx,
            "game_elapsed_seconds": out["game_elapsed_seconds"],
            "period": out["period"],
            "win_prob_pct": out["win_prob_pct"],
            "volume": out["volume"],
            "team_won": row["team_won"],
        }
    )


def build_per_second_frame(
    df: pd.DataFrame,
    max_games: Optional[int] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    required = {
        "kalshi_event",
        "team",
        "realworld_timestamp",
        "game_elapsed_seconds",
        "period",
        "win_prob_pct",
        "volume",
        "team_won",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input missing columns: {sorted(missing)}")

    df = df.copy()
    df["realworld_timestamp"] = pd.to_datetime(df["realworld_timestamp"])
    df = df.dropna(
        subset=[
            "kalshi_event",
            "team",
            "realworld_timestamp",
            "game_elapsed_seconds",
            "win_prob_pct",
            "volume",
            "team_won",
        ]
    )

    # Game order: start time (min timestamp), then event id for tie-break.
    event_start = df.groupby("kalshi_event")["realworld_timestamp"].min().sort_values(
        kind="mergesort"
    )
    event_ids: List[str] = list(event_start.index.astype(str))
    if max_games is not None:
        event_ids = event_ids[: int(max_games)]

    chunks: List[pd.DataFrame] = []
    total_events = len(event_ids)
    total_elapsed_seconds = 0.0
    for idx, event_id in enumerate(event_ids, start=1):
        event_start_time = time.perf_counter()
        if show_progress:
            progress = f"({idx:>4}/{total_events})"
            if idx == 1:
                eta_text = "ETA --"
            else:
                avg_seconds_per_game = total_elapsed_seconds / (idx - 1)
                remaining_games = total_events - idx + 1
                eta_text = f"ETA {_format_eta(avg_seconds_per_game * remaining_games)}"
            event_label = _short_label(event_id)
            print(f"Game {progress} | {eta_text:<12} | {event_label}")
        g = df[df["kalshi_event"].astype(str) == event_id]
        t_min = g["realworld_timestamp"].min()
        t_max = g["realworld_timestamp"].max()
        target_idx = _target_timestamps(t_min, t_max)
        if len(target_idx) == 0:
            continue

        teams = sorted(g["team"].astype(str).unique().tolist())
        for team in teams:
            tg = g[g["team"].astype(str) == team].sort_values(
                "realworld_timestamp", kind="mergesort"
            )
            chunks.append(_expand_team_series(tg, target_idx))
        total_elapsed_seconds += time.perf_counter() - event_start_time

    if not chunks:
        return pd.DataFrame(columns=list(df.columns))
    return pd.concat(chunks, ignore_index=True)


def _pivot_two_team_rows(expanded: pd.DataFrame) -> pd.DataFrame:
    """
    Convert per-team per-second rows into one row per (kalshi_event, realworld_timestamp).
    Assumes standard two-team games.
    """
    if expanded.empty:
        return pd.DataFrame(
            columns=[
                "kalshi_event",
                "realworld_timestamp",
                "game_elapsed_seconds",
                "period",
                "team_1",
                "team_2",
                "team_1_win_prob_pct",
                "team_2_win_prob_pct",
                "team_1_volume",
                "team_2_volume",
                "winning_team",
            ]
        )

    expanded = expanded.sort_values(
        ["kalshi_event", "realworld_timestamp", "team"], kind="mergesort"
    ).reset_index(drop=True)

    def _per_game(g: pd.DataFrame) -> pd.DataFrame:
        teams = sorted(g["team"].astype(str).unique().tolist())
        if len(teams) != 2:
            raise ValueError(
                f"Expected exactly 2 teams for {g['kalshi_event'].iloc[0]}, found {teams}"
            )
        team_1, team_2 = teams[0], teams[1]

        left = g[g["team"].astype(str) == team_1][
            [
                "kalshi_event",
                "realworld_timestamp",
                "game_elapsed_seconds",
                "period",
                "win_prob_pct",
                "volume",
                "team_won",
            ]
        ].rename(
            columns={
                "win_prob_pct": "team_1_win_prob_pct",
                "volume": "team_1_volume",
                "team_won": "team_1_won",
            }
        )
        right = g[g["team"].astype(str) == team_2][
            ["realworld_timestamp", "win_prob_pct", "volume", "team_won"]
        ].rename(
            columns={
                "win_prob_pct": "team_2_win_prob_pct",
                "volume": "team_2_volume",
                "team_won": "team_2_won",
            }
        )

        out = left.merge(right, on="realworld_timestamp", how="inner")
        out["team_1"] = team_1
        out["team_2"] = team_2
        out["winning_team"] = np.where(out["team_1_won"] == 1, team_1, team_2)
        return out[
            [
                "kalshi_event",
                "realworld_timestamp",
                "game_elapsed_seconds",
                "period",
                "team_1",
                "team_2",
                "team_1_win_prob_pct",
                "team_2_win_prob_pct",
                "team_1_volume",
                "team_2_volume",
                "winning_team",
            ]
        ]

    pieces: List[pd.DataFrame] = []
    skipped_events: List[str] = []
    for event_id, g in expanded.groupby("kalshi_event", sort=False):
        teams = sorted(g["team"].astype(str).unique().tolist())
        if len(teams) != 2:
            skipped_events.append(f"{event_id} ({teams})")
            continue
        pieces.append(_per_game(g))

    if skipped_events:
        print(
            "Skipped malformed games in pivot step "
            f"(expected 2 teams): {len(skipped_events)}"
        )
        preview = ", ".join(skipped_events[:5])
        if preview:
            suffix = " ..." if len(skipped_events) > 5 else ""
            print(f"Examples: {preview}{suffix}")

    if not pieces:
        return pd.DataFrame(
            columns=[
                "kalshi_event",
                "realworld_timestamp",
                "game_elapsed_seconds",
                "period",
                "team_1",
                "team_2",
                "team_1_win_prob_pct",
                "team_2_win_prob_pct",
                "team_1_volume",
                "team_2_volume",
                "winning_team",
            ]
        )
    return pd.concat(pieces, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--input",
        default=_default_input_path(script_dir),
        help="Path to all_games_merged_clean.csv",
    )
    parser.add_argument(
        "--output",
        default=_default_output_path(script_dir),
        help="Path to write per-second CSV",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Only process the first N games by start time (for debugging)",
    )
    args = parser.parse_args()

    inp = args.input if args.input.startswith("/") else os.path.normpath(
        os.path.join(script_dir, args.input)
    )
    out = args.output if args.output.startswith("/") else os.path.normpath(
        os.path.join(script_dir, args.output)
    )

    os.makedirs(os.path.dirname(out), exist_ok=True)

    df = pd.read_csv(inp)
    if df.empty:
        pd.DataFrame().to_csv(out, index=False)
        print(f"Input is empty. Wrote empty file: {out}")
        return

    df = df.copy()
    df["realworld_timestamp"] = pd.to_datetime(df["realworld_timestamp"])

    # Determine game order by event start, then assign each game to its start week.
    event_start = (
        df.groupby("kalshi_event")["realworld_timestamp"]
        .min()
        .sort_values(kind="mergesort")
    )
    event_ids: List[str] = list(event_start.index.astype(str))
    if args.max_games is not None:
        event_ids = event_ids[: int(args.max_games)]
        event_start = event_start.loc[event_ids]

    event_week_start = event_start.dt.to_period("W-SUN").dt.start_time

    wrote_files = 0
    for week_idx, week_start in enumerate(
        event_week_start.sort_values(kind="mergesort").unique(), start=1
    ):
        week_events = event_week_start[event_week_start == week_start].index.astype(str)
        week_df = df[df["kalshi_event"].astype(str).isin(week_events)].copy()
        if week_df.empty:
            continue

        print(f"\nProcessing week {pd.Timestamp(week_start).strftime('%Y-%m-%d')} ({len(week_events)} games)")
        expanded = build_per_second_frame(
            week_df,
            max_games=None,
            show_progress=True,
        )
        pivoted = _pivot_two_team_rows(expanded)

        week_out = _week_file_name(out, week_idx)
        pivoted.to_csv(week_out, index=False)
        wrote_files += 1
        print(f"Week complete. Wrote file: {week_out} ({len(pivoted)} rows)")

    print(f"\nDone. Wrote {wrote_files} weekly file(s) as each week completed.")


if __name__ == "__main__":
    main()
