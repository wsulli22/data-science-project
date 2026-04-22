#!/usr/bin/env python3
"""
data_documentation.py

Summary statistics describing the cleaned week-by-week dataset under
`0-Data/week_*_games.csv`. Prints (and optionally writes) a plain-text
documentation report covering:

  * Number of per-second data points
  * Number of unique games
  * Percentage of games occurring on each day of the week
  * Date range (earliest game start -> latest game start)
  * Percentage of games that ended in regulation vs. overtime
    (overtime total, and segmented by OT1 / OT2 / OT3)
  * Average length of regulation games, all games, and
    OT1 / OT2 / OT3 games (both in game-clock seconds and real-world minutes)

Each weekly CSV is streamed and reduced to one row per `kalshi_event`
so the full dataset never has to be held in memory at once.
"""

import glob
import os
from typing import Iterable

import numpy as np
import pandas as pd


# Rank periods so we can tell how far a game progressed by taking the max.
PERIOD_RANK = {
    "firstHalf": 0,
    "halfTime": 1,
    "secondHalf": 2,
    "preOT1": 3,
    "OT1": 4,
    "preOT2": 5,
    "OT2": 6,
    "preOT3": 7,
    "OT3": 8,
}

# Day-of-week order used when reporting percentages.
DAYS_OF_WEEK = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def _resolve_path(p: str) -> str:
    """Resolve paths relative to this script file."""
    script_dir = os.path.dirname(__file__)
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(script_dir, p))


def _classify_furthest_period(max_rank: int) -> str:
    """Return a bucket label from the max PERIOD_RANK seen in a game."""
    if max_rank >= PERIOD_RANK["OT3"]:
        return "OT3"
    if max_rank >= PERIOD_RANK["OT2"]:
        return "OT2"
    if max_rank >= PERIOD_RANK["OT1"]:
        return "OT1"
    return "Regulation"


def _reduce_week_file(path: str) -> pd.DataFrame:
    """Return one row per `kalshi_event` with stats needed for the report."""
    df = pd.read_csv(
        path,
        usecols=["kalshi_event", "realworld_timestamp", "game_elapsed_seconds", "period"],
        parse_dates=["realworld_timestamp"],
    )

    # Unknown periods (bad rows, header artifacts, etc.) get -1 and are ignored
    # when we take the max.
    df["period_rank"] = df["period"].map(PERIOD_RANK).fillna(-1).astype(int)

    grouped = df.groupby("kalshi_event", sort=False).agg(
        n_rows=("kalshi_event", "size"),
        start_ts=("realworld_timestamp", "min"),
        end_ts=("realworld_timestamp", "max"),
        max_elapsed_seconds=("game_elapsed_seconds", "max"),
        max_period_rank=("period_rank", "max"),
    )
    return grouped.reset_index()


def _combine_duplicated_games(per_game: pd.DataFrame) -> pd.DataFrame:
    """If the same `kalshi_event` appears in multiple week files, merge rows."""
    if per_game["kalshi_event"].is_unique:
        return per_game

    return (
        per_game.groupby("kalshi_event", sort=False)
        .agg(
            n_rows=("n_rows", "sum"),
            start_ts=("start_ts", "min"),
            end_ts=("end_ts", "max"),
            max_elapsed_seconds=("max_elapsed_seconds", "max"),
            max_period_rank=("max_period_rank", "max"),
        )
        .reset_index()
    )


def _format_pct(count: int, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{100.0 * count / total:.2f}%"


def _format_seconds(seconds: float) -> str:
    if np.isnan(seconds):
        return "n/a"
    minutes = seconds / 60.0
    return f"{seconds:,.1f} sec ({minutes:,.2f} min)"


def _build_report_lines(per_game: pd.DataFrame, data_point_count: int) -> list[str]:
    """Assemble the human-readable report as a list of lines."""
    total_games = len(per_game)

    # ── classify games by furthest period reached ────────────────────────────
    per_game = per_game.copy()
    per_game["end_bucket"] = per_game["max_period_rank"].map(_classify_furthest_period)

    # Real-world game duration in seconds (min->max of realworld_timestamp).
    per_game["realworld_duration_seconds"] = (
        per_game["end_ts"] - per_game["start_ts"]
    ).dt.total_seconds()

    # Day of week of game start.
    per_game["start_day"] = per_game["start_ts"].dt.day_name()

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("DATA DOCUMENTATION")
    lines.append("=" * 72)

    # ── top-line counts ──────────────────────────────────────────────────────
    lines.append("")
    lines.append("COUNTS")
    lines.append("-" * 72)
    lines.append(f"Number of per-second data points : {data_point_count:,}")
    lines.append(f"Number of unique games           : {total_games:,}")

    # ── date range ───────────────────────────────────────────────────────────
    earliest = per_game["start_ts"].min()
    latest = per_game["start_ts"].max()
    lines.append("")
    lines.append("DATE RANGE OF GAMES (by game start time)")
    lines.append("-" * 72)
    lines.append(f"Earliest game start : {earliest}")
    lines.append(f"Latest game start   : {latest}")
    if pd.notna(earliest) and pd.notna(latest):
        span_days = (latest - earliest).total_seconds() / 86400.0
        lines.append(f"Span                : {span_days:,.2f} days")

    # ── day-of-week breakdown ────────────────────────────────────────────────
    lines.append("")
    lines.append("PERCENTAGE OF GAMES BY DAY OF WEEK (based on game start date)")
    lines.append("-" * 72)
    day_counts = per_game["start_day"].value_counts()
    for day in DAYS_OF_WEEK:
        count = int(day_counts.get(day, 0))
        lines.append(f"  {day:<9s} : {count:>5d}  ({_format_pct(count, total_games)})")

    # ── ending-period breakdown ──────────────────────────────────────────────
    bucket_counts = per_game["end_bucket"].value_counts()
    regulation_count = int(bucket_counts.get("Regulation", 0))
    ot1_count = int(bucket_counts.get("OT1", 0))
    ot2_count = int(bucket_counts.get("OT2", 0))
    ot3_count = int(bucket_counts.get("OT3", 0))
    overtime_all_count = ot1_count + ot2_count + ot3_count

    lines.append("")
    lines.append("PERCENTAGE OF GAMES BY ENDING PERIOD")
    lines.append("-" * 72)
    lines.append(
        f"  Regulation (ended in 2nd half) : {regulation_count:>5d}  "
        f"({_format_pct(regulation_count, total_games)})"
    )
    lines.append(
        f"  Overtime (any OT)              : {overtime_all_count:>5d}  "
        f"({_format_pct(overtime_all_count, total_games)})"
    )
    lines.append(
        f"    - ended in OT1               : {ot1_count:>5d}  "
        f"({_format_pct(ot1_count, total_games)})"
    )
    lines.append(
        f"    - ended in OT2               : {ot2_count:>5d}  "
        f"({_format_pct(ot2_count, total_games)})"
    )
    lines.append(
        f"    - ended in OT3               : {ot3_count:>5d}  "
        f"({_format_pct(ot3_count, total_games)})"
    )

    # ── average length ───────────────────────────────────────────────────────
    def _avg_elapsed(mask: pd.Series) -> float:
        subset = per_game.loc[mask, "max_elapsed_seconds"]
        return float(subset.mean()) if len(subset) else float("nan")

    def _avg_realworld(mask: pd.Series) -> float:
        subset = per_game.loc[mask, "realworld_duration_seconds"]
        return float(subset.mean()) if len(subset) else float("nan")

    masks = {
        "All games": pd.Series(True, index=per_game.index),
        "Regulation only": per_game["end_bucket"] == "Regulation",
        "Overtime (any)": per_game["end_bucket"] != "Regulation",
        "OT1 (ended in OT1)": per_game["end_bucket"] == "OT1",
        "OT2 (ended in OT2)": per_game["end_bucket"] == "OT2",
        "OT3 (ended in OT3)": per_game["end_bucket"] == "OT3",
    }

    lines.append("")
    lines.append("AVERAGE GAME LENGTH")
    lines.append("-" * 72)
    lines.append(f"  {'Bucket':<22s} {'n':>5s}   {'game-clock length':<32s}   real-world duration")
    for label, mask in masks.items():
        n = int(mask.sum())
        elapsed = _avg_elapsed(mask)
        realworld = _avg_realworld(mask)
        lines.append(
            f"  {label:<22s} {n:>5d}   "
            f"{_format_seconds(elapsed):<32s}   {_format_seconds(realworld)}"
        )

    lines.append("")
    lines.append("=" * 72)
    return lines


def generate_data_documentation(
    input_glob: str = "../0-Data/week_*_games.csv",
    output_filename: str | None = "data_documentation.txt",
    input_files: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Compute dataset statistics and print/write a documentation report.

    Args:
        input_glob: Glob pattern (relative to this script) used to find the
            weekly per-second CSVs when `input_files` is not provided.
        output_filename: If not None, the report is also written to this path
            alongside this script.
        input_files: Optional explicit list of CSV paths to process. Useful for
            tests or for re-running against a subset of weeks.

    Returns:
        DataFrame with one row per unique `kalshi_event` and the raw
        aggregates used to build the report.
    """
    print("\nGENERATING DATA DOCUMENTATION\n")

    if input_files is None:
        pattern = _resolve_path(input_glob)
        files = sorted(glob.glob(pattern))
    else:
        files = [_resolve_path(p) for p in input_files]

    if not files:
        raise FileNotFoundError(
            f"No input CSVs found. Looked for '{input_glob}' relative to "
            f"{os.path.dirname(__file__)}."
        )

    per_game_frames: list[pd.DataFrame] = []
    total_data_points = 0

    for path in files:
        print(f"  scanning {os.path.basename(path)} ...", flush=True)
        reduced = _reduce_week_file(path)
        total_data_points += int(reduced["n_rows"].sum())
        per_game_frames.append(reduced)

    per_game = pd.concat(per_game_frames, ignore_index=True)
    per_game = _combine_duplicated_games(per_game)

    lines = _build_report_lines(per_game, total_data_points)
    report = "\n".join(lines)
    print(report)

    if output_filename is not None:
        out_path = _resolve_path(output_filename)
        with open(out_path, "w") as f:
            f.write(report + "\n")
        print(f"\nSaved report -> {out_path}")

    return per_game


if __name__ == "__main__":
    generate_data_documentation()
