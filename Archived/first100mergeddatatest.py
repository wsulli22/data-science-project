#!/usr/bin/env python3
"""
first100mergeddatatest.py

Processes the first 100 games in kalshi_espn_game_mappings.csv,
fetches and merges Kalshi + ESPN data for both teams in each game,
and outputs a single combined CSV: first_100_games_merged.csv

No intermediate per-game CSV files are created.
"""

import os
import sys
import time
import pandas as pd

from create_game_data_csv import lookup_espn_game_id
from get_kalshi_game_data import get_kalshi_game_data

# Import ESPN internal helpers directly to avoid the file-saving side effect
# in get_espn_game_timestamp_mapping
from get_espn_game_timestamp_mapings import (
    _fetch_all_plays,
    _parse_wallclock,
    _compute_game_elapsed,
    _validate_monotonicity,
)

MAPPINGS_CSV = "kalshi_espn_game_mappings.csv"
KALSHI_GAMES_TXT = "list_of_kalshi_games.txt"
OUTPUT_CSV = "first_100_games_merged.csv"
NUM_GAMES = 100


def fetch_espn_timestamp_mapping(espn_game_id):
    """
    Same logic as get_espn_game_timestamp_mapping but does NOT save a CSV file.
    """
    game_id = str(espn_game_id)
    plays = _fetch_all_plays(game_id)
    if not plays:
        raise ValueError(f"No plays returned for game {game_id}")

    rows = []
    for play in plays:
        wallclock_str = play.get("wallclock")
        period_info = play.get("period", {})
        clock_info = play.get("clock", {})

        period_number = period_info.get("number")
        clock_display = clock_info.get("displayValue")
        clock_value = clock_info.get("value")

        if wallclock_str is None or period_number is None or clock_value is None:
            continue

        wallclock_ts = _parse_wallclock(wallclock_str)
        if wallclock_ts is None:
            continue

        game_elapsed = _compute_game_elapsed(period_number, clock_value)
        rows.append({
            "wallclock_ts": wallclock_ts,
            "period": period_number,
            "clock_display": clock_display or "",
            "game_elapsed_seconds": game_elapsed,
        })

    if not rows:
        raise ValueError(f"No valid plays parsed for game {game_id}")

    df = pd.DataFrame(rows)
    df = df.drop_duplicates().sort_values("wallclock_ts").reset_index(drop=True)
    df = _validate_monotonicity(df)
    return df


def load_team_lookup():
    """
    Build a dict mapping kalshi_game_id -> (team1, team2) from list_of_kalshi_games.txt.
    """
    lookup = {}
    if not os.path.exists(KALSHI_GAMES_TXT):
        print(f"ERROR: {KALSHI_GAMES_TXT} not found. Run get_list_of_kalshi_games.py first.")
        sys.exit(1)

    with open(KALSHI_GAMES_TXT, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                lookup[parts[0]] = (parts[1], parts[2])
    return lookup


def build_game_team_df(kalshi_event_id, team_abbr, espn_raw):
    """
    Fetch Kalshi data for one game/team and merge with pre-fetched ESPN data.
    Returns a DataFrame or None on failure.
    """
    kalshi_event_id = kalshi_event_id.upper()
    team_abbr = team_abbr.upper()

    # Fetch Kalshi candlestick data
    kalshi_raw = get_kalshi_game_data(kalshi_event_id, team_abbr)
    kalshi_raw["wallclock_ts"] = pd.to_datetime(kalshi_raw["wallclock_ts"]).dt.tz_localize(None)
    kalshi_raw = kalshi_raw.sort_values("wallclock_ts").reset_index(drop=True)

    # Filter Kalshi candles to game window
    game_start = espn_raw["wallclock_ts"].min()
    game_end = espn_raw["wallclock_ts"].max()
    kalshi = kalshi_raw[
        (kalshi_raw["wallclock_ts"] >= game_start) &
        (kalshi_raw["wallclock_ts"] <= game_end)
    ].copy().reset_index(drop=True)

    # Backward merge
    merged = pd.merge_asof(
        kalshi[["wallclock_ts", "win_prob", "result"]].copy(),
        espn_raw[["wallclock_ts", "period", "game_elapsed_seconds"]],
        on="wallclock_ts",
        direction="backward",
    ).dropna(subset=["game_elapsed_seconds"])

    if len(merged) == 0:
        return None

    # Prepare output columns
    merged["win_prob_pct"] = (merged["win_prob"] * 100.0).round(2)
    merged["team_won"] = (merged["result"] == "yes").astype(int)
    merged["kalshi_event"] = kalshi_event_id
    merged["team"] = team_abbr

    return merged[["kalshi_event", "team", "game_elapsed_seconds",
                    "period", "win_prob_pct", "team_won"]]


def main():
    if not os.path.exists(MAPPINGS_CSV):
        print(f"ERROR: {MAPPINGS_CSV} not found.")
        sys.exit(1)

    mappings = pd.read_csv(MAPPINGS_CSV)
    first_100 = mappings.head(NUM_GAMES)
    print(f"Loaded {len(first_100)} games from {MAPPINGS_CSV}")

    team_lookup = load_team_lookup()
    print(f"Loaded team info for {len(team_lookup)} games from {KALSHI_GAMES_TXT}")
    print("=" * 70)

    all_dfs = []
    success_count = 0
    fail_count = 0
    skip_count = 0
    errors = []

    for idx, row in first_100.iterrows():
        kalshi_game_id = row["kalshi_game_id"]
        game_num = idx + 1

        if kalshi_game_id not in team_lookup:
            print(f"[{game_num}/{NUM_GAMES}] SKIP {kalshi_game_id} — no team info")
            skip_count += 1
            continue

        team1, team2 = team_lookup[kalshi_game_id]
        teams = [t for t in [team1, team2] if t]

        if not teams:
            print(f"[{game_num}/{NUM_GAMES}] SKIP {kalshi_game_id} — no team abbreviations")
            skip_count += 1
            continue

        # Look up ESPN game ID once per game
        espn_game_id = lookup_espn_game_id(kalshi_game_id)
        if espn_game_id is None:
            print(f"[{game_num}/{NUM_GAMES}] SKIP {kalshi_game_id} — no ESPN game ID")
            skip_count += 1
            continue

        # Fetch ESPN data once per game (no file saved)
        try:
            espn_raw = fetch_espn_timestamp_mapping(espn_game_id)
            espn_raw["wallclock_ts"] = pd.to_datetime(espn_raw["wallclock_ts"]).dt.tz_localize(None)
            espn_raw = espn_raw.sort_values("wallclock_ts").reset_index(drop=True)
        except Exception as e:
            fail_count += len(teams)
            for team in teams:
                errors.append(f"{kalshi_game_id} - {team}: ESPN fetch failed: {e}")
            print(f"[{game_num}/{NUM_GAMES}] ✗ ESPN fetch failed for {kalshi_game_id}: {e}")
            continue

        for team in teams:
            print(f"[{game_num}/{NUM_GAMES}] {kalshi_game_id} - {team} ... ", end="", flush=True)
            try:
                df = build_game_team_df(kalshi_game_id, team, espn_raw)
                if df is not None and len(df) > 0:
                    all_dfs.append(df)
                    success_count += 1
                    print(f"✓ {len(df)} rows")
                else:
                    fail_count += 1
                    errors.append(f"{kalshi_game_id} - {team}: no aligned data")
                    print("✗ no aligned data")
            except Exception as e:
                fail_count += 1
                errors.append(f"{kalshi_game_id} - {team}: {e}")
                print(f"✗ {e}")

            time.sleep(0.5)

    # Combine everything into one CSV
    print("\n" + "=" * 70)
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved {OUTPUT_CSV}  ({len(combined)} total rows, {success_count} game/team combos)")
    else:
        print("No data collected — nothing to save.")

    print(f"\n  Successful : {success_count}")
    print(f"  Failed     : {fail_count}")
    print(f"  Skipped    : {skip_count}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for err in errors:
            print(f"  - {err}")

    print("\nDone!")


if __name__ == "__main__":
    main()
