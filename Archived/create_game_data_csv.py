#!/usr/bin/env python3
"""
create_game_data_csv.py

Standalone script to create a game_data CSV file from a Kalshi event ID and team.

Usage:
    python create_game_data_csv.py <kalshi_event_id> <team_abbr> [output_dir]

Example:
    python create_game_data_csv.py KXNCAAMBGAME-26FEB10MILWIUIN MILW

This script:
1. Looks up the ESPN game ID from kalshi_espn_game_mappings.csv
2. Fetches Kalshi candlestick data
3. Fetches ESPN play-by-play timestamp mappings
4. Aligns Kalshi data to game clock time using backward merge
5. Outputs a CSV with columns: kalshi_event, team, game_elapsed_seconds, 
   period, win_prob_pct, team_won
"""

import sys
import os
import pandas as pd
import numpy as np

# Import required modules
from get_kalshi_game_data import get_kalshi_game_data
from get_espn_game_timestamp_mapings import get_espn_game_timestamp_mapping


MAPPINGS_CSV = "kalshi_espn_game_mappings.csv"


def lookup_espn_game_id(kalshi_event_id: str) -> str | None:
    """
    Look up the ESPN game ID for a given Kalshi event ID.
    
    Args:
        kalshi_event_id: Kalshi event ticker (e.g., "KXNCAAMBGAME-26FEB10MILWIUIN")
    
    Returns:
        ESPN game ID string, or None if not found
    """
    if not os.path.exists(MAPPINGS_CSV):
        raise FileNotFoundError(
            f"Mapping file not found: {MAPPINGS_CSV}\n"
            f"Run kalshi_espn_game_mapper2.py first to generate mappings."
        )
    
    mappings = pd.read_csv(MAPPINGS_CSV)
    match = mappings[mappings["kalshi_game_id"] == kalshi_event_id.upper()]
    
    if len(match) == 0:
        return None
    
    return str(match.iloc[0]["espn_game_id"])


def create_game_data_csv(kalshi_event_id: str, team_abbr: str, 
                         output_dir: str | None = None, verbose: bool = True) -> str:
    """
    Create a game_data CSV file for a given Kalshi event and team.
    
    Args:
        kalshi_event_id: Kalshi event ticker (e.g., "KXNCAAMBGAME-26FEB10MILWIUIN")
        team_abbr: Team abbreviation (e.g., "MILW")
        output_dir: Directory to save CSV. If None, saves in current directory.
        verbose: Whether to print progress messages
    
    Returns:
        Path to the created CSV file
    
    Raises:
        ValueError: If ESPN game ID not found or data fetch fails
    """
    kalshi_event_id = kalshi_event_id.upper()
    team_abbr = team_abbr.upper()
    
    if verbose:
        print(f"Creating game data CSV for {kalshi_event_id} - {team_abbr}")
        print("=" * 70)
    
    # Step 1: Look up ESPN game ID
    if verbose:
        print("\n1. Looking up ESPN game ID...")
    espn_game_id = lookup_espn_game_id(kalshi_event_id)
    if espn_game_id is None:
        raise ValueError(
            f"ESPN game ID not found for Kalshi event: {kalshi_event_id}\n"
            f"Check that this game is in {MAPPINGS_CSV}"
        )
    if verbose:
        print(f"   ✓ Found ESPN game ID: {espn_game_id}")
    
    # Step 2: Fetch Kalshi candlestick data
    if verbose:
        print("\n2. Fetching Kalshi candlestick data...")
    try:
        kalshi_raw = get_kalshi_game_data(kalshi_event_id, team_abbr)
        kalshi_raw["wallclock_ts"] = (
            pd.to_datetime(kalshi_raw["wallclock_ts"]).dt.tz_localize(None)
        )
        kalshi_raw = kalshi_raw.sort_values("wallclock_ts").reset_index(drop=True)
        if verbose:
            print(f"   ✓ Fetched {len(kalshi_raw)} Kalshi candles")
    except Exception as e:
        raise ValueError(f"Failed to fetch Kalshi data: {e}")
    
    # Step 3: Fetch ESPN play-by-play timestamp mappings
    if verbose:
        print("\n3. Fetching ESPN play-by-play data...")
    try:
        espn_raw = get_espn_game_timestamp_mapping(espn_game_id)
        espn_raw["wallclock_ts"] = (
            pd.to_datetime(espn_raw["wallclock_ts"]).dt.tz_localize(None)
        )
        espn_raw = espn_raw.sort_values("wallclock_ts").reset_index(drop=True)
        if verbose:
            print(f"   ✓ Fetched {len(espn_raw)} ESPN plays")
    except Exception as e:
        raise ValueError(f"Failed to fetch ESPN data: {e}")
    
    # Step 4: Filter Kalshi candles to only those during the game
    if verbose:
        print("\n4. Aligning data to game clock time...")
    game_start = espn_raw["wallclock_ts"].min()
    game_end = espn_raw["wallclock_ts"].max()
    kalshi = kalshi_raw[
        (kalshi_raw["wallclock_ts"] >= game_start) &
        (kalshi_raw["wallclock_ts"] <= game_end)
    ].copy().reset_index(drop=True)
    
    if verbose:
        print(f"   ✓ {len(kalshi)} Kalshi candles during game "
              f"(dropped {len(kalshi_raw) - len(kalshi)} pre/post-game)")
    
    # Step 5: Merge using backward merge strategy (recommended)
    merged = pd.merge_asof(
        kalshi[["wallclock_ts", "win_prob", "result"]].copy(),
        espn_raw[["wallclock_ts", "period", "game_elapsed_seconds"]],
        on="wallclock_ts",
        direction="backward",
    ).dropna(subset=["game_elapsed_seconds"])
    
    if len(merged) == 0:
        raise ValueError(
            "No data points after alignment. Check that Kalshi and ESPN data "
            "have overlapping timestamps."
        )
    
    if verbose:
        print(f"   ✓ Successfully aligned {len(merged)} data points")
    
    # Step 6: Prepare output DataFrame
    final = merged.copy()
    final["win_prob_pct"] = final["win_prob"] * 100.0
    final["team_won"] = (final["result"] == "yes").astype(int)
    final["kalshi_event"] = kalshi_event_id
    final["team"] = team_abbr
    
    # Select output columns
    OUTPUT_COLS = [
        "kalshi_event",
        "team",
        "game_elapsed_seconds",
        "period",
        "win_prob_pct",
        "team_won",
    ]
    output_df = final[OUTPUT_COLS].copy()
    output_df["win_prob_pct"] = output_df["win_prob_pct"].round(2)
    
    # Step 7: Save to CSV
    if output_dir is None:
        output_dir = os.getcwd()
    
    output_filename = f"game_data_{kalshi_event_id}_{team_abbr}.csv"
    output_path = os.path.join(output_dir, output_filename)
    output_df.to_csv(output_path, index=False)
    
    if verbose:
        print(f"\n5. Saved CSV:")
        print(f"   ✓ {output_path}")
        print(f"   ✓ {len(output_df)} rows")
        print(f"   ✓ Game time range: {output_df['game_elapsed_seconds'].min():.0f} - "
              f"{output_df['game_elapsed_seconds'].max():.0f} seconds")
        print(f"   ✓ Team {team_abbr} won: {'YES' if output_df['team_won'].iloc[0] == 1 else 'NO'}")
        print("\n" + "=" * 70)
        print("DONE")
        print("=" * 70)
    
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        print("\nUsage:")
        print(f"  python {sys.argv[0]} <kalshi_event_id> <team_abbr> [output_dir]")
        print("\nExample:")
        print(f"  python {sys.argv[0]} KXNCAAMBGAME-26FEB10MILWIUIN MILW")
        sys.exit(1)
    
    kalshi_event_id = sys.argv[1]
    team_abbr = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        output_path = create_game_data_csv(kalshi_event_id, team_abbr, output_dir)
        print(f"\n✓ Successfully created: {output_path}")
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
