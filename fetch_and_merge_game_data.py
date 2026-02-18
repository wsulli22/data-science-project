"""
fetch_and_merge_game_data.py

Provides a pipeline used by main.py:
  - fetch_and_merge_all_games(num_games)
    For each mapped game, fetches ESPN play-by-play + Kalshi candlestick data,
    merges them, and saves the combined result to a CSV.
"""

import os
import pandas as pd

from get_kalshi_game_data import get_kalshi_game_data

# Import ESPN internal helpers directly to avoid the file-saving side effect
# in get_espn_game_timestamp_mapping
from get_espn_game_timestamp_mapings import (
    _fetch_all_plays,
    _parse_wallclock,
    _compute_game_elapsed,
    _validate_monotonicity,
)

# Team lookup (loaded once, lazily per file)
_team_lookup_cache = {}


def _load_team_lookup(kalshi_games_file="GeneratedDataFiles/list_of_kalshi_game.txt"):
    """Build a dict mapping kalshi_game_id -> (team1, team2) from list_of_kalshi_games.txt."""
    # Cache lookups per file to avoid reloading
    if kalshi_games_file in _team_lookup_cache:
        return _team_lookup_cache[kalshi_games_file]

    _team_lookup = {}
    if not os.path.exists(kalshi_games_file):
        print(f"ERROR: {kalshi_games_file} not found.")
        _team_lookup_cache[kalshi_games_file] = _team_lookup
        return _team_lookup

    with open(kalshi_games_file, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                _team_lookup[parts[0]] = (parts[1], parts[2])
    
    _team_lookup_cache[kalshi_games_file] = _team_lookup
    return _team_lookup


# ---------------------------------------------------------------------------
# Single-game processor
# ---------------------------------------------------------------------------

def _fetch_espn_df(espn_game_id):
    """Fetch ESPN play-by-play timestamps for a game. Returns (df, discarded_count)."""
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
    before_mono = len(df)
    df = _validate_monotonicity(df)
    espn_discarded = before_mono - len(df)
    df["wallclock_ts"] = pd.to_datetime(df["wallclock_ts"]).dt.tz_localize(None)
    df = df.sort_values("wallclock_ts").reset_index(drop=True)

    return df, espn_discarded


def _fetch_kalshi_teams(kalshi_game_id, team_lookup):
    """Fetch Kalshi candlestick data for both teams. Returns (list_of_(team_abbr, df), diagnostic_or_None)."""
    if kalshi_game_id not in team_lookup:
        return [], "not in team lookup"

    team1, team2 = team_lookup[kalshi_game_id]
    teams = [t for t in [team1, team2] if t]

    if not teams:
        return [], "teams are empty in lookup"

    results = []
    errors = []
    for team in teams:
        try:
            kalshi_raw = get_kalshi_game_data(kalshi_game_id, team)
            kalshi_raw["wallclock_ts"] = pd.to_datetime(
                kalshi_raw["wallclock_ts"]
            ).dt.tz_localize(None)
            kalshi_raw = kalshi_raw.sort_values("wallclock_ts").reset_index(drop=True)
            results.append((team, kalshi_raw))
        except Exception as e:
            errors.append(f"{team}: {e}")
    
    if not results and errors:
        return [], f"Kalshi API failed ({'; '.join(errors)})"
    return results, None


def _merge_espn_kalshi(espn_df, kalshi_teams, kalshi_game_id, espn_discarded):
    """
    Merge ESPN and Kalshi data for one game (pure function, no globals).

    Returns:
        (DataFrame | None, good_count, discarded_count, stale_dropped,
         ot_dropped, ot_before, diagnostic_msg)
    """
    if espn_df is None:
        return None, 0, 0, 0, 0, 0, "No ESPN data"
    if not kalshi_teams:
        return None, 0, 0, 0, 0, 0, "No Kalshi teams found"

    all_dfs = []
    total_kalshi_in_window = 0
    total_stale = 0

    for team_abbr, kalshi_raw in kalshi_teams:
        game_start = espn_df["wallclock_ts"].min()
        game_end = espn_df["wallclock_ts"].max()
        kalshi = kalshi_raw[
            (kalshi_raw["wallclock_ts"] >= game_start)
            & (kalshi_raw["wallclock_ts"] <= game_end)
        ].copy().reset_index(drop=True)

        if len(kalshi) == 0:
            continue

        total_kalshi_in_window += len(kalshi)

        merged = pd.merge_asof(
            kalshi[["wallclock_ts", "win_prob", "volume", "result"]].copy(),
            espn_df[["wallclock_ts", "game_elapsed_seconds"]],
            on="wallclock_ts",
            direction="backward",
        ).dropna(subset=["game_elapsed_seconds"])

        if len(merged) == 0:
            continue

        # --- Deduplicate stale prices ---
        # When consecutive candles have the same win_prob and zero volume,
        # the price is just a stale repeat (no trades occurred). Collapse
        # these runs into a single observation to avoid inflating counts.
        before_dedup = len(merged)
        same_prob = merged["win_prob"] == merged["win_prob"].shift()
        zero_vol = merged["volume"] == 0
        stale = same_prob & zero_vol
        merged = merged[~stale].reset_index(drop=True)
        n_stale = before_dedup - len(merged)
        total_stale += n_stale

        merged["win_prob_pct"] = (merged["win_prob"] * 100.0).round(2)
        merged["team_won"] = (merged["result"] == "yes").astype(int)
        merged["kalshi_event"] = kalshi_game_id
        merged["team"] = team_abbr

        all_dfs.append(
            merged[
                [
                    "kalshi_event",
                    "team",
                    "game_elapsed_seconds",
                    "win_prob_pct",
                    "volume",
                    "team_won",
                ]
            ]
        )

    if not all_dfs:
        if total_kalshi_in_window == 0:
            return None, 0, 0, 0, 0, 0, "No Kalshi data in game time window"
        else:
            return None, 0, 0, 0, 0, 0, f"Merge failed ({total_kalshi_in_window} Kalshi rows, no matches)"

    result = pd.concat(all_dfs, ignore_index=True)

    n_before = len(result)
    result = result[result["game_elapsed_seconds"] <= 2400].copy()
    n_ot = n_before - len(result)

    good = len(result)
    discarded = (total_kalshi_in_window - good) + espn_discarded
    return result, good, discarded, total_stale, n_ot, n_before, None


def _process_single_game(kalshi_game_id, espn_game_id, team_lookup):
    """
    End-to-end fetch + merge for one game.

    Returns:
        (kalshi_game_id, merged_df_or_None, good, discarded, stale_dropped,
         ot_dropped, ot_before, error_msg_or_None, diagnostic_msg_or_None)
    """
    try:
        espn_df, espn_discarded = _fetch_espn_df(espn_game_id)
        kalshi_teams, fetch_diag = _fetch_kalshi_teams(kalshi_game_id, team_lookup)
        if fetch_diag:
            return (kalshi_game_id, None, 0, 0, 0, 0, 0, None, fetch_diag)
        merged, good, discarded, stale_dropped, ot_dropped, ot_before, diagnostic = \
            _merge_espn_kalshi(espn_df, kalshi_teams, kalshi_game_id, espn_discarded)
        return (kalshi_game_id, merged, good, discarded, stale_dropped,
                ot_dropped, ot_before, None, diagnostic)
    except Exception as e:
        return (kalshi_game_id, None, 0, 0, 0, 0, 0, str(e), None)


# ---------------------------------------------------------------------------
# Data Cleaning
# ---------------------------------------------------------------------------

def _clean_merged_data(df):
    """
    Clean and validate merged data before saving.
    
    Removes:
    - Rows with NaN in critical columns
    - Invalid win_prob_pct values (outside 0-100)
    - Invalid game_elapsed_seconds (negative or > 2400)
    - Invalid team_won values (not 0 or 1)
    - Duplicate rows
    
    Returns:
        (cleaned_df, stats_dict) where stats_dict contains counts of removed rows
    """
    if df is None or len(df) == 0:
        return df, {}
    
    original_count = len(df)
    stats = {
        "original": original_count,
        "nan_dropped": 0,
        "invalid_prob_dropped": 0,
        "invalid_time_dropped": 0,
        "invalid_team_won_dropped": 0,
        "invalid_volume_dropped": 0,
        "duplicates_dropped": 0,
        "final": 0
    }
    
    # Required columns check
    required_cols = ["kalshi_event", "team", "game_elapsed_seconds", "win_prob_pct", "volume", "team_won"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Drop rows with NaN in critical columns
    before_nan = len(df)
    df = df.dropna(subset=["win_prob_pct", "game_elapsed_seconds", "team_won", "kalshi_event", "team", "volume"])
    stats["nan_dropped"] = before_nan - len(df)
    
    # Validate win_prob_pct: should be between 0 and 100
    before_prob = len(df)
    df = df[(df["win_prob_pct"] >= 0) & (df["win_prob_pct"] <= 100)]
    stats["invalid_prob_dropped"] = before_prob - len(df)
    
    # Validate game_elapsed_seconds: should be non-negative and <= 2400 (regulation)
    before_time = len(df)
    df = df[(df["game_elapsed_seconds"] >= 0) & (df["game_elapsed_seconds"] <= 2400)]
    stats["invalid_time_dropped"] = before_time - len(df)
    
    # Validate team_won: should be 0 or 1
    before_team_won = len(df)
    df = df[df["team_won"].isin([0, 1])]
    stats["invalid_team_won_dropped"] = before_team_won - len(df)
    
    # Validate volume: should be non-negative
    before_volume = len(df)
    df = df[df["volume"] >= 0]
    stats["invalid_volume_dropped"] = before_volume - len(df)
    
    # Remove duplicates
    before_dup = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    stats["duplicates_dropped"] = before_dup - len(df)
    
    # Ensure correct data types (may introduce NaN for invalid values)
    df["game_elapsed_seconds"] = pd.to_numeric(df["game_elapsed_seconds"], errors="coerce")
    df["win_prob_pct"] = pd.to_numeric(df["win_prob_pct"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["team_won"] = pd.to_numeric(df["team_won"], errors="coerce")
    
    # Final check: drop any rows that became NaN after type conversion
    before_final_nan = len(df)
    df = df.dropna(subset=["win_prob_pct", "game_elapsed_seconds", "team_won", "volume"])
    if before_final_nan > len(df):
        stats["nan_dropped"] += (before_final_nan - len(df))
    
    # Convert to final types
    df["game_elapsed_seconds"] = df["game_elapsed_seconds"].astype(float)
    df["win_prob_pct"] = df["win_prob_pct"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["team_won"] = df["team_won"].astype(int)
    
    stats["final"] = len(df)
    
    return df, stats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_and_merge_all_games(num_games, mappings_file="GeneratedDataFiles/kalshi_espn_game_mappings.csv", kalshi_games_file="GeneratedDataFiles/list_of_kalshi_game.txt"):
    """
    For each mapped game (up to num_games), fetch ESPN and Kalshi data
    sequentially, merge them, and save the combined result to a CSV.

    Args:
        num_games: Maximum number of games to process.
        mappings_file: Path to the CSV file containing kalshi_espn_game_mappings. 
                      Defaults to "GeneratedDataFiles/kalshi_espn_game_mappings.csv".
        kalshi_games_file: Path to the text file containing list of Kalshi games with teams.
                          Defaults to "GeneratedDataFiles/list_of_kalshi_game.txt".
    """
    mappings = pd.read_csv(mappings_file)
    print(f"Loading team lookup from: {kalshi_games_file}")
    team_lookup = _load_team_lookup(kalshi_games_file)
    print(f"Loaded {len(team_lookup)} games with team data")

    rows_to_process = list(mappings.head(num_games).iterrows())
    max_id_width = max(len(row["kalshi_game_id"]) for _, row in rows_to_process)
    total_games = len(rows_to_process)

    print("\nMERGING ESPN PLAY-BY-PLAY AND KALSHI MARKET DATA (fetch_and_merge_game_data.py)\n")

    all_merged = []
    total_ot_dropped = 0
    total_ot_before = 0
    total_stale_dropped = 0

    for idx, (_, row) in enumerate(rows_to_process, 1):
        kalshi_game_id, merged, good, discarded, stale_dropped, ot_dropped, ot_before, error, diagnostic = \
            _process_single_game(row["kalshi_game_id"], row["espn_game_id"], team_lookup)

        kalshi_id = kalshi_game_id.ljust(max_id_width)
        if error:
            print(f"  ({idx}/{total_games}) Error processing {kalshi_id}: {error}")
        elif merged is not None:
            all_merged.append(merged)
            total_ot_dropped += ot_dropped
            total_ot_before += ot_before
            total_stale_dropped += stale_dropped
            stale_str = f", {stale_dropped} stale" if stale_dropped else ""
            print(f"  ({idx}/{total_games}) Processed {kalshi_id} ({good} good rows, {discarded} discarded{stale_str})")
        else:
            diag_str = f" ({diagnostic})" if diagnostic else ""
            print(f"  ({idx}/{total_games}) No aligned data for {kalshi_id}{diag_str}")

    if total_stale_dropped > 0 or total_ot_dropped > 0:
        print()
    if total_stale_dropped > 0:
        print(f"  Deduplicated {total_stale_dropped} stale-price rows (zero volume, unchanged probability)")
    if total_ot_dropped > 0:
        print(f"  Dropped {total_ot_dropped} overtime records ({total_ot_dropped/total_ot_before*100:.1f}%)")

    if all_merged:
        combined = pd.concat(all_merged, ignore_index=True)
        
        # Clean and validate the data before saving
        print("\n  Cleaning and validating merged data...")
        cleaned, clean_stats = _clean_merged_data(combined)
        
        # Print cleaning statistics
        if clean_stats["original"] > clean_stats["final"]:
            dropped_total = clean_stats["original"] - clean_stats["final"]
            print(f"  Data cleaning removed {dropped_total} rows:")
            if clean_stats["nan_dropped"] > 0:
                print(f"    - {clean_stats['nan_dropped']} rows with NaN values")
            if clean_stats["invalid_prob_dropped"] > 0:
                print(f"    - {clean_stats['invalid_prob_dropped']} rows with invalid win_prob_pct (not 0-100)")
            if clean_stats["invalid_time_dropped"] > 0:
                print(f"    - {clean_stats['invalid_time_dropped']} rows with invalid game_elapsed_seconds")
            if clean_stats["invalid_team_won_dropped"] > 0:
                print(f"    - {clean_stats['invalid_team_won_dropped']} rows with invalid team_won (not 0 or 1)")
            if clean_stats["invalid_volume_dropped"] > 0:
                print(f"    - {clean_stats['invalid_volume_dropped']} rows with invalid volume (negative)")
            if clean_stats["duplicates_dropped"] > 0:
                print(f"    - {clean_stats['duplicates_dropped']} duplicate rows")
        
        generated_data_dir = "GeneratedDataFiles"
        os.makedirs(generated_data_dir, exist_ok=True)
        output_file = os.path.join(generated_data_dir, "all_games_merged_clean.csv")
        cleaned.to_csv(output_file, index=False)
        print(f"\n  Saved {len(cleaned)} rows to GeneratedDataFiles/all_games_merged_clean.csv")
    else:
        print("\nNo data collected — nothing to save.")
