"""
fetch_and_merge_game_data.py

Provides a pipeline used by main.py:
  - fetch_and_merge_all_games(num_games)
    For each mapped game, fetches ESPN play-by-play + Kalshi candlestick data,
    merges them, and saves the combined result to a CSV.
"""

import os
from collections import Counter
from datetime import datetime, timezone

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

# Hard ceiling for validation:
# Regulation ends at 2400s, each OT adds 300s.
# Allow up to 4OT (end at 3600s) to avoid dropping valid overtime games.
MAX_GAME_ELAPSED_SECONDS = 2400 + 300 * 4


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


def _load_ot_espn_game_ids(
    espn_games_file="GeneratedDataFiles/list_of_espn_games.txt",
):
    """
    Load ESPN game IDs whose `ot_period` is > 0 from list_of_espn_games.txt.

    Assumes list format:
      game_id,team1,team2,winner,slug1,slug2,date,ot_period
    """
    ot_game_ids = set()
    if not os.path.exists(espn_games_file):
        print(f"WARNING: {espn_games_file} not found; cannot filter overtime games.")
        return ot_game_ids

    with open(espn_games_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue

            espn_game_id = parts[0].strip()
            ot_period = 0
            try:
                ot_period = int(parts[-1].strip())
            except (ValueError, IndexError):
                ot_period = 0

            if ot_period > 0:
                ot_game_ids.add(str(espn_game_id))

    return ot_game_ids


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
            "realworld_timestamp": wallclock_ts,
            "period": period_number,
            "clock_display": clock_display or "",
            "game_elapsed_seconds": game_elapsed,
        })

    if not rows:
        raise ValueError(f"No valid plays parsed for game {game_id}")

    df = pd.DataFrame(rows)
    df = df.drop_duplicates().sort_values("realworld_timestamp").reset_index(drop=True)
    before_mono = len(df)
    # Reuse ESPN monotonicity validator, which expects wallclock_ts internally.
    df = df.rename(columns={"realworld_timestamp": "wallclock_ts"})
    df = _validate_monotonicity(df)
    df = df.rename(columns={"wallclock_ts": "realworld_timestamp"})
    espn_discarded = before_mono - len(df)
    df["realworld_timestamp"] = pd.to_datetime(df["realworld_timestamp"]).dt.tz_localize(None)
    df = df.sort_values("realworld_timestamp").reset_index(drop=True)

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
            kalshi_raw["realworld_timestamp"] = pd.to_datetime(
                kalshi_raw["realworld_timestamp"]
            ).dt.tz_localize(None)
            kalshi_raw = kalshi_raw.sort_values("realworld_timestamp").reset_index(drop=True)
            results.append((team, kalshi_raw))
        except Exception as e:
            errors.append(f"{team}: {e}")
    
    if not results and errors:
        return [], f"Kalshi API failed ({'; '.join(errors)})"
    return results, None


def _merge_espn_kalshi(espn_df, kalshi_teams, kalshi_game_id, espn_discarded):
    """
    Merge ESPN and Kalshi data for one game (pure function, no globals).
    Includes Kalshi candlestick data during intermissions between consecutive ESPN
    periods (halftime and OT breaks). Candles in those gaps are pinned to the exact
    end-of-previous-period elapsed seconds.

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
    start_elapsed = float(espn_df["game_elapsed_seconds"].min())
    start_rows = espn_df[
        (espn_df["game_elapsed_seconds"].astype(float) - start_elapsed).abs() < 1e-9
    ].sort_values("realworld_timestamp", kind="mergesort")
    start_ts = start_rows["realworld_timestamp"].iloc[0] if len(start_rows) > 0 else None
    start_period = int(start_rows["period"].iloc[0]) if len(start_rows) > 0 else None

    terminal_elapsed = float(espn_df["game_elapsed_seconds"].max())
    terminal_rows = espn_df[
        (espn_df["game_elapsed_seconds"].astype(float) - terminal_elapsed).abs() < 1e-9
    ].sort_values("realworld_timestamp", kind="mergesort")
    terminal_ts = terminal_rows["realworld_timestamp"].iloc[0] if len(terminal_rows) > 0 else None
    terminal_period = int(terminal_rows["period"].iloc[0]) if len(terminal_rows) > 0 else None

    def _end_elapsed_for_period(prev_period_number: int) -> float:
        # Regulation halves are 1200s each; each OT period is 300s.
        if prev_period_number == 1:
            return 1200.0
        if prev_period_number == 2:
            return 2400.0
        # Period 3 => OT1 end at 2700, period 4 => OT2 end at 3000, etc.
        return 2400.0 + (prev_period_number - 2) * 300.0

    # Identify intermission windows between consecutive ESPN periods.
    # For each p where both p and p+1 exist, we take:
    #   (last wallclock play in period p) .... (first wallclock play in period p+1)
    # Kalshi candles inside that strict window are assigned elapsed=end(p).
    period_values = sorted(
        {int(p) for p in espn_df["period"].dropna().unique().tolist() if int(p) >= 1}
    )
    period_set = set(period_values)
    break_windows = []
    for p in period_values:
        if (p + 1) not in period_set:
            continue

        p_df = espn_df[espn_df["period"] == p]
        pnext_df = espn_df[espn_df["period"] == (p + 1)]
        if len(p_df) == 0 or len(pnext_df) == 0:
            continue

        start_wc = p_df["realworld_timestamp"].max()
        end_wc = pnext_df["realworld_timestamp"].min()
        if pd.notna(start_wc) and pd.notna(end_wc) and end_wc > start_wc:
            break_windows.append((start_wc, end_wc, p))

    for team_abbr, kalshi_raw in kalshi_teams:
        game_start = espn_df["realworld_timestamp"].min()
        game_end = espn_df["realworld_timestamp"].max()
        kalshi = kalshi_raw[
            (kalshi_raw["realworld_timestamp"] >= game_start)
            & (kalshi_raw["realworld_timestamp"] <= game_end)
        ].copy().reset_index(drop=True)

        if len(kalshi) == 0:
            continue

        total_kalshi_in_window += len(kalshi)

        # Split Kalshi data into in-game and intermission portions.
        # Intermission candles (halftime + OT breaks) get a pinned elapsed seconds value.
        in_game_kalshi = kalshi.copy()
        merged_breaks_parts = []

        for start_wc, end_wc, prev_period in break_windows:
            intermission_mask = (
                (in_game_kalshi["realworld_timestamp"] > start_wc)
                & (in_game_kalshi["realworld_timestamp"] < end_wc)
            )
            if not intermission_mask.any():
                continue

            intermission_kalshi = in_game_kalshi[intermission_mask].copy()
            intermission_kalshi["game_elapsed_seconds"] = float(_end_elapsed_for_period(prev_period))
            intermission_kalshi["period"] = 0  # Will be re-derived from elapsed seconds later
            merged_breaks_parts.append(
                intermission_kalshi[
                    ["realworld_timestamp", "win_prob", "volume", "result", "game_elapsed_seconds", "period"]
                ]
            )

            # Remove from in-game set so it doesn't get merged twice
            in_game_kalshi = in_game_kalshi[~intermission_mask].copy()

        merged_in_game = pd.DataFrame(
            columns=["realworld_timestamp", "win_prob", "volume", "result", "game_elapsed_seconds", "period"]
        )
        if len(in_game_kalshi) > 0:
            merged_in_game = pd.merge_asof(
                in_game_kalshi[["realworld_timestamp", "win_prob", "volume", "result"]].copy(),
                espn_df[["realworld_timestamp", "game_elapsed_seconds", "period"]],
                on="realworld_timestamp",
                direction="backward",
            ).dropna(subset=["game_elapsed_seconds"])

        merged_breaks = (
            pd.concat(merged_breaks_parts, ignore_index=True) if len(merged_breaks_parts) > 0
            else pd.DataFrame(columns=["realworld_timestamp", "win_prob", "volume", "result", "game_elapsed_seconds", "period"])
        )

        # Combine in-game and intermission data
        # Filter out empty DataFrames before concatenation to avoid FutureWarning
        dfs_to_concat = [df for df in [merged_in_game, merged_breaks] if len(df) > 0]
        if not dfs_to_concat:
            merged = pd.DataFrame(columns=["realworld_timestamp", "win_prob", "volume", "result", "game_elapsed_seconds", "period"])
        else:
            merged = pd.concat(dfs_to_concat, ignore_index=True)

        if len(merged) == 0:
            continue

        # Preserve initial ESPN game state (e.g., tip-off at 0s) when first
        # minute-bucket Kalshi row starts after the first ESPN play timestamp.
        if start_ts is not None and start_period is not None:
            has_start_elapsed = (
                (merged["game_elapsed_seconds"].astype(float) - start_elapsed).abs() < 1e-9
            ).any()
            if not has_start_elapsed:
                start_kalshi = kalshi[kalshi["realworld_timestamp"] >= start_ts].head(1)
                if len(start_kalshi) == 0:
                    start_kalshi = kalshi.head(1)
                if len(start_kalshi) > 0:
                    start_row = start_kalshi.iloc[0]
                    merged = pd.concat(
                        [
                            pd.DataFrame(
                                [
                                    {
                                        "realworld_timestamp": start_ts,
                                        "win_prob": start_row["win_prob"],
                                        "volume": start_row["volume"],
                                        "result": start_row["result"],
                                        "game_elapsed_seconds": start_elapsed,
                                        "period": start_period,
                                    }
                                ]
                            ),
                            merged,
                        ],
                        ignore_index=True,
                    )

        # Preserve terminal game state (e.g., period end at 2400s / OT end) even when
        # minute-bucket Kalshi timestamps fall slightly before the final ESPN play.
        if terminal_ts is not None and terminal_period is not None:
            has_terminal_elapsed = (
                (merged["game_elapsed_seconds"].astype(float) - terminal_elapsed).abs() < 1e-9
            ).any()
            if not has_terminal_elapsed:
                terminal_kalshi = kalshi[kalshi["realworld_timestamp"] <= terminal_ts].tail(1)
                if len(terminal_kalshi) > 0:
                    terminal_row = terminal_kalshi.iloc[0]
                    merged = pd.concat(
                        [
                            merged,
                            pd.DataFrame(
                                [
                                    {
                                        "realworld_timestamp": terminal_ts,
                                        "win_prob": terminal_row["win_prob"],
                                        "volume": terminal_row["volume"],
                                        "result": terminal_row["result"],
                                        "game_elapsed_seconds": terminal_elapsed,
                                        "period": terminal_period,
                                    }
                                ]
                            ),
                        ],
                        ignore_index=True,
                    )

        # Sort by realworld_timestamp to maintain chronological order
        merged = merged.sort_values("realworld_timestamp").reset_index(drop=True)

        # Keep zero-volume rows so no-trade minutes remain in the timeline.

        merged["win_prob_pct"] = (merged["win_prob"] * 100.0).round(2)
        merged["team_won"] = (merged["result"] == "yes").astype(int)
        merged["kalshi_event"] = kalshi_game_id
        merged["team"] = team_abbr

        all_dfs.append(
            merged[
                [
                    "kalshi_event",
                    "team",
                    "realworld_timestamp",
                    "game_elapsed_seconds",
                    "period",
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

    # Post-merge cleanup: fix monotonicity violations and finalize period labels
    # Sort by game, team, and realworld_timestamp to ensure chronological order
    result = result.sort_values(["kalshi_event", "team", "realworld_timestamp"]).reset_index(drop=True)

    # Coerce game_elapsed_seconds to float so we never have object/NaN causing all rows to drop later
    result["game_elapsed_seconds"] = pd.to_numeric(result["game_elapsed_seconds"], errors="coerce")
    result = result.dropna(subset=["game_elapsed_seconds"])
    if len(result) == 0:
        return None, 0, 0, 0, 0, 0, "All rows had invalid game_elapsed_seconds after merge"

    # Apply cummax per game/team to fix backward regressions
    result["game_elapsed_seconds"] = result.groupby(["kalshi_event", "team"])["game_elapsed_seconds"].cummax()

    def _label_and_truncate_game(g: pd.DataFrame) -> pd.DataFrame:
        """
        Label periods from elapsed-time boundary occurrences using game-wide
        timestamps (shared across both teams), and truncate after the first
        occurrence of the true game-end boundary.

        Rules:
        - firstHalf -> halfTime -> secondHalf from first/last occurrence of 1200.
          secondHalf starts at the last occurrence of 1200.
        - For OT boundaries B in [2400, 2700, 3000, ...]:
            first..(last-1) timestamp at B => preOTk
            last timestamp at B            => OTk
          where k=1 for B=2400, k=2 for B=2700, etc.
        - Truncate rows after the first occurrence of the final game-end boundary:
            regular game end: 2400
            1OT game end:     2700
            2OT game end:     3000
            ...
        """
        g = g.sort_values("realworld_timestamp", kind="mergesort").reset_index(drop=True)
        elapsed = g["game_elapsed_seconds"].astype(float)
        # Keep exactly one row per timestamp when deriving boundaries so both
        # teams share the same phase transitions.
        ts_frame = (
            g[["realworld_timestamp", "game_elapsed_seconds"]]
            .drop_duplicates(subset=["realworld_timestamp"], keep="last")
            .sort_values("realworld_timestamp", kind="mergesort")
            .reset_index(drop=True)
        )
        ts_elapsed = ts_frame["game_elapsed_seconds"].astype(float)
        ts_labels = pd.Series(index=ts_frame.index, data="firstHalf", dtype="string")

        def _ts_indices_at(boundary: float) -> list[int]:
            return ts_frame.index[(ts_elapsed - boundary).abs() < 1e-9].tolist()

        # secondHalf starts at the last occurrence of 1200; earlier 1200 rows are halfTime.
        idx_1200 = _ts_indices_at(1200.0)
        if idx_1200:
            first_1200 = idx_1200[0]
            last_1200 = idx_1200[-1]
            ts_labels.iloc[first_1200:last_1200] = "halfTime"
            ts_labels.iloc[last_1200:] = "secondHalf"

        max_elapsed = float(ts_elapsed.max()) if len(ts_elapsed) else 0.0
        # Number of overtime periods present from elapsed timeline.
        n_ot = 0 if max_elapsed <= 2400.0 else int((max_elapsed - 2400.0 - 1e-9) // 300.0) + 1

        # For each OT boundary, relabel by first/last occurrence timestamp.
        for ot_k in range(1, n_ot + 1):
            boundary = 2400.0 + (ot_k - 1) * 300.0
            idx_b = _ts_indices_at(boundary)
            if not idx_b:
                continue
            first_b = idx_b[0]
            last_b = idx_b[-1]
            if first_b < last_b:
                ts_labels.iloc[first_b:last_b] = f"preOT{ot_k}"
            ts_labels.iloc[last_b:] = f"OT{ot_k}"

        # Truncate after first occurrence of true game-end boundary.
        final_end_boundary = 2400.0 + 300.0 * n_ot
        idx_final_end = _ts_indices_at(final_end_boundary)
        if idx_final_end:
            first_end_idx = idx_final_end[0]
            end_ts = ts_frame.loc[first_end_idx, "realworld_timestamp"]
            g = g[g["realworld_timestamp"] <= end_ts].copy()

            ts_frame = ts_frame[ts_frame["realworld_timestamp"] <= end_ts].copy()
            ts_labels = ts_labels.loc[ts_frame.index].copy()

        label_map = pd.DataFrame(
            {
                "realworld_timestamp": ts_frame["realworld_timestamp"].to_numpy(),
                "period_label": ts_labels.astype(str).to_numpy(),
            }
        )
        g = g.merge(label_map, on="realworld_timestamp", how="left")
        g["period"] = g["period_label"].fillna("firstHalf")
        g = g.drop(columns=["period_label"])
        return g

    result = _label_and_truncate_game(result).reset_index(drop=True)

    good = len(result)
    # For reporting: count how many merged rows fall in overtime (elapsed > 40:00).
    ot_before = int((result["game_elapsed_seconds"] > 2400).sum())
    discarded = (total_kalshi_in_window - good) + espn_discarded
    return result, good, discarded, total_stale, 0, ot_before, None


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
    
    Note: `period` is a human-readable label (e.g. firstHalf, halfTime, secondHalf, OT1).
    
    Removes:
    - Rows with NaN in critical columns
    - Invalid win_prob_pct values (outside 0-100)
    - Invalid game_elapsed_seconds (negative or unreasonably large)
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
    required_cols = ["kalshi_event", "team", "realworld_timestamp", "game_elapsed_seconds", "period", "win_prob_pct", "volume", "team_won"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Diagnose which columns have NaN (so we know why rows are dropped)
    nan_counts = {c: int(df[c].isna().sum()) for c in required_cols}
    if any(nan_counts.values()):
        print("  NaN counts in required columns (before cleaning):")
        for c in required_cols:
            n = nan_counts[c]
            if n > 0:
                pct = 100.0 * n / len(df)
                print(f"    - {c}: {n:,} ({pct:.1f}%)")

    # Drop rows with NaN in critical columns
    before_nan = len(df)
    df = df.dropna(subset=["win_prob_pct", "game_elapsed_seconds", "team_won", "kalshi_event", "team", "volume", "realworld_timestamp", "period"])
    stats["nan_dropped"] = before_nan - len(df)
    
    # Validate win_prob_pct: should be between 0 and 100
    before_prob = len(df)
    df = df[(df["win_prob_pct"] >= 0) & (df["win_prob_pct"] <= 100)]
    stats["invalid_prob_dropped"] = before_prob - len(df)
    
    # Validate game_elapsed_seconds: should be non-negative and within a hard ceiling.
    before_time = len(df)
    df = df[(df["game_elapsed_seconds"] >= 0) & (df["game_elapsed_seconds"] <= MAX_GAME_ELAPSED_SECONDS)]
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
    df = df.dropna(subset=["win_prob_pct", "game_elapsed_seconds", "team_won", "volume", "realworld_timestamp", "period"])
    if before_final_nan > len(df):
        stats["nan_dropped"] += (before_final_nan - len(df))
    
    # Convert to final types
    df["game_elapsed_seconds"] = df["game_elapsed_seconds"].astype(float)
    df["win_prob_pct"] = df["win_prob_pct"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["team_won"] = df["team_won"].astype(int)
    df["period"] = df["period"].astype(str)
    # Keep realworld_timestamp as datetime
    if not pd.api.types.is_datetime64_any_dtype(df["realworld_timestamp"]):
        df["realworld_timestamp"] = pd.to_datetime(df["realworld_timestamp"])
    
    stats["final"] = len(df)
    
    return df, stats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _print_fetch_merge_issue_summary(
    *,
    issue_error_lines,
    issue_no_align_lines,
    error_reason_counts,
    no_align_diag_counts,
    issue_log_file,
    total_games,
    issue_error_count,
    issue_no_align_count,
):
    """Print deferred per-game failures and aggregate counts at end of run."""
    sep = "=" * 80
    failed = issue_error_count + issue_no_align_count
    ok = total_games - failed
    print(f"\n{sep}")
    print("FETCH / MERGE — ISSUES SUMMARY (games not merged successfully)")
    print(sep)
    print(
        f"Mapped games: {total_games}  |  merged OK: {ok}  |  "
        f"errors / no aligned data: {failed}"
    )
    print(f"Full per-game lines: {issue_log_file}\n")

    if issue_error_count:
        print(f"--- Processing errors ({issue_error_count}) ---")
        for line in issue_error_lines:
            print(line)
        print("\nCount by error message:")
        for msg, cnt in error_reason_counts.most_common():
            print(f"  {cnt:>5}×  {msg}")
        print()

    if issue_no_align_count:
        print(f"--- No merged / aligned data ({issue_no_align_count}) ---")
        for line in issue_no_align_lines:
            print(line)
        print("\nCount by reason:")
        for msg, cnt in no_align_diag_counts.most_common():
            print(f"  {cnt:>5}×  {msg}")
        print()

    if not failed:
        print("No processing errors and no missing aligned data.\n")
    print(sep)


def fetch_and_merge_all_games(
    num_games,
    mappings_file="GeneratedDataFiles/kalshi_espn_game_mappings.csv",
    kalshi_games_file="GeneratedDataFiles/list_of_kalshi_game.txt",
    overtime_games="all",
    espn_games_file="GeneratedDataFiles/list_of_espn_games.txt",
    issue_log_file=None,
):
    """
    For each mapped game (up to num_games), fetch ESPN and Kalshi data
    sequentially, merge them, and save the combined result to a CSV.

    Args:
        num_games: Maximum number of games to process.
        mappings_file: Path to the CSV file containing kalshi_espn_game_mappings. 
                      Defaults to "GeneratedDataFiles/kalshi_espn_game_mappings.csv".
        kalshi_games_file: Path to the text file containing list of Kalshi games with teams.
                          Defaults to "GeneratedDataFiles/list_of_kalshi_game.txt".
        overtime_games: Either "all" (default) or "only" to restrict processing to
                        games with ot_period > 0 in list_of_espn_games.txt.
        espn_games_file: Path to list_of_espn_games.txt used for overtime filtering.
        issue_log_file: Path to a text file where per-game errors and no-aligned-data
                        lines are written (same text as printed to the terminal).
                        If None, uses GeneratedDataFiles/fetch_merge_issues.txt.
    """
    mappings = pd.read_csv(mappings_file)

    if overtime_games == "only":
        ot_game_ids = _load_ot_espn_game_ids(espn_games_file=espn_games_file)
        mappings["espn_game_id"] = mappings["espn_game_id"].astype(str)
        before_n = len(mappings)
        mappings = mappings[mappings["espn_game_id"].isin(ot_game_ids)].copy()
        print(f"Filtering to overtime games only: {before_n} -> {len(mappings)} mappings")

    print(f"Loading team lookup from: {kalshi_games_file}")
    team_lookup = _load_team_lookup(kalshi_games_file)
    print(f"Loaded {len(team_lookup)} games with team data")

    rows_to_process = list(mappings.head(num_games).iterrows())
    if not rows_to_process:
        print("No games to process after filtering/mapping.")
        return

    max_id_width = max(len(row["kalshi_game_id"]) for _, row in rows_to_process)
    total_games = len(rows_to_process)

    print("\nMERGING ESPN PLAY-BY-PLAY AND KALSHI MARKET DATA (fetch_and_merge_game_data.py)\n")

    if issue_log_file is None:
        issue_log_file = os.path.join("GeneratedDataFiles", "fetch_merge_issues.txt")
    log_dir = os.path.dirname(os.path.abspath(issue_log_file))
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    all_merged = []
    total_ot_dropped = 0
    total_ot_before = 0
    total_stale_dropped = 0
    issue_error_count = 0
    issue_no_align_count = 0
    issue_error_lines = []
    issue_no_align_lines = []
    error_reason_counts: Counter = Counter()
    no_align_diag_counts: Counter = Counter()

    with open(issue_log_file, "w", encoding="utf-8") as issue_log:
        issue_log.write("fetch_and_merge_all_games — errors and no aligned data\n")
        issue_log.write(f"Started (UTC): {datetime.now(timezone.utc).isoformat()}\n")
        issue_log.write(f"Games to process: {total_games}\n\n")
        issue_log.flush()

        for idx, (_, row) in enumerate(rows_to_process, 1):
            kalshi_game_id, merged, good, discarded, stale_dropped, ot_dropped, ot_before, error, diagnostic = \
                _process_single_game(row["kalshi_game_id"], row["espn_game_id"], team_lookup)

            kalshi_id = kalshi_game_id.ljust(max_id_width)
            if error:
                line = f"  ({idx}/{total_games}) Error processing {kalshi_id}: {error}"
                issue_error_lines.append(line)
                error_reason_counts[error] += 1
                issue_log.write(line + "\n")
                issue_log.flush()
                issue_error_count += 1
            elif merged is not None:
                all_merged.append(merged)
                total_ot_dropped += ot_dropped
                total_ot_before += ot_before
                total_stale_dropped += stale_dropped
                print(f"  ({idx}/{total_games}) Processed {kalshi_id} ({good} good rows, {discarded} discarded)")
            else:
                diag_str = f" ({diagnostic})" if diagnostic else ""
                line = f"  ({idx}/{total_games}) No aligned data for {kalshi_id}{diag_str}"
                issue_no_align_lines.append(line)
                no_align_diag_counts[diagnostic if diagnostic else "(no detail)"] += 1
                issue_log.write(line + "\n")
                issue_log.flush()
                issue_no_align_count += 1

        if no_align_diag_counts:
            issue_log.write("\n--- Breakdown: no aligned data (by reason) ---\n")
            for msg, cnt in no_align_diag_counts.most_common():
                issue_log.write(f"  {cnt:>5}×  {msg}\n")
        if error_reason_counts:
            issue_log.write("\n--- Breakdown: processing errors (by message) ---\n")
            for msg, cnt in error_reason_counts.most_common():
                issue_log.write(f"  {cnt:>5}×  {msg}\n")
        issue_log.write(
            f"\n---\nSummary: {issue_error_count} error(s), "
            f"{issue_no_align_count} no aligned data\n"
        )
        issue_log.write(f"Finished (UTC): {datetime.now(timezone.utc).isoformat()}\n")

    if total_ot_dropped > 0:
        print()
    if total_ot_before > 0:
        print(f"  Included {total_ot_before} overtime rows (elapsed > 2400s)")
    elif total_ot_dropped > 0:
        # Defensive fallback if counters are changed later.
        print(f"  Dropped {total_ot_dropped} overtime records")

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

    _print_fetch_merge_issue_summary(
        issue_error_lines=issue_error_lines,
        issue_no_align_lines=issue_no_align_lines,
        error_reason_counts=error_reason_counts,
        no_align_diag_counts=no_align_diag_counts,
        issue_log_file=os.path.abspath(issue_log_file),
        total_games=total_games,
        issue_error_count=issue_error_count,
        issue_no_align_count=issue_no_align_count,
    )
