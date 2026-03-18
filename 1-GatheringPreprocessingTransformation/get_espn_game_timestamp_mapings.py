import requests
import pandas as pd
import logging
import os
import time
from datetime import datetime, timezone, timedelta

ESPN_GAME_ID = 401823374

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REGULATION_PERIOD_SECONDS = 1200  # 20 minutes per half
OVERTIME_PERIOD_SECONDS = 300     # 5 minutes per OT
NUM_REGULATION_PERIODS = 2


def get_espn_game_timestamp_mapping(espn_game_id: str | int) -> pd.DataFrame:
    """
    Fetch ESPN men's college basketball play-by-play data and build a mapping
    between real-world wallclock time and game-clock elapsed seconds.

    Args:
        espn_game_id: ESPN game ID (e.g. 401817686)

    Returns:
        pandas DataFrame with columns:
            wallclock_ts          – timezone-aware datetime (UTC)
            period                – period number (1, 2, 3=OT1, …)
            clock_display         – game clock as "mm:ss" remaining
            game_elapsed_seconds  – continuous elapsed seconds from tip-off
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
        clock_value = clock_info.get("value")  # seconds remaining (float)

        if wallclock_str is None or period_number is None or clock_value is None:
            continue

        # Parse wallclock
        wallclock_ts = _parse_wallclock(wallclock_str)
        if wallclock_ts is None:
            continue

        # Compute game elapsed seconds
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

    # Deduplicate and sort with stable sort and tie breakers
    df = (
        df.drop_duplicates()
        .sort_values(
            ["wallclock_ts", "period", "game_elapsed_seconds"],
            kind="mergesort"  # stable sort
        )
        .reset_index(drop=True)
    )

    # Collapse multiple rows at the same wallclock_ts and period (keep latest game state)
    # Group by both wallclock_ts and period to handle edge cases at halftime where
    # the same wallclock_ts might appear in two periods
    df = (
        df.groupby(["wallclock_ts", "period"], as_index=False)
        .tail(1)  # Keep the row with max game_elapsed_seconds at that timestamp/period
        .reset_index(drop=True)
    )

    # Replace period 2 "20:00" row with an artificial one derived from first non-20:00 P2 play
    df = _replace_period2_20_00_with_backtrack(df)

    # Validate monotonicity globally (game_elapsed_seconds should be non-decreasing)
    df = _validate_monotonicity(df)

    # Final sort by wallclock_ts (ascending) with stable sort for merge_asof compatibility
    df = df.sort_values("wallclock_ts", kind="mergesort").reset_index(drop=True)

    # Truncate at first occurrence of period 2 with clock 0:00
    df = _truncate_at_period2_end(df)

    # Save to CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, f"mapping_{game_id}.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved {len(df)} rows to {csv_path}")

    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_all_plays(game_id: str) -> list[dict]:
    """Fetch all plays from the ESPN play-by-play endpoint with pagination."""
    base_url = (
        "https://sports.core.api.espn.com/v2/sports/basketball/leagues/"
        f"mens-college-basketball/events/{game_id}/competitions/{game_id}/plays"
    )
    all_plays: list[dict] = []
    page = 1
    page_size = 500

    max_retries = 5
    retry_backoff = 2

    while True:
        params = {"limit": page_size, "page": page}
        resp = None
        for attempt in range(max_retries):
            try:
                resp = requests.get(base_url, params=params, timeout=20)
                resp.raise_for_status()
                break
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                wait = retry_backoff * (2 ** attempt)
                logger.debug(f"ESPN request failed ({e}), retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            except requests.exceptions.HTTPError as e:
                # Retry on server errors (500, 502, 503, 504) and rate limits (429)
                if resp is not None and resp.status_code in [429, 500, 502, 503, 504]:
                    wait = retry_backoff * (2 ** attempt)
                    logger.warning(f"ESPN API returned {resp.status_code} for game {game_id}, retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                else:
                    # Non-retryable HTTP error, re-raise
                    raise
        else:
            # Final attempt — let it raise on any error
            resp = requests.get(base_url, params=params, timeout=20)
            resp.raise_for_status()
        data = resp.json()

        items = data.get("items", [])
        all_plays.extend(items)
        logger.info(f"Page {page}: fetched {len(items)} plays (total so far: {len(all_plays)})")

        page_count = data.get("pageCount", 1)
        if page >= page_count:
            break
        page += 1

    logger.info(f"Total plays fetched: {len(all_plays)}")
    return all_plays


def _parse_wallclock(wallclock_str: str) -> datetime | None:
    """Parse an ESPN wallclock string into a timezone-aware UTC datetime."""
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%MZ"):
        try:
            dt = datetime.strptime(wallclock_str, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    logger.warning(f"Could not parse wallclock: {wallclock_str}")
    return None


def _compute_game_elapsed(period_number: int, seconds_remaining: float) -> float:
    """
    Convert period number and seconds remaining on the game clock into
    a continuous game-elapsed-seconds value.

    Regulation: 2 halves × 1200s each.
    Overtime:   each OT period is 300s, starting at period 3.
    """
    if period_number <= NUM_REGULATION_PERIODS:
        period_length = REGULATION_PERIOD_SECONDS
        elapsed_before = (period_number - 1) * REGULATION_PERIOD_SECONDS
    else:
        period_length = OVERTIME_PERIOD_SECONDS
        # All regulation time + prior OT periods
        elapsed_before = (
            NUM_REGULATION_PERIODS * REGULATION_PERIOD_SECONDS
            + (period_number - NUM_REGULATION_PERIODS - 1) * OVERTIME_PERIOD_SECONDS
        )

    elapsed_in_period = period_length - seconds_remaining
    return elapsed_before + elapsed_in_period


def _validate_monotonicity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure game_elapsed_seconds is non-decreasing over wallclock time globally.
    Drop rows that violate monotonicity and log context for auditing.
    """
    elapsed = df["game_elapsed_seconds"]
    violations = elapsed.diff() < 0
    
    if violations.any():
        n_bad = violations.sum()
        logger.warning(
            f"Dropping {n_bad} rows with decreasing game_elapsed_seconds over wallclock time"
        )
        
        # Log context for each violation (show row before and after)
        violation_indices = df.index[violations].tolist()
        for idx in violation_indices[:5]:  # Log up to 5 violations to avoid spam
            row_idx = df.index.get_loc(idx)
            if row_idx > 0:
                prev_row = df.iloc[row_idx - 1]
                curr_row = df.iloc[row_idx]
                logger.warning(
                    f"  Violation at index {row_idx}: "
                    f"wallclock={curr_row['wallclock_ts']}, "
                    f"period={curr_row['period']}, "
                    f"elapsed={prev_row['game_elapsed_seconds']:.1f} -> {curr_row['game_elapsed_seconds']:.1f}"
                )
        if len(violation_indices) > 5:
            logger.warning(f"  ... and {len(violation_indices) - 5} more violations")
        
        df = df[~violations].reset_index(drop=True)

    return df


def _replace_period2_20_00_with_backtrack(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove period 2 rows with clock_display "20:00" (often inaccurate) and add a single
    artificial period-2 start row by backtracking from the first non-20:00 period 2 play.
    """
    PERIOD2_START_ELAPSED = REGULATION_PERIOD_SECONDS  # 1200

    # Drop all period 2, 20:00 rows
    mask_p2_20 = (df["period"] == 2) & (df["clock_display"] == "20:00")
    if not mask_p2_20.any():
        return df
    df = df[~mask_p2_20].reset_index(drop=True)

    # First period 2 row (after drop) is the first non-20:00 P2 play
    p2_rows = df[df["period"] == 2]
    if p2_rows.empty:
        return df

    first_p2 = p2_rows.iloc[0]
    wallclock_ts = first_p2["wallclock_ts"]
    game_elapsed = first_p2["game_elapsed_seconds"]

    # Backtrack: at period 2 start, game_elapsed == 1200. Assume wall-clock ~ game-clock
    # over that short interval, so subtract (game_elapsed - 1200) seconds from wallclock.
    if game_elapsed <= PERIOD2_START_ELAPSED:
        # Clock still at or before 20:00 (e.g. data glitch); don't go backwards in wall time
        delta_seconds = 0.0
    else:
        delta_seconds = game_elapsed - PERIOD2_START_ELAPSED

    artificial_ts = wallclock_ts - timedelta(seconds=delta_seconds)

    new_row = pd.DataFrame([{
        "wallclock_ts": artificial_ts,
        "period": 2,
        "clock_display": "20:00",
        "game_elapsed_seconds": float(PERIOD2_START_ELAPSED),
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df = df.sort_values("wallclock_ts", kind="mergesort").reset_index(drop=True)
    logger.info(
        f"Replaced period 2 20:00 with backtrack from first P2 non-20:00: "
        f"artificial wallclock={artificial_ts}"
    )
    return df


def _truncate_at_period2_end(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate entries at the end of period 1 (halftime) and period 2 (end of regulation).
    Keeps only the first occurrence of period 1, 0:00 and period 2, 0:00, removing all subsequent duplicates.
    """
    original_len = len(df)
    
    # Find first row where period == 1 and clock_display == "0:00" (halftime)
    mask_p1 = (df["period"] == 1) & (df["clock_display"] == "0:00")
    if mask_p1.any():
        first_p1_end_idx = df.index[mask_p1][0]
        # Remove all subsequent rows that are also period 1, 0:00 (but keep period 2 rows)
        mask_p1_duplicates = (df.index > first_p1_end_idx) & mask_p1
        if mask_p1_duplicates.any():
            df = df[~mask_p1_duplicates].reset_index(drop=True)
            logger.info(f"Removed {original_len - len(df)} duplicate period 1, 0:00 entries")
            original_len = len(df)
    
    # Find first row where period == 2 and clock_display == "0:00" (end of regulation)
    mask_p2 = (df["period"] == 2) & (df["clock_display"] == "0:00")
    if mask_p2.any():
        first_p2_end_idx = df.index[mask_p2][0]
        # Remove all rows after first period 2, 0:00
        df = df.loc[:first_p2_end_idx].reset_index(drop=True)
        logger.info(f"Truncated data at first period 2, 0:00 occurrence (removed {original_len - len(df)} rows)")
    
    return df


if __name__ == "__main__":
    import sys

    # Use variable if set, otherwise use command line argument
    if ESPN_GAME_ID is not None:
        game_id = ESPN_GAME_ID
    elif len(sys.argv) >= 2:
        game_id = sys.argv[1]
    else:
        print("Usage: python get_espn_game_timestamp_mapings.py <espn_game_id>")
        print("Or set ESPN_GAME_ID variable at the top of the script")
        sys.exit(1)

    df = get_espn_game_timestamp_mapping(game_id)
    print(f"\n{df.to_string(max_rows=30)}")
