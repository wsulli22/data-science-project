import requests
import pandas as pd
import logging
import os
from datetime import datetime, timezone

ESPN_GAME_ID = 401817686

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

    # Deduplicate and sort
    df = df.drop_duplicates().sort_values("wallclock_ts").reset_index(drop=True)

    # Validate monotonicity within each period
    df = _validate_monotonicity(df)

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

    while True:
        params = {"limit": page_size, "page": page}
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
    Ensure game_elapsed_seconds is non-decreasing over wallclock time
    within each period. Drop rows that violate monotonicity and log them.
    """
    bad_mask = pd.Series(False, index=df.index)

    for period, group in df.groupby("period"):
        elapsed = group["game_elapsed_seconds"]
        violations = elapsed.diff() < 0
        n_bad = violations.sum()
        if n_bad > 0:
            logger.warning(
                f"Period {period}: {n_bad} rows violate monotonicity – dropping them"
            )
            bad_mask.loc[violations[violations].index] = True

    if bad_mask.any():
        df = df[~bad_mask].reset_index(drop=True)

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
