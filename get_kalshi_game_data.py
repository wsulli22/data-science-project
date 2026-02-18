import requests
import pandas as pd
import logging
import time
import re
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MAX_RETRIES = 5
RETRY_BACKOFF = 2  # seconds; doubles each retry

SERIES_TICKER = "KXNCAAMBGAME"


def get_kalshi_game_data(event_ticker: str, team_abbreviation: str) -> pd.DataFrame:
    """
    Fetch 1-minute candlestick data for a Kalshi college basketball team market.

    Args:
        event_ticker:      Kalshi event ticker (e.g. "KXNCAAMBGAME-26FEB10MILWIUIN")
        team_abbreviation: Team abbreviation (e.g. "IUIN")

    Returns:
        pandas DataFrame with columns:
            wallclock_ts       – timezone-aware datetime (UTC)
            end_period_ts      – unix timestamp of candlestick end
            win_prob_close     – last trade price as probability (0.0–1.0)
            win_prob_mean      – mean trade price as probability (0.0–1.0)
            win_prob_open      – first trade price as probability (0.0–1.0)
            win_prob_previous  – last trade price before this candle (0.0–1.0)
            volume             – contracts traded in this minute
            open_interest      – total outstanding contracts
            result             – final outcome for this team ("yes" = win, "no" = loss)
            team1_score        – final score for team 1 (if available)
            team2_score        – final score for team 2 (if available)
    """
    event_ticker = event_ticker.upper()
    team_abbreviation = team_abbreviation.upper()
    market_ticker = f"{event_ticker}-{team_abbreviation}"

    # Step 1: Get market info for time range and result
    market_info = _get_market_info(market_ticker)
    open_time = market_info["open_time"]
    close_time = market_info["close_time"]
    result = market_info["result"]

    # Step 1b: Get event info to fetch scores
    event_info = _get_event_info(event_ticker)
    team1_score = event_info.get("team1_score", None)
    team2_score = event_info.get("team2_score", None)

    # Convert to unix timestamps
    #start_ts = int(open_time.timestamp())
    #end_ts = int(close_time.timestamp())

    end_ts = int(close_time.timestamp())
    start_ts = end_ts - (6 * 3600)  # 6 hours in seconds

    logger.info(f"Market {market_ticker}: {open_time} → {close_time}, result={result}")
    if team1_score is not None and team2_score is not None:
        logger.info(f"Scores: {team1_score} - {team2_score}")

    # Step 2: Fetch all 1-minute candlesticks
    candlesticks = _fetch_candlesticks(market_ticker, start_ts, end_ts)
    if not candlesticks:
        raise ValueError(f"No candlestick data for {market_ticker}")

    logger.info(f"Fetched {len(candlesticks)} candlesticks")

    # Step 3: Parse into rows
    rows = []
    for candle in candlesticks:
        ts = candle.get("end_period_ts", 0)
        price = candle.get("price", {})
        volume = candle.get("volume", 0)
        oi = candle.get("open_interest", 0)

        rows.append({
            "wallclock_ts": datetime.fromtimestamp(ts, tz=timezone.utc),
            "end_period_ts": ts,
            "win_prob_close": _cents_to_prob(price.get("close")),
            "win_prob_mean": _cents_to_prob(price.get("mean")),
            "win_prob_open": _cents_to_prob(price.get("open")),
            "win_prob_previous": _cents_to_prob(price.get("previous")),
            "volume": volume,
            "open_interest": oi,
            "result": result,
            "team1_score": team1_score,
            "team2_score": team2_score,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("end_period_ts").reset_index(drop=True)

    # Forward-fill: when no trades occurred (close is null), use previous
    df["win_prob"] = df["win_prob_close"].fillna(df["win_prob_previous"]).infer_objects(copy=False)

    logger.info(
        f"Built DataFrame: {len(df)} rows, "
        f"non-null win_prob: {df['win_prob'].notna().sum()}, "
        f"result: {result}"
    )

    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _request_with_retry(url, headers, params=None, timeout=15):
    """Make a GET request with retry on 429 rate-limit errors and timeouts."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout)
            if resp.status_code == 429:
                wait = RETRY_BACKOFF * (2 ** attempt)
                logger.debug(f"Rate limited (429), retrying in {wait}s (attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            wait = RETRY_BACKOFF * (2 ** attempt)
            logger.debug(f"Request failed ({e}), retrying in {wait}s (attempt {attempt+1}/{MAX_RETRIES})")
            time.sleep(wait)
    # Final attempt — let it raise on any error
    resp = requests.get(url, headers=headers, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp


def _get_market_info(market_ticker: str) -> dict:
    """Fetch market metadata (open/close times, result)."""
    url = f"https://api.elections.kalshi.com/trade-api/v2/markets/{market_ticker}"
    headers = {"accept": "application/json"}
    resp = _request_with_retry(url, headers, timeout=15)
    market = resp.json().get("market", {})

    open_time = datetime.fromisoformat(market["open_time"].replace("Z", "+00:00"))
    close_time = datetime.fromisoformat(market["close_time"].replace("Z", "+00:00"))
    result = market.get("result", "")  # "yes" or "no"

    return {
        "open_time": open_time,
        "close_time": close_time,
        "result": result,
    }


def _get_event_info(event_ticker: str) -> dict:
    """Fetch event metadata including scores if available."""
    url = f"https://api.elections.kalshi.com/trade-api/v2/events/{event_ticker}"
    headers = {"accept": "application/json"}
    
    try:
        resp = _request_with_retry(url, headers, timeout=15)
        event = resp.json().get("event", {})
        
        # Try to extract scores from various possible fields
        team1_score = None
        team2_score = None
        
        # Check common score field names
        if "score" in event:
            score_data = event["score"]
            if isinstance(score_data, dict):
                team1_score = score_data.get("team1_score") or score_data.get("score1") or score_data.get("home_score")
                team2_score = score_data.get("team2_score") or score_data.get("score2") or score_data.get("away_score")
            elif isinstance(score_data, list) and len(score_data) >= 2:
                team1_score = score_data[0]
                team2_score = score_data[1]
        
        # Check for scores in metadata or outcome fields
        if team1_score is None:
            metadata = event.get("metadata", {})
            team1_score = metadata.get("team1_score") or metadata.get("score1") or metadata.get("home_score")
            team2_score = metadata.get("team2_score") or metadata.get("score2") or metadata.get("away_score")
        
        # Check outcome field
        if team1_score is None:
            outcome = event.get("outcome", {})
            if isinstance(outcome, dict):
                team1_score = outcome.get("team1_score") or outcome.get("score1")
                team2_score = outcome.get("team2_score") or outcome.get("score2")
        
        # Try to parse scores from description or subtitle if available
        if team1_score is None:
            subtitle = event.get("subtitle", "")
            description = event.get("description", "")
            # Look for score patterns like "85-72" or "Score: 85-72"
            score_pattern = r'(\d+)\s*[-–]\s*(\d+)'
            for text in [subtitle, description]:
                match = re.search(score_pattern, text)
                if match:
                    team1_score = int(match.group(1))
                    team2_score = int(match.group(2))
                    break
        
        # Convert to integers if they're strings
        if team1_score is not None:
            try:
                team1_score = int(team1_score) if isinstance(team1_score, (int, str)) and str(team1_score).isdigit() else None
            except (ValueError, TypeError):
                team1_score = None
        
        if team2_score is not None:
            try:
                team2_score = int(team2_score) if isinstance(team2_score, (int, str)) and str(team2_score).isdigit() else None
            except (ValueError, TypeError):
                team2_score = None
        
        return {
            "team1_score": team1_score,
            "team2_score": team2_score,
        }
    except Exception as e:
        logger.debug(f"Could not fetch event info for {event_ticker}: {e}")
        return {
            "team1_score": None,
            "team2_score": None,
        }


def _fetch_candlesticks(market_ticker: str, start_ts: int, end_ts: int) -> list[dict]:
    """Fetch all 1-minute candlesticks for a market."""
    series_ticker = SERIES_TICKER
    url = (
        f"https://api.elections.kalshi.com/trade-api/v2"
        f"/series/{series_ticker}/markets/{market_ticker}/candlesticks"
    )
    params = {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "period_interval": 1,
    }
    headers = {"accept": "application/json"}
    resp = _request_with_retry(url, headers, params=params, timeout=30)
    return resp.json().get("candlesticks", [])


def _cents_to_prob(cents) -> float | None:
    """Convert price in cents (0-100) to probability (0.0-1.0)."""
    if cents is None:
        return None
    return cents / 100.0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example usage
    EVENT_TICKER = "KXNCAAMBGAME-26FEB10MILWIUIN"
    TEAM = "MILW"

    df = get_kalshi_game_data(EVENT_TICKER, TEAM)
    print(f"\n{df.to_string(max_rows=30)}")
