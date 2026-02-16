import requests
import pandas as pd
import logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

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
    """
    event_ticker = event_ticker.upper()
    team_abbreviation = team_abbreviation.upper()
    market_ticker = f"{event_ticker}-{team_abbreviation}"

    # Step 1: Get market info for time range and result
    market_info = _get_market_info(market_ticker)
    open_time = market_info["open_time"]
    close_time = market_info["close_time"]
    result = market_info["result"]

    # Convert to unix timestamps
    #start_ts = int(open_time.timestamp())
    #end_ts = int(close_time.timestamp())

    end_ts = int(close_time.timestamp())
    start_ts = end_ts - (6 * 3600)  # 6 hours in seconds

    logger.info(f"Market {market_ticker}: {open_time} → {close_time}, result={result}")

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
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("end_period_ts").reset_index(drop=True)

    # Forward-fill: when no trades occurred (close is null), use previous
    df["win_prob"] = df["win_prob_close"].fillna(df["win_prob_previous"])

    logger.info(
        f"Built DataFrame: {len(df)} rows, "
        f"non-null win_prob: {df['win_prob'].notna().sum()}, "
        f"result: {result}"
    )

    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_market_info(market_ticker: str) -> dict:
    """Fetch market metadata (open/close times, result)."""
    url = f"https://api.elections.kalshi.com/trade-api/v2/markets/{market_ticker}"
    headers = {"accept": "application/json"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    market = resp.json().get("market", {})

    open_time = datetime.fromisoformat(market["open_time"].replace("Z", "+00:00"))
    close_time = datetime.fromisoformat(market["close_time"].replace("Z", "+00:00"))
    result = market.get("result", "")  # "yes" or "no"

    return {
        "open_time": open_time,
        "close_time": close_time,
        "result": result,
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
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
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
