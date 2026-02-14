import requests
from datetime import datetime


def candlestick(event_ticker, team_abbreviation, period_interval=1):
    """
    Get the latest 1-minute candlestick for a team's yes_ask price from the Kalshi API.
    
    Args:
        event_ticker: The event ticker (e.g., "KXLIGUE1GAME-26FEB06FCMLIL")
        team_abbreviation: The team abbreviation (e.g., "FCMLIL")
        period_interval: Candlestick period in minutes (1, 60, or 1440). Default: 1 (shortest)
    
    Returns:
        Dictionary with:
        - open: yes_ask open price
        - high: yes_ask high price
        - low: yes_ask low price
        - close: yes_ask close price
        - end_period_ts: End of period unix timestamp
        - timestamp: Human-readable timestamp
        Or None if no data available.
    """
    event_ticker = event_ticker.upper()
    team_abbreviation = team_abbreviation.upper()
    
    # Build the market ticker: {event_ticker}-{team_abbreviation}
    market_ticker = f"{event_ticker}-{team_abbreviation}"
    
    url = f"https://api.elections.kalshi.com/trade-api/v2/markets/{market_ticker}/candlesticks"
    params = {
        "period_interval": period_interval
    }
    headers = {"accept": "application/json"}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException:
        return None
    
    candlesticks = data.get("candlesticks", [])
    
    if not candlesticks:
        return None
    
    # Get the most recent candlestick (last in the list)
    latest = candlesticks[-1]
    yes_ask = latest.get("yes_ask", {})
    end_ts = latest.get("end_period_ts", 0)
    
    return {
        "open": yes_ask.get("open"),
        "high": yes_ask.get("high"),
        "low": yes_ask.get("low"),
        "close": yes_ask.get("close"),
        "end_period_ts": end_ts,
        "timestamp": datetime.utcfromtimestamp(end_ts).strftime("%Y-%m-%d %H:%M:%S") if end_ts else None
    }
