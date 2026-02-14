import requests
import json
import os
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_session_with_retries():
    """Create a requests session with retry logic."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def fetch_all_ncaamb_markets():
    """
    Fetch all KXNCAAMBGAME markets from the Kalshi API using cursor-based
    pagination, with no status filter so we can see everything available.
    """
    url = "https://api.elections.kalshi.com/trade-api/v2/markets"
    headers = {"accept": "application/json"}
    session = create_session_with_retries()

    all_markets = []
    cursor = None
    page = 0

    while True:
        page += 1
        params = {
            "series_ticker": "KXNCAAMBGAME",
            "limit": 1000,
        }
        if cursor:
            params["cursor"] = cursor

        try:
            response = session.get(url, headers=headers, params=params, timeout=15)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"  Request error on page {page}: {e}")
            break

        try:
            json_data = response.json()
        except json.JSONDecodeError as e:
            print(f"  JSON parse error on page {page}: {e}")
            break

        markets = json_data.get("markets", [])
        if not markets:
            break

        all_markets.extend(markets)
        print(f"  Page {page}: fetched {len(markets)} markets (total: {len(all_markets)})")

        cursor = json_data.get("cursor", None)
        if not cursor:
            break

    return all_markets

def get_finished_games_with_teams(all_markets, limit=None):
    """
    From a list of markets, extract unique game event_tickers that have ended,
    along with their team abbreviations. Returns list of (event_ticker, team1, team2, winner) tuples.
    """
    games = {}  # event_ticker -> {close_time, status, result, teams: set(), winner: str}

    for market in all_markets:
        event_ticker = market.get("event_ticker", "")
        if not event_ticker:
            continue

        ticker = market.get("ticker", "")
        status = market.get("status", "")
        close_time = market.get("close_time", "")
        result = market.get("result", "")

        # Extract team abbreviation from ticker
        # Ticker format is: {event_ticker}-{team_code}
        team_code = ""
        if ticker.startswith(event_ticker + '-'):
            team_code = ticker[len(event_ticker) + 1:]  # Remove "{event_ticker}-" prefix
        elif "-" in ticker:
            # Fallback: extract everything after the last "-"
            team_code = ticker.split("-")[-1]

        # Keep track of each game, update with info from any of its sub-markets
        if event_ticker not in games:
            games[event_ticker] = {
                "close_time": close_time,
                "status": status,
                "result": result,
                "teams": set(),
                "winner": ""
            }
        else:
            # Update if this sub-market has a result and prior didn't
            if result and not games[event_ticker]["result"]:
                games[event_ticker]["result"] = result
            if status in ("closed", "settled") and games[event_ticker]["status"] not in ("closed", "settled"):
                games[event_ticker]["status"] = status

        # If this team's market result is "yes", they are the winner
        if result == "yes" and team_code:
            games[event_ticker]["winner"] = team_code

        # Add team code to the set
        if team_code:
            games[event_ticker]["teams"].add(team_code)

    # Print summary of statuses found
    status_counts = {}
    for g in games.values():
        s = g["status"]
        status_counts[s] = status_counts.get(s, 0) + 1
    print(f"\n  Total unique games: {len(games)}")
    print(f"  Status breakdown: {json.dumps(status_counts, indent=4)}")

    # Filter to only finished games: status is closed/settled OR result is set
    finished = {
        ticker: info for ticker, info in games.items()
        if info["status"] in ("closed", "settled", "finalized") or info["result"]
    }

    print(f"  Finished games: {len(finished)}")

    # Sort by close_time descending (most recent first)
    sorted_games = sorted(
        finished.items(),
        key=lambda x: x[1]["close_time"] or "",
        reverse=True
    )

    # Extract event tickers with team abbreviations and winner
    result = []
    for ticker, info in sorted_games:
        teams = sorted(list(info["teams"]))  # Sort for consistent ordering
        winner = info.get("winner", "")
        if len(teams) >= 2:
            # Take first two teams (should be exactly 2 for basketball games)
            result.append((ticker, teams[0], teams[1], winner))
        elif len(teams) == 1:
            # Some games might only have one team market, use empty string for second
            result.append((ticker, teams[0], "", winner))
        else:
            # No teams found, use empty strings
            result.append((ticker, "", "", winner))

    if limit:
        return result[:limit]
    return result

def get_list_of_all_kalshi_college_basketball_games(limit=None):
    """
    Fetch all finished Kalshi college basketball games.
    
    Returns:
        List of tuples: (event_ticker, team1, team2, winner) for each finished game
    """
    all_markets = fetch_all_ncaamb_markets()
    if not all_markets:
        return []
    
    games_list = get_finished_games_with_teams(all_markets, limit=limit)
    return games_list

def main():
    print("Fetching ALL closed NCAAMB games from Kalshi...")
    print("-" * 60)

    # Step 1: Fetch all KXNCAAMBGAME markets (no status filter)
    all_markets = fetch_all_ncaamb_markets()

    if not all_markets:
        print("\nNo markets returned from API.")
        return

    # Step 2: Extract finished games with team abbreviations
    games_with_teams = get_finished_games_with_teams(all_markets)

    if not games_with_teams:
        print("\nNo finished games found.")
        return

    # Step 3: Write to text file in format: event_ticker, team1, team2, winner
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "list_of_kalshi_games.txt")

    with open(output_file, "w") as f:
        for event_ticker, team1, team2, winner in games_with_teams:
            f.write(f"{event_ticker},{team1},{team2},{winner}\n")

    print(f"\nWrote {len(games_with_teams)} games to {output_file}")

if __name__ == "__main__":
    main()
