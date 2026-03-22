import requests
import json
import os
import re
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Month abbreviation → number for parsing Kalshi tickers
MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
    "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

#NORMALLY GAMES THAT WERE POSTPONED OR CANCELLED
EXCLUDED_EVENT_TICKERS = {
    "KXNCAAMBGAME-26JAN26ACTARL",
    "KXNCAAMBGAME-26JAN25WAGLIU",
    "KXNCAAMBGAME-26JAN25PFWIUIN",
    "KXNCAAMBGAME-26FEB14MORGSCST",
    "KXNCAAMBGAME-26FEB14CHSLMC",
    "KXNCAAMBGAME-26FEB14NWNEB",
    "KXNCAAMBGAME-26JAN26ACTARL",
    "KXNCAAMBGAME-26JAN25WAGLIU",
    "KXNCAAMBGAME-26JAN25PFWIUIN",
    "KXNCAAMBGAME-26JAN25SIUEVAN",
    "KXNCAAMBGAME-25DEC21PEAYUMKC",
    "KXNCAAMBGAME-25DEC15EWUUSI",
    "KXNCAAMBGAME-26JAN25SFPANHC",
    "KXNCAAMBGAME-26MAR06UICMURR",
    "KXNCAAMBGAME-26JAN26WAGLIU",
    "KXNCAAMBGAME-26JAN24ETAMNWST",
    "KXNCAAMBGAME-26JAN12GRAMFAMU",
    "KXNCAAMBGAME-26JAN10UTMUSI",
    "KXNCAAMBGAME-25DEC13MSSTUTAH",
    "KXNCAAMBGAME-25DEC13NMSUTLSA",
    "KXNCAAMBGAME-25DEC03CBUBYU",
    "KXNCAAMBGAME-25DEC03LSUBC",
    "KXNCAAMBGAME-25NOV07WAGHALL",
    "KXNCAAMBGAME-25NOV07UABNCST",
    "KXNCAAMBGAME-25NOV03NWSTTXAM",
    "KXNCAAMBGAME-26JAN31WEBBCHSO",
    "KXNCAAMBGAME-26JAN02USDSF",
    "KXNCAAMBGAME-26JAN24MVSUGRAM",
    "KXNCAAMBGAME-26JAN24USAMRSH",
    "KXNCAAMBGAME-26JAN25USCWIS",
    "KXNCAAMBGAME-26JAN04ORSTWSU",



}

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

        cursor = json_data.get("cursor", None)
        if not cursor:
            break

    return all_markets

def fetch_event_data(event_ticker):
    """
    Fetch event data from Kalshi API to get sub_title, title, and full team names.
    
    Returns:
        dict with keys: sub_title, title, team1_full_name, team2_full_name
        or None if fetch fails
    """
    url = f"https://api.elections.kalshi.com/trade-api/v2/events/{event_ticker}"
    headers = {"accept": "application/json"}
    session = create_session_with_retries()
    
    try:
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        event = data.get("event", {})
        markets = data.get("markets", [])
        
        sub_title = event.get("sub_title", "")
        title = event.get("title", "")
        
        # Extract full team names from markets
        # Markets have yes_sub_title and no_sub_title fields (both contain the team name)
        team_names = {}
        for market in markets:
            ticker = market.get("ticker", "")
            if "-" in ticker:
                # Extract team abbreviation from ticker (last part after final dash)
                team_abbr = ticker.split("-")[-1]
                # Use yes_sub_title if available, otherwise no_sub_title
                team_name = market.get("yes_sub_title", "") or market.get("no_sub_title", "")
                if team_name and team_abbr:
                    team_names[team_abbr] = team_name
        
        return {
            "sub_title": sub_title,
            "title": title,
            "team_names": team_names
        }
    except Exception:
        return None


def get_finished_games_with_teams(all_markets):
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

    return result


def _parse_date_from_ticker(event_ticker):
    """Parse a date from a Kalshi event ticker. Returns datetime.date or None."""
    if not event_ticker.startswith("KXNCAAMBGAME-"):
        return None
    suffix = event_ticker[len("KXNCAAMBGAME-"):]
    match = re.match(r"(\d{2})([A-Z]{3})(\d{2})", suffix)
    if not match:
        return None
    yy, mon, dd = match.groups()
    month_num = MONTH_MAP.get(mon)
    if month_num is None:
        return None
    try:
        return datetime(2000 + int(yy), month_num, int(dd)).date()
    except ValueError:
        return None


def get_list_of_all_kalshi_college_basketball_games(date=None):
    """
    Fetch all finished Kalshi college basketball games, save to
    list_of_kalshi_games.txt, and return the list.

    Args:
        date: Optional cutoff date string (e.g. "2026-02-14").
              Only games on or before this date are included.

    Returns:
        List of tuples: (event_ticker, team1, team2, winner, date, sub_title, title, team1_full, team2_full) for each finished game with a Kalshi winner
    """
    print(f"\nFETCHING ALL NCAAMB GAMES ON KALSHI (get_list_of_kalshi_games.py)")
    print(f"\n  This may take a few seconds...")
    all_markets = fetch_all_ncaamb_markets()
    if not all_markets:
        print("\nNo markets returned from API.")
        return []

    games_list = get_finished_games_with_teams(all_markets)

    if not games_list:
        print("\nNo finished games found.")
        return []

    # Filter by date if specified
    if date:
        cutoff = datetime.strptime(date, "%Y-%m-%d").date()
        filtered = []
        for game in games_list:
            game_date = _parse_date_from_ticker(game[0])
            if game_date is not None and game_date <= cutoff:
                filtered.append(game)
        print(f"  Filtered to {len(filtered)} games on or before {date} (from {len(games_list)})")
        games_list = filtered

    before_winner = len(games_list)
    games_list = [g for g in games_list if (g[3] or "").strip()]
    dropped_no_winner = before_winner - len(games_list)
    if dropped_no_winner:
        print(f"  Dropped {dropped_no_winner} game(s) with no Kalshi winner (unsettled/empty result)")
    if not games_list:
        print("\nNo games left after requiring a resolved winner.")
        return []

    # Add dates and fetch event metadata (sub_title, title, full team names)
    print(f"\n  Fetching event metadata (sub_title, title, team names) for {len(games_list)} games...")
    games_with_metadata = []
    effective_index = 0
    for (event_ticker, team1, team2, winner) in games_list:
        # Skip any event tickers we explicitly do not want in the output
        if event_ticker in EXCLUDED_EVENT_TICKERS:
            continue

        effective_index += 1
        game_date = _parse_date_from_ticker(event_ticker)
        date_str = game_date.strftime("%Y-%m-%d") if game_date else ""
        
        # Fetch event data from API
        event_data = fetch_event_data(event_ticker)
        if event_data:
            sub_title = event_data.get("sub_title", "")
            title = event_data.get("title", "")
            team_names = event_data.get("team_names", {})
            team1_full = team_names.get(team1, "")
            team2_full = team_names.get(team2, "")
        else:
            sub_title = ""
            title = ""
            team1_full = ""
            team2_full = ""
        
        if effective_index % 100 == 0:
            print(f"    Processed {effective_index} games (excluding {len(EXCLUDED_EVENT_TICKERS)} explicitly skipped)...")
        
        games_with_metadata.append((event_ticker, team1, team2, winner, date_str, 
                                   sub_title, title, team1_full, team2_full))

    # Save to file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    generated_data_dir = os.path.join(script_dir, "GeneratedDataFiles")
    os.makedirs(generated_data_dir, exist_ok=True)
    output_file = os.path.join(generated_data_dir, "list_of_kalshi_game.txt")

    with open(output_file, "w") as f:
        for game in games_with_metadata:
            event_ticker, team1, team2, winner, date_str, sub_title, title, team1_full, team2_full = game
            # Escape commas in text fields by replacing with semicolons or using quotes
            # Using semicolons for simplicity
            sub_title_clean = sub_title.replace(",", ";")
            title_clean = title.replace(",", ";")
            team1_full_clean = team1_full.replace(",", ";")
            team2_full_clean = team2_full.replace(",", ";")
            f.write(f"{event_ticker},{team1},{team2},{winner},{date_str},{sub_title_clean},{title_clean},{team1_full_clean},{team2_full_clean}\n")

    filename = os.path.basename(output_file)
    print(f"\n  Wrote {len(games_with_metadata)} games to GeneratedDataFiles/{filename}")
    return games_with_metadata


if __name__ == "__main__":
    get_list_of_all_kalshi_college_basketball_games()
