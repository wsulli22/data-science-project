import requests
import os
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

END_DATE = "2026-02-14"

ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/scoreboard"
)


def create_session_with_retries():
    """Create a requests session with retry logic."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _fetch_single_day(date_obj):
    """Fetch all finished games for a single date from ESPN."""
    session = create_session_with_retries()
    date_str = date_obj.strftime("%Y%m%d")
    date_formatted = date_obj.strftime("%Y-%m-%d")
    params = {
        "dates": date_str,
        "groups": 50,
        "limit": 365,
    }

    try:
        resp = session.get(ESPN_SCOREBOARD_URL, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException:
        return []

    events = data.get("events", [])
    games = []

    for event in events:
        game_id = event.get("id", "")
        status = event.get("status", {}).get("type", {}).get("name", "")

        if status != "STATUS_FINAL":
            continue

        competitors = event.get("competitions", [{}])[0].get("competitors", [])
        teams = []
        winner = ""
        for comp in competitors:
            abbrev = comp.get("team", {}).get("abbreviation", "")
            if abbrev:
                teams.append(abbrev)
            if comp.get("winner") and abbrev:
                winner = abbrev

        # Sort teams alphabetically
        teams.sort()
        if len(teams) >= 2:
            games.append((game_id, teams[0], teams[1], winner, date_formatted))
        elif len(teams) == 1:
            games.append((game_id, teams[0], "", winner, date_formatted))
        else:
            games.append((game_id, "", "", winner, date_formatted))

    return games


def get_list_of_all_espn_college_basketball_games(
    start_date="2025-11-03",
    end_date=END_DATE
):
    """
    Fetch all ESPN men's college basketball games between start_date and
    end_date (inclusive) using the ESPN scoreboard API.
    Returns:
        List of tuples: (espn_game_id, team1_abbrev, team2_abbrev, winner_abbrev, date)
        where team1 and team2 are sorted alphabetically.
    """
    print(f"\nFETCHING ALL NCAAMB ESPN GAMES SINCE KALSHI'S NCAAMB INCEPTION ({start_date}) (get_list_of_espn_games.py)")
    print(f"\n  This may take a few seconds...")

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # Build list of all dates to fetch
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)

    # Fetch all dates sequentially
    all_games = []
    for d in dates:
        result = _fetch_single_day(d)
        if result:
            all_games.extend(result)

    # Deduplicate by game_id (ESPN may list a game on multiple dates)
    # Keep the first occurrence (which will have the date from when it was first seen)
    seen = set()
    unique_games = []
    for game in all_games:
        if game[0] not in seen:
            seen.add(game[0])
            unique_games.append(game)

    print(f"\n  Total finished NCAAMB games found: {len(unique_games)}")

    # Save to file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    generated_data_dir = os.path.join(script_dir, "GeneratedDataFiles")
    os.makedirs(generated_data_dir, exist_ok=True)
    output_file = os.path.join(generated_data_dir, "list_of_espn_games.txt")

    with open(output_file, "w") as f:
        for game in unique_games:
            game_id, team1, team2, winner, date_str = game
            f.write(f"{game_id},{team1},{team2},{winner},{date_str}\n")

    filename = os.path.basename(output_file)
    print(f"\n  Wrote {len(unique_games)} games to GeneratedDataFiles/{filename}")

    return unique_games


if __name__ == "__main__":
    games = get_list_of_all_espn_college_basketball_games()
