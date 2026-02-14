import requests
import os
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


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


def get_list_of_all_espn_college_basketball_games(
    start_date="2025-11-03",
    end_date="2026-02-14"
):
    """
    Fetch all ESPN men's college basketball games between start_date and
    end_date (inclusive) using the ESPN scoreboard API.

    Returns:
        List of tuples: (espn_game_id, team1_abbrev, team2_abbrev, winner_abbrev)
        where team1 and team2 are sorted alphabetically.
    """
    session = create_session_with_retries()
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    all_games = []
    current = start

    while current <= end:
        date_str = current.strftime("%Y%m%d")
        params = {
            "dates": date_str,
            "groups": 50,   # All Division I games
            "limit": 365,   # Ensure we get every game for the day
        }

        try:
            resp = session.get(url, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            print(f"  Error fetching {date_str}: {e}")
            current += timedelta(days=1)
            continue

        events = data.get("events", [])

        for event in events:
            game_id = event.get("id", "")
            status = event.get("status", {}).get("type", {}).get("name", "")

            # Only include finished games
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

            teams.sort()
            if len(teams) >= 2:
                all_games.append((game_id, teams[0], teams[1], winner))
            elif len(teams) == 1:
                all_games.append((game_id, teams[0], "", winner))
            else:
                all_games.append((game_id, "", "", winner))

        if events:
            print(f"  {current.strftime('%Y-%m-%d')}: {len(events)} games")

        current += timedelta(days=1)

    # Deduplicate by game_id (ESPN may list a game on multiple dates)
    seen = set()
    unique_games = []
    for game in all_games:
        if game[0] not in seen:
            seen.add(game[0])
            unique_games.append(game)

    print(f"\nTotal unique finished ESPN games: {len(unique_games)}")

    # Save to file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "list_of_espn_games.txt")

    with open(output_file, "w") as f:
        for game_id, team1, team2, winner in unique_games:
            f.write(f"{game_id},{team1},{team2},{winner}\n")

    print(f"Wrote {len(unique_games)} games to {output_file}")

    return unique_games


if __name__ == "__main__":
    games = get_list_of_all_espn_college_basketball_games()
