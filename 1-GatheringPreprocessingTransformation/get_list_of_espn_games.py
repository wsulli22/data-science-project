import requests
import os
import re
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

END_DATE = "2026-02-14"

ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/scoreboard"
)

# Some ESPN game IDs are known to be missing from the scoreboard API
# responses used above but still need to be included in our data.
MANUAL_ESPN_GAME_IDS = {
    "401803638",
    "401805178",
    "401808646",
    "401808712",
    "401808713",
    "401812370",
    "401812393",
    "401817640",
    "401827053",
    "401828693",
    "401829375",
    "401829453",
    "401830280",
}


def create_session_with_retries():
    """Create a requests session with retry logic."""
    import urllib3
    # Disable SSL warnings for retries (we're handling errors)
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        # Also retry on connection errors and SSL errors
        connect=3,
        read=3,
        redirect=3
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# Cache for team slugs to avoid duplicate API calls
_team_slug_cache = {}

def _infer_ot_period_from_status_type(status_type: dict) -> int:
    """
    Infer overtime period number from ESPN status.type fields.

    Examples:
      - "Final/OT" -> 1
      - "Final/2OT" -> 2
      - "Final/3OT" -> 3
    """
    if not isinstance(status_type, dict):
        return 0

    # ESPN tends to populate "detail" and/or "shortDetail" with values like:
    #   "Final", "Final/OT", "Final/2OT", ...
    fields = [
        status_type.get("detail", "") or "",
        status_type.get("shortDetail", "") or "",
        status_type.get("altDetail", "") or "",
    ]

    for field in fields:
        field_str = str(field)
        if "OT" not in field_str.upper():
            continue

        m = re.search(r"Final/(\d*)OT", field_str, flags=re.IGNORECASE)
        if m:
            n_str = m.group(1) or ""
            return int(n_str) if n_str else 1

        # Fallback: allow just "OT" or "2OT" without "Final/" prefix.
        m2 = re.search(r"(\d*)OT", field_str, flags=re.IGNORECASE)
        if m2:
            n_str = m2.group(1) or ""
            return int(n_str) if n_str else 1

    return 0

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
    except (requests.exceptions.RequestException, requests.exceptions.SSLError) as e:
        # Log the error but continue (retry logic should handle it)
        print(f"  Warning: Error fetching date {date_formatted}: {type(e).__name__}")
        return []

    events = data.get("events", [])
    games = []

    for event in events:
        game_id = event.get("id", "")
        status_type = event.get("status", {}).get("type", {}) or {}
        status = status_type.get("name", "")

        if status != "STATUS_FINAL":
            continue

        ot_period = _infer_ot_period_from_status_type(status_type)

        competitors = event.get("competitions", [{}])[0].get("competitors", [])
        team_data_list = []
        winner = ""
        for comp in competitors:
            team_data = comp.get("team", {})
            abbrev = team_data.get("abbreviation", "")
            # Try multiple ways to get the slug
            slug = team_data.get("slug", "")
            
            # If slug not in team object, try to get it from team $ref (with caching)
            if not slug and "$ref" in team_data:
                team_ref = team_data["$ref"]
                # Check cache first
                if team_ref in _team_slug_cache:
                    slug = _team_slug_cache[team_ref]
                # Only fetch if it's a team detail endpoint (not a reference object)
                elif "/teams/" in team_ref:
                    try:
                        team_ref_resp = session.get(team_ref, timeout=10)
                        if team_ref_resp.status_code == 200:
                            team_ref_data = team_ref_resp.json()
                            slug = team_ref_data.get("slug", "")
                            _team_slug_cache[team_ref] = slug
                    except:
                        pass
            
            # If still no slug, try to construct from displayName
            if not slug:
                display_name = team_data.get("displayName", "")
                if display_name:
                    # Try to construct slug from displayName
                    # Remove special characters and convert to lowercase with hyphens
                    slug = (display_name.lower()
                           .replace("'", "")
                           .replace(".", "")
                           .replace("&", "and")
                           .replace("(", "")
                           .replace(")", "")
                           .replace(",", "")
                           .replace("  ", " ")
                           .replace(" ", "-")
                           .strip("-"))
            
            if abbrev:
                team_data_list.append((abbrev, slug))
            if comp.get("winner") and abbrev:
                winner = abbrev

        # Sort teams alphabetically by abbreviation, keeping slugs aligned
        team_data_list.sort(key=lambda x: x[0])
        
        if len(team_data_list) >= 2:
            team1, slug1 = team_data_list[0]
            team2, slug2 = team_data_list[1]
            games.append((game_id, team1, team2, winner, slug1, slug2, date_formatted, ot_period))
        elif len(team_data_list) == 1:
            team1, slug1 = team_data_list[0]
            games.append((game_id, team1, "", winner, slug1, "", date_formatted, ot_period))
        else:
            games.append((game_id, "", "", winner, "", "", date_formatted, ot_period))

    return games


def get_list_of_all_espn_college_basketball_games(
    start_date="2025-11-03",
    end_date=END_DATE
):
    """
    Fetch all ESPN men's college basketball games between start_date and
    end_date (inclusive) using the ESPN scoreboard API.
    Returns:
        List of tuples: (espn_game_id, team1_abbrev, team2_abbrev, winner_abbrev, slug1, slug2, date, ot_period)
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

    # Ensure manually-specified ESPN game IDs are always present
    existing_ids = {g[0] for g in unique_games}
    manual_added = 0
    for mid in MANUAL_ESPN_GAME_IDS:
        if mid not in existing_ids:
            # Append with empty metadata; downstream code can fill details if needed
            unique_games.append((mid, "", "", "", "", "", "", 0))
            manual_added += 1

    if manual_added:
        print(f"  Added {manual_added} manual ESPN game IDs not returned by API")

    # Save to file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    generated_data_dir = os.path.join(script_dir, "GeneratedDataFiles")
    os.makedirs(generated_data_dir, exist_ok=True)
    output_file = os.path.join(generated_data_dir, "list_of_espn_games.txt")

    with open(output_file, "w") as f:
        for game in unique_games:
            game_id, team1, team2, winner, slug1, slug2, date_str, ot_period = game
            f.write(f"{game_id},{team1},{team2},{winner},{slug1},{slug2},{date_str},{ot_period}\n")

    filename = os.path.basename(output_file)
    print(f"\n  Wrote {len(unique_games)} games to GeneratedDataFiles/{filename}")

    return unique_games


if __name__ == "__main__":
    games = get_list_of_all_espn_college_basketball_games()
