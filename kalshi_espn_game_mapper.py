import requests
import csv
import re
import os
from datetime import datetime

FILE = "GeneratedDataFiles/list_of_kalshi_game.txt"
MAPPING_CORRECTIONS_CSV = "mapping_corrections.csv"
ESPN_TEAM_NAMES_CSV = "GeneratedDataFiles/full_team_names_espn.csv"

# Kalshi API endpoint for events
kalshiAPIURL = "https://api.elections.kalshi.com/trade-api/v2/events/"

# ESPN scoreboard API (used to get the date of a specific game)
ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/scoreboard"
)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

def load_mapping_from_csv():
    """
    Load team mapping from mapping_corrections.csv and full_team_names_espn.csv.
    
    Returns:
        dict: Mapping from ESPN abbreviation to Kalshi abbreviation
    """
    # First, load ESPN full name -> Kalshi abbreviation from mapping_corrections.csv
    espn_fullname_to_kalshi = {}
    try:
        with open(MAPPING_CORRECTIONS_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row_count = 0
            for row in reader:
                row_count += 1
                # Access columns - CSV reader handles spaces in headers
                # Try both with and without leading/trailing spaces in key names
                espn_full_name = (row.get("espn_full_name") or row.get("espn_full_name ") or "").strip()
                kalshi_abbr = (row.get("kalshi_abbreviation") or row.get(" kalshi_abbreviation") or "").strip()
                if espn_full_name and kalshi_abbr:
                    espn_fullname_to_kalshi[espn_full_name] = kalshi_abbr
            if row_count == 0:
                # File exists but is empty (only has header or is completely empty)
                pass  # This is okay, just means no manual corrections are set up
    except FileNotFoundError:
        print(f"Warning: {MAPPING_CORRECTIONS_CSV} not found. Using empty mapping.")
        return {}
    except Exception as e:
        print(f"Error reading {MAPPING_CORRECTIONS_CSV}: {e}")
        return {}
    
    # Second, load ESPN full name -> ESPN abbreviation from full_team_names_espn.csv
    espn_fullname_to_abbr = {}
    try:
        with open(ESPN_TEAM_NAMES_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Handle both possible column names (with spaces)
                espn_full_name = (row.get("full team name") or row.get("espn_full_name") or "").strip()
                espn_abbr = (row.get("espn team abbreviation") or row.get("espn_abbreviation") or "").strip()
                if espn_full_name and espn_abbr:
                    espn_fullname_to_abbr[espn_full_name] = espn_abbr
    except FileNotFoundError:
        print(f"Warning: {ESPN_TEAM_NAMES_CSV} not found. Cannot build full mapping.")
        return {}
    except Exception as e:
        print(f"Error reading {ESPN_TEAM_NAMES_CSV}: {e}")
        return {}
    
    # Combine: ESPN abbreviation -> Kalshi abbreviation
    espn_abbr_to_kalshi = {}
    for espn_full_name, kalshi_abbr in espn_fullname_to_kalshi.items():
        espn_abbr = espn_fullname_to_abbr.get(espn_full_name)
        if espn_abbr:
            espn_abbr_to_kalshi[espn_abbr] = kalshi_abbr
    
    print(f"  Loaded {len(espn_abbr_to_kalshi)} team mappings from CSV files")
    return espn_abbr_to_kalshi

# Global variables for team mappings (loaded lazily when needed)
espnAbbrToKalshiAbbr = {}
kalshiAbbrToEspnAbbr = {}

# Month abbreviation → number for parsing Kalshi tickers
MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
    "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def parse_kalshi_event_ticker(event_ticker):
    """
    Parse a Kalshi event ticker to extract date and team abbreviations.
    
    Format: KXNCAAMBGAME-{YY}{MON}{DD}{TEAM1}{TEAM2}
    Example: KXNCAAMBGAME-26FEB14FURVMI
    
    Returns:
        tuple: (date_str, teams_str) or None if parsing fails
               date_str is e.g. "26FEB14", teams_str is e.g. "FURVMI"
    """
    # Remove the sport code prefix
    if not event_ticker.startswith("KXNCAAMBGAME-"):
        return None
    
    suffix = event_ticker[len("KXNCAAMBGAME-"):]
    
    # Date format: YY + MON (3 letters) + DD (2 digits)
    match = re.match(r"(\d{2})([A-Z]{3})(\d{2})(.+)", suffix)
    if not match:
        return None
    
    yy, mon, dd, teams = match.groups()
    date_str = f"{yy}{mon}{dd}"
    
    return date_str, teams


def parse_kalshi_date(event_ticker):
    """
    Extract a Python date object from a Kalshi event ticker.

    Example: KXNCAAMBGAME-26FEB10MILWIUIN → datetime.date(2026, 2, 10)
    Returns None if parsing fails.
    """
    parsed = parse_kalshi_event_ticker(event_ticker)
    if parsed is None:
        return None
    date_str, _ = parsed    # e.g. "26FEB10"
    match = re.match(r"(\d{2})([A-Z]{3})(\d{2})", date_str)
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


def get_espn_game_date(espn_game_id):
    """
    Fetch the date of an ESPN game from the ESPN event summary endpoint.
    
    Returns:
        datetime.date or None
    """
    url = (
        "https://site.api.espn.com/apis/site/v2/sports/basketball/"
        f"mens-college-basketball/summary?event={espn_game_id}"
    )
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # date string like "2026-02-10T23:30Z"
        date_str = data.get("header", {}).get("competitions", [{}])[0].get("date", "")
        if date_str:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00")).date()
    except Exception:
        pass
    return None


def verify_kalshi_game_exists(event_ticker):
    """
    Verify that a Kalshi game exists by calling the Kalshi API.
    
    Args:
        event_ticker: Kalshi event ticker (e.g., KXNCAAMBGAME-26FEB14FURVMI)
        
    Returns:
        bool: True if game exists (no error in response), False otherwise
    """
    try:
        response = requests.get(kalshiAPIURL + event_ticker, headers=headers, timeout=10)
        response.raise_for_status()
        kalshi_json = response.json()
        
        # Check if there's an error in the response
        if "error" in kalshi_json:
            return False
        return True
    except requests.exceptions.RequestException:
        return False


def convert_kalshi_to_espn_abbr(kalshi_abbr):
    """
    Convert a Kalshi team abbreviation to ESPN abbreviation.
    
    Args:
        kalshi_abbr: Kalshi team abbreviation
        
    Returns:
        str: ESPN team abbreviation (same as input if no conversion found)
    """
    return kalshiAbbrToEspnAbbr.get(kalshi_abbr, kalshi_abbr)


def find_matching_espn_game(kalshi_team1, kalshi_team2, espn_games,
                            kalshi_date=None):
    """
    Find a matching ESPN game for given Kalshi teams, using date as the primary filter.
    
    Args:
        kalshi_team1: First team abbreviation from Kalshi
        kalshi_team2: Second team abbreviation from Kalshi
        espn_games:   List of ESPN games, each as (game_id, team1, team2, winner, date)
                     where date is a datetime.date object or None
        kalshi_date:  datetime.date parsed from the Kalshi ticker (required for reliable matching).
                      If provided, filters ESPN games by date first, then by teams.
        
    Returns:
        str or None: ESPN game ID if match found with matching date and teams, None otherwise
    """
    # Convert Kalshi abbreviations to ESPN abbreviations
    espn_team1 = convert_kalshi_to_espn_abbr(kalshi_team1)
    espn_team2 = convert_kalshi_to_espn_abbr(kalshi_team2)
    
    # If date is provided, use it as the primary filter
    if kalshi_date is not None:
        # First filter by date, then by teams
        for game_data in espn_games:
            # Unpack game data
            game_id, team1, team2, winner, espn_date = game_data[:5]
            
            # Skip games without dates
            if espn_date is None:
                continue
            # Check if dates match
            if espn_date == kalshi_date:
                # Date matches - now check teams
                if team1 == espn_team1 and team2 == espn_team2:
                    return game_id
                elif team1 == espn_team2 and team2 == espn_team1:
                    return game_id
        # No match found with matching date and teams
        return None
    
    # No date available — fall back to team matching only (less reliable)
    for game_data in espn_games:
        # Unpack game data
        game_id, team1, team2, winner, espn_date = game_data[:5]
        
        if team1 == espn_team1 and team2 == espn_team2:
            return game_id
        elif team1 == espn_team2 and team2 == espn_team1:
            return game_id
    
    return None


def _process_single_kalshi_game(event_ticker, kalshi_team1, kalshi_team2, kalshi_date, espn_games):
    """
    Process a single Kalshi game: verify it exists via API, then find a
    matching ESPN game using date as the primary filter.

    Returns:
        (event_ticker, kalshi_team1, kalshi_team2, espn_game_id_or_None, status)
        where status is "matched", "not_found_in_api", or "no_espn_match".
    """
    if not verify_kalshi_game_exists(event_ticker):
        return (event_ticker, kalshi_team1, kalshi_team2, None, "not_found_in_api")

    espn_game_id = find_matching_espn_game(
        kalshi_team1, kalshi_team2, espn_games, kalshi_date=kalshi_date
    )

    if espn_game_id:
        return (event_ticker, kalshi_team1, kalshi_team2, espn_game_id, "matched")
    else:
        return (event_ticker, kalshi_team1, kalshi_team2, None, "no_espn_match")


def map_kalshi_and_espn_game_ids(limit=None):
    """
    Map Kalshi games to ESPN games.
    
    Starting from Kalshi games, verify each exists via API, then find matching ESPN game.
    Creates a CSV file with format: kalshi_game_id, espn_game_id
    
    Args:
        limit: Optional limit on number of games to process. Stops after finding
               this many successful mappings.
    """
    print("\nFINDING ESPN MATCHES FOR EACH KALSHI GAME (kalshi_espn_game_mapper.py)")
    
    # Load team mappings from CSV files
    global espnAbbrToKalshiAbbr, kalshiAbbrToEspnAbbr
    espnAbbrToKalshiAbbr = load_mapping_from_csv()
    kalshiAbbrToEspnAbbr = {v: k for k, v in espnAbbrToKalshiAbbr.items()}
    
    # Load Kalshi games with dates
    kalshi_games = []
    try:
        with open(FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) >= 3:
                    event_ticker = parts[0]
                    team1 = parts[1]
                    team2 = parts[2]
                    # Parse date from file (format: YYYY-MM-DD)
                    kalshi_date = None
                    if len(parts) >= 5:
                        date_str = parts[4].strip()
                        try:
                            kalshi_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                        except (ValueError, IndexError):
                            # Fall back to parsing from ticker if date not in file
                            kalshi_date = parse_kalshi_date(event_ticker)
                    else:
                        # Fall back to parsing from ticker if date not in file
                        kalshi_date = parse_kalshi_date(event_ticker)
                    
                    kalshi_games.append((event_ticker, team1, team2, kalshi_date))
    except FileNotFoundError:
        print(f"Error: {FILE} not found")
        return
    
    # Load ESPN games with dates
    espn_games = []
    try:
        with open("GeneratedDataFiles/list_of_espn_games.txt", "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) >= 3:
                    game_id = parts[0]
                    team1 = parts[1]
                    team2 = parts[2]
                    winner = parts[3] if len(parts) > 3 else ""
                    # Parse date from file (format: YYYY-MM-DD)
                    espn_date = None
                    if len(parts) >= 5:
                        date_str = parts[4].strip()
                        try:
                            espn_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                        except (ValueError, IndexError):
                            pass  # Keep as None if parsing fails
                    
                    espn_games.append((game_id, team1, team2, winner, espn_date))
    except FileNotFoundError:
        print("Error: GeneratedDataFiles/list_of_espn_games.txt not found")
        return
    
    
    print(f"\n  Loaded {len(kalshi_games)} Kalshi games and {len(espn_games)} ESPN games\n")
    
    # Calculate max width of event tickers for alignment
    max_ticker_width = max(len(event_ticker) for event_ticker, _, _, _ in kalshi_games)
    
    # --- Process games sequentially ---
    mappings = []
    not_found_in_api = []
    no_espn_match = []
    total_games = len(kalshi_games)

    for idx, (event_ticker, t1, t2, kalshi_date) in enumerate(kalshi_games, 1):
        if limit and len(mappings) >= limit:
            print(f"\n  Reached limit of {limit} successful game mappings. Stopping.")
            break

        event_ticker, kalshi_team1, kalshi_team2, espn_game_id, status = \
            _process_single_kalshi_game(event_ticker, t1, t2, kalshi_date, espn_games)

        aligned_ticker = event_ticker.ljust(max_ticker_width)

        if status == "not_found_in_api":
            print(f"  ({idx}/{total_games}) Skipping {event_ticker}: not found in Kalshi API")
            not_found_in_api.append((event_ticker, kalshi_team1, kalshi_team2))
        elif status == "no_espn_match":
            print(f"  ({idx}/{total_games}) No ESPN match found for {aligned_ticker} ({kalshi_team1} vs {kalshi_team2})")
            no_espn_match.append((event_ticker, kalshi_team1, kalshi_team2))
        else:
            mappings.append((event_ticker, espn_game_id))
            print(f"  ({idx}/{total_games}) Mapped {aligned_ticker} -> {espn_game_id}")
    
    # Write to CSV file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    generated_data_dir = os.path.join(script_dir, "GeneratedDataFiles")
    os.makedirs(generated_data_dir, exist_ok=True)
    output_file = os.path.join(generated_data_dir, "kalshi_espn_game_mappings.csv")
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kalshi_game_id", "espn_game_id"])
        for kalshi_id, espn_id in mappings:
            writer.writerow([kalshi_id, espn_id])
    
    filename = os.path.basename(output_file)
    print(f"\n  Wrote {len(mappings)} mappings to GeneratedDataFiles/{filename}")
    
    # Write unmatched games to file
    unmatched_file = os.path.join(generated_data_dir, "unmatched_kalshi_games.txt")
    with open(unmatched_file, "w") as f:
        if not_found_in_api:
            f.write(f"Games not found in Kalshi API ({len(not_found_in_api)}):\n")
            f.write("=" * 60 + "\n")
            for event_ticker, team1, team2 in not_found_in_api:
                f.write(f"{event_ticker},{team1},{team2}\n")
            f.write("\n")
        
        if no_espn_match:
            f.write(f"Games with no ESPN match ({len(no_espn_match)}):\n")
            f.write("=" * 60 + "\n")
            for event_ticker, team1, team2 in no_espn_match:
                f.write(f"{event_ticker},{team1},{team2}\n")
    
    if not_found_in_api or no_espn_match:
        unmatched_filename = os.path.basename(unmatched_file)
        print(f"  Wrote {len(not_found_in_api) + len(no_espn_match)} unmatched games to GeneratedDataFiles/{unmatched_filename}")
    
    
    if not_found_in_api:
        print(f"\nGames not found in Kalshi API ({len(not_found_in_api)}):")
        for event_ticker, team1, team2 in not_found_in_api:
            print(f"  {event_ticker} ({team1} vs {team2})")
    
    
    return mappings


if __name__ == "__main__":
    map_kalshi_and_espn_game_ids()
