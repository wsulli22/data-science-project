#!/usr/bin/env python3
"""
Verify mappings for previously unmatched games.

This script:
1. Loads games from unmatched_kalshi_games_GOOD.txt
2. Checks if they now have mappings in kalshi_espn_game_mappings_GOOD.csv
3. Creates a report showing which are matched and which are still unmatched
4. Includes team information and links for verification
"""

import csv
import requests
import os
from kalshi_espn_game_mapper import (
    find_matching_espn_game,
    parse_kalshi_date,
    convert_kalshi_to_espn_abbr
)

# File paths
UNMATCHED_FILE = "GeneratedDataFiles/unmatched_kalshi_games_GOOD.txt"
MAPPINGS_FILE = "GeneratedDataFiles/kalshi_espn_game_mappings_GOOD.csv"
KALSHI_TEAMS_CSV = "GeneratedDataFiles/full_team_names_kalshi.csv"
ESPN_TEAMS_CSV = "GeneratedDataFiles/full_team_names_espn.csv"
ESPN_GAMES_FILE = "GeneratedDataFiles/list_of_espn_games.txt"
OUTPUT_FILE = "GeneratedDataFiles/unmatched_games_verification_report.csv"

# Kalshi API endpoint
KALSHI_MARKET_URL = "https://api.elections.kalshi.com/trade-api/v2/markets"

headers = {
    "accept": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

def load_team_lookups():
    """Load team name lookups from CSV files."""
    # Load Kalshi: abbreviation -> full name
    kalshi_abbr_to_name = {}
    try:
        with open(KALSHI_TEAMS_CSV, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header if present
            for row in reader:
                if len(row) >= 2:
                    full_name = row[0].strip()
                    abbr = row[1].strip()
                    if abbr and full_name:
                        if abbr not in kalshi_abbr_to_name:
                            kalshi_abbr_to_name[abbr] = full_name
    except FileNotFoundError:
        print(f"Warning: {KALSHI_TEAMS_CSV} not found")
    except Exception as e:
        print(f"Error reading {KALSHI_TEAMS_CSV}: {e}")
    
    # Load ESPN: abbreviation -> full name
    espn_abbr_to_name = {}
    try:
        with open(ESPN_TEAMS_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                full_name = (row.get("full team name") or row.get("espn_full_name") or "").strip()
                abbr = (row.get("espn team abbreviation") or row.get("espn_abbreviation") or "").strip()
                if abbr and full_name:
                    if abbr not in espn_abbr_to_name:
                        espn_abbr_to_name[abbr] = full_name
    except FileNotFoundError:
        print(f"Warning: {ESPN_TEAMS_CSV} not found")
    except Exception as e:
        print(f"Error reading {ESPN_TEAMS_CSV}: {e}")
    
    return kalshi_abbr_to_name, espn_abbr_to_name

def get_kalshi_team_name_from_api(kalshi_game_id, team_abbr):
    """Try to get full team name from Kalshi API."""
    market_ticker = f"{kalshi_game_id}-{team_abbr}"
    url = f"{KALSHI_MARKET_URL}/{market_ticker}"
    
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        market = data.get("market", {})
        title = market.get("title", "") or ""
        
        if title:
            if title.startswith("Will "):
                end = title.find(" win")
                if end > 5:
                    return title[5:end].strip()
            elif " at " in title:
                parts = title.split(" at ")
                if len(parts) >= 2:
                    team1 = parts[0].strip()
                    team2 = parts[1].replace(" Winner?", "").replace(" Winner", "").strip()
                    if team_abbr.upper() in team1.upper()[:len(team_abbr)+3]:
                        return team1
                    elif team_abbr.upper() in team2.upper()[:len(team_abbr)+3]:
                        return team2
    except Exception:
        pass
    
    return None

def get_espn_team_info(espn_game_id):
    """Get team information from ESPN API."""
    url = (
        "https://site.api.espn.com/apis/site/v2/sports/basketball/"
        f"mens-college-basketball/summary?event={espn_game_id}"
    )
    
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        teams = []
        competitions = data.get("header", {}).get("competitions", [])
        if competitions:
            competitors = competitions[0].get("competitors", [])
            for comp in competitors:
                team = comp.get("team", {})
                full_name = (
                    team.get("displayName", "") or 
                    team.get("name", "") or 
                    team.get("fullName", "") or
                    (team.get("location", "") + " " + team.get("name", "")).strip()
                )
                abbr = team.get("abbreviation", "") or team.get("shortDisplayName", "") or ""
                if full_name:
                    teams.append((full_name, abbr))
        
        return teams
    except Exception:
        return []

def load_existing_mappings():
    """Load existing mappings from the GOOD file."""
    mappings = {}
    try:
        with open(MAPPINGS_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                kalshi_id = row.get("kalshi_game_id", "").strip()
                espn_id = row.get("espn_game_id", "").strip()
                if kalshi_id and espn_id:
                    mappings[kalshi_id] = espn_id
    except FileNotFoundError:
        print(f"Warning: {MAPPINGS_FILE} not found")
    except Exception as e:
        print(f"Error reading {MAPPINGS_FILE}: {e}")
    
    return mappings

def load_unmatched_games():
    """Load all unmatched games (both sections)."""
    games = []
    try:
        with open(UNMATCHED_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
            in_section = False
            for line in lines:
                line = line.strip()
                if line.startswith("Games"):
                    in_section = True
                    continue
                if not line or line.startswith("="):
                    continue
                
                if in_section:
                    parts = line.split(",")
                    if len(parts) >= 3:
                        event_ticker = parts[0].strip()
                        team1 = parts[1].strip()
                        team2 = parts[2].strip()
                        games.append((event_ticker, team1, team2))
    except FileNotFoundError:
        print(f"Error: {UNMATCHED_FILE} not found")
        return []
    except Exception as e:
        print(f"Error reading {UNMATCHED_FILE}: {e}")
        return []
    
    return games

def verify_unmatched_games_mappings():
    """Verify mappings for previously unmatched games."""
    print("\nVERIFYING MAPPINGS FOR PREVIOUSLY UNMATCHED GAMES")
    print("=" * 60)
    
    # Load team lookups
    print("Loading team name lookups...")
    kalshi_abbr_to_name, espn_abbr_to_name = load_team_lookups()
    print(f"  Loaded {len(kalshi_abbr_to_name)} Kalshi teams, {len(espn_abbr_to_name)} ESPN teams")
    
    # Load existing mappings
    print("\nLoading existing mappings...")
    existing_mappings = load_existing_mappings()
    print(f"  Loaded {len(existing_mappings)} existing mappings")
    
    # Load ESPN games for matching attempts
    espn_games = []
    try:
        with open(ESPN_GAMES_FILE, "r") as f:
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
                    espn_games.append((game_id, team1, team2, winner))
    except FileNotFoundError:
        print(f"Warning: {ESPN_GAMES_FILE} not found")
    
    print(f"  Loaded {len(espn_games)} ESPN games for matching")
    
    # Load unmatched games
    print("\nLoading previously unmatched games...")
    unmatched_games = load_unmatched_games()
    print(f"  Found {len(unmatched_games)} previously unmatched games\n")
    
    # Process each game
    report_rows = []
    matched_count = 0
    still_unmatched_count = 0
    
    for idx, (kalshi_game_id, kalshi_team1_abbr, kalshi_team2_abbr) in enumerate(unmatched_games, 1):
        print(f"Processing {idx}/{len(unmatched_games)}: {kalshi_game_id}")
        
        # Check if now matched
        espn_game_id = existing_mappings.get(kalshi_game_id)
        is_matched = espn_game_id is not None
        
        if is_matched:
            matched_count += 1
            status = "MATCHED"
        else:
            still_unmatched_count += 1
            status = "STILL UNMATCHED"
            # Try to find a match
            if espn_games:
                kalshi_date = parse_kalshi_date(kalshi_game_id)
                espn_game_id = find_matching_espn_game(
                    kalshi_team1_abbr, kalshi_team2_abbr, espn_games, kalshi_date=kalshi_date
                )
        
        # Get Kalshi team names
        kalshi_team1_name = kalshi_abbr_to_name.get(kalshi_team1_abbr, "")
        kalshi_team2_name = kalshi_abbr_to_name.get(kalshi_team2_abbr, "")
        
        # Try API if not in lookup
        if not kalshi_team1_name:
            kalshi_team1_name = get_kalshi_team_name_from_api(kalshi_game_id, kalshi_team1_abbr) or kalshi_team1_abbr
        if not kalshi_team2_name:
            kalshi_team2_name = get_kalshi_team_name_from_api(kalshi_game_id, kalshi_team2_abbr) or kalshi_team2_abbr
        
        # Get ESPN team info
        espn_team1_name = ""
        espn_team2_name = ""
        espn_team1_abbr = ""
        espn_team2_abbr = ""
        
        if espn_game_id:
            espn_teams = get_espn_team_info(espn_game_id)
            if len(espn_teams) >= 2:
                espn_team1_name, espn_team1_abbr = espn_teams[0]
                espn_team2_name, espn_team2_abbr = espn_teams[1]
        else:
            # Try to get ESPN abbreviations from conversion
            espn_team1_abbr = convert_kalshi_to_espn_abbr(kalshi_team1_abbr)
            espn_team1_name = espn_abbr_to_name.get(espn_team1_abbr, "")
            espn_team2_abbr = convert_kalshi_to_espn_abbr(kalshi_team2_abbr)
            espn_team2_name = espn_abbr_to_name.get(espn_team2_abbr, "")
        
        # Construct links
        if kalshi_game_id.startswith("KXNCAAMBGAME-"):
            kalshi_suffix = kalshi_game_id[len("KXNCAAMBGAME-"):].lower()
            kalshi_link = f"https://kalshi.com/markets/kxncaambgame/mens-college-basketball-mens-game/kxncaambgame-{kalshi_suffix}"
        else:
            kalshi_link = f"https://kalshi.com/markets/kxncaambgame/mens-college-basketball-mens-game/{kalshi_game_id.lower()}"
        
        espn_link = f"https://www.espn.com/mens-college-basketball/game/_/gameId/{espn_game_id}" if espn_game_id else ""
        
        # Create rows for both teams
        # Team 1
        report_rows.append({
            "status": status,
            "kalshi_game_id": kalshi_game_id,
            "espn_game_id": espn_game_id or "",
            "espn_full_name": espn_team1_name,
            "kalshi_full_name": kalshi_team1_name,
            "espn_abbreviation": espn_team1_abbr,
            "kalshi_abbreviation": kalshi_team1_abbr,
            "espn_game_link": espn_link,
            "kalshi_game_link": kalshi_link
        })
        
        # Team 2
        report_rows.append({
            "status": status,
            "kalshi_game_id": kalshi_game_id,
            "espn_game_id": espn_game_id or "",
            "espn_full_name": espn_team2_name,
            "kalshi_full_name": kalshi_team2_name,
            "espn_abbreviation": espn_team2_abbr,
            "kalshi_abbreviation": kalshi_team2_abbr,
            "espn_game_link": espn_link,
            "kalshi_game_link": kalshi_link
        })
    
    # Write to CSV
    output_dir = os.path.dirname(OUTPUT_FILE) or "GeneratedDataFiles"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "status", "kalshi_game_id", "espn_game_id",
            "espn_full_name", "kalshi_full_name", 
            "espn_abbreviation", "kalshi_abbreviation", 
            "espn_game_link", "kalshi_game_link"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)
    
    print(f"\n✓ Summary:")
    print(f"  Matched: {matched_count} games")
    print(f"  Still Unmatched: {still_unmatched_count} games")
    print(f"  Total: {len(unmatched_games)} games")
    print(f"\n✓ Wrote {len(report_rows)} rows ({len(unmatched_games)} games) to {OUTPUT_FILE}")
    print("=" * 60)

if __name__ == "__main__":
    verify_unmatched_games_mappings()
