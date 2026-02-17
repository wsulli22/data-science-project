#!/usr/bin/env python3
"""
Read unmatched ESPN games and extract all unique full team names and abbreviations.

This script:
1. Loads unmatched ESPN game IDs from unmatched_espn_games_GOOD.txt
2. For each game ID, fetches full team names and abbreviations from ESPN API
3. Collects all unique team names with their abbreviations
4. Writes them to full_team_names.txt (one per row: full_name, abbreviation)
"""

import urllib.request
import urllib.parse
import json
import time
import os
import ssl
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# File paths
UNMATCHED_ESPN_FILE = "GeneratedDataFiles/unmatched_espn_games_GOOD.txt"
OUTPUT_FILE = "GeneratedDataFiles/full_team_names.txt"

# ESPN API endpoint
ESPN_SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/summary"
)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

def load_unmatched_espn_game_ids():
    """Load ESPN game IDs from the unmatched games file."""
    game_ids = []
    try:
        with open(UNMATCHED_ESPN_FILE, "r") as f:
            for line in f:
                line = line.strip()
                # Skip header lines
                if not line or line.startswith("Games") or line.startswith("="):
                    continue
                parts = line.split(",")
                if len(parts) >= 1:
                    game_id = parts[0].strip()
                    if game_id:
                        game_ids.append(game_id)
    except FileNotFoundError:
        print(f"Error: {UNMATCHED_ESPN_FILE} not found")
        return []
    
    print(f"Loaded {len(game_ids)} unmatched ESPN game IDs")
    return game_ids

def get_team_names_from_espn(espn_game_id):
    """
    Fetch full team names and abbreviations for an ESPN game.
    
    Returns:
        List of tuples (full_name, abbreviation), or empty list if fetch fails
    """
    url = f"{ESPN_SUMMARY_URL}?event={espn_game_id}"
    
    try:
        req = urllib.request.Request(url, headers=headers)
        # Create SSL context that doesn't verify certificates (for testing)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        with urllib.request.urlopen(req, timeout=10, context=ssl_context) as response:
            data = json.loads(response.read().decode())
        
        team_info = []
        # Extract teams from the header/competitions section
        competitions = data.get("header", {}).get("competitions", [])
        if competitions:
            competitors = competitions[0].get("competitors", [])
            for comp in competitors:
                team = comp.get("team", {})
                # Try multiple fields for full name
                full_name = (
                    team.get("displayName", "") or 
                    team.get("name", "") or 
                    team.get("fullName", "") or
                    (team.get("location", "") + " " + team.get("name", "")).strip()
                )
                # Get abbreviation
                abbreviation = team.get("abbreviation", "") or team.get("shortDisplayName", "") or ""
                if full_name:
                    team_info.append((full_name, abbreviation))
        
        return team_info
    except Exception as e:
        print(f"  Warning: Failed to fetch team names for game {espn_game_id}: {e}")
        return []

def process_single_game(game_id, unique_team_names, lock, processed_count):
    """Process a single game and update shared state with team names found."""
    team_info = get_team_names_from_espn(game_id)
    
    new_teams = []
    with lock:
        for full_name, abbreviation in team_info:
            if (full_name, abbreviation) not in unique_team_names:
                unique_team_names.add((full_name, abbreviation))
                new_teams.append((full_name, abbreviation))
        processed_count[0] += 1
        total_unique = len(unique_team_names)
        
        # Print new teams found (thread-safe printing)
        for full_name, abbreviation in new_teams:
            abbrev_str = f" ({abbreviation})" if abbreviation else ""
            print(f"    ✓ New team found: {full_name}{abbrev_str} (Total: {total_unique} unique teams)")

def collect_unique_team_names(limit=None, num_workers=10):
    """Collect all unique team names from unmatched ESPN games using parallel workers."""
    print("\nCOLLECTING UNIQUE TEAM NAMES FROM UNMATCHED ESPN GAMES")
    print("=" * 60)
    
    # Load game IDs
    game_ids = load_unmatched_espn_game_ids()
    if not game_ids:
        return set()
    
    # Limit to first N games if specified
    if limit:
        game_ids = game_ids[:limit]
        print(f"Limiting to first {limit} games for testing")
    
    # Thread-safe data structures
    unique_team_names = set()
    lock = Lock()
    processed_count = [0]  # Use list to allow modification in nested function
    total_games = len(game_ids)
    
    print(f"\nFetching team names for {total_games} games using {num_workers} workers...")
    print("(Processing in parallel for faster execution)\n")
    
    # Process games in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_game = {
            executor.submit(process_single_game, game_id, unique_team_names, lock, processed_count): game_id
            for game_id in game_ids
        }
        
        # Process completed tasks
        for future in as_completed(future_to_game):
            game_id = future_to_game[future]
            try:
                future.result()  # Wait for completion
            except Exception as e:
                print(f"  Error processing game {game_id}: {e}")
            
            # Print progress summary periodically
            with lock:
                current = processed_count[0]
                if current % 50 == 0 or current == total_games:
                    print(f"\n  Progress: {current}/{total_games} games processed | {len(unique_team_names)} unique teams found\n")
    
    print(f"\nFound {len(unique_team_names)} unique team names")
    return unique_team_names

def write_team_names_to_file(team_names):
    """Write unique team names and abbreviations to output file, one per row."""
    output_dir = os.path.dirname(OUTPUT_FILE) or "GeneratedDataFiles"
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort team names alphabetically by full name
    sorted_teams = sorted(team_names, key=lambda x: x[0])
    
    with open(OUTPUT_FILE, "w") as f:
        for full_name, abbreviation in sorted_teams:
            abbrev_str = f", {abbreviation}" if abbreviation else ""
            f.write(f"{full_name}{abbrev_str}\n")
    
    print(f"\nWrote {len(sorted_teams)} unique team names to {OUTPUT_FILE}")
    print("=" * 60)

if __name__ == "__main__":
    # Process all games with 10 workers
    unique_names = collect_unique_team_names(num_workers=10)
    if unique_names:
        write_team_names_to_file(unique_names)
