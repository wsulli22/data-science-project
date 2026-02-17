#!/usr/bin/env python3
"""
Read unmatched Kalshi games and extract all unique full team names and abbreviations.

This script:
1. Loads unmatched Kalshi game IDs from unmatched_kalshi_games_GOOD.txt
2. For each game ID, extracts team abbreviations and fetches full team names from Kalshi API
3. Collects all unique team names with their abbreviations
4. Writes them to full_team_names_kalshi.csv (one per row: full_name, abbreviation)
"""

import requests
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# File paths
UNMATCHED_KALSHI_FILE = "GeneratedDataFiles/unmatched_kalshi_games_GOOD.txt"
OUTPUT_FILE = "GeneratedDataFiles/full_team_names_kalshi.csv"

# Kalshi API endpoint
KALSHI_MARKET_URL = "https://api.elections.kalshi.com/trade-api/v2/markets"

headers = {
    "accept": "application/json",
}

def load_unmatched_kalshi_game_ids():
    """Load Kalshi game IDs and team abbreviations from the unmatched games file."""
    games = []
    try:
        with open(UNMATCHED_KALSHI_FILE, "r") as f:
            for line in f:
                line = line.strip()
                # Skip header lines
                if not line or line.startswith("Games") or line.startswith("="):
                    continue
                parts = line.split(",")
                if len(parts) >= 3:
                    game_id = parts[0].strip()
                    team1 = parts[1].strip()
                    team2 = parts[2].strip()
                    if game_id:
                        games.append((game_id, team1, team2))
    except FileNotFoundError:
        print(f"Error: {UNMATCHED_KALSHI_FILE} not found")
        return []
    
    print(f"Loaded {len(games)} unmatched Kalshi games")
    return games

def get_team_name_from_kalshi_market(market_ticker, team_abbreviation):
    """
    Fetch full team name from a Kalshi market.
    
    Args:
        market_ticker: Full market ticker (e.g., "KXNCAAMBGAME-26FEB14GONZSCU-GONZ")
        team_abbreviation: Team abbreviation to help identify which team in the title
    
    Returns:
        Full team name from market title/subtitle, or empty string if not found
    """
    url = f"{KALSHI_MARKET_URL}/{market_ticker}"
    
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        market = data.get("market", {})
        # Try to extract team name from title or subtitle
        title = market.get("title", "") or ""
        subtitle = market.get("subtitle", "") or ""
        
        full_name = ""
        
        if title:
            # Common patterns:
            # "Will {Team} win?" -> extract {Team}
            # "{Team1} at {Team2} Winner?" -> need to identify which team
            # "{Team} to win" -> extract {Team}
            
            # First, try simple patterns like "Will X win?"
            if title.startswith("Will ") and (" win" in title or " Winner" in title):
                # Extract text between "Will " and " win"
                start = 5  # After "Will "
                end = title.find(" win")
                if end == -1:
                    end = title.find(" Winner")
                if end > start:
                    full_name = title[start:end].strip()
            # Pattern: "{Team1} at {Team2} Winner?" - need to parse both teams
            elif " at " in title and " Winner" in title:
                # Split by " at " to get both teams
                parts = title.split(" at ")
                if len(parts) == 2:
                    # Remove " Winner?" from the second part
                    team1 = parts[0].strip()
                    team2 = parts[1].replace(" Winner?", "").replace(" Winner", "").strip()
                    
                    # Try to match which team corresponds to the abbreviation
                    # This is a heuristic - we'll use the one that seems more likely
                    # For now, prefer team1 (first team mentioned)
                    # In the future, we could try to match based on abbreviation patterns
                    full_name = team1  # Default to first team
            # Pattern: "{Team} to win" or "{Team} wins"
            elif " to win" in title:
                full_name = title.replace(" to win", "").strip()
            elif " wins" in title and not title.startswith("Will "):
                full_name = title.replace(" wins", "").strip()
            else:
                # Fallback: try to clean up common suffixes
                title_clean = title.replace(" Winner?", "").replace(" Winner", "").replace(" win?", "").replace(" to win", "").replace(" wins", "").strip()
                if title_clean and len(title_clean) > 3:
                    full_name = title_clean
        
        if not full_name and subtitle:
            full_name = subtitle.strip()
        
        return full_name
    except Exception as e:
        # Silently fail - we'll use abbreviation as fallback
        return ""

def get_team_info_from_kalshi(kalshi_game_id, team1_abbr, team2_abbr, target_abbr):
    """
    Fetch full team name and abbreviation for a Kalshi game team.
    Fetches both markets to properly match team names when title contains both teams.
    
    Args:
        kalshi_game_id: Kalshi event ticker (e.g., "KXNCAAMBGAME-26FEB14GONZSCU")
        team1_abbr: First team abbreviation (e.g., "GONZ")
        team2_abbr: Second team abbreviation (e.g., "SCU")
        target_abbr: The team abbreviation we're looking for (e.g., "GONZ")
    
    Returns:
        Tuple (full_name, abbreviation)
    """
    # Fetch both markets to get both titles
    market1_ticker = f"{kalshi_game_id}-{team1_abbr}"
    market2_ticker = f"{kalshi_game_id}-{team2_abbr}"
    
    title1 = get_team_name_from_kalshi_market(market1_ticker, team1_abbr)
    title2 = get_team_name_from_kalshi_market(market2_ticker, team2_abbr)
    
    # If titles contain " at ", parse to extract both team names
    full_name = ""
    if title1 and " at " in title1:
        parts = title1.split(" at ")
        if len(parts) == 2:
            team1_name = parts[0].strip()
            team2_name = parts[1].replace(" Winner?", "").replace(" Winner", "").strip()
            # Match based on which abbreviation we're looking for
            if target_abbr == team1_abbr:
                full_name = team1_name
            elif target_abbr == team2_abbr:
                full_name = team2_name
    elif title1:
        # Single team name, use it if it's for the target team
        if target_abbr == team1_abbr:
            full_name = title1
        elif not full_name and title2:
            full_name = title2
    
    # If we still don't have a name, try the other market
    if not full_name and title2:
        if " at " in title2:
            parts = title2.split(" at ")
            if len(parts) == 2:
                team1_name = parts[0].strip()
                team2_name = parts[1].replace(" Winner?", "").replace(" Winner", "").strip()
                if target_abbr == team1_abbr:
                    full_name = team1_name
                elif target_abbr == team2_abbr:
                    full_name = team2_name
        else:
            if target_abbr == team2_abbr:
                full_name = title2
    
    # If we couldn't get full name, use abbreviation as fallback
    if not full_name:
        full_name = target_abbr
    
    return (full_name, target_abbr)

def process_single_game(game_data, unique_team_names, lock, processed_count):
    """Process a single game and update shared state with team names found."""
    kalshi_game_id, team1_abbr, team2_abbr = game_data
    
    new_teams = []
    with lock:
        # Process team 1 - fetch both markets to properly parse team names
        if team1_abbr:
            team1_info = get_team_info_from_kalshi(kalshi_game_id, team1_abbr, team2_abbr, team1_abbr)
            if team1_info not in unique_team_names:
                unique_team_names.add(team1_info)
                new_teams.append(team1_info)
        
        # Process team 2
        if team2_abbr:
            team2_info = get_team_info_from_kalshi(kalshi_game_id, team1_abbr, team2_abbr, team2_abbr)
            if team2_info not in unique_team_names:
                unique_team_names.add(team2_info)
                new_teams.append(team2_info)
        
        processed_count[0] += 1
        total_unique = len(unique_team_names)
        
        # Print new teams found (thread-safe printing)
        for full_name, abbreviation in new_teams:
            abbrev_str = f" ({abbreviation})" if abbreviation else ""
            print(f"    ✓ New team found: {full_name}{abbrev_str} (Total: {total_unique} unique teams)")

def collect_unique_team_names(limit=None, num_workers=10):
    """Collect all unique team names from unmatched Kalshi games using parallel workers."""
    print("\nCOLLECTING UNIQUE TEAM NAMES FROM UNMATCHED KALSHI GAMES")
    print("=" * 60)
    
    # Load game IDs
    games = load_unmatched_kalshi_game_ids()
    if not games:
        return set()
    
    # Limit to first N games if specified
    if limit:
        games = games[:limit]
        print(f"Limiting to first {limit} games for testing")
    
    # Thread-safe data structures
    unique_team_names = set()
    lock = Lock()
    processed_count = [0]  # Use list to allow modification in nested function
    total_games = len(games)
    
    print(f"\nFetching team names for {total_games} games using {num_workers} workers...")
    print("(Processing in parallel for faster execution)\n")
    
    # Process games in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_game = {
            executor.submit(process_single_game, game_data, unique_team_names, lock, processed_count): game_data
            for game_data in games
        }
        
        # Process completed tasks
        for future in as_completed(future_to_game):
            game_data = future_to_game[future]
            try:
                future.result()  # Wait for completion
            except Exception as e:
                print(f"  Error processing game {game_data[0]}: {e}")
            
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
