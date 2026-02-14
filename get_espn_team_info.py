import requests
from typing import Dict, List, Tuple, Optional


def get_espn_team_info(espn_game_id: str | int) -> Dict[str, any]:
    """
    Fetch team information (full names and abbreviations) for an ESPN game.
    
    Args:
        espn_game_id: ESPN game ID (e.g. 401817686)
        
    Returns:
        Dictionary with the following structure:
        {
            'game_id': str,
            'away_team': {
                'abbreviation': str,
                'full_name': str,
                'is_home': bool
            },
            'home_team': {
                'abbreviation': str,
                'full_name': str,
                'is_home': bool
            }
        }
    """
    game_id = str(espn_game_id)
    
    # Use the ESPN core API endpoint for competition details
    # This matches the pattern used in get_espn_game_timestamp_mapings.py
    competition_url = (
        f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/"
        f"mens-college-basketball/events/{game_id}/competitions/{game_id}"
    )
    
    try:
        resp = requests.get(competition_url, timeout=20)
        resp.raise_for_status()
        competition = resp.json()
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to fetch game {game_id}: {e}")
    
    # Extract competitors from competition data
    competitors = competition.get("competitors", [])
    
    if len(competitors) < 2:
        raise ValueError(f"Expected 2 competitors, found {len(competitors)} for game {game_id}")
    
    result = {
        'game_id': game_id,
        'away_team': {},
        'home_team': {}
    }
    
    for comp in competitors:
        # The team data might be a reference, so we may need to fetch it
        team_ref = comp.get("team", {})
        
        # If team is a reference (has $ref), we need to fetch it
        if "$ref" in team_ref:
            try:
                team_resp = requests.get(team_ref["$ref"], timeout=20)
                team_resp.raise_for_status()
                team = team_resp.json()
            except requests.exceptions.RequestException:
                # Fallback: try to extract from the reference or use minimal data
                team = team_ref
        else:
            team = team_ref
        
        is_home = comp.get("homeAway", "").lower() == "home"
        
        # Try multiple possible fields for full name
        full_name = (
            team.get("displayName", "") or 
            team.get("name", "") or 
            team.get("fullName", "") or
            team.get("location", "") + " " + team.get("name", "")
        ).strip()
        
        team_info = {
            'abbreviation': team.get("abbreviation", ""),
            'full_name': full_name,
            'is_home': is_home
        }
        
        if is_home:
            result['home_team'] = team_info
        else:
            result['away_team'] = team_info
    
    return result


def get_espn_team_names_and_abbreviations(espn_game_id: str | int) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Simplified function that returns just the team names and abbreviations.
    
    Args:
        espn_game_id: ESPN game ID (e.g. 401817686)
        
    Returns:
        Tuple of two dictionaries:
        - away_team: {'abbreviation': str, 'full_name': str}
        - home_team: {'abbreviation': str, 'full_name': str}
    """
    info = get_espn_team_info(espn_game_id)
    
    away = {
        'abbreviation': info['away_team']['abbreviation'],
        'full_name': info['away_team']['full_name']
    }
    
    home = {
        'abbreviation': info['home_team']['abbreviation'],
        'full_name': info['home_team']['full_name']
    }
    
    return away, home


if __name__ == "__main__":
    import sys
    
    # Test with the example game ID from the codebase
    test_game_id = 401817686
    
    if len(sys.argv) >= 2:
        test_game_id = sys.argv[1]
    
    try:
        info = get_espn_team_info(test_game_id)
        print(f"\nGame ID: {info['game_id']}")
        print(f"\nAway Team:")
        print(f"  Abbreviation: {info['away_team']['abbreviation']}")
        print(f"  Full Name: {info['away_team']['full_name']}")
        print(f"\nHome Team:")
        print(f"  Abbreviation: {info['home_team']['abbreviation']}")
        print(f"  Full Name: {info['home_team']['full_name']}")
        
        # Also demonstrate the simplified function
        print("\n" + "="*50)
        print("Using simplified function:")
        away, home = get_espn_team_names_and_abbreviations(test_game_id)
        print(f"Away: {away['full_name']} ({away['abbreviation']})")
        print(f"Home: {home['full_name']} ({home['abbreviation']})")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
