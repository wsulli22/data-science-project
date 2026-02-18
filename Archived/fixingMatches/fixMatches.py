import csv
import os
import re
import requests
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool, cpu_count

def load_espn_teams(espn_file):
    """Load ESPN team data: abbreviation -> full name"""
    abbrev_to_full = {}
    full_to_abbrevs = defaultdict(set)
    
    with open(espn_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            full_name = row['full team name'].strip()
            abbrev = row['espn team abbreviation'].strip()
            abbrev_to_full[abbrev] = full_name
            full_to_abbrevs[full_name].add(abbrev)
    
    return abbrev_to_full, full_to_abbrevs

def load_kalshi_teams(kalshi_file):
    """Load Kalshi team data: abbreviation -> full name (multiple abbrevs per team)"""
    abbrev_to_full = {}
    full_to_abbrevs = defaultdict(set)
    
    with open(kalshi_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                full_name = row[0].strip()
                abbrev = row[1].strip()
                abbrev_to_full[abbrev] = full_name
                full_to_abbrevs[full_name].add(abbrev)
    
    return abbrev_to_full, full_to_abbrevs

# Month abbreviation → number for parsing Kalshi tickers
MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
    "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# Headers for ESPN API requests
ESPN_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

def parse_kalshi_date(kalshi_id):
    """
    Extract a Python date object from a Kalshi event ticker.
    
    Example: KXNCAAMBGAME-26FEB10MILWIUIN → datetime.date(2026, 2, 10)
    Returns None if parsing fails.
    """
    if not kalshi_id.startswith("KXNCAAMBGAME-"):
        return None
    
    suffix = kalshi_id[len("KXNCAAMBGAME-"):]
    # Date format: YY + MON (3 letters) + DD (2 digits)
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
        resp = requests.get(url, headers=ESPN_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # date string like "2026-02-10T23:30Z"
        date_str = data.get("header", {}).get("competitions", [{}])[0].get("date", "")
        if date_str:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00")).date()
    except Exception:
        pass
    return None

def dates_match_within_tolerance(kalshi_date, espn_date, tolerance_days=5):
    """
    Check if two dates are within tolerance_days of each other.
    Returns tuple: (is_within_tolerance, is_exact_match)
    """
    if kalshi_date is None or espn_date is None:
        return (True, False)  # If either date is missing, don't filter by date
    
    date_diff = abs((espn_date - kalshi_date).days)
    is_exact = (date_diff == 0)
    is_within_tolerance = (date_diff <= tolerance_days)
    
    return (is_within_tolerance, is_exact)

def score_name_similarity(name1, name2):
    """
    Score how similar two team names are (0-100).
    Higher score = better match.
    """
    if not name1 or not name2:
        return 0
    
    norm1 = normalize_team_name(name1)
    norm2 = normalize_team_name(name2)
    
    # Exact match
    if norm1 == norm2:
        return 100
    
    # Word-based matching
    words1 = set(norm1.split())
    words2 = set(norm2.split())
    
    if not words1 or not words2:
        return 0
    
    # Calculate word overlap
    common_words = words1 & words2
    total_words = words1 | words2
    
    if not total_words:
        return 0
    
    # Base score from word overlap
    word_overlap_score = (len(common_words) / len(total_words)) * 80
    
    # Bonus for substring matches (one name contains the other)
    if norm1 in norm2 or norm2 in norm1:
        word_overlap_score += 15
    
    # Bonus for significant word matches (words longer than 3 chars)
    significant_words1 = {w for w in words1 if len(w) > 3}
    significant_words2 = {w for w in words2 if len(w) > 3}
    if significant_words1 and significant_words2:
        significant_common = significant_words1 & significant_words2
        if significant_common:
            word_overlap_score += min(5, len(significant_common) * 2)
    
    return min(100, int(word_overlap_score))

def find_best_pairing(kalshi_team1_full, kalshi_team2_full, espn_team1_full, espn_team2_full):
    """
    Determine the best pairing between Kalshi and ESPN teams.
    Returns (kalshi_team1_matched, espn_team1_matched, kalshi_team2_matched, espn_team2_matched, reversed)
    where reversed indicates if we swapped the ESPN teams.
    """
    # Option 1: Direct pairing (Kalshi 1 ↔ ESPN 1, Kalshi 2 ↔ ESPN 2)
    score1 = score_name_similarity(kalshi_team1_full, espn_team1_full) + \
             score_name_similarity(kalshi_team2_full, espn_team2_full)
    
    # Option 2: Reversed pairing (Kalshi 1 ↔ ESPN 2, Kalshi 2 ↔ ESPN 1)
    score2 = score_name_similarity(kalshi_team1_full, espn_team2_full) + \
             score_name_similarity(kalshi_team2_full, espn_team1_full)
    
    # Choose the better pairing
    if score2 > score1:
        # Reversed pairing is better
        return (kalshi_team1_full, espn_team2_full, kalshi_team2_full, espn_team1_full, True)
    else:
        # Direct pairing is better
        return (kalshi_team1_full, espn_team1_full, kalshi_team2_full, espn_team2_full, False)

def normalize_team_name(name):
    """Normalize team name for matching"""
    # Remove common suffixes and normalize
    name = name.upper()
    # Remove common suffixes
    for suffix in [' WILDCATS', ' TIGERS', ' EAGLES', ' BULLDOGS', ' LIONS', 
                   ' BEARS', ' COUGARS', ' WOLVES', ' HORNETS', ' HAWKS',
                   ' RAMS', ' PANTHERS', ' KNIGHTS', ' CRUSADERS', ' SAINTS',
                   ' MOUNTAINEERS', ' RED WOLVES', ' GOLDEN EAGLES', ' GOLDEN LIONS',
                   ' SUN DEVILS', ' RAZORBACKS', ' BLACK KNIGHTS', ' BRUINS',
                   ' CARDINALS', ' BISON', ' BULLS', ' COLONELS', ' DEMONS',
                   ' FLAMES', ' GATORS', ' GOVERNORS', ' HUSKIES', ' JAYHAWKS',
                   ' LADY VOLS', ' LONGHORNS', ' MAVERICKS', ' MINERS', ' OWLS',
                   ' PIRATES', ' RED RAIDERS', ' SEMINOLES', ' SPARTANS', ' TAR HEELS',
                   ' TERRIERS', ' TROJANS', ' VOLUNTEERS', ' WOLVERINES', ' YELLOW JACKETS',
                   ' ZIPS', ' AGGIES', ' BOBCATS', ' CHIEFTAINS', ' COMETS',
                   ' CRIMSON TIDE', ' DUKES', ' FIGHTING IRISH', ' GOLDEN BEARS',
                   ' HOKIES', ' HURRICANES', ' JAYHAWKS', ' LOBOS', ' MONARCHS',
                   ' NITTANY LIONS', ' ORANGE', ' QUAKERS', ' REBELS', ' SCARLET KNIGHTS',
                   ' SHOCKERS', ' SOONERS', ' TERRAPINS', ' THUNDERING HERD', ' TIDE',
                   ' TORNADOES', ' UCONN', ' WILDCATS', ' WOLFPACK']:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
    
    # Remove state/college indicators
    name = name.replace(' ST.', ' STATE')
    name = name.replace(' ST ', ' STATE ')
    name = name.replace(' STATE', '')
    name = name.replace(' UNIVERSITY', '')
    name = name.replace(' UNIV', '')
    name = name.replace(' COLLEGE', '')
    name = name.replace(' U.', '')
    name = name.replace(' U ', ' ')
    
    return name.strip()

def print_formatted_match(kalshi_id, kalshi_date, espn_id, espn_date,
                          espn_team1_full, espn_team1_abbrev, kalshi_team1_full, kalshi_team1_abbrev,
                          espn_team2_full, espn_team2_abbrev, kalshi_team2_full, kalshi_team2_abbrev,
                          confidence):
    """Print a formatted match output"""
    # Format dates
    kalshi_date_str = f"({kalshi_date})" if kalshi_date else ""
    espn_date_str = f"({espn_date})" if espn_date else ""
    
    # Check if dates match exactly and calculate day difference
    is_exact = True
    day_diff = 0
    if kalshi_date and espn_date:
        _, is_exact = dates_match_within_tolerance(kalshi_date, espn_date)
        day_diff = abs((espn_date - kalshi_date).days)
    
    # Calculate column widths for alignment
    # First column: game IDs (with some padding)
    id_col_width = max(len(kalshi_id), len(espn_id), 20)
    # Second column: dates (with padding)
    date_col_width = max(len(kalshi_date_str), len(espn_date_str), 15)
    
    # Print Kalshi game ID and Team 1
    line1 = f"{kalshi_id:<{id_col_width}} {kalshi_date_str:<{date_col_width}} {espn_team1_full} ({espn_team1_abbrev})"
    print(f"\n{line1}")
    
    # Print ESPN game ID and Kalshi Team 1
    line2 = f"{espn_id:<{id_col_width}} {espn_date_str:<{date_col_width}} {kalshi_team1_full} ({kalshi_team1_abbrev})"
    print(line2)
    
    # Print Team 2 (aligned to team column)
    indent = " " * (id_col_width + date_col_width + 2)  # IDs + dates + 2 spaces
    line3 = f"{indent}{espn_team2_full} ({espn_team2_abbrev})"
    print(line3)
    
    line4 = f"{indent}{kalshi_team2_full} ({kalshi_team2_abbrev})"
    confidence_msg = f" (confidence: {confidence})"
    if not is_exact and day_diff > 0:
        if day_diff == 1:
            confidence_msg += f" [1 day off]"
        else:
            confidence_msg += f" [{day_diff} days off]"
    print(f"{line4}{confidence_msg}")

def load_unmatched_kalshi_games(kalshi_games_file):
    """Load unmatched Kalshi games, returns games_list"""
    games = []
    with open(kalshi_games_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('Games') and not line.startswith('='):
                parts = line.split(',')
                if len(parts) >= 3:
                    kalshi_id = parts[0].strip()
                    team1_abbrev = parts[1].strip()
                    team2_abbrev = parts[2].strip()
                    games.append((kalshi_id, team1_abbrev, team2_abbrev))
    return games

def load_all_espn_games(espn_games_file):
    """Load all ESPN games from file, returns (games_list, espn_id_to_date_dict)"""
    games = []
    espn_id_to_date = {}
    with open(espn_games_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('Games') and not line.startswith('='):
                parts = line.split(',')
                if len(parts) >= 3:
                    espn_id = parts[0].strip()
                    team1_abbrev = parts[1].strip()
                    team2_abbrev = parts[2].strip()
                    games.append((espn_id, team1_abbrev, team2_abbrev))
                    # Extract date if available (5th field)
                    if len(parts) >= 5:
                        date_str = parts[4].strip()
                        try:
                            espn_id_to_date[espn_id] = datetime.strptime(date_str, '%Y-%m-%d').date()
                        except (ValueError, IndexError):
                            pass
    return games, espn_id_to_date

def load_matched_espn_ids(matched_games_file):
    """Load set of matched ESPN game IDs from mappings file"""
    matched_espn_ids = set()
    if not os.path.exists(matched_games_file):
        return matched_espn_ids
    
    with open(matched_games_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            espn_id = row.get('espn_game_id', '').strip()
            if espn_id:
                matched_espn_ids.add(espn_id)
    return matched_espn_ids

def find_all_matching_espn_games(kalshi_team1_full, kalshi_team2_full, kalshi_team1_abbrevs, kalshi_team2_abbrevs, 
                                 kalshi_team1_orig_abbrev, kalshi_team2_orig_abbrev,
                                 espn_games, espn_abbrev_to_full, espn_full_to_abbrevs, espn_id_to_date=None, kalshi_date=None):
    """Find all matching ESPN games based on abbreviations first, then team names. Returns list of matches sorted by confidence."""
    
    # Convert abbreviation sets to lists for easier checking
    kalshi1_abbrev_list = list(kalshi_team1_abbrevs) if isinstance(kalshi_team1_abbrevs, set) else kalshi_team1_abbrevs.split(', ')
    kalshi2_abbrev_list = list(kalshi_team2_abbrevs) if isinstance(kalshi_team2_abbrevs, set) else kalshi_team2_abbrevs.split(', ')
    
    # Normalize Kalshi team names
    kalshi1_norm = normalize_team_name(kalshi_team1_full)
    kalshi2_norm = normalize_team_name(kalshi_team2_full)
    
    all_matches = []
    
    # First pass: look for exact abbreviation matches (highest priority)
    exact_abbrev_matches = []
    
    for espn_id, espn_team1_abbrev, espn_team2_abbrev in espn_games:
        # Check for exact abbreviation match first (from original game data)
        exact_direct = (espn_team1_abbrev == kalshi_team1_orig_abbrev and espn_team2_abbrev == kalshi_team2_orig_abbrev)
        exact_reversed = (espn_team1_abbrev == kalshi_team2_orig_abbrev and espn_team2_abbrev == kalshi_team1_orig_abbrev)
        
        if exact_direct or exact_reversed:
            espn_team1_full = espn_abbrev_to_full.get(espn_team1_abbrev, '')
            espn_team2_full = espn_abbrev_to_full.get(espn_team2_abbrev, '')
            if espn_team1_full and espn_team2_full:
                # Verify date matches if we have Kalshi date
                if kalshi_date is not None and espn_id_to_date:
                    espn_date = espn_id_to_date.get(espn_id)
                    if espn_date:
                        is_within_tolerance, is_exact = dates_match_within_tolerance(kalshi_date, espn_date)
                        if not is_within_tolerance:
                            continue  # Date doesn't match within tolerance, skip this candidate
                        if not is_exact:
                            # Date is within tolerance but not exact - will be noted when displaying
                            pass
                exact_abbrev_matches.append((espn_id, espn_team1_full, espn_team1_abbrev, espn_team2_full, espn_team2_abbrev, exact_reversed, 100))
    
    # Add exact matches to all_matches
    all_matches.extend(exact_abbrev_matches)
    
    # If we found exact abbreviation matches with dates, return only those
    if exact_abbrev_matches and kalshi_date is not None:
        return sorted(all_matches, key=lambda x: x[6], reverse=True)  # Sort by confidence
    
    # Second pass: look for matches in abbreviation sets and names
    for espn_id, espn_team1_abbrev, espn_team2_abbrev in espn_games:
        # Skip if already in exact matches
        if any(match[0] == espn_id for match in all_matches):
            continue
            
        # Get full names for ESPN teams
        espn_team1_full = espn_abbrev_to_full.get(espn_team1_abbrev, '')
        espn_team2_full = espn_abbrev_to_full.get(espn_team2_abbrev, '')
        
        if not espn_team1_full or not espn_team2_full:
            continue
        
        # Get all ESPN abbreviations for these teams
        espn1_abbrevs = espn_full_to_abbrevs.get(espn_team1_full, {espn_team1_abbrev})
        espn2_abbrevs = espn_full_to_abbrevs.get(espn_team2_full, {espn_team2_abbrev})
        
        # Normalize ESPN team names
        espn1_norm = normalize_team_name(espn_team1_full)
        espn2_norm = normalize_team_name(espn_team2_full)
        
        # Check for abbreviation matches in sets
        abbrev_match1_direct = kalshi_team1_orig_abbrev in espn1_abbrevs or any(abbrev in espn1_abbrevs for abbrev in kalshi1_abbrev_list)
        abbrev_match2_direct = kalshi_team2_orig_abbrev in espn2_abbrevs or any(abbrev in espn2_abbrevs for abbrev in kalshi2_abbrev_list)
        abbrev_match1_reversed = kalshi_team1_orig_abbrev in espn2_abbrevs or any(abbrev in espn2_abbrevs for abbrev in kalshi1_abbrev_list)
        abbrev_match2_reversed = kalshi_team2_orig_abbrev in espn1_abbrevs or any(abbrev in espn1_abbrevs for abbrev in kalshi2_abbrev_list)
        
        score1 = 0
        score2 = 0
        
        # Direct order: check abbreviation matches first
        if abbrev_match1_direct:
            score1 += 3  # High weight for abbreviation match
        elif kalshi1_norm == espn1_norm:  # Exact normalized name match
            score1 += 2
        elif abs(len(kalshi1_norm) - len(espn1_norm)) <= 2 and len(kalshi1_norm) >= 5 and (kalshi1_norm in espn1_norm or espn1_norm in kalshi1_norm):
            score1 += 1  # Only allow substring if lengths are close and names are substantial
        
        if abbrev_match2_direct:
            score1 += 3
        elif kalshi2_norm == espn2_norm:
            score1 += 2
        elif abs(len(kalshi2_norm) - len(espn2_norm)) <= 2 and len(kalshi2_norm) >= 5 and (kalshi2_norm in espn2_norm or espn2_norm in kalshi2_norm):
            score1 += 1
        
        # Reversed order
        if abbrev_match1_reversed:
            score2 += 3
        elif kalshi1_norm == espn2_norm:
            score2 += 2
        elif abs(len(kalshi1_norm) - len(espn2_norm)) <= 2 and len(kalshi1_norm) >= 5 and (kalshi1_norm in espn2_norm or espn2_norm in kalshi1_norm):
            score2 += 1
        
        if abbrev_match2_reversed:
            score2 += 3
        elif kalshi2_norm == espn1_norm:
            score2 += 2
        elif abs(len(kalshi2_norm) - len(espn1_norm)) <= 2 and len(kalshi2_norm) >= 5 and (kalshi2_norm in espn1_norm or espn1_norm in kalshi2_norm):
            score2 += 1
        
        score = max(score1, score2)
        
        # Calculate confidence score (0-100 scale)
        # Base score is 0-6, convert to 0-100 scale
        # Perfect match (both abbrevs) = 100, both name matches = 80, one abbrev + one name = 70, etc.
        confidence = 0
        if score >= 6:  # Both teams have abbreviation matches
            confidence = 95
        elif score >= 5:  # One abbrev + one perfect name match
            confidence = 85
        elif score >= 4:  # Both teams have name matches or one abbrev match
            if abbrev_match1_direct or abbrev_match2_direct or abbrev_match1_reversed or abbrev_match2_reversed:
                confidence = 75
            else:
                confidence = 65
        elif score >= 3:  # One team has strong match
            confidence = 55
        elif score >= 2:  # Both teams have weak matches
            confidence = 45
        
        # Require both teams to match with high confidence (score >= 5 for both abbreviation matches, or >= 4 with at least one abbreviation)
        if score >= 5 or (score >= 4 and (abbrev_match1_direct or abbrev_match2_direct or abbrev_match1_reversed or abbrev_match2_reversed)):
            # Verify date matches if we have Kalshi date
            if kalshi_date is not None and espn_id_to_date:
                espn_date = espn_id_to_date.get(espn_id)
                if espn_date:
                    is_within_tolerance, is_exact = dates_match_within_tolerance(kalshi_date, espn_date)
                    if not is_within_tolerance:
                        continue  # Date doesn't match within tolerance, skip this candidate
                    if not is_exact:
                        # Date is within tolerance but not exact - will be noted when displaying
                        pass
            
            # Add this match to the list
            if score == score1:
                all_matches.append((espn_id, espn_team1_full, espn_team1_abbrev, espn_team2_full, espn_team2_abbrev, False, confidence))
            else:
                all_matches.append((espn_id, espn_team2_full, espn_team2_abbrev, espn_team1_full, espn_team1_abbrev, True, confidence))
    
    # Return all matches sorted by confidence (highest first)
    return sorted(all_matches, key=lambda x: x[6], reverse=True) if all_matches else []

def find_closest_match_any_confidence(kalshi_team1_full, kalshi_team2_full, kalshi_team1_abbrevs, kalshi_team2_abbrevs, 
                                     kalshi_team1_orig_abbrev, kalshi_team2_orig_abbrev,
                                     espn_games, espn_abbrev_to_full, espn_full_to_abbrevs, espn_id_to_date=None, kalshi_date=None):
    """Find the closest match from ESPN games with any confidence level (even very low)."""
    
    # Convert abbreviation sets to lists for easier checking
    kalshi1_abbrev_list = list(kalshi_team1_abbrevs) if isinstance(kalshi_team1_abbrevs, set) else kalshi_team1_abbrevs.split(', ')
    kalshi2_abbrev_list = list(kalshi_team2_abbrevs) if isinstance(kalshi_team2_abbrevs, set) else kalshi_team2_abbrevs.split(', ')
    
    # Normalize Kalshi team names
    kalshi1_norm = normalize_team_name(kalshi_team1_full)
    kalshi2_norm = normalize_team_name(kalshi_team2_full)
    
    best_match = None
    best_score = -1
    candidates = []  # Store top candidates before checking dates
    
    for espn_id, espn_team1_abbrev, espn_team2_abbrev in espn_games:
        # Get full names for ESPN teams
        espn_team1_full = espn_abbrev_to_full.get(espn_team1_abbrev, '')
        espn_team2_full = espn_abbrev_to_full.get(espn_team2_abbrev, '')
        
        if not espn_team1_full or not espn_team2_full:
            continue
        
        # Get all ESPN abbreviations for these teams
        espn1_abbrevs = espn_full_to_abbrevs.get(espn_team1_full, {espn_team1_abbrev})
        espn2_abbrevs = espn_full_to_abbrevs.get(espn_team2_full, {espn_team2_abbrev})
        
        # Normalize ESPN team names
        espn1_norm = normalize_team_name(espn_team1_full)
        espn2_norm = normalize_team_name(espn_team2_full)
        
        # Check for abbreviation matches
        abbrev_match1_direct = kalshi_team1_orig_abbrev in espn1_abbrevs or any(abbrev in espn1_abbrevs for abbrev in kalshi1_abbrev_list)
        abbrev_match2_direct = kalshi_team2_orig_abbrev in espn2_abbrevs or any(abbrev in espn2_abbrevs for abbrev in kalshi2_abbrev_list)
        abbrev_match1_reversed = kalshi_team1_orig_abbrev in espn2_abbrevs or any(abbrev in espn2_abbrevs for abbrev in kalshi1_abbrev_list)
        abbrev_match2_reversed = kalshi_team2_orig_abbrev in espn1_abbrevs or any(abbrev in espn1_abbrevs for abbrev in kalshi2_abbrev_list)
        
        score1 = 0
        score2 = 0
        
        # Direct order: check abbreviation matches first
        if abbrev_match1_direct:
            score1 += 3
        elif kalshi1_norm == espn1_norm:
            score1 += 2
        elif kalshi1_norm in espn1_norm or espn1_norm in kalshi1_norm:
            score1 += 1
        elif len(set(kalshi1_norm.split()) & set(espn1_norm.split())) > 0:
            score1 += 0.5  # At least one word in common
        
        if abbrev_match2_direct:
            score1 += 3
        elif kalshi2_norm == espn2_norm:
            score1 += 2
        elif kalshi2_norm in espn2_norm or espn2_norm in kalshi2_norm:
            score1 += 1
        elif len(set(kalshi2_norm.split()) & set(espn2_norm.split())) > 0:
            score1 += 0.5
        
        # Reversed order
        if abbrev_match1_reversed:
            score2 += 3
        elif kalshi1_norm == espn2_norm:
            score2 += 2
        elif kalshi1_norm in espn2_norm or espn2_norm in kalshi1_norm:
            score2 += 1
        elif len(set(kalshi1_norm.split()) & set(espn2_norm.split())) > 0:
            score2 += 0.5
        
        if abbrev_match2_reversed:
            score2 += 3
        elif kalshi2_norm == espn1_norm:
            score2 += 2
        elif kalshi2_norm in espn1_norm or espn1_norm in kalshi2_norm:
            score2 += 1
        elif len(set(kalshi2_norm.split()) & set(espn1_norm.split())) > 0:
            score2 += 0.5
        
        score = max(score1, score2)
        
        # Store candidates with score > 0 (only check dates for promising matches)
        if score > 0:
            if score == score1:
                candidates.append((score, espn_id, espn_team1_full, espn_team1_abbrev, espn_team2_full, espn_team2_abbrev, False))
            else:
                candidates.append((score, espn_id, espn_team2_full, espn_team2_abbrev, espn_team1_full, espn_team1_abbrev, True))
    
    # Sort candidates by score (highest first) and only check dates for top 10 to avoid too many API calls
    candidates.sort(reverse=True, key=lambda x: x[0])
    top_candidates = candidates[:10] if len(candidates) > 10 else candidates
    
    # Check dates for top candidates and find best match
    for score, espn_id, espn_team1_full, espn_team1_abbrev, espn_team2_full, espn_team2_abbrev, reversed_order in top_candidates:
        date_penalty = 0
        if kalshi_date is not None and espn_id_to_date:
            espn_date = espn_id_to_date.get(espn_id)
            if espn_date:
                is_within_tolerance, is_exact = dates_match_within_tolerance(kalshi_date, espn_date)
                if not is_within_tolerance:
                    date_penalty = -1  # Penalize if date is way off
                elif not is_exact:
                    date_penalty = -0.5  # Small penalty if not exact
        
        final_score = score + date_penalty
        
        if final_score > best_score:
            best_score = final_score
            # Calculate confidence (0-100 scale, but can be very low)
            confidence = max(0, min(100, int(final_score * 15)))  # Scale the score to 0-100
            best_match = (espn_id, espn_team1_full, espn_team1_abbrev, espn_team2_full, espn_team2_abbrev, reversed_order, confidence)
    
    # If no match found from top candidates with dates, use best candidate without date check
    if best_match is None and candidates:
        score, espn_id, espn_team1_full, espn_team1_abbrev, espn_team2_full, espn_team2_abbrev, reversed_order = candidates[0]
        confidence = max(0, min(100, int(score * 15)))
        best_match = (espn_id, espn_team1_full, espn_team1_abbrev, espn_team2_full, espn_team2_abbrev, reversed_order, confidence)
    
    return best_match

def process_single_game(args):
    """Worker function to process a single Kalshi game. Returns (idx, matches, output_lines)."""
    (idx, total_games, kalshi_id, kalshi_team1_abbrev, kalshi_team2_abbrev,
     espn_games, espn_abbrev_to_full, espn_full_to_abbrevs, espn_id_to_date,
     kalshi_abbrev_to_full, kalshi_full_to_abbrevs) = args
    
    matches = []
    output_lines = []
    kalshi_date = parse_kalshi_date(kalshi_id)
    
    # Status message for each game
    status_msg = f"[{idx}/{total_games}] {kalshi_id} ({kalshi_team1_abbrev} vs {kalshi_team2_abbrev})"
    if kalshi_date:
        status_msg += f" [{kalshi_date}]"
    status_msg += " ... "
    
    match_found = False
    # First, try to find ESPN match by exact abbreviation match (highest priority)
    espn_match = None
    for espn_id, espn_team1_abbrev, espn_team2_abbrev in espn_games:
        if (espn_team1_abbrev == kalshi_team1_abbrev and espn_team2_abbrev == kalshi_team2_abbrev) or \
           (espn_team1_abbrev == kalshi_team2_abbrev and espn_team2_abbrev == kalshi_team1_abbrev):
            espn_team1_full = espn_abbrev_to_full.get(espn_team1_abbrev, '')
            espn_team2_full = espn_abbrev_to_full.get(espn_team2_abbrev, '')
            if espn_team1_full and espn_team2_full:
                # Verify date matches if we have Kalshi date
                if kalshi_date is not None and espn_id_to_date:
                    espn_date = espn_id_to_date.get(espn_id)
                    if espn_date:
                        is_within_tolerance, is_exact = dates_match_within_tolerance(kalshi_date, espn_date)
                        if not is_within_tolerance:
                            continue  # Date doesn't match within tolerance, skip this candidate
                
                reversed_order = (espn_team1_abbrev == kalshi_team2_abbrev)
                espn_match = (espn_id, espn_team1_full, espn_team1_abbrev, espn_team2_full, espn_team2_abbrev, reversed_order)
                break
    
    # If we found an exact ESPN match, use it to find the correct Kalshi teams
    if espn_match:
        espn_id, espn_team1_full, espn_team1_abbrev, espn_team2_full, espn_team2_abbrev, reversed_order = espn_match
        confidence = 100  # Exact abbreviation match = perfect confidence
        
        # Find Kalshi teams that match the ESPN teams
        espn1_norm = normalize_team_name(espn_team1_full)
        espn2_norm = normalize_team_name(espn_team2_full)
        
        kalshi_team1_full = None
        kalshi_team2_full = None
        best_score1 = 0
        best_score2 = 0
        
        kalshi_team1_candidates = [name for name, abbrevs in kalshi_full_to_abbrevs.items() if kalshi_team1_abbrev in abbrevs]
        kalshi_team2_candidates = [name for name, abbrevs in kalshi_full_to_abbrevs.items() if kalshi_team2_abbrev in abbrevs]
        
        # Match team 1
        espn1_words = set(espn1_norm.split())
        for candidate in kalshi_team1_candidates:
            candidate_norm = normalize_team_name(candidate)
            candidate_words = set(candidate_norm.split())
            score = 0
            if candidate_norm == espn1_norm:
                score = 10
            elif len(espn1_words & candidate_words) >= 2:
                score = 8
            elif len(espn1_words & candidate_words) >= 1:
                score = 5
            elif abs(len(candidate_norm) - len(espn1_norm)) <= 3 and len(candidate_norm) >= 5:
                if candidate_norm in espn1_norm or espn1_norm in candidate_norm:
                    score = 3
            if score > best_score1:
                best_score1 = score
                kalshi_team1_full = candidate
        
        # Match team 2
        espn2_words = set(espn2_norm.split())
        for candidate in kalshi_team2_candidates:
            candidate_norm = normalize_team_name(candidate)
            candidate_words = set(candidate_norm.split())
            score = 0
            if candidate_norm == espn2_norm:
                score = 10
            elif len(espn2_words & candidate_words) >= 2:
                score = 8
            elif len(espn2_words & candidate_words) >= 1:
                score = 5
            elif abs(len(candidate_norm) - len(espn2_norm)) <= 3 and len(candidate_norm) >= 5:
                if candidate_norm in espn2_norm or espn2_norm in candidate_norm:
                    score = 3
            if score > best_score2:
                best_score2 = score
                kalshi_team2_full = candidate
        
        if not kalshi_team1_full and kalshi_team1_candidates:
            kalshi_team1_full = kalshi_team1_candidates[0]
        if not kalshi_team2_full and kalshi_team2_candidates:
            kalshi_team2_full = kalshi_team2_candidates[0]
        
        if kalshi_team1_full and kalshi_team2_full:
            kalshi_team1_abbrevs_set = kalshi_full_to_abbrevs.get(kalshi_team1_full, {kalshi_team1_abbrev})
            kalshi_team2_abbrevs_set = kalshi_full_to_abbrevs.get(kalshi_team2_full, {kalshi_team2_abbrev})
            kalshi_team1_abbrevs = ', '.join(sorted(kalshi_team1_abbrevs_set))
            kalshi_team2_abbrevs = ', '.join(sorted(kalshi_team2_abbrevs_set))
            
            espn_date = espn_id_to_date.get(espn_id) if espn_id_to_date else None
            day_diff = 0
            if kalshi_date and espn_date:
                day_diff = abs((espn_date - kalshi_date).days)
            
            matches.append({
                'kalshi_id': kalshi_id,
                'kalshi_team1_full': kalshi_team1_full,
                'kalshi_team1_abbrevs': kalshi_team1_abbrevs,
                'kalshi_team2_full': kalshi_team2_full,
                'kalshi_team2_abbrevs': kalshi_team2_abbrevs,
                'espn_id': espn_id,
                'espn_team1_full': espn_team1_full,
                'espn_team1_abbrev': espn_team1_abbrev,
                'espn_team2_full': espn_team2_full,
                'espn_team2_abbrev': espn_team2_abbrev,
                'reversed': reversed_order,
                'confidence': confidence,
                'match_type': 'exact',
                'kalshi_date': kalshi_date,
                'espn_date': espn_date,
                'day_difference': day_diff
            })
            match_found = True
            output_lines.append(status_msg + "✓ MATCHED")
            return (idx, matches, output_lines)
    
    # If no exact abbreviation match, fall back to the previous method
    kalshi_team1_candidates = []
    kalshi_team2_candidates = []
    
    for full_name, abbrevs in kalshi_full_to_abbrevs.items():
        if kalshi_team1_abbrev in abbrevs:
            kalshi_team1_candidates.append(full_name)
        if kalshi_team2_abbrev in abbrevs:
            kalshi_team2_candidates.append(full_name)
    
    if not kalshi_team1_candidates or not kalshi_team2_candidates:
        # Try to find closest match even without candidates
        if not match_found:
            kalshi_team1_full_fallback = kalshi_abbrev_to_full.get(kalshi_team1_abbrev, kalshi_team1_abbrev)
            kalshi_team2_full_fallback = kalshi_abbrev_to_full.get(kalshi_team2_abbrev, kalshi_team2_abbrev)
            
            closest_match = find_closest_match_any_confidence(
                kalshi_team1_full_fallback, kalshi_team2_full_fallback, {kalshi_team1_abbrev}, {kalshi_team2_abbrev},
                kalshi_team1_abbrev, kalshi_team2_abbrev,
                espn_games, espn_abbrev_to_full, espn_full_to_abbrevs, espn_id_to_date, kalshi_date
            )
            if closest_match:
                espn_id, espn_team1_full, espn_team1_abbrev, espn_team2_full, espn_team2_abbrev, reversed_order, confidence = closest_match
                espn_date = espn_id_to_date.get(espn_id) if espn_id_to_date else None
                day_diff = 0
                if kalshi_date and espn_date:
                    day_diff = abs((espn_date - kalshi_date).days)
                
                matches.append({
                    'kalshi_id': kalshi_id,
                    'kalshi_team1_full': kalshi_team1_full_fallback,
                    'kalshi_team1_abbrevs': kalshi_team1_abbrev,
                    'kalshi_team2_full': kalshi_team2_full_fallback,
                    'kalshi_team2_abbrevs': kalshi_team2_abbrev,
                    'espn_id': espn_id,
                    'espn_team1_full': espn_team1_full,
                    'espn_team1_abbrev': espn_team1_abbrev,
                    'espn_team2_full': espn_team2_full,
                    'espn_team2_abbrev': espn_team2_abbrev,
                    'reversed': reversed_order,
                    'confidence': confidence,
                    'match_type': 'closest',
                    'kalshi_date': kalshi_date,
                    'espn_date': espn_date,
                    'day_difference': day_diff
                })
                output_lines.append(status_msg + "✓ CLOSEST MATCH")
            else:
                output_lines.append(status_msg + "✗ NO MATCHES")
        return (idx, matches, output_lines)
    
    # Try all combinations to find all potential matches
    all_proposed_matches = []
    
    for kalshi_team1_full in kalshi_team1_candidates:
        for kalshi_team2_full in kalshi_team2_candidates:
            kalshi_team1_abbrevs_set = kalshi_full_to_abbrevs.get(kalshi_team1_full, {kalshi_team1_abbrev})
            kalshi_team2_abbrevs_set = kalshi_full_to_abbrevs.get(kalshi_team2_full, {kalshi_team2_abbrev})
            
            matches_list = find_all_matching_espn_games(kalshi_team1_full, kalshi_team2_full, kalshi_team1_abbrevs_set, kalshi_team2_abbrevs_set,
                                                       kalshi_team1_abbrev, kalshi_team2_abbrev,
                                                       espn_games, espn_abbrev_to_full, espn_full_to_abbrevs, espn_id_to_date, kalshi_date)
            
            for match in matches_list:
                all_proposed_matches.append({
                    'match': match,
                    'kalshi_team1_full': kalshi_team1_full,
                    'kalshi_team1_abbrevs_set': kalshi_team1_abbrevs_set,
                    'kalshi_team2_full': kalshi_team2_full,
                    'kalshi_team2_abbrevs_set': kalshi_team2_abbrevs_set
                })
    
    # Sort all proposed matches by confidence
    all_proposed_matches.sort(key=lambda x: x['match'][6], reverse=True)
    
    if all_proposed_matches:
        # Only save the BEST match (first one, highest confidence)
        best_prop_match = all_proposed_matches[0]
        match = best_prop_match['match']
        kalshi_team1_full = best_prop_match['kalshi_team1_full']
        kalshi_team1_abbrevs_set = best_prop_match['kalshi_team1_abbrevs_set']
        kalshi_team2_full = best_prop_match['kalshi_team2_full']
        kalshi_team2_abbrevs_set = best_prop_match['kalshi_team2_abbrevs_set']
        kalshi_team1_abbrevs = ', '.join(sorted(kalshi_team1_abbrevs_set))
        kalshi_team2_abbrevs = ', '.join(sorted(kalshi_team2_abbrevs_set))
        
        espn_id, espn_team1_full, espn_team1_abbrev, espn_team2_full, espn_team2_abbrev, reversed_order, confidence = match
        espn_date = espn_id_to_date.get(espn_id) if espn_id_to_date else None
        day_diff = 0
        if kalshi_date and espn_date:
            day_diff = abs((espn_date - kalshi_date).days)
        
        matches.append({
            'kalshi_id': kalshi_id,
            'kalshi_team1_full': kalshi_team1_full,
            'kalshi_team1_abbrevs': kalshi_team1_abbrevs,
            'kalshi_team2_full': kalshi_team2_full,
            'kalshi_team2_abbrevs': kalshi_team2_abbrevs,
            'espn_id': espn_id,
            'espn_team1_full': espn_team1_full,
            'espn_team1_abbrev': espn_team1_abbrev,
            'espn_team2_full': espn_team2_full,
            'espn_team2_abbrev': espn_team2_abbrev,
            'reversed': reversed_order,
            'confidence': confidence,
            'match_type': 'proposed',
            'kalshi_date': kalshi_date,
            'espn_date': espn_date,
            'day_difference': day_diff
        })
        output_lines.append(status_msg + f"✓ MATCHED (best of {len(all_proposed_matches)} proposed)")
    else:
        # No matches found - try to find closest match
        if not match_found:
            closest_match = None
            closest_match_info = None
            best_closest_score = -1
            
            for kalshi_team1_full in kalshi_team1_candidates:
                for kalshi_team2_full in kalshi_team2_candidates:
                    kalshi_team1_abbrevs_set = kalshi_full_to_abbrevs.get(kalshi_team1_full, {kalshi_team1_abbrev})
                    kalshi_team2_abbrevs_set = kalshi_full_to_abbrevs.get(kalshi_team2_full, {kalshi_team2_abbrev})
                    
                    match = find_closest_match_any_confidence(
                        kalshi_team1_full, kalshi_team2_full, kalshi_team1_abbrevs_set, kalshi_team2_abbrevs_set,
                        kalshi_team1_abbrev, kalshi_team2_abbrev,
                        espn_games, espn_abbrev_to_full, espn_full_to_abbrevs, espn_id_to_date, kalshi_date
                    )
                    
                    if match and match[6] > best_closest_score:
                        best_closest_score = match[6]
                        closest_match = match
                        closest_match_info = {
                            'kalshi_team1_full': kalshi_team1_full,
                            'kalshi_team1_abbrevs_set': kalshi_team1_abbrevs_set,
                            'kalshi_team2_full': kalshi_team2_full,
                            'kalshi_team2_abbrevs_set': kalshi_team2_abbrevs_set
                        }
            
            if closest_match and closest_match_info:
                kalshi_team1_full = closest_match_info['kalshi_team1_full']
                kalshi_team1_abbrevs_set = closest_match_info['kalshi_team1_abbrevs_set']
                kalshi_team2_full = closest_match_info['kalshi_team2_full']
                kalshi_team2_abbrevs_set = closest_match_info['kalshi_team2_abbrevs_set']
                kalshi_team1_abbrevs = ', '.join(sorted(kalshi_team1_abbrevs_set))
                kalshi_team2_abbrevs = ', '.join(sorted(kalshi_team2_abbrevs_set))
                
                espn_id, espn_team1_full, espn_team1_abbrev, espn_team2_full, espn_team2_abbrev, reversed_order, confidence = closest_match
                espn_date = espn_id_to_date.get(espn_id) if espn_id_to_date else None
                day_diff = 0
                if kalshi_date and espn_date:
                    day_diff = abs((espn_date - kalshi_date).days)
                
                matches.append({
                    'kalshi_id': kalshi_id,
                    'kalshi_team1_full': kalshi_team1_full,
                    'kalshi_team1_abbrevs': kalshi_team1_abbrevs,
                    'kalshi_team2_full': kalshi_team2_full,
                    'kalshi_team2_abbrevs': kalshi_team2_abbrevs,
                    'espn_id': espn_id,
                    'espn_team1_full': espn_team1_full,
                    'espn_team1_abbrev': espn_team1_abbrev,
                    'espn_team2_full': espn_team2_full,
                    'espn_team2_abbrev': espn_team2_abbrev,
                    'reversed': reversed_order,
                    'confidence': confidence,
                    'match_type': 'closest',
                    'kalshi_date': kalshi_date,
                    'espn_date': espn_date,
                    'day_difference': day_diff
                })
                output_lines.append(status_msg + "✓ CLOSEST MATCH")
            else:
                output_lines.append(status_msg + "✗ NO MATCHES")
    
    return (idx, matches, output_lines)

def format_kalshi_url(kalshi_id):
    """Convert Kalshi ID to URL format"""
    # Convert to lowercase and format
    url_id = kalshi_id.lower()
    return f"https://kalshi.com/markets/kxncaambgame/mens-college-basketball-mens-game/{url_id}"

def format_espn_url(espn_id):
    """Convert ESPN ID to URL format"""
    return f"https://www.espn.com/mens-college-basketball/game/_/gameId/{espn_id}"

def generate_html_output(unique_games):
    """Generate HTML file with approval interface"""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kalshi-ESPN Team Pairings</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        .controls {
            margin: 20px 0;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin-right: 10px;
        }
        button:hover {
            background-color: #45a049;
        }
        button.secondary {
            background-color: #2196F3;
        }
        button.secondary:hover {
            background-color: #0b7dda;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th {
            background-color: #4CAF50;
            color: white;
            padding: 12px;
            text-align: left;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        td {
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        tr.approved {
            background-color: #e8f5e9;
        }
        a {
            color: #2196F3;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        input[type="checkbox"] {
            width: 20px;
            height: 20px;
            cursor: pointer;
        }
        .confidence-high {
            color: #4CAF50;
            font-weight: bold;
        }
        .confidence-medium {
            color: #FF9800;
        }
        .confidence-low {
            color: #f44336;
        }
        .match-type {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }
        .match-type.exact {
            background-color: #4CAF50;
            color: white;
        }
        .match-type.proposed {
            background-color: #2196F3;
            color: white;
        }
        .match-type.closest {
            background-color: #FF9800;
            color: white;
        }
        .match-yes {
            background-color: #4CAF50;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
            text-align: center;
        }
        .match-no {
            background-color: #f44336;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
            text-align: center;
        }
        .match-na {
            background-color: #9E9E9E;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
            text-align: center;
        }
        #approvedOutput {
            margin-top: 20px;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
            display: none;
        }
        #approvedOutput textarea {
            width: 100%;
            min-height: 200px;
            font-family: monospace;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Kalshi-ESPN Game Matches</h1>
        <div class="controls">
            <button onclick="selectAll()">Select All</button>
            <button onclick="deselectAll()">Deselect All</button>
            <button class="secondary" onclick="showApproved()">Show Approved Pairs</button>
            <button class="secondary" onclick="copyApproved()">Copy Approved to Clipboard</button>
            <span id="selectedCount" style="margin-left: 20px; font-weight: bold;">0 selected</span>
        </div>
        <table id="pairingsTable">
            <thead>
                <tr>
                    <th>Approve</th>
                    <th>Confidence</th>
                    <th>Match Type</th>
                    <th>Kalshi ID</th>
                    <th>ESPN ID</th>
                    <th>Kalshi Team 1</th>
                    <th>ESPN Team 1</th>
                    <th>Kalshi Team 2</th>
                    <th>ESPN Team 2</th>
                    <th>Kalshi Abbrevs</th>
                    <th>ESPN Abbrevs</th>
                    <th>Kalshi Date</th>
                    <th>ESPN Date</th>
                    <th>Day Diff</th>
                    <th>Date Match</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for i, game in enumerate(unique_games):
        kalshi_url = format_kalshi_url(game['kalshi_id'])
        espn_url = format_espn_url(game['espn_id'])
        
        # Determine confidence class
        confidence = game['confidence']
        if confidence >= 80:
            conf_class = "confidence-high"
        elif confidence >= 50:
            conf_class = "confidence-medium"
        else:
            conf_class = "confidence-low"
        
        # Format dates
        kalshi_date_str = str(game['kalshi_date']) if game['kalshi_date'] else ''
        espn_date_str = str(game['espn_date']) if game['espn_date'] else ''
        
        # Check if dates match
        date_match = "N/A"
        date_match_class = "match-na"
        if game.get('kalshi_date') and game.get('espn_date'):
            if game['kalshi_date'] == game['espn_date']:
                date_match = "Yes"
                date_match_class = "match-yes"
            else:
                date_match = "No"
                date_match_class = "match-no"
        elif not game.get('kalshi_date') and not game.get('espn_date'):
            date_match = "N/A"
            date_match_class = "match-na"
        
        html += f"""
                <tr id="row_{i}">
                    <td><input type="checkbox" id="check_{i}" onchange="updateCount()"></td>
                    <td class="{conf_class}">{confidence}</td>
                    <td><span class="match-type {game['match_type']}">{game['match_type']}</span></td>
                    <td><a href="{kalshi_url}" target="_blank">{game['kalshi_id']}</a></td>
                    <td><a href="{espn_url}" target="_blank">{game['espn_id']}</a></td>
                    <td>{game['kalshi_team1_full']}</td>
                    <td>{game['espn_team1_full']}</td>
                    <td>{game['kalshi_team2_full']}</td>
                    <td>{game['espn_team2_full']}</td>
                    <td>{game['kalshi_team1_abbrevs']}, {game['kalshi_team2_abbrevs']}</td>
                    <td>{game['espn_team1_abbrev']}, {game['espn_team2_abbrev']}</td>
                    <td>{kalshi_date_str}</td>
                    <td>{espn_date_str}</td>
                    <td>{game['day_difference']}</td>
                    <td><span class="{date_match_class}">{date_match}</span></td>
                </tr>
"""
    
    html += """            </tbody>
        </table>
        <div id="approvedOutput">
            <h3>Approved Pairs (CSV format):</h3>
            <textarea id="approvedText" readonly></textarea>
        </div>
    </div>
    
    <script>
        function updateCount() {
            const checkboxes = document.querySelectorAll('input[type="checkbox"]:checked');
            const count = checkboxes.length;
            document.getElementById('selectedCount').textContent = count + ' selected';
            
            // Update row styling
            checkboxes.forEach(cb => {
                const rowId = cb.id.replace('check_', 'row_');
                document.getElementById(rowId).classList.add('approved');
            });
            
            // Remove styling from unchecked rows
            document.querySelectorAll('input[type="checkbox"]').forEach(cb => {
                if (!cb.checked) {
                    const rowId = cb.id.replace('check_', 'row_');
                    document.getElementById(rowId).classList.remove('approved');
                }
            });
        }
        
        function selectAll() {
            document.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = true);
            updateCount();
        }
        
        function deselectAll() {
            document.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = false);
            updateCount();
        }
        
        function showApproved() {
            const checkboxes = document.querySelectorAll('input[type="checkbox"]:checked');
            const output = document.getElementById('approvedOutput');
            const textarea = document.getElementById('approvedText');
            
            if (checkboxes.length === 0) {
                alert('No pairs selected!');
                return;
            }
            
            let csv = 'Kalshi Team 1,ESPN Team 1,Kalshi Team 2,ESPN Team 2\\n';
            checkboxes.forEach(cb => {
                const rowId = cb.id.replace('check_', 'row_');
                const row = document.getElementById(rowId);
                const cells = row.getElementsByTagName('td');
                const kalshiTeam1 = cells[5].textContent;
                const espnTeam1 = cells[6].textContent;
                const kalshiTeam2 = cells[7].textContent;
                const espnTeam2 = cells[8].textContent;
                csv += `"${kalshiTeam1}","${espnTeam1}","${kalshiTeam2}","${espnTeam2}"\\n`;
            });
            
            textarea.value = csv;
            output.style.display = 'block';
            textarea.select();
        }
        
        function copyApproved() {
            const checkboxes = document.querySelectorAll('input[type="checkbox"]:checked');
            
            if (checkboxes.length === 0) {
                alert('No pairs selected!');
                return;
            }
            
            let csv = 'Kalshi Team 1,ESPN Team 1,Kalshi Team 2,ESPN Team 2\\n';
            checkboxes.forEach(cb => {
                const rowId = cb.id.replace('check_', 'row_');
                const row = document.getElementById(rowId);
                const cells = row.getElementsByTagName('td');
                const kalshiTeam1 = cells[5].textContent;
                const espnTeam1 = cells[6].textContent;
                const kalshiTeam2 = cells[7].textContent;
                const espnTeam2 = cells[8].textContent;
                csv += `"${kalshiTeam1}","${espnTeam1}","${kalshiTeam2}","${espnTeam2}"\\n`;
            });
            
            navigator.clipboard.writeText(csv).then(() => {
                alert('Approved pairs copied to clipboard!');
            }).catch(err => {
                // Fallback for older browsers
                const textarea = document.createElement('textarea');
                textarea.value = csv;
                document.body.appendChild(textarea);
                textarea.select();
                document.execCommand('copy');
                document.body.removeChild(textarea);
                alert('Approved pairs copied to clipboard!');
            });
        }
        
        // Initialize count on load
        updateCount();
    </script>
</body>
</html>"""
    
    return html

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    generated_data_dir = os.path.join(script_dir, '..', 'GeneratedDataFiles')
    
    # File paths (relative to GeneratedDataFiles)
    espn_teams_file = os.path.join(generated_data_dir, 'full_team_names_espn.csv')
    kalshi_teams_file = os.path.join(generated_data_dir, 'full_team_names_kalshi.csv')
    # Read unmatched Kalshi games from GeneratedDataFiles
    unmatched_kalshi_file = os.path.join(generated_data_dir, 'unmatched_kalshi_games.txt')
    # Load all ESPN games and matched games to determine unmatched ESPN games
    all_espn_games_file = os.path.join(generated_data_dir, 'list_of_espn_games.txt')
    matched_games_file = os.path.join(generated_data_dir, 'kalshi_espn_game_mappings.csv')
    
    print("Loading data...")
    # Load team mappings
    espn_abbrev_to_full, espn_full_to_abbrevs = load_espn_teams(espn_teams_file)
    kalshi_abbrev_to_full, kalshi_full_to_abbrevs = load_kalshi_teams(kalshi_teams_file)
    
    # Load unmatched Kalshi games
    kalshi_games = load_unmatched_kalshi_games(unmatched_kalshi_file)
    
    # Load all ESPN games and determine unmatched ones
    print("Determining unmatched ESPN games...")
    all_espn_games, espn_id_to_date = load_all_espn_games(all_espn_games_file)
    matched_espn_ids = load_matched_espn_ids(matched_games_file)
    
    # Filter to get unmatched ESPN games
    espn_games = [game for game in all_espn_games if game[0] not in matched_espn_ids]
    
    print(f"Loaded {len(kalshi_games)} unmatched Kalshi games")
    print(f"Loaded {len(espn_games)} unmatched ESPN games")
    print(f"Loaded {len(espn_abbrev_to_full)} ESPN team abbreviations")
    print(f"Loaded {len(kalshi_abbrev_to_full)} Kalshi team abbreviations")
    
    # Find matches using multiprocessing
    total_games = len(kalshi_games)
    num_workers = 20
    print(f"\nProcessing {total_games} Kalshi games using {num_workers} workers...")
    print()
    
    # Prepare arguments for worker function
    worker_args = []
    for idx, (kalshi_id, kalshi_team1_abbrev, kalshi_team2_abbrev) in enumerate(kalshi_games, 1):
        worker_args.append((
            idx, total_games, kalshi_id, kalshi_team1_abbrev, kalshi_team2_abbrev,
            espn_games, espn_abbrev_to_full, espn_full_to_abbrevs, espn_id_to_date,
            kalshi_abbrev_to_full, kalshi_full_to_abbrevs
        ))
    
    # Process games in parallel and print as results come in (in order)
    matches = []
    results_dict = {}  # Store results by index
    next_idx_to_print = 1  # Track next index to print
    
    with Pool(processes=num_workers) as pool:
        # Use imap_unordered to get results as they complete
        for result in pool.imap_unordered(process_single_game, worker_args):
            idx, result_matches, output_lines = result
            results_dict[idx] = (result_matches, output_lines)
            
            # Print results in order as they become available
            while next_idx_to_print in results_dict:
                result_matches, output_lines = results_dict.pop(next_idx_to_print)
                for line in output_lines:
                    print(line)
                matches.extend(result_matches)
                next_idx_to_print += 1
    
    print()  # Blank line after all processing
    
    # Sort matches by confidence score (highest to lowest)
    matches.sort(key=lambda x: x.get('confidence', 0), reverse=True)
    
    # Output results as HTML
    output_file = os.path.join(script_dir, 'matched_games.html')
    
    # Group matches by game (kalshi_id + espn_id) to show both teams together
    game_matches = {}
    
    for match in matches:
        # Get the original Kalshi team names and abbreviations
        kalshi_team1_full = match['kalshi_team1_full']
        kalshi_team2_full = match['kalshi_team2_full']
        kalshi_team1_abbrevs = match['kalshi_team1_abbrevs']
        kalshi_team2_abbrevs = match['kalshi_team2_abbrevs']
        
        # Get ESPN team names
        espn_team1_full = match['espn_team1_full']
        espn_team2_full = match['espn_team2_full']
        espn_team1_abbrev = match['espn_team1_abbrev']
        espn_team2_abbrev = match['espn_team2_abbrev']
        
        # Find the best pairing based on full name similarity
        best_kalshi1, best_espn1, best_kalshi2, best_espn2, pairing_reversed = find_best_pairing(
            kalshi_team1_full, kalshi_team2_full, espn_team1_full, espn_team2_full
        )
        
        # Determine which Kalshi abbreviations to use
        if best_kalshi1 == kalshi_team1_full:
            best_kalshi1_abbrevs = kalshi_team1_abbrevs
            best_kalshi2_abbrevs = kalshi_team2_abbrevs
        else:
            best_kalshi1_abbrevs = kalshi_team2_abbrevs
            best_kalshi2_abbrevs = kalshi_team1_abbrevs
        
        # Determine which ESPN abbreviations to use
        if best_espn1 == espn_team1_full:
            best_espn1_abbrev = espn_team1_abbrev
            best_espn2_abbrev = espn_team2_abbrev
        else:
            best_espn1_abbrev = espn_team2_abbrev
            best_espn2_abbrev = espn_team1_abbrev
        
        # Group by game (kalshi_id + espn_id)
        game_key = (match['kalshi_id'], match['espn_id'])
        if game_key not in game_matches:
            game_matches[game_key] = {
                'confidence': match.get('confidence', 0),
                'match_type': match.get('match_type', 'unknown'),
                'kalshi_id': match['kalshi_id'],
                'espn_id': match['espn_id'],
                'kalshi_date': match.get('kalshi_date', ''),
                'espn_date': match.get('espn_date', ''),
                'day_difference': match.get('day_difference', 0),
                'kalshi_team1_full': best_kalshi1,
                'kalshi_team2_full': best_kalshi2,
                'espn_team1_full': best_espn1,
                'espn_team2_full': best_espn2,
                'kalshi_team1_abbrevs': best_kalshi1_abbrevs,
                'kalshi_team2_abbrevs': best_kalshi2_abbrevs,
                'espn_team1_abbrev': best_espn1_abbrev,
                'espn_team2_abbrev': best_espn2_abbrev,
                'reversed': pairing_reversed
            }
    
    # Convert to list for HTML generation
    unique_games = list(game_matches.values())
    
    # Generate HTML
    html_content = generate_html_output(unique_games)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\nFound {len(matches)} matches out of {len(kalshi_games)} Kalshi games")
    print(f"Found {len(unique_games)} unique game matches")
    print(f"Results saved to {output_file}")

if __name__ == '__main__':
    main()
