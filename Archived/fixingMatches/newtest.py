import re
import requests
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

try:
    from rapidfuzz import fuzz
    def fuzzy(a: str, b: str) -> float:
        return float(fuzz.token_set_ratio(a, b))
except Exception:
    from difflib import SequenceMatcher
    def fuzzy(a: str, b: str) -> float:
        return 100.0 * SequenceMatcher(None, a, b).ratio()

ESPN_NCAAM_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"

KALSHI_BASE = "https://trading-api.kalshi.com/trade-api/v2"
KALSHI_HEADERS = {
    # Fill this in with your working Kalshi auth headers
    # Example:
    # "Authorization": "Bearer YOUR_TOKEN"
}

MONTHS = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}

STOP_WORDS = {
    "university", "college", "of", "the", "at",
    "men", "mens", "women", "womens", "vs", "vs.", "v", "@", "and"
}

def normalize_team_name(name: str) -> str:
    s = (name or "").lower().strip()
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    s = s.replace("st ", "saint ")
    s = s.replace("st. ", "saint ")

    tokens = [t for t in s.split() if t not in STOP_WORDS]
    s = " ".join(tokens)

    toks = s.split()
    if len(toks) <= 3:
        return s

    multiword_prefixes = {"texas", "new", "north", "south", "east", "west", "central", "saint", "mount"}
    keep = 4 if toks[0] in multiword_prefixes else 3
    return " ".join(toks[:keep])

def parse_kalshi_date_from_ticker(ticker: str) -> date:
    t = ticker.strip()
    if "-" not in t:
        raise ValueError(f"Unexpected ticker format: {ticker}")
    rest = t.split("-", 1)[1]  # 26JAN17LMCCHS
    yy = int(rest[0:2])
    mon = rest[2:5].lower()
    dd = int(rest[5:7])
    if mon not in MONTHS:
        raise ValueError(f"Bad month token: {mon}")
    return date(2000 + yy, MONTHS[mon], dd)

def parse_espn_iso(dt_str: str) -> Optional[datetime]:
    if not dt_str:
        return None
    try:
        if dt_str.endswith("Z"):
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return datetime.fromisoformat(dt_str)
    except Exception:
        return None

def fetch_kalshi_market(ticker: str) -> Dict[str, Any]:
    url = f"{KALSHI_BASE}/markets/{ticker}"
    r = requests.get(url, headers=KALSHI_HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()

def extract_kalshi_teams(market: Dict[str, Any]) -> Tuple[str, str]:
    title = ""
    if isinstance(market, dict):
        m = market.get("market") if "market" in market else market
        title = (m.get("title") or m.get("event_title") or "").strip()

    if not title:
        raise ValueError("Could not find a usable title in Kalshi market response")

    # Try common separators
    for sep in [" at ", " vs ", " vs. ", " v ", " @ "]:
        if sep in title.lower():
            parts = re.split(sep, title, flags=re.IGNORECASE)
            if len(parts) >= 2:
                return parts[0].strip(), parts[1].strip()

    raise ValueError(f"Could not parse teams from Kalshi title: {title}")

def fetch_espn_scoreboard_window(anchor_date: date, window_days: int = 2, groups: int = 50, limit: int = 1500) -> List[Dict[str, Any]]:
    start_dt = anchor_date - timedelta(days=window_days)
    end_dt = anchor_date + timedelta(days=window_days)
    params = {
        "dates": f"{start_dt.strftime('%Y%m%d')}-{end_dt.strftime('%Y%m%d')}",
        "groups": groups,
        "limit": limit,
    }
    r = requests.get(ESPN_NCAAM_SCOREBOARD, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("events") or []

def extract_home_away(ev: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    comps = ev.get("competitions") or []
    if not comps:
        return None
    competitors = comps[0].get("competitors") or []
    home = away = None
    for c in competitors:
        if c.get("homeAway") == "home":
            home = c
        elif c.get("homeAway") == "away":
            away = c
    if not home or not away:
        return None
    return away, home

def score_event(kalshi_a: str, kalshi_b: str, ev: Dict[str, Any], anchor: date) -> Optional[Dict[str, Any]]:
    ha = extract_home_away(ev)
    if not ha:
        return None
    away_c, home_c = ha
    away_team = away_c.get("team") or {}
    home_team = home_c.get("team") or {}

    espn_away_names = [
        away_team.get("shortDisplayName"),
        away_team.get("displayName"),
        away_team.get("location"),
        away_team.get("name"),
    ]
    espn_home_names = [
        home_team.get("shortDisplayName"),
        home_team.get("displayName"),
        home_team.get("location"),
        home_team.get("name"),
    ]

    kal_a_n = normalize_team_name(kalshi_a)
    kal_b_n = normalize_team_name(kalshi_b)

    def best_name_match(kal_norm: str, espn_names: List[Optional[str]]) -> float:
        best = 0.0
        for nm in espn_names:
            if not nm:
                continue
            best = max(best, fuzzy(kal_norm, normalize_team_name(nm)))
        return best

    s_ah = best_name_match(kal_a_n, espn_away_names) + best_name_match(kal_b_n, espn_home_names)
    s_ha = best_name_match(kal_a_n, espn_home_names) + best_name_match(kal_b_n, espn_away_names) - 5.0

    team_score = max(s_ah, s_ha)
    orientation = "A=away,B=home" if s_ah >= s_ha else "A=home,B=away"

    ev_time = parse_espn_iso(ev.get("date"))
    if not ev_time:
        return None
    day_diff = abs((ev_time.date() - anchor).days)
    date_bonus = max(0.0, 30.0 - 10.0 * day_diff)

    total = team_score + date_bonus

    return {
        "event_id": ev.get("id"),
        "event_name": ev.get("name"),
        "event_date": ev.get("date"),
        "score_team": team_score,
        "score_date": date_bonus,
        "score_total": total,
        "orientation": orientation,
        "away": away_team.get("displayName"),
        "home": home_team.get("displayName"),
        "away_abbr": away_team.get("abbreviation"),
        "home_abbr": home_team.get("abbreviation"),
    }

def connect_kalshi_to_espn_by_names(
    kalshi_ticker: str,
    kalshi_anchor_date: date,
    window_days: int = 2,
    top_n: int = 5
) -> Dict[str, Any]:
    market = fetch_kalshi_market(kalshi_ticker)
    team_a, team_b = extract_kalshi_teams(market)

    events = fetch_espn_scoreboard_window(kalshi_anchor_date, window_days=window_days)

    scored: List[Dict[str, Any]] = []
    for ev in events:
        s = score_event(team_a, team_b, ev, kalshi_anchor_date)
        if s:
            scored.append(s)

    scored.sort(key=lambda x: x["score_total"], reverse=True)
    best = scored[0] if scored else None
    second = scored[1] if len(scored) > 1 else None
    gap = (best["score_total"] - second["score_total"]) if best and second else None

    accepted = False
    if best:
        accepted = (best["score_team"] >= 170.0) and (gap is None or gap >= 10.0)

    summary_url = None
    if best and best.get("event_id"):
        summary_url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary?event={best['event_id']}"

    return {
        "kalshi_ticker": kalshi_ticker,
        "kalshi_teams": [team_a, team_b],
        "normalized_kalshi_teams": [normalize_team_name(team_a), normalize_team_name(team_b)],
        "best_match": best,
        "accepted": accepted,
        "gap": gap,
        "candidates": scored[:top_n],
        "espn_summary_url": summary_url,
    }

if __name__ == "__main__":
    kalshi_ticker = "KXNCAAMBGAME-26JAN17LMCCHS"
    anchor = parse_kalshi_date_from_ticker(kalshi_ticker)

    out = connect_kalshi_to_espn_by_names(
        kalshi_ticker=kalshi_ticker,
        kalshi_anchor_date=anchor,
        window_days=3,
        top_n=8,
    )

    print("KALSHI TEAMS:", out["kalshi_teams"])
    print("NORMALIZED:", out["normalized_kalshi_teams"])
    print("ACCEPTED:", out["accepted"])
    print("GAP:", out["gap"])
    print("BEST:", out["best_match"])
    print("SUMMARY:", out["espn_summary_url"])
    print("TOP CANDIDATES:")
    for c in out["candidates"]:
        print(c["score_total"], c["score_team"], c["score_date"], c["event_id"], c["event_name"], c["event_date"])
