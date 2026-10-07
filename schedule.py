"""Daily NHL schedule and results from the public NHL API (api-web.nhle.com, no key needed)."""
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

API = "https://api-web.nhle.com/v1"
ET = ZoneInfo("America/New_York")

ABBR_TO_NST = {
    "ANA": "Anaheim Ducks", "BOS": "Boston Bruins", "BUF": "Buffalo Sabres",
    "CGY": "Calgary Flames", "CAR": "Carolina Hurricanes", "CHI": "Chicago Blackhawks",
    "COL": "Colorado Avalanche", "CBJ": "Columbus Blue Jackets", "DAL": "Dallas Stars",
    "DET": "Detroit Red Wings", "EDM": "Edmonton Oilers", "FLA": "Florida Panthers",
    "LAK": "Los Angeles Kings", "MIN": "Minnesota Wild", "MTL": "Montreal Canadiens",
    "NSH": "Nashville Predators", "NJD": "New Jersey Devils", "NYI": "New York Islanders",
    "NYR": "New York Rangers", "OTT": "Ottawa Senators", "PHI": "Philadelphia Flyers",
    "PIT": "Pittsburgh Penguins", "SJS": "San Jose Sharks", "SEA": "Seattle Kraken",
    "STL": "St Louis Blues", "TBL": "Tampa Bay Lightning", "TOR": "Toronto Maple Leafs",
    "UTA": "Utah Mammoth", "VAN": "Vancouver Canucks", "VGK": "Vegas Golden Knights",
    "WSH": "Washington Capitals", "WPG": "Winnipeg Jets",
}


def today_et():
    """Today's date in Eastern time (the NHL's schedule day), as YYYY-MM-DD."""
    return datetime.now(ET).strftime("%Y-%m-%d")


def _get(path):
    for attempt in range(4):
        try:
            r = requests.get(API + path, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == 3:
                raise RuntimeError(f"NHL API request failed for {path}: {e}") from e
            time.sleep(2 ** attempt * 3)


def _team(t):
    abbr = t["abbrev"]
    if abbr in ABBR_TO_NST:
        return ABBR_TO_NST[abbr]
    return f"{t['placeName']['default']} {t['commonName']['default']}".replace(".", "")


def get_games(date=None, regular_season_only=True):
    """All NHL games on `date` (YYYY-MM-DD, default today ET) with status and scores if played.

    gameState: FUT/PRE = not started, LIVE/CRIT = in progress, OFF/FINAL = finished.
    """
    date = date or today_et()
    data = _get(f"/schedule/{date}")
    day = next((d for d in data.get("gameWeek", []) if d["date"] == date), None)
    rows = []
    for g in (day or {}).get("games", []):
        if regular_season_only and g.get("gameType") != 2:
            continue
        start = pd.Timestamp(g["startTimeUTC"]).tz_convert(ET)
        final = g["gameState"] in ("OFF", "FINAL")
        last = (g.get("gameOutcome") or {}).get("lastPeriodType")
        rows.append({
            "nhl_game_id": g["id"],
            "date": date,
            "start_et": start.strftime("%I:%M %p").lstrip("0"),
            "away_team": _team(g["awayTeam"]),
            "home_team": _team(g["homeTeam"]),
            "away_abbr": g["awayTeam"]["abbrev"],
            "home_abbr": g["homeTeam"]["abbrev"],
            "state": g["gameState"],
            "away_score": g["awayTeam"].get("score") if final else None,
            "home_score": g["homeTeam"].get("score") if final else None,
            "result_type": last if final else None,  # REG / OT / SO
        })
    return pd.DataFrame(rows, columns=[
        "nhl_game_id", "date", "start_et", "away_team", "home_team", "away_abbr", "home_abbr",
        "state", "away_score", "home_score", "result_type"])


if __name__ == "__main__":
    import sys
    print(get_games(sys.argv[1] if len(sys.argv) > 1 else None).to_string(index=False))
