import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from teams import ABBR_TO_NST

API = "https://api-web.nhle.com/v1"
ET = ZoneInfo("America/New_York")


def today_et():
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
