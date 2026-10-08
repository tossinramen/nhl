import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
NHL_CACHE = DATA / "cache" / "nhl"
ODDS_CACHE = DATA / "cache" / "odds"
NHL_API = "https://api-web.nhle.com/v1"
AN_API = "https://api.actionnetwork.com/web/v1/scoreboard/nhl"
FIRST_SEASON = 2018 
UA = {"User-Agent": "Mozilla/5.0 (nhlmodel personal research)"}

AN_BOOKS = {15: "close", 30: "open"} 


def _get_json(url, params=None, tries=5):
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(2 * 2 ** attempt)
    raise RuntimeError(f"request failed: {url} {params}")


def season_start_year(d):
    d = pd.Timestamp(d)
    return d.year if d.month >= 8 else d.year - 1



def list_season_games(year):
    """Every regular-season game of the season starting in `year`, from the weekly schedule."""
    games, d, end = {}, date(year, 9, 20), date(year + 1, 7, 15)
    while d <= end:
        data = _get_json(f"{NHL_API}/schedule/{d:%Y-%m-%d}") or {}
        for day in data.get("gameWeek", []):
            for g in day.get("games", []):
                if g.get("gameType") == 2:
                    games[g["id"]] = {"nhl_game_id": g["id"], "date": day["date"], "state": g["gameState"]}
        d += timedelta(days=7)
    return pd.DataFrame(list(games.values()))


def _toi(s):
    if not s or ":" not in s:
        return 0.0
    m, sec = s.split(":")
    return int(m) + int(sec) / 60


def fetch_boxscore(game_id):
    """Boxscore JSON for a finished game, cached forever once final."""
    path = NHL_CACHE / f"{game_id}.json.gz"
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    box = _get_json(f"{NHL_API}/gamecenter/{game_id}/boxscore")
    if box and box.get("gameState") in ("OFF", "FINAL"):
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(box, f)
    return box


def parse_boxscore(box):
    gid = box["id"]
    home, away = box["homeTeam"], box["awayTeam"]
    game = {
        "nhl_game_id": gid, "season": str(box["season"]), "date": box["gameDate"],
        "home_abbr": home["abbrev"], "away_abbr": away["abbrev"],
        "home_score": home.get("score"), "away_score": away.get("score"),
        "home_sog": home.get("sog"), "away_sog": away.get("sog"),
        "result_type": (box.get("gameOutcome") or {}).get("lastPeriodType"),
    }
    skaters, goalies = [], []
    stats = box.get("playerByGameStats", {})
    for side, team in (("homeTeam", home), ("awayTeam", away)):
        t = stats.get(side, {})
        for grp in ("forwards", "defense"):
            for p in t.get(grp, []):
                skaters.append({
                    "nhl_game_id": gid, "team": team["abbrev"], "player_id": p["playerId"],
                    "name": p["name"]["default"], "pos": "D" if grp == "defense" else "F",
                    "toi": _toi(p.get("toi")), "goals": p.get("goals", 0), "assists": p.get("assists", 0),
                    "sog": p.get("sog", 0), "blocks": p.get("blockedShots", 0), "hits": p.get("hits", 0),
                    "pim": p.get("pim", 0), "plus_minus": p.get("plusMinus", 0),
                })
        for p in t.get("goalies", []):
            goalies.append({
                "nhl_game_id": gid, "team": team["abbrev"], "player_id": p["playerId"],
                "name": p["name"]["default"], "toi": _toi(p.get("toi")), "starter": bool(p.get("starter")),
                "shots_against": p.get("shotsAgainst", 0), "goals_against": p.get("goalsAgainst", 0),
            })
    return game, skaters, goalies


def update_nhl(first_season=FIRST_SEASON, workers=4):
    """Fetch boxscores for every finished regular-season game not yet cached; rebuild nhl_*.csv."""
    NHL_CACHE.mkdir(parents=True, exist_ok=True)
    this_season = season_start_year(date.today())
    listed = DATA / "cache" / "nhl_game_list.csv"
    old = pd.read_csv(listed) if listed.exists() else pd.DataFrame(columns=["nhl_game_id", "date", "state", "season"])
    frames = [old[old["season"] < this_season]]  
    for year in range(first_season, this_season + 1):
        if year < this_season and (old["season"] == year).any():
            continue
        print(f"listing {year}-{year + 1} schedule")
        df = list_season_games(year)
        df["season"] = year
        frames.append(df)
    games = pd.concat(frames, ignore_index=True).drop_duplicates("nhl_game_id", keep="last")
    games.to_csv(listed, index=False)

    finished = games[games["state"].isin(["OFF", "FINAL"])]["nhl_game_id"].tolist()
    todo = [g for g in finished if not (NHL_CACHE / f"{g}.json.gz").exists()]
    print(f"{len(finished)} finished games, {len(todo)} boxscores to fetch")
    done = 0
    with ThreadPoolExecutor(workers) as ex:
        for _ in ex.map(fetch_boxscore, todo):
            done += 1
            if done % 250 == 0:
                print(f"  {done}/{len(todo)}")

    outs = {"games": DATA / "nhl_games.csv", "skaters": DATA / "nhl_skaters.csv", "goalies": DATA / "nhl_goalies.csv"}
    have = set(pd.read_csv(outs["games"])["nhl_game_id"]) if all(p.exists() for p in outs.values()) else set()
    g_rows, s_rows, gl_rows = [], [], []
    for gid in finished:
        path = NHL_CACHE / f"{gid}.json.gz"
        if gid in have or not path.exists():
            continue
        with gzip.open(path, "rt", encoding="utf-8") as f:
            g, s, gl = parse_boxscore(json.load(f))
        g_rows.append(g)
        s_rows.extend(s)
        gl_rows.extend(gl)
    new = {"games": g_rows, "skaters": s_rows, "goalies": gl_rows}
    for name, path in outs.items():
        df = pd.DataFrame(new[name])
        if have:
            df = pd.concat([pd.read_csv(path), df], ignore_index=True)
        if name == "games":
            df = df.sort_values(["date", "nhl_game_id"])
        df.to_csv(path, index=False)
    print(f"added {len(g_rows)} games to nhl_games.csv / nhl_skaters.csv / nhl_goalies.csv")




def action_network_day(d):
    """All NHL games Action Network lists for date d (YYYYMMDD or date), with open/close lines."""
    ds = pd.Timestamp(d).strftime("%Y%m%d")
    data = _get_json(AN_API, {"period": "game", "bookIds": ",".join(map(str, AN_BOOKS)), "date": ds}) or {}
    rows = []
    for g in data.get("games", []):
        teams = {t["id"]: t for t in g.get("teams", [])}
        home, away = teams.get(g.get("home_team_id")), teams.get(g.get("away_team_id"))
        if not home or not away:
            continue
        row = {"an_game_id": g["id"], "an_date": ds, "start_time": g.get("start_time"), "an_status": g.get("status"),
               "home_name": home["full_name"], "away_name": away["full_name"]}
        for o in g.get("odds") or []:
            tag = AN_BOOKS.get(o.get("book_id"))
            if tag and o.get("type", "game") == "game":
                for k in ("ml_home", "ml_away", "total", "over", "under"):
                    row[f"{tag}_{k}"] = o.get(k)
        rows.append(row)
    return rows


def update_odds(first_season=FIRST_SEASON):
    """Cache Action Network lines for every past game day (skips days already cached)."""
    ODDS_CACHE.mkdir(parents=True, exist_ok=True)
    games = pd.read_csv(DATA / "nhl_games.csv")
    days = sorted(set(games["date"]))
    today = date.today().strftime("%Y-%m-%d")
    todo = [d for d in days if d < today and not (ODDS_CACHE / f"{d}.csv").exists()]
    print(f"{len(days)} game days, {len(todo)} odds days to fetch")
    for i, d in enumerate(todo, 1):
        rows = action_network_day(d)
        pd.DataFrame(rows).to_csv(ODDS_CACHE / f"{d}.csv", index=False)
        time.sleep(0.4)
        if i % 100 == 0:
            print(f"  {i}/{len(todo)}")
    build_odds_table(games)


def build_odds_table(games=None):
    from teams import franchise, name_to_abbr
    games = pd.read_csv(DATA / "nhl_games.csv") if games is None else games
    parts = []
    for p in sorted(ODDS_CACHE.glob("*.csv")):
        try:
            df = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            continue
        df["date"] = p.stem
        parts.append(df)
    if not parts:
        return
    odds = pd.concat(parts, ignore_index=True)

    odds["home_fr"] = odds["home_name"].map(name_to_abbr).map(franchise)
    odds["away_fr"] = odds["away_name"].map(name_to_abbr).map(franchise)
    games = games.assign(home_fr=games["home_abbr"].map(franchise), away_fr=games["away_abbr"].map(franchise))
    keys = ["date", "home_fr", "away_fr"]
    cols = [c for c in odds.columns if c.startswith(("close_", "open_"))]
    m = games[["nhl_game_id", "home_abbr", "away_abbr"] + keys].merge(
        odds[keys + cols].drop_duplicates(keys), on=keys, how="inner").drop(columns=["home_fr", "away_fr"])
    m.to_csv(DATA / "odds.csv", index=False)
    print(f"wrote odds.csv ({len(m)} games with lines, {m['close_ml_home'].notna().sum()} with closing moneyline)")


if __name__ == "__main__":
    update_nhl()
    update_odds()
