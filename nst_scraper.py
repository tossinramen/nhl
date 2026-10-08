import argparse
import gzip
import os
import pickle
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

BASE = "https://data.naturalstattrick.com"
FROM_SEASON = "20252026"
THRU_SEASON = "20262027"
STYPE = 2  
LIST_SITS = ["all", "5v5", "ev", "pp", "pk"]

DATA = Path(__file__).parent / "data"
CACHE = DATA / "cache"
GAME_CACHE = CACHE / "games"


FAST_INTERVAL = 4.0
SLOW_INTERVAL = 20.5
FAST_TOKEN_FLOOR = 200
BACKOFF = 300

GAME_TABLE_KINDS = {
    "st": "skater_individual",
    "oi": "skater_onice",
    "sh": "skater_shifts",
    "fl": "forward_lines",
}


class NSTClient:
    def __init__(self, key):
        self.s = requests.Session()
        self.s.headers.update({"nst-key": key, "User-Agent": "nhlmodel-scraper (personal research)"})
        self.last = 0.0
        self.tokens = None

    def get(self, path):
        for attempt in range(6):
            interval = FAST_INTERVAL if self.tokens is None or self.tokens > FAST_TOKEN_FLOOR else SLOW_INTERVAL
            wait = self.last + interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            try:
                r = self.s.get(BASE + path, timeout=120)
            except requests.RequestException as e:
                print(f"  network error: {e}; retrying in 60s")
                self.last = time.monotonic()
                time.sleep(60)
                continue
            self.last = time.monotonic()
            if r.headers.get("tokens-remaining"):
                self.tokens = int(r.headers["tokens-remaining"])
            if r.status_code == 200 and "<table" in r.text:
                return r.text
            if "you have been blocked" in r.text.lower():
                sys.exit(f"Blocked by Cloudflare on {path}. Check NST_ACCESS_KEY / key approval status.")
            print(f"  HTTP {r.status_code} (tokens={self.tokens}) on {path}; backing off {BACKOFF}s (attempt {attempt + 1})")
            time.sleep(BACKOFF)
        raise RuntimeError(f"Failed to fetch {path}")


def text(cell):
    return cell.get_text(" ", strip=True).replace("\xa0", " ")


def table_rows(t):
    header, rows = None, []
    for tr in t.find_all("tr"):
        if tr.find_parent("table") is not t:
            continue
        cells = tr.find_all(["th", "td"], recursive=False)
        if not cells:
            continue
        if header is None:
            header = [text(c) for c in cells]
        else:
            rows.append(cells)
    return header, rows


def numify(df):
    for col in df.columns:
        if df[col].dtype != object:
            continue
        s = df[col].str.rstrip("%").replace({"-": None, "": None})
        conv = pd.to_numeric(s, errors="coerce")
        if conv.notna().sum() == s.notna().sum():
            df[col] = conv
    return df


def mmss_to_min(v):
    if isinstance(v, str) and ":" in v:
        m, s = v.split(":")
        return int(m) + int(s) / 60
    return v


def simple_table(t):
    header, rows = table_rows(t)
    data = [[text(c) for c in r] for r in rows if len(r) == len(header)]
    return pd.DataFrame(data, columns=header)




def parse_games_list(html, sit):
    soup = BeautifulSoup(html, "lxml")
    t = soup.find("table", id="teams")
    header, rows = table_rows(t)
    header = [h if h else "links" for h in header]
    out = []
    for r in rows:
        if len(r) != len(header):
            continue
        rec = dict(zip(header, (text(c) for c in r)))
        link = r[header.index("links")].find("a", href=re.compile(r"game\.php"))
        m = re.search(r"season=(\d+)&(?:amp;)?game=(\d+)", link["href"])
        rec["season"], rec["game_id"] = m.group(1), int(m.group(2))
        toi = r[header.index("TOI")].get("data-sort")
        if toi:
            rec["TOI"] = toi
        del rec["links"]
        out.append(rec)
    df = numify(pd.DataFrame(out))
    df["season"] = df["season"].astype(str)
    df["date"] = df["Game"].str.slice(0, 10)
    df["situation"] = sit
    return df




def parse_game(html, season, game_id):
    soup = BeautifulSoup(html, "lxml")
    key = {"season": season, "game_id": game_id}

    title = soup.title.get_text(strip=True)
    m = re.match(r"(.+?) @ (.+?), (\d{4}-\d{2}-\d{2})", title)
    away, home, date = m.groups()
    h2 = [p for p in soup.find("h2").get_text("|", strip=True).split("|") if p]
    sm = re.match(r"(\d+)\s*-\s*(\d+)", h2[1])
    meta = pd.DataFrame([{**key, "date": date, "away_team": away, "home_team": home,
                          "away_score": int(sm.group(1)), "home_score": int(sm.group(2)),
                          "status": " ".join(h2[2:])}])

    def full_name(short):
        for n in (home, away):
            if n == short or n.endswith(" " + short):
                return n
        return short

    abbr_team = {}
    for t in soup.find_all("table", id=re.compile(r"^tb(.+)stall$")):
        abbr = re.match(r"^tb(.+)stall$", t["id"]).group(1)
        label = t.find_previous(string=re.compile(r"\S - Individual\s*$"))
        abbr_team[abbr] = full_name(label.strip().rsplit(" - ", 1)[0]) if label else abbr


    periods = []
    for t in soup.find_all("table", id=re.compile(r"^tbts")):
        sit = t["id"][4:]
        header, rows = table_rows(t)
        for r in rows:
            team = full_name(text(r[0]))
            cols = [c.get_text("|", strip=True).split("|") for c in r[1:]]
            for i in range(len(cols[0])):
                rec = {**key, "team": team, "situation": sit}
                for h, vals in zip(header[1:], cols):
                    rec[h] = vals[i] if i < len(vals) else None
                periods.append(rec)
    periods = numify(pd.DataFrame(periods))
    if "TOI" in periods:
        periods["TOI"] = periods["TOI"].map(mmss_to_min)

    out = {"games": meta, "team_periods": periods}
    frames = {name: [] for name in GAME_TABLE_KINDS.values()}
    frames["goalies"] = []
    for abbr, team in abbr_team.items():
        pat = re.compile(rf"^tb{re.escape(abbr)}(st|oi|sh|fl)(.+)$")
        for t in soup.find_all("table", id=pat):
            kind, sit = pat.match(t["id"]).groups()
            if kind == "st" and sit.startswith("g"):
                continue
            df = simple_table(t)
            if kind == "sh":
                df = df[~df["Player"].isin(["Forwards", "Defense"])]
            df.insert(0, "situation", sit)
            df.insert(0, "team", team)
            for k, v in reversed(key.items()):
                df.insert(0, k, v)
            frames[GAME_TABLE_KINDS[kind]].append(df)
            if kind == "st":
                g = t.find_next("table", id=re.compile(r"^tb.+stg"))
                if g is not None and g["id"].startswith(f"tb{abbr}stg"):
                    gdf = simple_table(g)
                    gdf.insert(0, "situation", sit)
                    gdf.insert(0, "team", team)
                    for k, v in reversed(key.items()):
                        gdf.insert(0, k, v)
                    frames["goalies"].append(gdf)
    for name, dfs in frames.items():
        out[name] = numify(pd.concat(dfs, ignore_index=True)) if dfs else pd.DataFrame()
    return out




def cache_path(season, game_id):
    return GAME_CACHE / f"{season}_{game_id}.pkl.gz"


def fetch_lists(client):
    lists = {}
    for sit in LIST_SITS:
        path = (f"/games.php?fromseason={FROM_SEASON}&thruseason={THRU_SEASON}"
                f"&stype={STYPE}&sit={sit}&loc=B&team=All&rate=n")
        print(f"Fetching games list sit={sit}")
        df = parse_games_list(client.get(path), sit)
        print(f"  {len(df)} team-game rows")
        df.to_pickle(CACHE / f"games_list_{sit}.pkl")
        lists[sit] = df
    return lists


def scrape_games(client, games, limit):
    todo = [g for g in games if not cache_path(*g).exists()]
    if limit:
        todo = todo[:limit]
    print(f"{len(games)} games listed, {len(todo)} to scrape")
    failed = []
    for i, (season, gid) in enumerate(todo, 1):
        try:
            html = client.get(f"/game.php?season={season}&game={gid}")
            parsed = parse_game(html, season, gid)
            status = parsed["games"]["status"].iloc[0]
            if not status.startswith("Final"):
                raise ValueError(f"game not final yet (status={status!r}); will retry next update")
        except Exception as e:  
            print(f"  [{i}/{len(todo)}] {season} {gid} FAILED: {e}")
            failed.append(f"{season},{gid},{e}")
            continue
        with gzip.open(cache_path(season, gid), "wb") as f:
            pickle.dump(parsed, f)
        print(f"  [{i}/{len(todo)}] {season} {gid} ok (tokens={client.tokens})")
    if failed:
        (DATA / "failed_games.txt").write_text("\n".join(failed) + "\n")
        print(f"{len(failed)} games failed, see data/failed_games.txt")


def widen_lists(lists):
    """Per-situation games lists -> one row per team-game with {sit}_{stat} columns."""
    keys = ["season", "game_id", "Team"]
    wide = None
    for sit, df in lists.items():
        stats = df.drop(columns=["situation", "Game", "date", "Attendance"], errors="ignore")
        stats = stats.rename(columns={c: f"{sit}_{c}" for c in stats.columns if c not in keys})
        wide = stats if wide is None else wide.merge(stats, on=keys, how="outer")
    base = lists.get("all", next(iter(lists.values())))[["season", "game_id", "Team", "Game", "date", "Attendance"]]
    return base.merge(wide, on=keys, how="right").rename(columns={"Team": "team"})


def fetch_history(client, first_year, last_year):
    hist = CACHE / "hist"
    hist.mkdir(parents=True, exist_ok=True)
    frames = []
    for year in range(first_year, last_year + 1):
        season = f"{year}{year + 1}"
        lists = {}
        for sit in LIST_SITS:
            p = hist / f"games_list_{season}_{sit}.pkl"
            if not p.exists():
                print(f"Fetching {season} games list sit={sit}")
                path = (f"/games.php?fromseason={season}&thruseason={season}"
                        f"&stype={STYPE}&sit={sit}&loc=B&team=All&rate=n")
                parse_games_list(client.get(path), sit).to_pickle(p)
            lists[sit] = pd.read_pickle(p)
        frames.append(widen_lists(lists))
        print(f"  {season}: {len(frames[-1])} team-game rows")
    out = DATA / "team_games_hist.csv"
    pd.concat(frames, ignore_index=True).sort_values(["season", "game_id", "team"]).to_csv(out, index=False)
    print(f"wrote {out.relative_to(DATA.parent)}")


def combine():
    lists = {s: pd.read_pickle(CACHE / f"games_list_{s}.pkl")
             for s in LIST_SITS if (CACHE / f"games_list_{s}.pkl").exists()}
    parts = {}
    for p in sorted(GAME_CACHE.glob("*.pkl.gz")):
        with gzip.open(p, "rb") as f:
            for name, df in pickle.load(f).items():
                parts.setdefault(name, []).append(df)
    tables = {n: pd.concat(d, ignore_index=True) for n, d in parts.items()}

    if lists:
        team_games = widen_lists(lists)
        if "games" in tables:
            g = tables["games"][["season", "game_id", "home_team", "away_team"]]
            team_games = team_games.merge(g, on=["season", "game_id"], how="left")
            team_games["is_home"] = team_games["team"] == team_games["home_team"]
            team_games["opponent"] = team_games["home_team"].where(~team_games["is_home"], team_games["away_team"])
            team_games = team_games.drop(columns=["home_team", "away_team"])
        tables["team_games"] = team_games.sort_values(["season", "game_id", "team"])

    for name, df in tables.items():
        if not df.empty:
            out = DATA / f"{name}.csv"
            df.to_csv(out, index=False)
            print(f"wrote {out.relative_to(DATA.parent)}: {len(df)} rows, {df.shape[1]} cols, "
                  f"{out.stat().st_size / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max new games to scrape this run")
    ap.add_argument("--combine-only", action="store_true")
    ap.add_argument("--history", nargs=2, type=int, metavar=("FIRST_YEAR", "LAST_YEAR"),
                    help="only fetch team-level game lists for past seasons, e.g. --history 2018 2024")
    args = ap.parse_args()

    GAME_CACHE.mkdir(parents=True, exist_ok=True)
    if not args.combine_only:
        load_dotenv()
        key = os.getenv("NST_ACCESS_KEY")
        if not key:
            sys.exit("NST_ACCESS_KEY missing from .env")
        client = NSTClient(key)
        if args.history:
            fetch_history(client, *args.history)
            return
        lists = fetch_lists(client)
        ref = lists["all"]
        today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        ref = ref[ref["date"] < today]
        games = sorted({(s, g) for s, g in zip(ref["season"], ref["game_id"])})
        scrape_games(client, games, args.limit)
    combine()


if __name__ == "__main__":
    main()
