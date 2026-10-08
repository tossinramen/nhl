import re
import time
import unicodedata
from functools import lru_cache

import requests
from bs4 import BeautifulSoup

ROTOWIRE_LINEUPS = "https://www.rotowire.com/hockey/nhl-lineups.php"
ROTOWIRE_INJURIES = "https://www.rotowire.com/hockey/tables/injury-report.php?team=ALL&pos=ALL"
ESPN_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/injuries"
NHL_API = "https://api-web.nhle.com/v1"
UA = {"User-Agent": "Mozilla/5.0 (nhlmodel personal research)"}


OUT = {"IR", "IR-LT", "IR-NR", "OUT", "SUSP", "SUSPENDED", "INJURED RESERVE", "INJURED_RESERVE", "LTIR"}
QUESTIONABLE = {"DTD", "GTD", "DAY-TO-DAY", "DAY TO DAY", "QUESTIONABLE"}


def _get(url, as_json=False, tries=3):
    for attempt in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            r.raise_for_status()
            return r.json() if as_json else r.text
        except (requests.RequestException, ValueError):
            if attempt == tries - 1:
                raise
            time.sleep(3 * (attempt + 1))


def norm_name(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", "", s.replace("-", " "))
    return " ".join(s.split())


def availability(status):
    """Chance a player with this injury status plays: 0 out, 0.5 questionable, 1 otherwise."""
    s = str(status or "").strip().upper()
    if s in OUT:
        return 0.0
    if s in QUESTIONABLE:
        return 0.5
    return 1.0



def _parse_side(ul):
    side = {"goalie": None, "goalie_status": None, "injuries": []}
    hl = ul.select_one("li.lineup__player-highlight")
    if hl:
        a = hl.select_one("a")
        name = a.get_text(strip=True) if a else None
        if name and "." in name:  
            m = re.search(r"/player/([a-z\-']+)-\d+", a.get("href", ""))
            if m:
                name = m.group(1).replace("-", " ").title()
        side["goalie"] = name
        st = hl.select_one("div[class*='is-']")
        side["goalie_status"] = st.get_text(strip=True) if st else None
    in_injuries = False
    for li in ul.find_all("li", recursive=False):
        classes = li.get("class", [])
        if "lineup__title" in classes:
            in_injuries = "INJURIES" in li.get_text(strip=True).upper()
            continue
        if in_injuries and "lineup__player" in classes:
            a = li.select_one("a")
            inj = li.select_one(".lineup__inj")
            pos = li.select_one(".lineup__pos")
            side["injuries"].append({
                "name": a.get("title") or a.get_text(strip=True),
                "pos": pos.get_text(strip=True) if pos else None,
                "status": inj.get_text(strip=True) if inj else None,
            })
    return side


def parse_rotowire(html):
    """{(away_abbr, home_abbr): {...goalies, injuries, line...}} from a lineups page."""
    soup = BeautifulSoup(html, "lxml")
    games = {}
    for g in soup.select("div.lineup.is-nhl"):
        abbrs = [a.get_text(strip=True) for a in g.select(".lineup__abbr")]
        visit, home = g.select_one("ul.lineup__list.is-visit"), g.select_one("ul.lineup__list.is-home")
        if len(abbrs) != 2 or not visit or not home:
            continue
        info = {"away": _parse_side(visit), "home": _parse_side(home), "line": None, "total": None}
        for item in g.select(".lineup__odds-item"):
            txt = item.get_text(" ", strip=True).replace("\xa0", " ")
            if txt.upper().startswith("LINE"):
                info["line"] = txt[4:].strip()
            m = re.search(r"O/U\s*([\d.]+)", txt)
            if m:
                info["total"] = float(m.group(1))
        games[(abbrs[0], abbrs[1])] = info
    return games


def rotowire_lineups(matchups):
    """RotoWire lineup info for the given {(away, home)} matchups.

    The page's "today" rolls over during the day (late at night it still shows the previous day),
    so try the default page and the ?date=tomorrow page and keep whichever matches the schedule.
    """
    matchups = set(matchups)
    best = {}
    for suffix in ("", "?date=tomorrow"):
        try:
            games = parse_rotowire(_get(ROTOWIRE_LINEUPS + suffix))
        except requests.RequestException:
            continue
        hit = {k: v for k, v in games.items() if k in matchups}
        if len(hit) > len(best):
            best = hit
        if len(best) == len(matchups):
            break
    return best


def rotowire_injuries():
    """{team_abbr: [{name, pos, status, injury}]} from RotoWire's injury report."""
    out = {}
    for x in _get(ROTOWIRE_INJURIES, as_json=True):
        out.setdefault(x["team"], []).append(
            {"name": x["player"], "pos": x.get("position"), "status": x.get("status"), "injury": x.get("injury")})
    return out


def espn_injuries():
    """Fallback injury source: {team_abbr: [{name, status}]}."""
    from teams import name_to_abbr
    out = {}
    for team in _get(ESPN_INJURIES, as_json=True).get("injuries", []):
        abbr = name_to_abbr(team.get("displayName"))
        if not abbr:
            continue
        for x in team.get("injuries", []):
            out.setdefault(abbr, []).append({"name": x["athlete"]["displayName"], "pos": None,
                                             "status": x.get("status"), "injury": x.get("shortComment")})
    return out




@lru_cache(maxsize=64)
def roster(abbr):
    """Current NHL roster (includes injured reserve): list of {id, name, pos}."""
    try:
        d = _get(f"{NHL_API}/roster/{abbr}/current", as_json=True)
    except requests.RequestException:
        return []
    players = []
    for grp, pos in (("forwards", "F"), ("defensemen", "D"), ("goalies", "G")):
        for p in d.get(grp, []):
            players.append({"id": p["id"], "pos": pos,
                            "name": f"{p['firstName']['default']} {p['lastName']['default']}"})
    return players


def resolve(abbr, name, recent=None):
    """NHL player id for `name` on team `abbr`, or None.

    recent: optional {player_id: "F. Lastname"} of players seen in boxscores, used when a player
    is missing from the current roster (e.g. just traded or waived).
    """
    target = norm_name(name)
    for p in roster(abbr):
        if norm_name(p["name"]) == target:
            return p["id"]
    parts = target.split()
    if not parts:
        return None
    first, last = parts[0], parts[-1]
    cands = [p for p in roster(abbr) if norm_name(p["name"]).split()[-1] == last
             and norm_name(p["name"]).startswith(first[0])]
    if len(cands) == 1:
        return cands[0]["id"]
    if recent:
        hits = [pid for pid, short in recent.items()
                if norm_name(short).split()[-1:] == [last] and norm_name(short).startswith(first[0])]
        if len(hits) == 1:
            return hits[0]
    return None


def day_info(matchups):
    """Everything known before puck drop for a set of (away, home) matchups.

    Returns {(away, home): {"away": side, "home": side, "line", "total", "source"}} where side has
    goalie (name), goalie_status, and injuries [{name, pos, status}] for that team.
    """
    matchups = list(matchups)
    info = {}
    try:
        info = rotowire_lineups(matchups)
        source = "rotowire"
    except Exception:  
        source = None

    league_inj = None
    for k in matchups:
        if k in info:
            info[k]["source"] = source
            continue
        if league_inj is None:
            try:
                league_inj, src = rotowire_injuries(), "rotowire-injury-report"
            except Exception:  # noqa: BLE001
                league_inj, src = espn_injuries(), "espn"
        info[k] = {"away": {"goalie": None, "goalie_status": None, "injuries": league_inj.get(k[0], [])},
                   "home": {"goalie": None, "goalie_status": None, "injuries": league_inj.get(k[1], [])},
                   "line": None, "total": None, "source": src}
    return info


if __name__ == "__main__":
    import sys
    import schedule
    games = schedule.get_games(sys.argv[1] if len(sys.argv) > 1 else None)
    info = day_info(zip(games["away_abbr"], games["home_abbr"]))
    for (a, h), x in info.items():
        print(f"{a} @ {h}  [{x['source']}]  line {x['line']}  O/U {x['total']}")
        for side, abbr in (("away", a), ("home", h)):
            s = x[side]
            inj = ", ".join(f"{i['name']} ({i['status']})" for i in s["injuries"]) or "none"
            print(f"   {abbr}: G {s['goalie']} ({s['goalie_status']}) | out: {inj}")
