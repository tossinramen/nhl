
from collections import Counter

import numpy as np
import pandas as pd

import lineups
import model
import odds
import schedule
from teams import franchise

SLOTS = {"F": 12, "D": 6}


def project_side(st, abbr, info, date):
    
    recent = st.recent_players(abbr, date)
    avail, injured = {}, []
    for inj in info.get("injuries", []):
        pid = lineups.resolve(abbr, inj["name"], recent)
        a = lineups.availability(inj["status"])
        if pid is not None:
            avail[pid] = min(avail.get(pid, 1.0), a)
        if a < 1:
            injured.append({"name": inj["name"], "status": inj["status"], "pid": pid,
                            "rating": st.skater.rating(pid) if pid in st.skater.name else None})

    roster = [p for p in lineups.roster(abbr) if p["pos"] in SLOTS]
    if not roster:
        roster = [{"id": pid, "pos": st.skater.pos.get(pid, "F"), "name": n}
                  for pid, n in recent.items() if pid in st.skater.pos]
    lineup, dressed = 0.0, []
    for pos, slots in SLOTS.items():
        cands = sorted((p for p in roster if p["pos"] == pos),
                       key=lambda p: st.skater.avg_toi(p["id"], pos), reverse=True)
        left = float(slots)
        for p in cands:
            w = min(avail.get(p["id"], 1.0), left)
            if w <= 0:
                continue
            lineup += w * st.skater.rating(p["id"], pos)
            dressed.append(p["id"])
            left -= w
            if left <= 0:
                break
        lineup += max(left, 0) * model.SKATER_PRIOR[pos]

    goalie_id, goalie_name, status = None, info.get("goalie"), info.get("goalie_status")
    if goalie_name:
        goalie_id = lineups.resolve(abbr, goalie_name, recent)
    if goalie_id is None:
        starts = Counter(g for g in st.recent_starters[franchise(abbr)] if avail.get(g, 1.0) > 0)
        if starts:
            goalie_id = starts.most_common(1)[0][0]
            goalie_name, status = st.goalie.name.get(goalie_id), "Projected (recent starts)"

    starts = Counter(st.recent_starters[franchise(abbr)])
    injured.sort(key=lambda x: (-starts.get(x["pid"], 0), -(x["rating"] or 0)))
    return {"goalie_id": goalie_id, "goalie": goalie_name, "goalie_status": status,
            "goalie_rating": st.goalie.rating(goalie_id), "lineup": lineup,
            "lineup_typ": st.lineup_typ.get(franchise(abbr), lineup), "injured": injured}


def actual_side(st, gid, abbr, skaters, goalies):

    sk = skaters[(skaters["nhl_game_id"] == gid) & (skaters["team"] == abbr) & (skaters["toi"] > 0)]
    gl = goalies[(goalies["nhl_game_id"] == gid) & (goalies["team"] == abbr) & (goalies["toi"] > 0)]
    if sk.empty or gl.empty:
        return None
    starter = gl.sort_values(["starter", "toi"]).iloc[-1]
    gid_ = int(starter["player_id"])
    return {"goalie_id": gid_, "goalie": starter["name"], "goalie_status": "Actual starter",
            "goalie_rating": st.goalie.rating(gid_),
            "lineup": sum(st.skater.rating(p, pos) for p, pos in zip(sk["player_id"], sk["pos"])),
            "lineup_typ": st.lineup_typ.get(franchise(abbr)), "injured": []}


def forecast(date):
    games = schedule.get_games(date)
    if games.empty:
        return games, None
    hist, st = model.build(before=date)
    fitted = model.Model().fit(hist[model.train_mask(hist)])
    season = model.season_of(date)
    past = date < schedule.today_et()
    if past:
        skaters = pd.read_csv(model.DATA / "nhl_skaters.csv")
        goalies = pd.read_csv(model.DATA / "nhl_goalies.csv")
        info = {}
    else:
        info = lineups.day_info(zip(games["away_abbr"], games["home_abbr"]))

    rows = []
    for g in games.itertuples(index=False):
        gi = info.get((g.away_abbr, g.home_abbr), {})
        row = g._asdict()
        row["info_source"] = gi.get("source")
        for side, abbr in (("h", g.home_abbr), ("a", g.away_abbr)):
            proj = actual_side(st, g.nhl_game_id, abbr, skaters, goalies) if past else None
            if proj is None:
                proj = project_side(st, abbr, gi.get("home" if side == "h" else "away", {}), date)
            else:
                row["info_source"] = "boxscore"
            for m, v in zip(model.METRICS, st.team.rating(franchise(abbr), season)):
                row[f"{side}_{m}"] = v
            row[f"{side}_goalie"] = proj["goalie_rating"]
            row[f"{side}_goalie_id"] = proj["goalie_id"]
            row[f"{side}_goalie_name"] = proj["goalie"]
            row[f"{side}_goalie_status"] = proj["goalie_status"]
            row[f"{side}_lineup"] = proj["lineup"]
            row[f"{side}_lineup_typ"] = proj["lineup_typ"]
            row[f"{side}_b2b"] = int(st.rest(abbr, date) == 1)
            row[f"{side}_out"] = "; ".join(f"{x['name']} ({x['status']})" for x in proj["injured"][:4])
        rows.append(row)
    df = fitted.predict(model.add_features(pd.DataFrame(rows)))

    lines = odds.day_lines(date)
    if not lines.empty:
        df = df.merge(lines, on=["home_abbr", "away_abbr"], how="left")
    for c in ("book_p_home", "book_total", "book_ml_home", "book_ml_away", "pm_p_home"):
        if c not in df:
            df[c] = np.nan
    df["market_p_home"] = df["book_p_home"].fillna(df["pm_p_home"])
    df["edge_home"] = df["p_home"] - df["market_p_home"]
    df["final_p_home"] = model.blend(df["p_home"], df["market_p_home"])
    df["total_line"] = df["book_total"].fillna(6.5)
    df["p_over_line"] = model.p_over_line(df["pred_total"], df["total_line"])
    df["pick"] = np.where(df["final_p_home"] >= 0.5, df["home_abbr"], df["away_abbr"])
    df["pick_prob"] = np.maximum(df["final_p_home"], 1 - df["final_p_home"])
    return df, hist
