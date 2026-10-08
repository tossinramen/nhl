from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from teams import franchise, name_to_abbr

DATA = Path(__file__).parent / "data"

TEAM_HALFLIFE = 20
SEASON_CARRY = 0.5
TEAM_PRIOR = 5
GOALIE_HALFLIFE = 60
GOALIE_PRIOR_MIN = 900
GOALIE_PRIOR_GSAX60 = -0.15
SKATER_HALFLIFE = 60
SKATER_PRIOR_GAMES = 10
SKATER_PRIOR = {"F": 0.25, "D": 0.2}
W_MODEL = 0.4

METRICS = ["xgf60", "xga60", "gf60", "ga60", "sf60", "sa60", "pp_xgf", "pk_xga", "finish", "save"]
OFF = ["xgf60", "gf60", "sf60", "pp_xgf", "finish"]
DEF = ["xga60", "ga60", "sa60", "pk_xga", "save"]

WIN_FEATURES = [f"d_{m}" for m in METRICS] + ["d_goalie", "d_lineup", "h_b2b", "a_b2b"]
GOAL_FEATURES = ([f"own_{m}" for m in OFF] + [f"opp_{m}" for m in DEF]
                 + ["opp_goalie", "own_lineup", "opp_lineup", "is_home", "own_b2b", "opp_b2b"])


def season_of(date):
    d = pd.Timestamp(date)
    y = d.year if d.month >= 8 else d.year - 1
    return f"{y}{y + 1}"


def game_score(g, a, sog, blk, pm, pim):
    return 0.75 * g + 0.7 * a + 0.075 * sog + 0.05 * blk + 0.15 * pm - 0.075 * pim


def team_stats():
    """NST team-level stats per (nhl_game_id, abbr), all seasons."""
    parts = [pd.read_csv(DATA / f) for f in ("team_games_hist.csv", "team_games.csv") if (DATA / f).exists()]
    tg = pd.concat(parts, ignore_index=True)
    tg["abbr"] = tg["team"].map(name_to_abbr)
    tg["nhl_game_id"] = tg["season"].astype(str).str[:4].astype(int) * 1_000_000 + tg["game_id"]
    toi = tg["all_TOI"].clip(lower=1)
    out = pd.DataFrame({"nhl_game_id": tg["nhl_game_id"], "abbr": tg["abbr"]})
    out["xgf60"] = tg["all_xGF"] / toi * 60
    out["xga60"] = tg["all_xGA"] / toi * 60
    out["gf60"] = tg["all_GF"] / toi * 60
    out["ga60"] = tg["all_GA"] / toi * 60
    out["sf60"] = tg["all_SF"] / toi * 60
    out["sa60"] = tg["all_SA"] / toi * 60
    out["pp_xgf"] = tg["pp_xGF"].fillna(0)
    out["pk_xga"] = tg["pk_xGA"].fillna(0)
    out["finish"] = tg["all_GF"] - tg["all_xGF"]
    out["save"] = tg["all_xGA"] - tg["all_GA"]
    out["xga"] = tg["all_xGA"]
    out["sa"] = tg["all_SA"]
    return out.drop_duplicates(["nhl_game_id", "abbr"], keep="last").set_index(["nhl_game_id", "abbr"])


def load():
    games = pd.read_csv(DATA / "nhl_games.csv")
    games["season"] = games["season"].astype(str)
    games["date"] = pd.to_datetime(games["date"])
    games = games.dropna(subset=["home_score", "away_score"]).sort_values(["date", "nhl_game_id"])
    skaters = pd.read_csv(DATA / "nhl_skaters.csv")
    goalies = pd.read_csv(DATA / "nhl_goalies.csv")
    return games, team_stats(), skaters, goalies


def load_odds():
    from odds import market_probs
    cols = ["nhl_game_id", "close_p_home", "close_total", "close_p_over", "close_ml_home", "close_ml_away"]
    path = DATA / "odds.csv"
    if not path.exists():
        return pd.DataFrame(columns=cols, dtype=float)
    return market_probs(pd.read_csv(path), "close")[cols]


class TeamRatings:
    def __init__(self):
        self.decay = 0.5 ** (1 / TEAM_HALFLIFE)
        self.S, self.W, self.season = {}, {}, {}
        self.league_sum, self.league_n = np.zeros(len(METRICS)), 0

    def _carried(self, team, season):
        S, W = self.S.get(team, np.zeros(len(METRICS))), self.W.get(team, 0.0)
        if team in self.season and self.season[team] != season:
            S, W = S * SEASON_CARRY, W * SEASON_CARRY
        return S, W

    def rating(self, team, season):
        S, W = self._carried(team, season)
        mean = self.league_sum / self.league_n if self.league_n else np.zeros(len(METRICS))
        return (S + TEAM_PRIOR * mean) / (W + TEAM_PRIOR)

    def update(self, team, season, x):
        S, W = self._carried(team, season)
        self.S[team], self.W[team], self.season[team] = self.decay * S + x, self.decay * W + 1, season
        self.league_sum += x
        self.league_n += 1


class GoalieRatings:
    def __init__(self):
        self.decay = 0.5 ** (1 / GOALIE_HALFLIFE)
        self.gsax, self.toi = defaultdict(float), defaultdict(float)
        self.name, self.team = {}, {}

    def rating(self, pid):
        if pid is None:
            return GOALIE_PRIOR_GSAX60
        num = self.gsax[pid] + GOALIE_PRIOR_MIN * GOALIE_PRIOR_GSAX60 / 60
        return num / (self.toi[pid] + GOALIE_PRIOR_MIN) * 60

    def update(self, pid, gsax, toi):
        self.gsax[pid] = self.decay * self.gsax[pid] + gsax
        self.toi[pid] = self.decay * self.toi[pid] + toi


class SkaterRatings:
    def __init__(self):
        self.decay = 0.5 ** (1 / SKATER_HALFLIFE)
        self.gs, self.w, self.toi = defaultdict(float), defaultdict(float), defaultdict(float)
        self.pos, self.name, self.team, self.last = {}, {}, {}, {}

    def rating(self, pid, pos=None):
        pos = pos or self.pos.get(pid, "F")
        return (self.gs[pid] + SKATER_PRIOR_GAMES * SKATER_PRIOR[pos]) / (self.w[pid] + SKATER_PRIOR_GAMES)

    def avg_toi(self, pid, pos="F"):
        if self.w[pid] > 0:
            return self.toi[pid] / self.w[pid]
        return 10.0 if pos == "F" else 14.0

    def update(self, pid, gs, toi):
        self.gs[pid] = self.decay * self.gs[pid] + gs
        self.w[pid] = self.decay * self.w[pid] + 1
        self.toi[pid] = self.decay * self.toi[pid] + toi


class State:
    def __init__(self):
        self.team = TeamRatings()
        self.goalie = GoalieRatings()
        self.skater = SkaterRatings()
        self.lineup_typ = {}
        self.last_date = {}
        self.recent_starters = defaultdict(lambda: deque(maxlen=10))
        self.xg_per_shot = [0.0, 0.0]

    def rest(self, abbr, date):
        last = self.last_date.get(franchise(abbr))
        return 99 if last is None else (pd.Timestamp(date) - last).days

    def recent_players(self, abbr, date, days=60):
        cutoff = pd.Timestamp(date) - pd.Timedelta(days=days)
        names = {pid: n for pid, n in self.skater.name.items()
                 if self.skater.team.get(pid) == abbr and self.skater.last.get(pid, cutoff) > cutoff}
        names.update({pid: n for pid, n in self.goalie.name.items() if self.goalie.team.get(pid) == abbr})
        return names


def _by_team(df, cols):
    out = defaultdict(list)
    for key, *vals in zip(zip(df["nhl_game_id"], df["team"]), *(df[c] for c in cols)):
        out[key].append(tuple(vals))
    return out


def build(before=None):
    games, stats, skaters, goalies = load()
    if before is not None:
        games = games[games["date"] < pd.Timestamp(before)]
    skaters = skaters[skaters["toi"] > 0].assign(
        gs=lambda d: game_score(d["goals"], d["assists"], d["sog"], d["blocks"], d["plus_minus"], d["pim"]))
    sk = _by_team(skaters, ["player_id", "pos", "gs", "toi", "name"])
    gl = _by_team(goalies, ["player_id", "name", "toi", "starter", "shots_against", "goals_against"])
    team_x = dict(zip(stats.index, stats[METRICS].to_numpy(dtype=float)))
    team_xga_sa = dict(zip(stats.index, stats[["xga", "sa"]].to_numpy(dtype=float)))
    st = State()
    lineup_decay = 0.5 ** (1 / 10)
    rows = []

    for g in games.itertuples(index=False):
        gid, season, date = g.nhl_game_id, g.season, g.date
        row = {"nhl_game_id": gid, "season": season, "date": date, "home_abbr": g.home_abbr,
               "away_abbr": g.away_abbr, "home_score": g.home_score, "away_score": g.away_score,
               "result_type": g.result_type}
        for side, abbr in (("h", g.home_abbr), ("a", g.away_abbr)):
            fr = franchise(abbr)
            for m, v in zip(METRICS, st.team.rating(fr, season)):
                row[f"{side}_{m}"] = v
            played = [r for r in gl.get((gid, abbr), []) if r[2] > 0]
            starter = None
            if played:
                flagged = [r for r in played if r[3]]
                starter = int((flagged or sorted(played, key=lambda r: r[2]))[-1][0])
            row[f"{side}_goalie_id"] = starter
            row[f"{side}_goalie"] = st.goalie.rating(starter)
            usual = Counter(st.recent_starters[fr]).most_common(1)
            row[f"{side}_goalie_usual"] = st.goalie.rating(usual[0][0] if usual else None)
            lineup = sum(st.skater.rating(pid, pos) for pid, pos, *_ in sk.get((gid, abbr), []))
            row[f"{side}_lineup"] = lineup
            row[f"{side}_lineup_typ"] = st.lineup_typ.get(fr, lineup)
            row[f"{side}_b2b"] = int(st.rest(abbr, date) == 1)
        rows.append(row)

        for abbr in (g.home_abbr, g.away_abbr):
            fr = franchise(abbr)
            key = (gid, abbr)
            x, xs = team_x.get(key), team_xga_sa.get(key)
            if x is not None:
                st.team.update(fr, season, x)
                st.xg_per_shot[0] += xs[0]
                st.xg_per_shot[1] += xs[1]
            for pid, name, toi, starter, sa, ga in gl.get(key, []):
                st.goalie.name[pid], st.goalie.team[pid] = name, abbr
                if toi <= 0:
                    continue
                if xs is not None and xs[1] > 0:
                    xga = xs[0] * sa / xs[1]
                else:
                    xga = sa * (st.xg_per_shot[0] / st.xg_per_shot[1] if st.xg_per_shot[1] else 0.1)
                st.goalie.update(pid, xga - ga, toi)
                if starter:
                    st.recent_starters[fr].append(pid)
            dressed = sk.get(key, [])
            if dressed:
                lineup = 0.0
                for pid, pos, gs, toi, name in dressed:
                    lineup += st.skater.rating(pid, pos)
                    st.skater.update(pid, gs, toi)
                    st.skater.pos[pid], st.skater.name[pid] = pos, name
                    st.skater.team[pid], st.skater.last[pid] = abbr, date
                prev = st.lineup_typ.get(fr, lineup)
                st.lineup_typ[fr] = lineup_decay * prev + (1 - lineup_decay) * lineup
            st.last_date[fr] = date

    hist = add_features(pd.DataFrame(rows))
    hist["home_win"] = (hist["home_score"] > hist["away_score"]).astype(int)
    hist["total"] = hist["home_score"] + hist["away_score"]
    return hist, st


def add_features(df):
    df = df.copy()
    for m in METRICS:
        df[f"d_{m}"] = df[f"h_{m}"] - df[f"a_{m}"]
    df["d_goalie"] = df["h_goalie"] - df["a_goalie"]
    df["d_lineup"] = df["h_lineup"] - df["a_lineup"]
    return df


def goal_rows(df):
    out = []
    for own, opp, home in (("h", "a", 1), ("a", "h", 0)):
        r = pd.DataFrame(index=df.index)
        for m in OFF:
            r[f"own_{m}"] = df[f"{own}_{m}"]
        for m in DEF:
            r[f"opp_{m}"] = df[f"{opp}_{m}"]
        r["opp_goalie"] = df[f"{opp}_goalie"]
        r["own_lineup"] = df[f"{own}_lineup"]
        r["opp_lineup"] = df[f"{opp}_lineup"]
        r["is_home"] = home
        r["own_b2b"] = df[f"{own}_b2b"]
        r["opp_b2b"] = df[f"{opp}_b2b"]
        if "home_score" in df:
            r["goals"] = df["home_score" if home else "away_score"]
        out.append(r)
    return out


class Model:
    def __init__(self, win_features=WIN_FEATURES, goal_features=GOAL_FEATURES, C=0.05):
        self.win_features, self.goal_features, self.C = win_features, goal_features, C

    def fit(self, hist):
        self.win = make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=1000))
        self.win.fit(hist[self.win_features], hist["home_win"])
        rows = pd.concat(goal_rows(hist), ignore_index=True)
        self.goals = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=1000))
        self.goals.fit(rows[self.goal_features], rows["goals"])
        return self

    def predict(self, df):
        df = df.copy()
        df["p_home"] = self.win.predict_proba(df[self.win_features])[:, 1]
        home_rows, away_rows = goal_rows(df)
        df["xg_home"] = self.goals.predict(home_rows[self.goal_features])
        df["xg_away"] = self.goals.predict(away_rows[self.goal_features])
        df["pred_total"] = df["xg_home"] + df["xg_away"]
        return df


def p_over(total_mean, line):
    return 1 - poisson.cdf(np.floor(line), total_mean)


def train_mask(hist):
    return hist["date"] >= hist["date"].min() + pd.Timedelta(days=45)


def backtest(hist=None, start="2019-10-01", refit_days=14, model_kw=None):
    if hist is None:
        hist, _ = build()
    usable = train_mask(hist)
    days = sorted(hist.loc[hist["date"] >= start, "date"].unique())
    out, model, fitted = [], None, None
    for d in days:
        if model is None or (d - fitted).days >= refit_days:
            model = Model(**(model_kw or {})).fit(hist[usable & (hist["date"] < d)])
            fitted = d
        out.append(model.predict(hist[hist["date"] == d]))
    return pd.concat(out).merge(load_odds(), on="nhl_game_id", how="left")


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def evaluate(p, col="p_home"):
    y = p["home_win"]
    return {"games": len(p), "accuracy": float(((p[col] >= 0.5) == y).mean()),
            "log_loss": logloss(p[col], y), "brier": float(((p[col] - y) ** 2).mean())}


def p_over_line(total_mean, line):
    over = p_over(total_mean, line)
    push = np.where(line % 1 == 0, p_over(total_mean, line - 1) - over, 0)
    return over / (1 - push)


def evaluate_totals(p):
    s = p[p["close_total"].notna()]
    line, total = s["close_total"], s["total"]
    decided = total != line
    model_over = p_over_line(s["pred_total"], line)[decided]
    went_over = (total > line)[decided]
    return {"games": int(decided.sum()),
            "model_mae": float((s["pred_total"] - total).abs().mean()),
            "line_mae": float((line - total).abs().mean()),
            "over_under_acc": float(((model_over >= 0.5) == went_over).mean()),
            "model_ll": logloss(model_over, went_over),
            "market_ll": logloss(s["close_p_over"][decided], went_over)}


def blend(p_model, p_market, w=W_MODEL):
    p_model, p_market = np.asarray(p_model, float), np.asarray(p_market, float)
    lm, lk = np.log(p_model / (1 - p_model)), np.log(p_market / (1 - p_market))
    out = 1 / (1 + np.exp(-(w * lm + (1 - w) * lk)))
    return np.where(np.isnan(p_market), p_model, out)


def blend_market(p, w=W_MODEL):
    return blend(p["p_home"], p["close_p_home"], w)
