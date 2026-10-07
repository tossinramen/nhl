from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).parent / "data"

HALFLIFE = 20      
SEASON_CARRY = 0.5 
PRIOR_GAMES = 5    
MAX_GOALS = 15

METRICS = ["xgf60", "xga60", "gf60", "ga60", "sf60", "sa60", "pp_xgf", "pk_xga", "finish", "save"]
OFF = ["xgf60", "gf60", "sf60", "pp_xgf", "finish"] 
DEF = ["xga60", "ga60", "sa60", "pk_xga", "save"]    
FEATURES = [f"own_{m}" for m in OFF] + [f"opp_{m}" for m in DEF] + ["is_home", "own_b2b", "opp_b2b"]


def season_of(date):
    d = pd.Timestamp(date)
    y = d.year if d.month >= 8 else d.year - 1
    return f"{y}{y + 1}"


def load_team_games():
    """One row per team per game with per-game metrics and final goals (SO winner counts, like betting totals)."""
    tg = pd.read_csv(DATA / "team_games.csv")
    g = pd.read_csv(DATA / "games.csv")
    tg = tg.merge(g[["season", "game_id", "home_score", "away_score", "status"]], on=["season", "game_id"])
    tg["season"] = tg["season"].astype(str)
    tg["date"] = pd.to_datetime(tg["date"])
    tg["goals"] = np.where(tg["is_home"], tg["home_score"], tg["away_score"])
    toi = tg["all_TOI"].clip(lower=1)
    out = tg[["season", "game_id", "date", "team", "opponent", "is_home", "goals"]].copy()
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
    return out.sort_values(["date", "game_id", "team"]).reset_index(drop=True)


class Ratings:
    """Sequential EWMA ratings per team, shrunk toward the running league average."""

    def __init__(self, halflife=HALFLIFE, carry=SEASON_CARRY, prior=PRIOR_GAMES):
        self.decay = 0.5 ** (1 / halflife)
        self.carry, self.prior = carry, prior
        self.S, self.W, self.season, self.last_date = {}, {}, {}, {}
        self.league_sum = np.zeros(len(METRICS))
        self.league_n = 0

    def _league_mean(self):
        return self.league_sum / self.league_n if self.league_n else np.zeros(len(METRICS))

    def rating(self, team, season):
        S = self.S.get(team, np.zeros(len(METRICS)))
        W = self.W.get(team, 0.0)
        if team in self.season and self.season[team] != season:
            S, W = S * self.carry, W * self.carry
        return (S + self.prior * self._league_mean()) / (W + self.prior)

    def update(self, team, season, date, x):
        S = self.S.get(team, np.zeros(len(METRICS)))
        W = self.W.get(team, 0.0)
        if team in self.season and self.season[team] != season:
            S, W = S * self.carry, W * self.carry
        self.S[team] = self.decay * S + x
        self.W[team] = self.decay * W + 1
        self.season[team], self.last_date[team] = season, date
        self.league_sum += x
        self.league_n += 1

    def rested_days(self, team, date):
        last = self.last_date.get(team)
        return 99 if last is None else (pd.Timestamp(date) - last).days


def _pair_rows(game, ra, rb, rest_a, rest_b):
    """Model rows for both sides of a matchup: game = dict(home_team, away_team, ...)."""
    rows = []
    for team, opp, own_r, opp_r, own_rest, opp_rest, home in (
        (game["home_team"], game["away_team"], ra, rb, rest_a, rest_b, 1),
        (game["away_team"], game["home_team"], rb, ra, rest_b, rest_a, 0),
    ):
        row = {"team": team, "opponent": opp, "is_home": home,
               "own_b2b": int(own_rest == 1), "opp_b2b": int(opp_rest == 1)}
        for m in OFF:
            row[f"own_{m}"] = own_r[METRICS.index(m)]
        for m in DEF:
            row[f"opp_{m}"] = opp_r[METRICS.index(m)]
        rows.append(row)
    return rows


def build_history(before=None, **rating_kw):
    """Walk through played games (only those dated before `before`, if given);
    return (pre-game feature rows, final Ratings state)."""
    tg = load_team_games()
    if before is not None:
        tg = tg[tg["date"] < pd.Timestamp(before)]
    R = Ratings(**rating_kw)
    rows = []
    for (season, gid), g in tg.groupby(["season", "game_id"], sort=False):
        h = g[g["is_home"]].iloc[0]
        a = g[~g["is_home"]].iloc[0]
        date = h["date"]
        pair = _pair_rows({"home_team": h["team"], "away_team": a["team"]},
                          R.rating(h["team"], season), R.rating(a["team"], season),
                          R.rested_days(h["team"], date), R.rested_days(a["team"], date))
        pair[0]["goals"], pair[1]["goals"] = h["goals"], a["goals"]
        for p in pair:
            p.update(season=season, game_id=gid, date=date)
        rows.extend(pair)
        for r in (h, a):
            R.update(r["team"], season, date, r[METRICS].to_numpy(dtype=float))
   
    return pd.DataFrame(rows), R


def fit(hist):
    model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=1000))
    model.fit(hist[FEATURES], hist["goals"])
    return model


def outcome_probs(lam_home, lam_away):
    k = np.arange(MAX_GOALS + 1)
    ph, pa = poisson.pmf(k, lam_home), poisson.pmf(k, lam_away)
    grid = np.outer(ph, pa)
    p_home = np.tril(grid, -1).sum() + 0.5 * np.trace(grid)
    tot = np.add.outer(k, k)
    return {"p_home": p_home, "p_over_5.5": grid[tot > 5.5].sum(), "p_over_6.5": grid[tot > 6.5].sum()}


def predict_games(model, games, R):
    out = []
    for _, g in games.iterrows():
        season = season_of(g["date"])
        pair = _pair_rows(g, R.rating(g["home_team"], season), R.rating(g["away_team"], season),
                          R.rested_days(g["home_team"], g["date"]), R.rested_days(g["away_team"], g["date"]))
        lam_h, lam_a = model.predict(pd.DataFrame(pair)[FEATURES])
        out.append({**g.to_dict(), "xg_home": lam_h, "xg_away": lam_a, **outcome_probs(lam_h, lam_a)})
    df = pd.DataFrame(out)
    if df.empty:
        return df
    df["pred_total"] = df["xg_home"] + df["xg_away"]
    df["pick"] = np.where(df["p_home"] >= 0.5, df["home_team"], df["away_team"])
    df["pick_prob"] = np.maximum(df["p_home"], 1 - df["p_home"])
    return df


def backtest(start="2025-11-15", **rating_kw):
    hist, _ = build_history(**rating_kw)
    days = sorted(hist.loc[hist["date"] >= start, "date"].unique())
    preds = []
    model, trained_through = None, None
    for d in days:
        if model is None or (d - trained_through).days >= 7:
            model = fit(hist[hist["date"] < d])
            trained_through = d
        day = hist[hist["date"] == d].copy()
        day["lam"] = model.predict(day[FEATURES])
        preds.append(day)
    p = pd.concat(preds)
    home = p[p["is_home"] == 1].set_index(["season", "game_id"])
    away = p[p["is_home"] == 0].set_index(["season", "game_id"])
    g = home[["date", "team", "goals", "lam"]].join(
        away[["team", "goals", "lam"]], rsuffix="_away").rename(
        columns={"team": "home_team", "goals": "home_goals", "lam": "xg_home",
                 "team_away": "away_team", "goals_away": "away_goals", "lam_away": "xg_away"})
    probs = pd.DataFrame([outcome_probs(h, a) for h, a in zip(g["xg_home"], g["xg_away"])], index=g.index)
    g = g.join(probs)
    g["home_won"] = (g["home_goals"] > g["away_goals"]).astype(int)
    g["total"] = g["home_goals"] + g["away_goals"]
    g["pred_total"] = g["xg_home"] + g["xg_away"]
    return g.reset_index()


def score_backtest(g, train_hist=None):
    p = g["p_home"].clip(1e-6, 1 - 1e-6)
    y = g["home_won"]
    base_rate = y.mean()
    res = {
        "games": len(g),
        "win_accuracy": ((p >= 0.5) == y).mean(),
        "always_home_accuracy": max(base_rate, 1 - base_rate),
        "log_loss": -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean(),
        "log_loss_coinflip": np.log(2),
        "brier": ((p - y) ** 2).mean(),
        "total_mae": (g["pred_total"] - g["total"]).abs().mean(),
        "total_mae_league_avg": (g["total"].expanding().mean().shift().fillna(6.2) - g["total"]).abs().mean(),
        "over_6.5_hit_rate": ((g["p_over_6.5"] >= 0.5) == (g["total"] > 6.5)).mean(),
    }
    conf = g.assign(conf=np.maximum(p, 1 - p), right=((p >= 0.5) == y))
    conf["bucket"] = pd.cut(conf["conf"], [0.5, 0.55, 0.6, 0.65, 1.0], include_lowest=True)
    calib = conf.groupby("bucket", observed=True).agg(games=("right", "size"), predicted=("conf", "mean"),
                                                      actual=("right", "mean"))
    return res, calib
