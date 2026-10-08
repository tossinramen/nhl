import argparse
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import collect
import model
import schedule

ROOT = Path(__file__).parent
PRED = ROOT / "predictions"
LOG = PRED / "graded.csv"
EDGE_BET = 0.05


def pct(x):
    return "  -" if pd.isna(x) else f"{x:.0%}"


def dec_odds(ml):
    return np.where(ml < 0, 1 + 100 / -ml, 1 + ml / 100)


def cmd_update(_args):
    r = subprocess.run([sys.executable, str(ROOT / "nst_scraper.py")], cwd=ROOT)
    if r.returncode:
        sys.exit(f"NST update failed (exit {r.returncode})")
    collect.update_nhl()
    collect.update_odds()


def check_freshness(date):
    games = pd.read_csv(model.DATA / "nhl_games.csv")
    last = games["date"].max()
    yesterday = (pd.Timestamp(date) - timedelta(days=1)).strftime("%Y-%m-%d")
    if last < yesterday:
        print(f"WARNING: data only runs through {last}. Run `python nhl.py update` first so "
              f"ratings, lineups and back-to-back flags include recent games.\n")


def goalie_label(name, status):
    if not isinstance(name, str):
        return "?"
    mark = "*" if isinstance(status, str) and status.lower().startswith("confirmed") else ""
    return name.split()[-1] + mark


def model_edge(df):
    """Side the model likes more than the market does, by how much, and its moneyline."""
    home = (df["p_home"] >= df["market_p_home"]).to_numpy()
    side = np.where(home, df["home_abbr"], df["away_abbr"])
    edge = (df["p_home"] - df["market_p_home"]).abs().to_numpy()
    ml = np.where(home, df["book_ml_home"], df["book_ml_away"]).astype(float)
    return side, edge, ml


def cmd_predict(args):
    from forecast import forecast
    date = args.date or schedule.today_et()
    check_freshness(date)
    df, hist = forecast(date)
    if df.empty:
        print(f"No regular-season NHL games on {date}.")
        return

    print(f"NHL predictions for {date}  (trained on {len(hist):,} games through {hist['date'].max():%Y-%m-%d})\n")
    print(f"{'Time':>8}  {'Game':<10} {'Pick':<4} {'Win%':>4}  {'Model':>5} {'Market':>6}  {'Model edge':<12} "
          f"{'Score':>7}  {'Total':>5} {'Line':>4} {'Over':>4}  Goalies (away / home)")
    side, edge, _ = model_edge(df)
    for i, (_, r) in enumerate(df.iterrows()):
        home_pick = r["pick"] == r["home_abbr"]
        model_p = r["p_home"] if home_pick else 1 - r["p_home"]
        mkt = r["market_p_home"] if home_pick else 1 - r["market_p_home"]
        e = "-" if np.isnan(edge[i]) else f"{side[i]} +{edge[i]:.1%}" + (" BET" if edge[i] >= EDGE_BET else "")
        print(f"{r['start_et']:>8}  {r['away_abbr']} @ {r['home_abbr']:<4} {r['pick']:<4} {r['pick_prob']:>4.0%}  "
              f"{model_p:>5.0%} {pct(mkt):>6}  {e:<12} {r['xg_away']:.1f}-{r['xg_home']:.1f}  "
              f"{r['pred_total']:>5.2f} {r['total_line']:>4} {r['p_over_line']:>4.0%}  "
              f"{goalie_label(r['a_goalie_name'], r['a_goalie_status'])} / "
              f"{goalie_label(r['h_goalie_name'], r['h_goalie_status'])}")

    out_lines = [f"  {abbr}: {r[f'{s}_out']}" for _, r in df.iterrows()
                 for s, abbr in (("a", r["away_abbr"]), ("h", r["home_abbr"]))
                 if isinstance(r[f"{s}_out"], str) and r[f"{s}_out"]]
    if out_lines:
        print("\nInjuries the model accounted for (most important first):")
        print("\n".join(out_lines))
    print(f"\nWin% = final win chance of the pick: {model.W_MODEL:.0%} model / {1 - model.W_MODEL:.0%} market blend "
          f"(model alone if no odds).\nModel / Market = each one's win chance for the pick "
          f"(market = no-vig sportsbook consensus, else Polymarket).\nModel edge = team the model likes more than "
          f"the market; BET = edge >= {EDGE_BET:.0%} (historically only profitable at early/opening prices).\n"
          f"Score/Total = model's expected goals. Over = model's chance the total goes over the line. "
          f"* = goalie confirmed.")

    PRED.mkdir(exist_ok=True)
    out = PRED / f"{date}.csv"
    df.to_csv(out, index=False)
    print(f"saved {out.relative_to(ROOT)}")


def cmd_grade(args):
    date = args.date or (pd.Timestamp(schedule.today_et()) - timedelta(days=1)).strftime("%Y-%m-%d")
    f = PRED / f"{date}.csv"
    if not f.exists():
        print(f"No saved predictions for {date} ({f.relative_to(ROOT)} missing).")
        return
    preds = pd.read_csv(f).drop(columns=["away_score", "home_score", "result_type", "state"], errors="ignore")
    results = schedule.get_games(date)
    done = results[results["state"].isin(["OFF", "FINAL"])]
    g = preds.merge(done[["nhl_game_id", "away_score", "home_score", "result_type"]], on="nhl_game_id")
    if g.empty:
        print(f"No finished games yet for {date}.")
        return
    g[["away_score", "home_score"]] = g[["away_score", "home_score"]].astype(int)
    g["winner"] = np.where(g["home_score"] > g["away_score"], g["home_abbr"], g["away_abbr"])
    g["correct"] = g["pick"] == g["winner"]
    g["model_correct"] = np.where(g["p_home"] >= 0.5, g["home_abbr"], g["away_abbr"]) == g["winner"]
    has_mkt = g["market_p_home"].notna()
    g["market_pick"] = np.where(g["market_p_home"] >= 0.5, g["home_abbr"], g["away_abbr"])
    g["market_correct"] = np.where(has_mkt, g["market_pick"] == g["winner"], np.nan)
    g["total"] = g["home_score"] + g["away_score"]
    g["total_err"] = g["pred_total"] - g["total"]
    side, edge, ml = model_edge(g)
    g["bet_side"] = side
    g["bet"] = (edge >= EDGE_BET) & ~np.isnan(ml)
    g["bet_profit"] = np.where(g["bet"], np.where(side == g["winner"], dec_odds(ml) - 1, -1.0), 0.0)

    print(f"Results for {date}:\n")
    for _, r in g.iterrows():
        mark = "OK  " if r["correct"] else "MISS"
        ot = f" ({r['result_type']})" if r["result_type"] in ("OT", "SO") else ""
        bet = f" | BET {r['bet_side']} {r['bet_profit']:+.2f}u" if r["bet"] else ""
        print(f"  {mark} {r['away_abbr']} {r['away_score']} @ {r['home_abbr']} {r['home_score']}{ot:<5}"
              f" | pick {r['pick']} {r['pick_prob']:.0%} (market fav {r['market_pick']}) | total {r['total']} "
              f"vs {r['pred_total']:.1f}{bet}")
    n_mkt = int(has_mkt.sum())
    print(f"\n  {date}: picks {g['correct'].sum()}/{len(g)}, model alone {g['model_correct'].sum()}/{len(g)}, "
          f"market favorite {int(np.nansum(g['market_correct']))}/{n_mkt}")

    log = pd.concat([pd.read_csv(LOG), g]) if LOG.exists() else g
    log = log.drop_duplicates("nhl_game_id", keep="last").sort_values(["date", "nhl_game_id"])
    log.to_csv(LOG, index=False)
    m = log["market_correct"].notna()
    home_won = (log["winner"] == log["home_abbr"]).astype(int)
    print(f"  season to date ({len(log)} games): picks {log['correct'].mean():.1%}, model alone "
          f"{log['model_correct'].mean():.1%}, market favorite {log.loc[m, 'market_correct'].mean():.1%}")
    print(f"  log loss (lower is better): picks {model.logloss(log.loc[m, 'final_p_home'], home_won[m]):.3f}, "
          f"model {model.logloss(log.loc[m, 'p_home'], home_won[m]):.3f}, "
          f"market {model.logloss(log.loc[m, 'market_p_home'], home_won[m]):.3f}")
    bets = log[log["bet"].astype(bool)]
    if len(bets):
        print(f"  BET flags (edge >= {EDGE_BET:.0%}): {int((bets['bet_profit'] > 0).sum())}-"
              f"{int((bets['bet_profit'] < 0).sum())}, {bets['bet_profit'].sum():+.2f} units "
              f"({bets['bet_profit'].mean():+.1%} ROI) at the odds when predicted")
    print(f"  [{LOG.relative_to(ROOT)}]")


def cmd_daily(args):
    cmd_update(args)
    print()
    cmd_grade(argparse.Namespace(date=None))
    print()
    cmd_predict(argparse.Namespace(date=None))


def cmd_backtest(_args):
    hist, _ = model.build()
    p = model.backtest(hist)
    print(f"Walk-forward backtest {p['date'].min():%Y-%m-%d} to {p['date'].max():%Y-%m-%d} "
          f"(each day predicted by a model trained only on earlier games)\n")
    p["season"] = p["season"].astype(str)
    has = p["close_p_home"].notna()
    print(f"{'Season':<10}{'Games':>6}  {'Model acc':>9} {'Market acc':>10}  {'Model LL':>8} {'Market LL':>9}")
    for season, s in list(p.groupby("season")) + [("ALL", p)]:
        s = s[s["close_p_home"].notna()]
        if s.empty:
            continue
        mo, mk = model.evaluate(s), model.evaluate(s, "close_p_home")
        print(f"{season:<10}{len(s):>6}  {mo['accuracy']:>9.1%} {mk['accuracy']:>10.1%}  "
              f"{mo['log_loss']:>8.4f} {mk['log_loss']:>9.4f}")

    t = model.evaluate_totals(p)
    print(f"\nTotals vs closing line ({t['games']} games, pushes excluded): model MAE {t['model_mae']:.3f} vs "
          f"line MAE {t['line_mae']:.3f}; model over/under calls right {t['over_under_acc']:.1%}; "
          f"log loss model {t['model_ll']:.4f} vs market {t['market_ll']:.4f}")

    b = p[has].copy()
    home_pick = b["p_home"] >= 0.5
    pick_p = np.where(home_pick, b["p_home"], 1 - b["p_home"])
    mkt = np.where(home_pick, b["close_p_home"], 1 - b["close_p_home"])
    price = dec_odds(np.where(home_pick, b["close_ml_home"], b["close_ml_away"]).astype(float))
    won = np.where(home_pick, b["home_win"] == 1, b["home_win"] == 0)
    print("\nFlat 1-unit bets on the model's pick at closing odds, when model beats market by >= edge:")
    for e in (0.0, 0.02, 0.03, 0.05, 0.08):
        sel = (pick_p - mkt) >= e
        profit = np.where(won[sel], price[sel] - 1, -1).sum()
        print(f"  edge >= {e:.0%}: {sel.sum():>5} bets, win {won[sel].mean():.1%}, "
              f"{profit:+.1f} units, ROI {profit / max(sel.sum(), 1):+.1%}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("update").set_defaults(fn=cmd_update)
    for name, fn in (("predict", cmd_predict), ("grade", cmd_grade)):
        sp = sub.add_parser(name)
        sp.add_argument("date", nargs="?", help="YYYY-MM-DD")
        sp.set_defaults(fn=fn)
    sub.add_parser("daily").set_defaults(fn=cmd_daily)
    sub.add_parser("backtest").set_defaults(fn=cmd_backtest)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
