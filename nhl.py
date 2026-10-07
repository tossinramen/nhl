import argparse
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

import model
import schedule

ROOT = Path(__file__).parent
PRED = ROOT / "predictions"
LOG = PRED / "graded.csv"


def short(name):
    return name.split()[-1]


def cmd_update(_args):
    r = subprocess.run([sys.executable, str(ROOT / "nst_scraper.py")], cwd=ROOT)
    if r.returncode:
        sys.exit(f"update failed (exit {r.returncode})")


def check_freshness(date):
    games = pd.read_csv(model.DATA / "games.csv")
    last = games["date"].max()
    yesterday = (pd.Timestamp(date) - timedelta(days=1)).strftime("%Y-%m-%d")
    if last < yesterday:
        print(f"WARNING: data only runs through {last}. Run `python nhl.py update` first so "
              f"ratings and back-to-back flags include recent games.\n")


def cmd_predict(args):
    date = args.date or schedule.today_et()
    games = schedule.get_games(date)
    if games.empty:
        print(f"No regular-season NHL games on {date}.")
        return
    check_freshness(date)
    hist, ratings = model.build_history(before=date)
    m = model.fit(hist)
    p = model.predict_games(m, games, ratings)

    print(f"NHL predictions for {date}  (model trained on {len(hist) // 2} games "
          f"through {hist['date'].max():%Y-%m-%d})\n")
    print(f"{'Time':>8}  {'Matchup':<28} {'Pick':<14} {'Win%':>5}  {'Score':>9}  {'Total':>5}  {'O5.5':>5} {'O6.5':>5}")
    for _, r in p.iterrows():
        matchup = f"{short(r['away_team'])} @ {short(r['home_team'])}"
        score = f"{r['xg_away']:.1f}-{r['xg_home']:.1f}"
        print(f"{r['start_et']:>8}  {matchup:<28} {short(r['pick']):<14} {r['pick_prob']:>5.0%}  {score:>9}  "
              f"{r['pred_total']:>5.2f}  {r['p_over_5.5']:>5.0%} {r['p_over_6.5']:>5.0%}")
    print("\nScore = expected goals away-home. O5.5/O6.5 = probability the total goes over that line.")

    PRED.mkdir(exist_ok=True)
    out = PRED / f"{date}.csv"
    p.drop(columns=["state", "away_score", "home_score", "result_type"]).to_csv(out, index=False)
    print(f"saved {out.relative_to(ROOT)}")


def cmd_grade(args):
    date = args.date or (pd.Timestamp(schedule.today_et()) - timedelta(days=1)).strftime("%Y-%m-%d")
    f = PRED / f"{date}.csv"
    if not f.exists():
        print(f"No saved predictions for {date} ({f.relative_to(ROOT)} missing).")
        return
    preds = pd.read_csv(f)
    results = schedule.get_games(date)
    done = results[results["state"].isin(["OFF", "FINAL"])]
    g = preds.merge(done[["nhl_game_id", "away_score", "home_score", "result_type"]], on="nhl_game_id")
    if g.empty:
        print(f"No finished games yet for {date}.")
        return
    g[["away_score", "home_score"]] = g[["away_score", "home_score"]].astype(int)
    g["winner"] = g["home_team"].where(g["home_score"] > g["away_score"], g["away_team"])
    g["correct"] = g["pick"] == g["winner"]
    g["total"] = g["home_score"] + g["away_score"]
    g["total_err"] = g["pred_total"] - g["total"]

    print(f"Results for {date}:\n")
    for _, r in g.iterrows():
        mark = "OK  " if r["correct"] else "MISS"
        ot = f" ({r['result_type']})" if r["result_type"] in ("OT", "SO") else ""
        print(f"  {mark} {short(r['away_team'])} {r['away_score']} @ {short(r['home_team'])} {r['home_score']}{ot}"
              f"  | picked {short(r['pick'])} {r['pick_prob']:.0%} | total {r['total']} vs pred {r['pred_total']:.2f}")
    print(f"\n  {date}: {g['correct'].sum()}/{len(g)} winners, total MAE {g['total_err'].abs().mean():.2f}")

    log = pd.concat([pd.read_csv(LOG), g]) if LOG.exists() else g
    log = log.drop_duplicates("nhl_game_id", keep="last").sort_values(["date", "nhl_game_id"])
    log.to_csv(LOG, index=False)
    print(f"  season to date: {log['correct'].sum()}/{len(log)} winners ({log['correct'].mean():.1%}), "
          f"total MAE {log['total_err'].abs().mean():.2f}  [{LOG.relative_to(ROOT)}]")


def cmd_daily(args):
    cmd_update(args)
    print()
    cmd_grade(argparse.Namespace(date=None))
    print()
    cmd_predict(argparse.Namespace(date=None))


def cmd_backtest(_args):
    g = model.backtest()
    res, calib = model.score_backtest(g)
    print(f"Walk-forward backtest, {g['date'].min():%Y-%m-%d} to {g['date'].max():%Y-%m-%d} "
          f"(each day predicted using only earlier games)\n")
    for k, v in res.items():
        print(f"  {k:<24} {v:.3f}" if isinstance(v, float) else f"  {k:<24} {v}")
    print("\nCalibration (how often picks at each confidence level won):")
    print(calib.to_string(float_format=lambda x: f"{x:.3f}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
