# NHL Model

## Commands

```
python nhl.py daily              # update data -> grade yesterday's picks -> predict today
python nhl.py update             # scrape any new completed games from Natural Stat Trick, rebuild data/*.csv
python nhl.py predict [DATE]     # predict winners + totals for DATE (YYYY-MM-DD, default: today ET)
python nhl.py grade [DATE]       # score saved predictions for DATE (default: yesterday) against real results
python nhl.py backtest           # walk-forward accuracy test on all data so far

python schedule.py [DATE]        # print the NHL schedule/results for a day
python nst_scraper.py --limit N  # scrape at most N new games this run
python nst_scraper.py --combine-only  # rebuild data/*.csv from the cache without scraping
```

## Setup

```
pip install -r requirements.txt
```

Create a `.env` file with your Natural Stat Trick access key (needed for `update` only):

```
NST_ACCESS_KEY=your-key-here
```

## Daily workflow

Run `python nhl.py daily` each day before games start. It:

1. **Updates** the data with any games finished since the last run. Already-scraped games are cached in
   `data/cache/games/` and skipped, and games from today or not yet marked final are never cached, so it is
   safe to run any time.
2. **Grades** yesterday's predictions against final scores from the NHL API and appends them to
   `predictions/graded.csv`, printing the season-to-date record.
3. **Predicts** today's games and saves them to `predictions/YYYY-MM-DD.csv`.

Example output:

```
    Time  Matchup                      Pick            Win%      Score  Total   O5.5  O6.5
 7:30 PM  Avalanche @ Jets             Avalanche        58%    3.6-3.1   6.69    66%   50%
```

- **Pick / Win%**: the predicted winner and its win probability (including OT/SO)
- **Score**: expected goals, away-home
- **Total**: expected combined goals (shootout winner counts as a goal, as with betting totals)
- **O5.5 / O6.5**: probability the total goes over that line

## How it works

| File | Purpose |
|---|---|
| `nst_scraper.py` | Scrapes game-level team, skater, goalie and line stats from Natural Stat Trick into `data/` |
| `schedule.py` | Fetches each day's schedule and results from the public NHL API (`api-web.nhle.com`, no key) |
| `model.py` | Team ratings, Poisson model, and walk-forward backtest |
| `nhl.py` | Command line entry point |

**Model.** Each team carries exponentially weighted ratings (20-game half-life) of its all-situations
xG, goal and shot rates for and against, power-play xG, penalty-kill xG against, finishing (goals − xG)
and goaltending (xGA − goals against). Ratings are shrunk toward the league average and carry half their
weight into a new season. A Poisson regression turns "this team's offense vs. that team's defense" plus
home ice and back-to-backs into expected goals for each side; the two goal distributions give the win
probability (regulation ties split 50/50 for OT/SO) and the total.

Predictions for any date only use games played *before* that date, so past dates can be re-predicted
and graded honestly.

## Accuracy

Walk-forward backtest over 1,081 games (Nov 2025 – Oct 2026), each day predicted using only earlier games:

| Metric | Model | Baseline |
|---|---|---|
| Winner accuracy | 56.3% | 52.3% (always pick home) |
| Log loss | 0.680 | 0.693 (coin flip) |
| Total goals MAE | 1.85 | 1.84 (league average) |



