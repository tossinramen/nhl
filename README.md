# NHL Model

## Commands

```
python nhl.py daily              # update data -> grade yesterday -> predict today
python nhl.py update             # pull new games: NST stats, NHL boxscores (lineups/goalies), closing odds
python nhl.py predict [DATE]     # winners + totals for DATE (YYYY-MM-DD, default: today ET)
python nhl.py grade [DATE]       # score saved predictions for DATE (default: yesterday) vs results and the market
python nhl.py backtest           # walk-forward test vs the betting market, 2019-20 to now

python lineups.py [DATE]         # starting goalies + injuries for a day (RotoWire, ESPN fallback)
python odds.py [DATE]            # sportsbook consensus + Polymarket lines for a day
python schedule.py [DATE]        # NHL schedule/results for a day
python collect.py                # rebuild NHL boxscore + odds history from scratch (first-time setup)
python nst_scraper.py --history 2018 2024   # NST team-level stats for past seasons (35 requests)
python nst_scraper.py --limit N             # scrape at most N new NST games this run
python nst_scraper.py --combine-only        # rebuild data/*.csv from the NST cache without scraping
```

## Setup

```
pip install -r requirements.txt
```

Create a `.env` file with your Natural Stat Trick access key (needed for `update`):

```
NST_ACCESS_KEY=your-key-here
```

First-time history (already done in this repo's `data/`): `python nst_scraper.py --history 2018 2024`
then `python collect.py` (~10k NHL boxscores + ~1.4k days of odds, about 45 minutes).

## Reading the predictions

```
    Time  Game       Pick Win%  Model Market  Model edge     Score  Total Line Over  Goalies (away / home)
 7:30 PM  COL @ WPG  COL   64%    68%    62%  COL +6.2% BET 3.8-2.7   6.43  6.5  46%  Blackwood / Skinner
```

- **Pick / Win%**: the final call: 40% model / 60% market blend (model alone when no odds are available)
- **Model / Market**: each one's win chance for the pick; market = no-vig sportsbook consensus, else Polymarket
- **Model edge**: the team the model likes more than the market and by how much; **BET** = edge of 5% or more
- **Score / Total**: the model's expected goals; **Line / Over**: sportsbook total and the model's chance of the over
- **Goalies**: projected starters (`*` = confirmed); injured players the model removed are listed under the table

Predicting a past date uses the lineups and goalies that actually played plus the closing odds, so
re-running an old day is a fair test (ratings still only use games before that day).

## How it works

| File | Purpose |
|---|---|
| `nst_scraper.py` | Natural Stat Trick: team xG/shots/goals per game (plus player and line data) |
| `collect.py` | NHL API boxscores (who dressed, starting goalie) and Action Network odds history |
| `lineups.py` | Day-of starting goalies and injuries (RotoWire lineups page + injury report, ESPN fallback) |
| `odds.py` | Day-of odds: Action Network consensus and Polymarket |
| `model.py` | Ratings, win/goals models, backtest |
| `forecast.py` | Projects each team's goalie and lineup for a day and runs the model |
| `teams.py` | Team names/abbreviations across sources (incl. Arizona -> Utah) |

**Ratings**, updated after every game, using only earlier games:
- **Team**: exponentially weighted xG, goals and shots for/against per 60, power play / penalty kill xG,
  finishing and saving vs expected (NST, all situations)
- **Goalie**: goals saved above expected per 60, shrunk toward slightly below average for small samples
- **Lineup**: each skater's rating from his own game logs (follows him across trades); a team's lineup
  strength is the sum over the 18 skaters expected to dress, so injuries and roster changes count immediately

**Projected lineup** for today: current NHL roster minus players RotoWire lists as out/IR (day-to-day = 50%),
top 12 F + 6 D by ice time. **Goalie**: RotoWire's confirmed/expected starter, else the team's most frequent
recent starter.

**Models**: logistic regression on home-minus-away rating differences (win probability), Poisson regression
for each team's goals (score and total). Win probability is then blended with the market.

## Accuracy

Walk-forward backtest, 8,558 games (2019-20 to Oct 2026), each day predicted by a model trained only on
earlier games, compared with the closing betting line on the same games:

| Season | Model | Market favorite | Model log loss | Market log loss |
|---|---|---|---|---|
| 2019-20 | 56.3% | 56.2% | 0.6805 | 0.6788 |
| 2020-21 | 61.6% | 62.5% | 0.6563 | 0.6524 |
| 2021-22 | 64.8% | 64.9% | 0.6390 | 0.6425 |
| 2022-23 | 61.1% | 62.0% | 0.6547 | 0.6518 |
| 2023-24 | 61.3% | 61.4% | 0.6531 | 0.6556 |
| 2024-25 | 60.6% | 60.7% | 0.6571 | 0.6579 |
| 2025-26 | 55.3% | 55.3% | 0.6857 | 0.6814 |
| **All** | **60.1%** | **60.4%** | **0.6606** | **0.6599** |

- The model is roughly as good as the closing line; the 40/60 blend beats both on log loss (0.6566 vs
  0.6572 market, 2020-21 onward). Accuracy swings by season with parity: 2025-26 was hard for everyone.
- Adding starting goalie + lineup ratings improved log loss from 0.6588 to 0.6563 over team stats alone
  (tuned on 2019-23 only).
- **Betting**: at closing odds there is no edge (ROI about -0.3% at a 3% edge, 95% CI -3% to +3%). Against
  opening lines (2023-24 onward), using only information available early (usual goalie and lineup),
  bets with a 5%+ edge returned +6.6% over 1,190 bets (95% CI +1.5% to +11.7%). Any edge is in betting
  early, before the market absorbs goalie and injury news; it is not guaranteed and early limits are low.
- **Totals**: the model matches the closing total (MAE 1.84 vs 1.84) and calls overs/unders 51.8% right,
  below the ~52.4% needed to profit at -110. Treat totals as informational.

## Limitations

- The backtest uses the lineups and goalies that actually played; live predictions use projections, which
  are usually but not always right (check for `*` confirmed goalies close to game time).
- RotoWire, Action Network and Polymarket are unofficial sources; if a page changes, `lineups.py` / `odds.py`
  fall back or skip that source and predictions still run.
- Regular season only.
