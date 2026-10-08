import json

import numpy as np
import pandas as pd
import requests

from collect import action_network_day
from teams import name_to_abbr, nickname_to_abbr

GAMMA = "https://gamma-api.polymarket.com/events"


def american_to_prob(ml):
    """Implied probability (with vig) of an American moneyline price."""
    ml = pd.to_numeric(ml, errors="coerce")
    return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))


def no_vig(p_a, p_b):
    p_a, p_b = np.asarray(p_a, float), np.asarray(p_b, float)
    return p_a / (p_a + p_b)


def market_probs(df, prefix="close"):
    df = df.copy()
    h, a = american_to_prob(df[f"{prefix}_ml_home"]), american_to_prob(df[f"{prefix}_ml_away"])
    df[f"{prefix}_p_home"] = no_vig(h, a)
    if f"{prefix}_over" in df:
        o, u = american_to_prob(df[f"{prefix}_over"]), american_to_prob(df[f"{prefix}_under"])
        df[f"{prefix}_p_over"] = no_vig(o, u)
    return df


def sportsbook_lines(date):
    rows = action_network_day(date)
    if not rows:
        return pd.DataFrame(columns=["home_abbr", "away_abbr"])
    df = pd.DataFrame(rows)
    df["home_abbr"] = df["home_name"].map(name_to_abbr)
    df["away_abbr"] = df["away_name"].map(name_to_abbr)
    keep = ["home_abbr", "away_abbr", "close_ml_home", "close_ml_away", "close_total", "close_over", "close_under"]
    df = df[[c for c in keep if c in df]].dropna(subset=["close_ml_home"])
    df = market_probs(df, "close")
    return df.rename(columns=lambda c: c.replace("close_", "book_"))


def _mid(m, i):
    prices = json.loads(m.get("outcomePrices") or "[]")
    bid, ask = m.get("bestBid"), m.get("bestAsk")
    if i == 0 and bid is not None and ask is not None and 0 < float(ask) - float(bid) < 0.1:
        return (float(bid) + float(ask)) / 2
    return float(prices[i]) if len(prices) > i else np.nan


def polymarket_lines(date):
    events, offset = [], 0
    while offset < 1000:
        r = requests.get(GAMMA, params={"tag_slug": "nhl", "closed": "false", "limit": 100, "offset": offset},
                         timeout=30)
        r.raise_for_status()
        batch = r.json()
        events += batch
        if len(batch) < 100:
            break
        offset += 100
    rows = []
    for e in events:
        if e.get("eventDate") != date or not e.get("slug", "").startswith("nhl-"):
            continue
        row = {"pm_volume": 0.0}
        for m in e.get("markets", []):
            kind = m.get("sportsMarketType")
            outcomes = json.loads(m.get("outcomes") or "[]")
            row["pm_volume"] += float(m.get("volume") or 0)
            if kind == "moneyline" and len(outcomes) == 2:
                a, h = (nickname_to_abbr(t.strip()) for t in e["title"].split(" vs. "))
                p0 = _mid(m, 0)
                first = nickname_to_abbr(outcomes[0])
                row.update(home_abbr=h, away_abbr=a, pm_p_home=p0 if first == h else 1 - p0)
            elif kind == "totals" and m.get("line") is not None and outcomes[:1] == ["Over"]:
                row[f"pm_p_over_{float(m['line'])}"] = _mid(m, 0)
        if "home_abbr" in row:
            rows.append(row)
    return pd.DataFrame(rows)


def day_lines(date):
    out = None
    for fn in (sportsbook_lines, polymarket_lines):
        try:
            df = fn(date)
        except Exception as e:  
            print(f"  (odds source {fn.__name__} unavailable: {e})")
            continue
        if df.empty:
            continue
        out = df if out is None else out.merge(df, on=["home_abbr", "away_abbr"], how="outer")
    return out if out is not None else pd.DataFrame(columns=["home_abbr", "away_abbr"])


if __name__ == "__main__":
    import sys
    import schedule
    pd.set_option("display.width", 200)
    print(day_lines(sys.argv[1] if len(sys.argv) > 1 else schedule.today_et()).round(3).to_string(index=False))
