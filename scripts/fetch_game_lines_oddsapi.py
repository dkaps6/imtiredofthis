#!/usr/bin/env python3
# scripts/fetch_game_lines_oddsapi.py
# Optional helper that hits The Odds API (https://the-odds-api.com/) to fetch moneylines/totals/spreads
# Requires env var ODDS_API_KEY. Writes outputs/odds_game.csv
# This script is optional; use build_game_lines_from_schedule.py if you prefer to avoid external APIs.

import os, sys, json, time
import pandas as pd
import numpy as np
import urllib.request
from urllib.parse import urlencode

OUT = "outputs/odds_game.csv"

def _normalize_team_names(s: pd.Series) -> pd.Series:
    if s is None:
        return s
    norm = s.astype(str).str.upper().str.strip()
    aliases = {"WSH":"WAS","WDC":"WAS","JAC":"JAX","ARZ":"ARI","AZ":"ARI","LA":"LAR","LVR":"LV","OAK":"LV","SFO":"SF","TAM":"TB","GBP":"GB","KAN":"KC","NOS":"NO","SD":"LAC"}
    return norm.replace(aliases)

def american_to_prob(odds):
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if o > 0:
        return 100.0 / (o + 100.0)
    else:
        return (-o) / (-o + 100.0)

def devig_two_way(p_home, p_away):
    if np.isnan(p_home) or np.isnan(p_away): 
        return (np.nan, np.nan)
    s = p_home + p_away
    if s <= 0: 
        return (np.nan, np.nan)
    return (p_home / s, p_away / s)

def fetch_json(url):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read().decode("utf-8"))

def main():
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        print("[oddsapi] ODDS_API_KEY env var not set; skipping", file=sys.stderr)
        sys.exit(0)

    # Fetch upcoming events for NFL
    params = {"apiKey": api_key, "sport": "americanfootball_nfl", "region": "us", "mkt": "h2h,spreads,totals", "dateFormat": "iso"}
    base = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
    url = base + "?" + urlencode(params)
    try:
        data = fetch_json(url)
    except Exception as e:
        print(f"[oddsapi] fetch failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Parse
    rows = []
    for game in data:
        eid = game.get("id")
        commence = game.get("commence_time")
        home = game.get("home_team")
        away = game.get("away_team")
        bm = game.get("bookmakers", [])
        # Take first bookmaker as reference
        h2h, spread, total = None, None, None
        if bm:
            for mk in bm[0].get("markets", []):
                key = mk.get("key")
                if key == "h2h": h2h = mk
                elif key == "spreads": spread = mk
                elif key == "totals": total = mk
        row = {"event_id": eid, "home_team": home, "away_team": away, "commence_time": commence,
               "home_moneyline": np.nan, "away_moneyline": np.nan, "home_spread": np.nan, "total": np.nan}

        # Moneyline
        if h2h and h2h.get("outcomes"):
            for oc in h2h["outcomes"]:
                if oc.get("name") == home: row["home_moneyline"] = oc.get("price")
                elif oc.get("name") == away: row["away_moneyline"] = oc.get("price")

        # Spread
        if spread and spread.get("outcomes"):
            for oc in spread["outcomes"]:
                if oc.get("name") == home: row["home_spread"] = oc.get("point")
        # Total
        if total and total.get("outcomes"):
            # totals list 'Over' and 'Under'; we just store the point
            pts = total["outcomes"][0].get("point")
            row["total"] = pts

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["home_team"] = _normalize_team_names(df["home_team"])
        df["away_team"] = _normalize_team_names(df["away_team"])
        # implied win probs
        p_home = df["home_moneyline"].map(american_to_prob)
        p_away = df["away_moneyline"].map(american_to_prob)
        df["home_wp"], df["away_wp"] = zip(*[devig_two_way(h, a) for h,a in zip(p_home, p_away)])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"[oddsapi] wrote {OUT} rows={len(df)}")

if __name__ == "__main__":
    main()
