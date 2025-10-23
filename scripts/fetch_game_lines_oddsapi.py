#!/usr/bin/env python3
# scripts/fetch_game_lines_oddsapi.py
import os, sys, json
import pandas as pd, numpy as np
import urllib.request
from urllib.parse import urlencode

OUT = "outputs/odds_game.csv"

def _normalize_team_names(s: pd.Series) -> pd.Series:
    if s is None:
        return s
    norm = s.astype(str).str.upper().str.strip()
    aliases = {"WSH":"WAS","WDC":"WAS","JAC":"JAX","ARZ":"ARI","AZ":"ARI",
               "LA":"LAR","LVR":"LV","OAK":"LV","SFO":"SF","TAM":"TB",
               "GBP":"GB","KAN":"KC","NOS":"NO","SD":"LAC"}
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
        print("[oddsapi] ODDS_API_KEY not set; skipping", file=sys.stderr)
        sys.exit(0)  # don’t fail pipeline

    # Correct param names: regions + markets
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds?" + urlencode(params)

    try:
        data = fetch_json(url)
    except Exception as e:
        print(f"[oddsapi] fetch failed: {e}", file=sys.stderr)
        sys.exit(0)  # soft-fail

    if not isinstance(data, list) or not data:
        print("[oddsapi] no data returned; writing empty file", file=sys.stderr)
        pd.DataFrame().to_csv(OUT, index=False)
        sys.exit(0)

    rows = []
    for game in data:
        eid = game.get("id")
        commence = game.get("commence_time")
        home = game.get("home_team")
        away = game.get("away_team")
        bm = game.get("bookmakers", [])
        h2h = next((m for m in bm[0].get("markets", []) if m.get("key")=="h2h"), None) if bm else None
        spreads = next((m for m in bm[0].get("markets", []) if m.get("key")=="spreads"), None) if bm else None
        totals  = next((m for m in bm[0].get("markets", []) if m.get("key")=="totals"), None) if bm else None

        row = {"event_id": eid, "home_team": home, "away_team": away, "commence_time": commence,
               "home_moneyline": np.nan, "away_moneyline": np.nan, "home_spread": np.nan, "total": np.nan}

        if h2h and h2h.get("outcomes"):
            for oc in h2h["outcomes"]:
                if oc.get("name") == home: row["home_moneyline"] = oc.get("price")
                if oc.get("name") == away: row["away_moneyline"] = oc.get("price")

        if spreads and spreads.get("outcomes"):
            for oc in spreads["outcomes"]:
                if oc.get("name") == home: row["home_spread"] = oc.get("point")

        if totals and totals.get("outcomes"):
            pts = totals["outcomes"][0].get("point")
            row["total"] = pts

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["home_team"] = _normalize_team_names(df["home_team"])
        df["away_team"] = _normalize_team_names(df["away_team"])
        p_home = df["home_moneyline"].map(american_to_prob)
        p_away = df["away_moneyline"].map(american_to_prob)
        df["home_wp"], df["away_wp"] = zip(*[devig_two_way(h, a) for h,a in zip(p_home, p_away)])

    os.makedirs("outputs", exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"[oddsapi] wrote {OUT} rows={len(df)}")

if __name__ == "__main__":
    main()
