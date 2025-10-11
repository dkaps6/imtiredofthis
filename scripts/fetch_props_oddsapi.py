# scripts/fetch_props_oddsapi.py
import argparse
import os
import sys
import time
import datetime as dt
from pathlib import Path
from typing import List, Dict, Any

import requests
import pandas as pd

BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"

def _iso_bounds_for_date(d: str) -> Dict[str, str]:
    # d is YYYY-MM-DD (local naive); convert to Z bounds
    day = dt.date.fromisoformat(d)
    start = dt.datetime.combine(day, dt.time(0, 0))
    end = start + dt.timedelta(days=1)
    return {
        "commenceTimeFrom": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commenceTimeTo": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dateFormat": "iso",
    }

def _normalize_event_rows(
    payload: List[Dict[str, Any]],
    market_key: str,
    allowed_books: List[str],
) -> List[Dict[str, Any]]:
    """
    Flatten Odds API JSON into rows:
    event_id, commence_time, book, market, player, line, side, odds
    For non-player markets (h2h/spreads/totals) we still emit rows but player will be blank.
    """
    rows = []
    for ev in payload:
        event_id = ev.get("id")
        commence_time = ev.get("commence_time")

        for bk in (ev.get("bookmakers") or []):
            book_key = bk.get("key")
            if allowed_books and book_key not in allowed_books:
                continue

            for mk in (bk.get("markets") or []):
                mk_key = mk.get("key")
                if mk_key != market_key:
                    continue
                for oc in (mk.get("outcomes") or []):
                    # Outcome fields vary by market. Props typically have name (player), price, point
                    player = oc.get("name") or ""
                    line = oc.get("point")
                    price = oc.get("price")
                    # 'side' for props usually "Over"/"Under"; for h2h it's a team/participant
                    side = oc.get("description") or oc.get("name") or ""

                    rows.append({
                        "event_id": event_id,
                        "commence_time": commence_time,
                        "book": book_key,
                        "market": mk_key,
                        "player": player,
                        "line": line,
                        "side": side,
                        "odds": price,
                    })
    return rows

def fetch_market(
    api_key: str,
    market: str,
    books: List[str],
    date_str: str = "",
    region: str = "us",
    odds_format: str = "american",
    pause_sec: float = 0.35,
) -> pd.DataFrame:
    params = {
        "apiKey": api_key,
        "regions": region,
        "oddsFormat": odds_format,
        "bookmakers": ",".join(books) if books else "",
        "markets": market,
    }

    # If --date is supplied, convert to ISO bounds (The Odds API doesn’t accept --date directly)
    if date_str:
        params.update(_iso_bounds_for_date(date_str))

    url = BASE
    print(f"[oddsapi] request market={market} -> {url}")
    r = requests.get(url, params=params, timeout=20)

    # Log rate-limit headers if present
    xr = r.headers.get("x-requests-remaining")
    xu = r.headers.get("x-requests-used")
    if xr is not None and xu is not None:
        print(f"[oddsapi] x-requests-remaining: {xr}")
        print(f"[oddsapi] x-requests-used: {xu}")

    if r.status_code == 200:
        data = r.json() if r.text else []
        rows = _normalize_event_rows(data, market, books)
        df = pd.DataFrame(rows)
        print(f"[oddsapi] market={market} rows={len(df)}")
        time.sleep(pause_sec)  # be nice to the API
        return df

    # Soft-fail on 401/422/other
    print(f"[oddsapi] error market={market}: {r.status_code} {r.text[:200]}")
    time.sleep(pause_sec)
    return pd.DataFrame(columns=["event_id","commence_time","book","market","player","line","side","odds"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", default="draftkings,fanduel,betmgm,caesars")
    ap.add_argument("--markets", default="")  # comma-separated; if blank we do a sane default set
    ap.add_argument("--date", default="")     # YYYY-MM-DD optional
    ap.add_argument("--out", default="outputs/props_raw.csv")
    a = ap.parse_args()

    api_key = os.getenv("ODDS_API_KEY", "")
    if not api_key:
        print("[oddsapi] ERROR: ODDS_API_KEY is not set")
        sys.exit(2)

    books = [b.strip() for b in a.books.split(",") if b.strip()]
    # If no markets passed, use a curated set (ask for one at a time)
    if a.markets.strip():
        markets = [m.strip() for m in a.markets.split(",") if m.strip()]
    else:
        markets = [
            # game odds (featured)
            "h2h", "spreads", "totals",
            # popular player props (ask one-by-one)
            "player_pass_yds",
            "player_rush_yds",
            "player_rec_yds",
            "player_receptions",
            "player_passing_tds",
            "player_anytime_td",
            # add/remove per need:
            # "player_rush_rec_yds", "player_two_or_more_tds",
        ]

    # Loop one market at a time and merge
    frames = []
    for m in markets:
        df = fetch_market(api_key, m, books=books, date_str=a.date)
        if not df.empty:
            frames.append(df)

    if frames:
        props = pd.concat(frames, ignore_index=True)
    else:
        # still write an empty, but well-formed CSV so downstream doesn't explode
        props = pd.DataFrame(columns=["event_id","commence_time","book","market","player","line","side","odds"])

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    props.to_csv(a.out, index=False)
    print(f"[oddsapi] wrote props={len(props)} -> {a.out}")

    # Additionally write game-only odds if present
    game_df = props[props["market"].isin(["h2h", "spreads", "totals"])]
    game_out = "outputs/odds_game.csv"
    game_df.to_csv(game_out, index=False)
    print(f"[oddsapi] wrote game-only odds -> {game_out}")

if __name__ == "__main__":
    main()
