#!/usr/bin/env python3
from __future__ import annotations
import sys
import os
from pathlib import Path
from datetime import datetime, timezone
import requests
import pandas as pd

API = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
KEY = os.getenv("ODDS_API_KEY", "")

OUT_PROPS = Path("outputs/props_raw.csv")
OUT_GAME  = Path("outputs/odds_game.csv")

GAME_MARKETS   = ["h2h","spreads","totals"]
PLAYER_MARKETS = [
    "player_receiving_yards","player_receptions","player_rushing_yards",
    "player_rush_and_receive_yards","player_passing_yards","player_passing_tds",
    "player_anytime_td"
]

def _unix_start_of_day(iso_date: str) -> int:
    dt = datetime.fromisoformat(iso_date).replace(tzinfo=timezone.utc)
    return int(dt.timestamp())

def _get(params: dict) -> requests.Response:
    r = requests.get(API, params=params, timeout=20)
    r.raise_for_status()
    return r

def fetch(date: str|None, books: list[str]) -> tuple[pd.DataFrame,pd.DataFrame]:
    if not KEY:
        print("[oddsapi] ERROR: ODDS_API_KEY missing", flush=True)
        return pd.DataFrame(), pd.DataFrame()

    base = dict(apiKey=KEY, regions="us", oddsFormat="american", bookmakers=",".join(books))

    # Game lines: OK to use commenceTimeFrom (UNIX)
    base_game = base.copy()
    if date:
        base_game["commenceTimeFrom"] = _unix_start_of_day(date)

    # 1) Game markets
    game_df = pd.DataFrame()
    try:
        p = base_game | {"markets": ",".join(GAME_MARKETS)}
        print(f"[oddsapi] game markets={p['markets']}")
        r = _get(p)
        game_df = pd.json_normalize(r.json())
    except requests.HTTPError as e:
        print(f"[oddsapi] WARNING game markets failed: {e}", flush=True)

    # 2) Player props (no commenceTimeFrom — some accounts/regions 422 if you include it)
    props = []
    for mkt in PLAYER_MARKETS:
        try:
            p = base | {"markets": mkt}
            print(f"[oddsapi] probe market={mkt}")
            r = _get(p)
            for game in r.json():
                gid = game.get("id"); commence = game.get("commence_time")
                for bm in game.get("bookmakers", []):
                    book = bm.get("key")
                    for mk in bm.get("markets", []):
                        if mk.get("key") != mkt: 
                            continue
                        for o in mk.get("outcomes", []):
                            props.append({
                                "event_id": gid,
                                "commence_time": commence,
                                "book": book,
                                "market": mkt,
                                "player": o.get("name"),
                                "line": o.get("point"),
                                "odds": o.get("price")
                            })
        except requests.HTTPError as e:
            print(f"[oddsapi] error market={mkt}: {e}", flush=True)
            continue

    props_df = pd.DataFrame(props)

    # Always write CSVs with headers
    OUT_PROPS.parent.mkdir(parents=True, exist_ok=True)
    if props_df.empty:
        pd.DataFrame(columns=[
            "event_id","commence_time","book","market","player","line","odds"
        ]).to_csv(OUT_PROPS, index=False)
    else:
        props_df.to_csv(OUT_PROPS, index=False)

    if game_df is None or game_df.empty:
        pd.DataFrame(columns=["id","commence_time","home_team","away_team","bookmakers"]).to_csv(OUT_GAME, index=False)
    else:
        game_df.to_csv(OUT_GAME, index=False)

    print(f"[oddsapi] wrote props={len(props_df)} games={len(game_df)}")
    return props_df, game_df

def cli(books: list[str], date: str|None, out: str|None, season: int|None) -> int:
    # season is optional; included only to be flexible with other callers
    props_df, _ = fetch(date, books)
    if out:
        try:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            pd.read_csv(OUT_PROPS).to_csv(out, index=False)
        except Exception:
            pass
    return 0

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", default="draftkings,fanduel,betmgm,caesars")
    ap.add_argument("--date", default="")
    ap.add_argument("--out", default="outputs/props_raw.csv")
    ap.add_argument("--season", type=int, default=None)  # OPTIONAL
    a = ap.parse_args()
    sys.exit(cli([b.strip() for b in a.books.split(",") if b.strip()], a.date or None, a.out, a.season))
