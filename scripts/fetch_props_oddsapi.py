#!/usr/bin/env python3
from __future__ import annotations
import sys, os
from pathlib import Path
from datetime import datetime, timezone
import time
import requests
import pandas as pd

API_BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
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

def _GET(path: str, params: dict, label: str):
    try:
        r = requests.get(path, params=params, timeout=20)
        r.raise_for_status()
        return r
    except requests.HTTPError as e:
        print(f"[oddsapi] {label} failed: {e} url={r.url if 'r' in locals() else path}", flush=True)
        return None
    except Exception as e:
        print(f"[oddsapi] {label} error: {e}", flush=True)
        return None

def _write_headers():
    OUT_PROPS.parent.mkdir(parents=True, exist_ok=True)
    if not OUT_PROPS.exists() or OUT_PROPS.stat().st_size == 0:
        pd.DataFrame(columns=[
            "event_id","commence_time","book","market","player","line","odds"
        ]).to_csv(OUT_PROPS, index=False)
    if not OUT_GAME.exists() or OUT_GAME.stat().st_size == 0:
        pd.DataFrame(columns=["id","commence_time","home_team","away_team","bookmakers"]).to_csv(OUT_GAME, index=False)

def fetch(date: str|None, books: list[str]) -> tuple[pd.DataFrame,pd.DataFrame]:
    if not KEY:
        print("[oddsapi] ERROR: ODDS_API_KEY missing", flush=True)
        _write_headers(); return pd.DataFrame(), pd.DataFrame()

    # ---------- GAME LINES ----------
    game_df = pd.DataFrame()
    games_url = f"{API_BASE}/odds"
    # Try parameterizations in order:
    game_param_sets = []

    # A) Most strict (UNIX since you passed --date)
    if date:
        game_param_sets.append(dict(
            apiKey=KEY, regions="us", oddsFormat="american",
            bookmakers=",".join(books), markets=",".join(GAME_MARKETS),
            commenceTimeFrom=_unix_start_of_day(date)
        ))
    # B) No commenceTimeFrom (some accounts/regions reject it)
    game_param_sets.append(dict(
        apiKey=KEY, regions="us", oddsFormat="american",
        bookmakers=",".join(books), markets=",".join(GAME_MARKETS),
    ))
    # C) Split markets one-by-one (some accounts reject multi-market combos)
    for m in GAME_MARKETS:
        game_param_sets.append(dict(
            apiKey=KEY, regions="us", oddsFormat="american",
            bookmakers=",".join(books), markets=m,
        ))

    for i,params in enumerate(game_param_sets,1):
        r = _GET(games_url, params, f"game attempt#{i}")
        if r is None: 
            continue
        try:
            df = pd.json_normalize(r.json())
            if not df.empty:
                game_df = df
                break
        except Exception as e:
            print(f"[oddsapi] game parse error: {e}", flush=True)
            continue

    # ---------- PLAYER PROPS ----------
    props_rows = []
    # Try each market; for robustness, try all books together, then each book separately
    for mkt in PLAYER_MARKETS:
        # 1) all books together
        param_sets = [dict(
            apiKey=KEY, regions="us", oddsFormat="american",
            bookmakers=",".join(books), markets=mkt
        )]
        # 2) each book individually (fallback)
        param_sets += [dict(apiKey=KEY, regions="us", oddsFormat="american", bookmakers=b, markets=mkt) for b in books]

        got_any = False
        for i,params in enumerate(param_sets,1):
            r = _GET(games_url, params, f"props {mkt} attempt#{i}")
            if r is None: 
                continue
            try:
                js = r.json()
                added = 0
                for game in js:
                    gid = game.get("id"); commence = game.get("commence_time")
                    for bm in game.get("bookmakers", []):
                        book = bm.get("key")
                        for mk in bm.get("markets", []):
                            if mk.get("key") != mkt: 
                                continue
                            for o in mk.get("outcomes", []):
                                props_rows.append({
                                    "event_id": gid,
                                    "commence_time": commence,
                                    "book": book,
                                    "market": mkt,
                                    "player": o.get("name"),
                                    "line": o.get("point"),
                                    "odds": o.get("price")
                                })
                                added += 1
                if added > 0:
                    got_any = True
                    break
            except Exception as e:
                print(f"[oddsapi] props parse error ({mkt}): {e}", flush=True)
                continue
        if not got_any:
            print(f"[oddsapi] props {mkt}: no data (after retries)")

    props_df = pd.DataFrame(props_rows)

    # ---------- WRITE ----------
    _write_headers()
    if not props_df.empty: props_df.to_csv(OUT_PROPS, index=False)
    if game_df is not None and not game_df.empty: game_df.to_csv(OUT_GAME, index=False)

    print(f"[oddsapi] wrote props={len(props_df)} games={0 if game_df is None else len(game_df)}")
    return props_df, game_df

def cli(books: list[str], date: str|None, out: str|None, season: int|None) -> int:
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
    ap.add_argument("--season", type=int, default=None)  # optional
    a = ap.parse_args()
    sys.exit(cli([b.strip() for b in a.books.split(",") if b.strip()], a.date or None, a.out, a.season))
