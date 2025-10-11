# scripts/fetch_props_oddsapi.py
import os, sys, json, time
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
import requests
import pandas as pd

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
BASE = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"

# keep this tight; expand when your plan confirms access
DEFAULT_BOOKS = ["draftkings", "fanduel"]  # fewer books = fewer 422s
DEFAULT_MARKETS = [
    # player markets – fetch one-at-a-time
    "player_pass_yds",
    "player_rec_yds",
    "player_rush_yds",
    "player_receptions",
    "player_anytime_td",
    # add more as your plan allows
]

def _req(url: str, params: Dict[str, Any]) -> requests.Response:
    r = requests.get(url, params=params, timeout=30)
    return r

def _flatten_game_lines(events: List[Dict[str, Any]], date: Optional[str]) -> pd.DataFrame:
    rows = []
    for ev in events:
        event_id = ev.get("id")
        commence_time = ev.get("commence_time")
        for bm in ev.get("bookmakers", []):
            book = bm.get("key")
            for mk in bm.get("markets", []):
                market = mk.get("key")
                for outc in mk.get("outcomes", []):
                    rows.append({
                        "event_id": event_id,
                        "book": book,
                        "market": market,
                        "commence_time": commence_time,
                        "team": outc.get("name"),
                        "price_american": outc.get("price"),
                        "point": outc.get("point"),
                        "date": date or "",
                        "kind": "game"
                    })
    return pd.DataFrame(rows)

def _flatten_player_market(market: str, events: List[Dict[str, Any]], date: Optional[str]) -> pd.DataFrame:
    rows = []
    for ev in events:
        event_id = ev.get("id")
        commence_time = ev.get("commence_time")
        home = ev.get("home_team")
        away = ev.get("away_team")
        for bm in ev.get("bookmakers", []):
            book = bm.get("key")
            for mk in bm.get("markets", []):
                if mk.get("key") != market:
                    continue
                for outc in mk.get("outcomes", []):
                    # The Odds API encodes player name in 'description' or 'name' depending on market
                    player = outc.get("description") or outc.get("name")
                    # over/under vs yes/no
                    ou = (outc.get("name") or "").lower()
                    side = "over" if "over" in ou else ("under" if "under" in ou else (ou if ou in ("yes","no") else "other"))
                    rows.append({
                        "event_id": event_id,
                        "home_team": home,
                        "away_team": away,
                        "book": book,
                        "market": market,
                        "player": player,
                        "side": side,
                        "line": outc.get("point"),
                        "price_american": outc.get("price"),
                        "commence_time": commence_time,
                        "date": date or "",
                        "kind": "player"
                    })
    return pd.DataFrame(rows)

def _fetch_game_lines(date: Optional[str], books: List[str]) -> pd.DataFrame:
    params = {
        "regions": "us",
        "oddsFormat": "american",
        "bookmakers": ",".join(books),
        "apiKey": ODDS_API_KEY,
        "dateFormat": "iso",
        # markets=h2h,spreads,totals
        "markets": "h2h,spreads,totals",
    }
    if date:
        params["commenceTimeFrom"] = f"{date}"
    url = f"{BASE}/sports/{SPORT}/odds"
    r = _req(url, params)
    if r.status_code != 200:
        print(f"[oddsapi] game lines error: {r.status_code} {r.text[:300]}")
        return pd.DataFrame()
    data = r.json()
    return _flatten_game_lines(data, date)

def _fetch_player_market(market: str, date: Optional[str], books: List[str]) -> pd.DataFrame:
    params = {
        "regions": "us",
        "oddsFormat": "american",
        "bookmakers": ",".join(books),
        "apiKey": ODDS_API_KEY,
        "dateFormat": "iso",
        "markets": market,        # one market per request
    }
    if date:
        params["commenceTimeFrom"] = f"{date}"
    url = f"{BASE}/sports/{SPORT}/odds"
    r = _req(url, params)
    if r.status_code != 200:
        print(f"[oddsapi] {market} error: {r.status_code} {r.text[:300]}")
        return pd.DataFrame()
    data = r.json()
    return _flatten_player_market(market, data, date)

def _maybe_fanduel_fallback(market: str, date: Optional[str]) -> pd.DataFrame:
    # Optional offline fallback if you keep CSV dumps under external/fanduel/
    if not date:
        return pd.DataFrame()
    fp = Path(f"external/fanduel/{market}_{date}.csv")
    if fp.exists():
        try:
            df = pd.read_csv(fp)
            print(f"[oddsapi] fallback: loaded {fp} rows={len(df)}")
            df["kind"] = df.get("kind", "player")
            return df
        except Exception as e:
            print(f"[oddsapi] fallback read error {fp}: {e}")
    return pd.DataFrame()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", default=",".join(DEFAULT_BOOKS))
    ap.add_argument("--markets", default="")   # optional override; else DEFAULT_MARKETS
    ap.add_argument("--date", default="")      # YYYY-MM-DD
    ap.add_argument("--out", default="outputs/props_raw.csv")
    args = ap.parse_args()

    if not ODDS_API_KEY:
        print("[oddsapi] ERROR: ODDS_API_KEY is not set")
        sys.exit(1)

    Path("outputs").mkdir(parents=True, exist_ok=True)

    books = [b.strip() for b in args.books.split(",") if b.strip()]
    markets = [m.strip() for m in args.markets.split(",") if m.strip()] or DEFAULT_MARKETS
    date = args.date or None

    # 1) game lines once
    game_df = _fetch_game_lines(date, books)
    game_out = "outputs/odds_game.csv"
    if not game_df.empty:
        game_df.to_csv(game_out, index=False)
        print(f"[oddsapi] wrote game odds → {game_out} rows={len(game_df)}")
    else:
        print("[oddsapi] WARNING: no game lines pulled")

    # 2) loop markets individually and merge
    frames = []
    for mkt in markets:
        print(f"[oddsapi] fetching player market={mkt}")
        df = _fetch_player_market(mkt, date, books)
        if df.empty:
            # try simple per-book fallback (sometimes a single book is valid)
            if len(books) > 1:
                for b in books:
                    df = _fetch_player_market(mkt, date, [b])
                    if not df.empty:
                        break
            # try FanDuel offline CSV fallback
            if df.empty:
                df = _maybe_fanduel_fallback(mkt, date)

        if df.empty:
            print(f"[oddsapi] no rows for market={mkt} (skipped)")
            continue
        frames.append(df)
        time.sleep(0.35)  # small spacing to be nice to API

    props = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if props.empty:
        print("[oddsapi] no supported player markets detected; writing empty CSV")
        props.to_csv(args.out, index=False)
        sys.exit(0)

    # normalize basic fields
    base_cols = [
        "event_id","commence_time","date","kind",
        "home_team","away_team","book","market",
        "player","side","line","price_american",
    ]
    for c in base_cols:
        if c not in props.columns:
            props[c] = None

    props = props[base_cols]
    props.to_csv(args.out, index=False)
    print(f"[oddsapi] wrote props → {args.out} rows={len(props)}")

if __name__ == "__main__":
    main()
