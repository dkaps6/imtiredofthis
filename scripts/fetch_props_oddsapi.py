# scripts/fetch_props_oddsapi.py
import argparse
import os
import sys
import time
import datetime as dt
from pathlib import Path
from typing import List, Dict, Any, Tuple

import requests
import pandas as pd

BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"

def _iso_bounds_for_date(d: str) -> Dict[str, str]:
    day = dt.date.fromisoformat(d)
    start = dt.datetime.combine(day, dt.time(0, 0))
    end = start + dt.timedelta(days=1)
    return {
        "commenceTimeFrom": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commenceTimeTo": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dateFormat": "iso",
    }

def _normalize_event_rows(payload: List[Dict[str, Any]], market_key: str, allowed_books: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for ev in payload or []:
        event_id = ev.get("id")
        commence_time = ev.get("commence_time")
        for bk in (ev.get("bookmakers") or []):
            book_key = bk.get("key")
            if allowed_books and book_key not in allowed_books:
                continue
            for mk in (bk.get("markets") or []):
                if mk.get("key") != market_key:
                    continue
                for oc in (mk.get("outcomes") or []):
                    rows.append({
                        "event_id": ev.get("id"),
                        "commence_time": commence_time,
                        "book": book_key,
                        "market": market_key,
                        "player": (oc.get("name") or ""),
                        "line": oc.get("point"),
                        "side": oc.get("description") or oc.get("name") or "",
                        "odds": oc.get("price"),
                    })
    return rows

def fetch_market(api_key: str, market: str, books: List[str], date_str: str = "", region: str = "us", odds_format: str = "american", pause_sec: float = 0.35) -> Tuple[pd.DataFrame, int]:
    params = {
        "apiKey": api_key,
        "regions": region,
        "oddsFormat": odds_format,
        "bookmakers": ",".join(books) if books else "",
        "markets": market,
    }
    if date_str:
        params.update(_iso_bounds_for_date(date_str))

    url = BASE
    print(f"[oddsapi] request market={market} -> {url}")
    r = requests.get(url, params=params, timeout=20)

    # Rate-limit headers
    xr = r.headers.get("x-requests-remaining")
    xu = r.headers.get("x-requests-used")
    if xr is not None and xu is not None:
        print(f"[oddsapi] x-requests-remaining: {xr}")
        print(f"[oddsapi] x-requests-used: {xu}")

    status = r.status_code
    if status == 200:
        data = r.json() if r.text else []
        rows = _normalize_event_rows(data, market, books)
        df = pd.DataFrame(rows)
        print(f"[oddsapi] market={market} rows={len(df)}")
        time.sleep(pause_sec)
        return df, status

    print(f"[oddsapi] error market={market}: {status} {r.text[:220]}")
    time.sleep(pause_sec)
    # return empty df with same schema; also return status so caller knows if it was 401
    cols = ["event_id","commence_time","book","market","player","line","side","odds"]
    return pd.DataFrame(columns=cols), status

# ---------- FanDuel fallback (CSV-first; optional JSON later) ----------
def fallback_fanduel_csv(market: str, date_str: str, fd_dir: str) -> pd.DataFrame:
    """
    Looks for external/fanduel/props_{market}_{YYYY-MM-DD}.csv and normalizes to:
    event_id,commence_time,book,market,player,line,side,odds
    Expected columns if you provide your own CSV:
      event_id, commence_time, player, line, side, odds
    'book' is set to 'fanduel'; 'market' uses the requested market key.
    """
    if not fd_dir:
        return pd.DataFrame()
    Path(fd_dir).mkdir(parents=True, exist_ok=True)
    fname = f"props_{market}_{date_str}.csv" if date_str else f"props_{market}.csv"
    path = Path(fd_dir) / fname
    if not path.exists():
        print(f"[fd-fallback] no file at {path}")
        return pd.DataFrame()

    try:
        raw = pd.read_csv(path)
    except Exception as e:
        print(f"[fd-fallback] failed reading {path}: {e}")
        return pd.DataFrame()

    # best-effort normalization
    out = pd.DataFrame({
        "event_id": raw.get("event_id"),
        "commence_time": raw.get("commence_time"),
        "book": "fanduel",
        "market": market,
        "player": raw.get("player", ""),
        "line": raw.get("line"),
        "side": raw.get("side", ""),
        "odds": raw.get("odds"),
    })
    out = out.dropna(subset=["odds"])
    print(f"[fd-fallback] merged {len(out)} rows from {path.name}")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", default="draftkings,fanduel,betmgm,caesars")
    ap.add_argument("--markets", default="")
    ap.add_argument("--date", default="")
    ap.add_argument("--out", default="outputs/props_raw.csv")
    ap.add_argument("--fd-fallback-dir", default="external/fanduel")  # <— new
    a = ap.parse_args()

    api_key = os.getenv("ODDS_API_KEY", "")
    if not api_key:
        print("[oddsapi] ERROR: ODDS_API_KEY is not set")
        sys.exit(2)

    books = [b.strip() for b in a.books.split(",") if b.strip()]
    if a.markets.strip():
        markets = [m.strip() for m in a.markets.split(",") if m.strip()]
    else:
        markets = ["h2h","spreads","totals","player_pass_yds","player_rush_yds","player_rec_yds","player_receptions","player_passing_tds","player_anytime_td"]

    frames = []
    for m in markets:
        df, status = fetch_market(api_key, m, books=books, date_str=a.date)
        if df.empty:
            # Soft fallback if Odds API returns 401/empty: try FanDuel CSV drop-in
            fd_df = fallback_fanduel_csv(m, a.date, a.fd_fallback_dir)
            if not fd_df.empty:
                df = fd_df
        if not df.empty:
            frames.append(df)

    props = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["event_id","commence_time","book","market","player","line","side","odds"]
    )

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    props.to_csv(a.out, index=False)
    print(f"[oddsapi] wrote props={len(props)} -> {a.out}")

    game_df = props[props["market"].isin(["h2h","spreads","totals"])]
    game_out = "outputs/odds_game.csv"
    game_df.to_csv(game_out, index=False)
    print(f"[oddsapi] wrote game-only odds -> {game_out}")

if __name__ == "__main__":
    main()
