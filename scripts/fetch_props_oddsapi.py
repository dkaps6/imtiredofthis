#!/usr/bin/env python3
from __future__ import annotations

import os, sys, time, json, math, argparse, logging
from pathlib import Path
from typing import Iterable, Dict, Any, List, Optional, Tuple

import requests
import pandas as pd

# ------------------------- CONFIG -------------------------

SPORT = "americanfootball_nfl"        # The Odds API sport key (v4)
BASE  = "https://api.the-odds-api.com/v4"
REGION_DEFAULT = "us"
ODDS_TIMEOUT_S = 25
BACKOFF_S = [0.5, 1.0, 2.0, 3.5, 5.0]  # backoff for 429/5xx

# Safe default market set (extend as needed)
DEFAULT_MARKETS = [
    # player props – call one market per request:
    "player_pass_yds",
    "player_rec_yds",
    "player_rush_yds",
    "player_receptions",
    # uncomment/add as your plan/books support
    # "player_anytime_td",
    # "player_rush_rec_yds",
    # "player_pass_tds",
    # "player_rush_att",
]

# Game odds we always fetch (single request)
GAME_MARKETS = ["h2h", "spreads", "totals"]

# ------------------------- LOGGING ------------------------

log = logging.getLogger("fetch_props_oddsapi")
log.setLevel(logging.INFO)
h = logging.StreamHandler(sys.stdout)
h.setFormatter(logging.Formatter("[oddsapi] %(message)s"))
log.addHandler(h)

# ------------------------- HTTP --------------------------

def _get(url: str, params: dict, *, max_retries=5) -> Tuple[int, Optional[Any], dict]:
    """GET with simple backoff. Returns (status, json_or_none, headers)."""
    for i in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=ODDS_TIMEOUT_S)
            if r.status_code == 200:
                try:
                    return 200, r.json(), r.headers
                except Exception:
                    return 200, None, r.headers
            if r.status_code in (401, 403, 404, 422):
                # hard failure — do not retry
                return r.status_code, _try_json(r), r.headers
            if r.status_code in (429, 500, 502, 503, 504):
                wait = BACKOFF_S[min(i, len(BACKOFF_S)-1)]
                log.info(f"HTTP {r.status_code} → backing off {wait}s (url={url})")
                time.sleep(wait)
                continue
            # other codes — return as-is
            return r.status_code, _try_json(r), r.headers
        except requests.RequestException as e:
            wait = BACKOFF_S[min(i, len(BACKOFF_S)-1)]
            log.info(f"Request error: {e} → retry in {wait}s")
            time.sleep(wait)
    return 0, None, {}

def _try_json(r: requests.Response):
    try:
        return r.json()
    except Exception:
        return {"text": r.text[:500]}

def _read_limit_headers(headers: dict) -> dict:
    out = {}
    for k in ("x-requests-remaining", "x-requests-used", "x-requests-apikey-remaining"):
        if k in headers:
            out[k] = headers.get(k)
    return out

# ------------------------- NORMALIZERS --------------------

def _normalize_game_rows(events: list, books_filter: set[str]) -> pd.DataFrame:
    """
    Flatten v4 /odds response (game odds) to rows.
    """
    rows = []
    for ev in events or []:
        eid = ev.get("id")
        commence = ev.get("commence_time")
        home = ev.get("home_team")
        away = ev.get("away_team")
        for bm in ev.get("bookmakers", []):
            book = (bm.get("title") or "").strip().lower().replace(" ", "_")
            if books_filter and book not in books_filter:
                continue
            for mk in bm.get("markets", []):
                market = mk.get("key")
                for oc in mk.get("outcomes", []):
                    rows.append({
                        "event_id": eid,
                        "commence_time": commence,
                        "home_team": home,
                        "away_team": away,
                        "book": book,
                        "market": market,
                        "name": oc.get("name"),
                        "price_american": oc.get("price"),
                        "point": oc.get("point"),
                    })
    return pd.DataFrame.from_records(rows)

def _normalize_player_rows(events: list, books_filter: set[str], market_key: str) -> pd.DataFrame:
    """
    Flatten v4 /events/{id}/odds?markets=player_* response(s) to rows.
    Each outcome has Over/Under with a 'description' (player).
    """
    recs: List[dict] = []
    for ev in events or []:
        eid = ev.get("id")
        commence = ev.get("commence_time")
        # v4 returns bookmakers at the event level per request
        for bm in ev.get("bookmakers", []):
            book = (bm.get("title") or "").strip().lower().replace(" ", "_")
            if books_filter and book not in books_filter:
                continue
            for mk in bm.get("markets", []):
                if mk.get("key") != market_key:
                    continue
                for oc in mk.get("outcomes", []):
                    side = (oc.get("name") or "").upper()    # "OVER" / "UNDER"
                    player = oc.get("description")            # player name
                    price = oc.get("price")
                    point = oc.get("point")                   # the line
                    if not player:
                        # Some books put player name in "participant"
                        player = oc.get("participant")
                    recs.append({
                        "event_id": eid,
                        "commence_time": commence,
                        "book": book,
                        "market": market_key,
                        "player": player,
                        "side": side,
                        "line": point,
                        "price_american": price,
                    })
    df = pd.DataFrame.from_records(recs)
    # clean types
    if not df.empty:
        df["line"] = pd.to_numeric(df["line"], errors="coerce")
        df["price_american"] = pd.to_numeric(df["price_american"], errors="coerce")
    return df

# ------------------------- CORE FETCHERS ------------------

def fetch_upcoming_events(api_key: str, region: str, books: set[str]) -> list:
    """
    Fetch a list of upcoming events using a cheap game market (h2h).
    We only need the event id & commence_time.
    """
    url = f"{BASE}/sports/{SPORT}/odds"
    params = {
        "apiKey": api_key,
        "regions": region,
        "markets": "h2h",
        "oddsFormat": "american",
    }
    status, js, headers = _get(url, params)
    log.info(f"events status={status} limit={_read_limit_headers(headers)}")
    if status != 200 or not isinstance(js, list):
        log.info(f"failed to fetch events: {js}")
        return []
    # Optionally filter out events that have no bookmakers in our list
    if books:
        filtered = []
        for ev in js:
            keep = False
            for bm in ev.get("bookmakers", []):
                book = (bm.get("title") or "").strip().lower().replace(" ", "_")
                if book in books:
                    keep = True; break
            if keep:
                filtered.append(ev)
        return filtered
    return js

def fetch_game_odds(api_key: str, region: str, books: set[str]) -> pd.DataFrame:
    """Single call for H2H/spreads/totals (useful for script)."""
    url = f"{BASE}/sports/{SPORT}/odds"
    params = {
        "apiKey": api_key,
        "regions": region,
        "markets": ",".join(GAME_MARKETS),
        "oddsFormat": "american",
    }
    status, js, headers = _get(url, params)
    log.info(f"game-odds status={status} limit={_read_limit_headers(headers)}")
    if status != 200 or not isinstance(js, list):
        log.info(f"failed to fetch game odds: {js}")
        return pd.DataFrame()
    return _normalize_game_rows(js, books)

def fetch_market_for_events(api_key: str, region: str, books: set[str], event_ids: list[str], market_key: str) -> pd.DataFrame:
    """
    For each event, call /events/{id}/odds?markets=<market_key>
    """
    all_rows: List[pd.DataFrame] = []
    for eid in event_ids:
        url = f"{BASE}/sports/{SPORT}/events/{eid}/odds"
        params = {
            "apiKey": api_key,
            "regions": region,
            "markets": market_key,
            "oddsFormat": "american",
        }
        status, js, headers = _get(url, params)
        lim = _read_limit_headers(headers)
        log.info(f"{market_key} eid={eid} status={status} limit={lim}")
        if status == 200 and isinstance(js, dict):
            df = _normalize_player_rows([js], books, market_key)
            if not df.empty:
                all_rows.append(df)
        elif status in (401, 403, 422):
            # unauthorized / plan doesn't include this market – skip cleanly
            log.info(f"skip market={market_key} for eid={eid}: {js}")
        else:
            log.info(f"market fetch error eid={eid} market={market_key}: {js}")
    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)

# ------------------------- UTILITIES ----------------------

def wide_over_under(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional: convert (OVER/UNDER) rows -> wide columns (over_odds/under_odds).
    Pricing can also pivot; this just makes debugging easy.
    """
    if df.empty: return df
    key = ["event_id","commence_time","book","market","player","line"]
    over  = df[df["side"]=="OVER"].groupby(key, as_index=False)["price_american"].first().rename(columns={"price_american":"over_odds"})
    under = df[df["side"]=="UNDER"].groupby(key, as_index=False)["price_american"].first().rename(columns={"price_american":"under_odds"})
    out = over.merge(under, on=key, how="outer")
    return out

# ------------------------- CLI ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", default="draftkings,fanduel,betmgm,caesars",
                    help="Comma separated bookmaker slugs (case/space-insensitive).")
    ap.add_argument("--markets", default=",".join(DEFAULT_MARKETS),
                    help="Comma separated player markets; fetched one per request.")
    ap.add_argument("--region", default=REGION_DEFAULT)
    ap.add_argument("--out", default="outputs/props_raw.csv")
    ap.add_argument("--out_game", default="outputs/odds_game.csv")
    args = ap.parse_args()

    api_key = os.getenv("ODDS_API_KEY", "").strip()
    if not api_key:
        log.info("ERROR: ODDS_API_KEY not set in environment")
        sys.exit(2)

    books = {b.strip().lower().replace(" ", "_") for b in args.books.split(",") if b.strip()}
    markets = [m.strip() for m in args.markets.split(",") if m.strip()]

    Path("outputs").mkdir(parents=True, exist_ok=True)

    # 1) Fetch once: upcoming events + game odds block
    events = fetch_upcoming_events(api_key, args.region, books)
    event_ids = [e.get("id") for e in events if e.get("id")]
    if not event_ids:
        log.info("No upcoming events found (check date window/plans).")
    game_df = fetch_game_odds(api_key, args.region, books)
    if not game_df.empty:
        game_df.to_csv(args.out_game, index=False)
        log.info(f"wrote {args.out_game} rows={len(game_df)}")
    else:
        pd.DataFrame().to_csv(args.out_game, index=False)
        log.info(f"wrote empty {args.out_game}")

    # 2) For each market, fetch per-event odds and append
    frames: List[pd.DataFrame] = []
    for mk in markets:
        if not event_ids:
            break
        log.info(f"=== MARKET {mk} ===")
        df = fetch_market_for_events(api_key, args.region, books, event_ids, mk)
        if not df.empty:
            frames.append(df)

    if not frames:
        # write an empty but schema-correct file
        pd.DataFrame(columns=["event_id","commence_time","book","market","player","side","line","price_american"]).to_csv(args.out, index=False)
        log.info(f"wrote empty {args.out}")
        return

    props = pd.concat(frames, ignore_index=True)
    # optional wide view for convenience (kept as additional merge target if needed)
    wide = wide_over_under(props)

    # Save the long version (pricing can pivot internally)
    props.to_csv(args.out, index=False)
    log.info(f"wrote {args.out} rows={len(props)}")

    # Also drop a wide snapshot for debugging (not used by pricing directly)
    wide_out = Path(args.out).with_name("props_raw_wide.csv")
    wide.to_csv(wide_out, index=False)
    log.info(f"wrote {wide_out} rows={len(wide)}")

def fetch_odds(**kwargs):
    """Compatibility shim for engine.run_pipeline"""
    from scripts.fetch_props_oddsapi import fetch_props_oddsapi
    return fetch_props_oddsapi(**kwargs)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", default="draftkings,fanduel,betmgm,caesars")
    ap.add_argument("--markets", default="player_pass_yds")
    ap.add_argument("--region", default="us")
    ap.add_argument("--date", default="")  # ← ADD THIS LINE
    ap.add_argument("--out", default="outputs/props_raw.csv")
    ap.add_argument("--out_game", default="outputs/odds_game.csv")
    args = ap.parse_args()

    fetch_odds(
        books=args.books.split(","),
        markets=args.markets.split(","),
        region=args.region,
        out=args.out,
        out_game=args.out_game,
        date=args.date,  # ← ADD THIS LINE TOO
    )
