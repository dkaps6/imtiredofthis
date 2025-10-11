#!/usr/bin/env python3
from __future__ import annotations

import os, sys, time, json, argparse, logging
from pathlib import Path
from typing import List, Tuple, Optional, Any, Dict

import requests
import pandas as pd

# ------------------------- CONFIG -------------------------

SPORT = "americanfootball_nfl"          # The Odds API sport key (v4)
BASE  = "https://api.the-odds-api.com/v4"
REGION_DEFAULT = "us"
TIMEOUT_S = 25
BACKOFF_S = [0.6, 1.2, 2.0, 3.5, 5.0]    # simple backoff on 429/5xx
GAME_MARKETS = ["h2h", "spreads", "totals"]

# ADDED: alias map for renamed markets in v4
MARKET_ALIASES = {
    # old_name              : new_name
    "player_rec_yds": "player_receiving_yards",
}

# ------------------------- LOGGING ------------------------

log = logging.getLogger("oddsapi")
log.setLevel(logging.INFO)
h = logging.StreamHandler(sys.stdout)
h.setFormatter(logging.Formatter("[oddsapi] %(message)s"))
log.addHandler(h)

# ------------------------- HTTP --------------------------

def _get(url: str, params: dict, max_retries: int = 5) -> Tuple[int, Optional[Any], dict]:
    for i in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT_S)
            if r.status_code == 200:
                try:
                    return 200, r.json(), r.headers
                except Exception:
                    return 200, None, r.headers
            if r.status_code in (401, 403, 404, 422):
                return r.status_code, _try_json(r), r.headers
            if r.status_code in (429, 500, 502, 503, 504):
                wait = BACKOFF_S[min(i, len(BACKOFF_S)-1)]
                log.info(f"HTTP {r.status_code} → backoff {wait}s: {url}")
                time.sleep(wait)
                continue
            return r.status_code, _try_json(r), r.headers
        except requests.RequestException as e:
            wait = BACKOFF_S[min(i, len(BACKOFF_S)-1)]
            log.info(f"Request error: {e} → retry {wait}s")
            time.sleep(wait)
    return 0, None, {}

def _try_json(r: requests.Response):
    try:
        return r.json()
    except Exception:
        return {"text": r.text[:500]}

def _lim(headers: dict) -> dict:
    out = {}
    for k in ("x-requests-remaining", "x-requests-used", "x-requests-apikey-remaining"):
        if k in headers:
            out[k] = headers.get(k)
    return out

# ------------------------- NORMALIZERS --------------------

def _normalize_game_rows(events: list, books_filter: set[str]) -> pd.DataFrame:
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
    recs: List[Dict[str, Any]] = []
    for ev in events or []:
        eid = ev.get("id")
        commence = ev.get("commence_time")
        for bm in ev.get("bookmakers", []):
            book = (bm.get("title") or "").strip().lower().replace(" ", "_")
            if books_filter and book not in books_filter:
                continue
            for mk in bm.get("markets", []):
                if mk.get("key") != market_key:
                    continue
                for oc in mk.get("outcomes", []):
                    side = (oc.get("name") or "").upper()    # "OVER"/"UNDER" or "YES"/"NO"
                    if market_key == "player_anytime_td":     # normalize YES/NO to OVER/UNDER
                        if side == "YES": side = "OVER"
                        if side == "NO":  side = "UNDER"
                    player = oc.get("description") or oc.get("participant")
                    recs.append({
                        "event_id": eid,
                        "commence_time": commence,
                        "book": book,
                        "market": market_key,
                        "player": player,
                        "side": side,
                        "line": oc.get("point"),
                        "price_american": oc.get("price"),
                    })
    df = pd.DataFrame.from_records(recs)
    if not df.empty:
        df["line"] = pd.to_numeric(df["line"], errors="coerce")
        df["price_american"] = pd.to_numeric(df["price_american"], errors="coerce")
    return df

def _wide_over_under(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: 
        return df
    key = ["event_id","commence_time","book","market","player","line"]
    over  = df[df["side"]=="OVER"].groupby(key, as_index=False)["price_american"].first().rename(columns={"price_american":"over_odds"})
    under = df[df["side"]=="UNDER"].groupby(key, as_index=False)["price_american"].first().rename(columns={"price_american":"under_odds"})
    return over.merge(under, on=key, how="outer")

# ------------------------- CORE FETCHERS ------------------

def _fetch_events_by_h2h(api_key: str, region: str, books: set[str]) -> list:
    url = f"{BASE}/sports/{SPORT}/odds"
    params = {
        "apiKey": api_key,
        "regions": region,
        "markets": "h2h",
        "oddsFormat": "american",
    }
    status, js, headers = _get(url, params)
    log.info(f"events status={status} limit={_lim(headers)}")
    if status != 200 or not isinstance(js, list):
        log.info(f"failed to fetch events: {js}")
        return []
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

def _fetch_game_odds(api_key: str, region: str, books: set[str]) -> pd.DataFrame:
    url = f"{BASE}/sports/{SPORT}/odds"
    params = {
        "apiKey": api_key,
        "regions": region,
        "markets": ",".join(GAME_MARKETS),
        "oddsFormat": "american",
    }
    status, js, headers = _get(url, params)
    log.info(f"game-odds status={status} limit={_lim(headers)}")
    if status != 200 or not isinstance(js, list):
        log.info(f"failed to fetch game odds: {js}")
        return pd.DataFrame()
    return _normalize_game_rows(js, books)

def _fetch_market_for_events(api_key: str, region: str, books: set[str], event_ids: list[str], market_key: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for eid in event_ids:
        url = f"{BASE}/sports/{SPORT}/events/{eid}/odds"
        params = {
            "apiKey": api_key,
            "regions": region,
            "markets": market_key,
            "oddsFormat": "american",
        }
        status, js, headers = _get(url, params)
        log.info(f"{market_key} eid={eid} status={status} limit={_lim(headers)}")
        if status == 200 and isinstance(js, dict):
            df = _normalize_player_rows([js], books, market_key)
            if not df.empty:
                frames.append(df)
        elif status in (401, 403, 422):
            log.info(f"skip market={market_key} for eid={eid}: {js}")
        else:
            log.info(f"market fetch error eid={eid} market={market_key}: {js}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

# ------------------------- PUBLIC ENTRY -------------------

def fetch_odds(
    *,
    books: List[str],
    markets: List[str],
    region: str = REGION_DEFAULT,
    date: str = "",
    out: str = "outputs/props_raw.csv",
    out_game: str = "outputs/odds_game.csv",
) -> None:
    """
    Public entry used by engine. Pulls game odds once, then per-market player props.
    """
    api_key = os.getenv("ODDS_API_KEY", "").strip()
    if not api_key:
        log.info("ERROR: ODDS_API_KEY not set")
        sys.exit(2)

    books_set = {b.strip().lower().replace(" ", "_") for b in books if b.strip()}
    markets = [m.strip() for m in markets if m.strip()]

    Path("outputs").mkdir(parents=True, exist_ok=True)

    # 1) events + game odds
    events = _fetch_events_by_h2h(api_key, region, books_set)
    event_ids = [e.get("id") for e in events if e.get("id")]
    game_df = _fetch_game_odds(api_key, region, books_set)
    if not game_df.empty:
        game_df.to_csv(out_game, index=False)
        log.info(f"wrote {out_game} rows={len(game_df)}")
    else:
        pd.DataFrame().to_csv(out_game, index=False)
        log.info(f"wrote empty {out_game}")

    # ADDED: normalize/alias markets and drop any game markets that slipped in
    normalized_markets: List[str] = []
    for mk in markets:
        if mk in GAME_MARKETS:
            log.info(f"skip non-player market in --markets: {mk}")
            continue
        resolved = MARKET_ALIASES.get(mk, mk)
        if resolved != mk:
            log.info(f"alias: {mk} → {resolved}")
        normalized_markets.append(resolved)
    markets = normalized_markets

    # 2) per-market player props
    frames: List[pd.DataFrame] = []
    for mk in markets:
        if not event_ids:
            break
        log.info(f"=== MARKET {mk} ===")
        df = _fetch_market_for_events(api_key, region, books_set, event_ids, mk)
        if not df.empty:
            frames.append(df)

    if not frames:
        # write schema-correct empty
        pd.DataFrame(columns=["event_id","commence_time","book","market","player","side","line","price_american"]).to_csv(out, index=False)
        log.info(f"wrote empty {out}")
        return

    props = pd.concat(frames, ignore_index=True)
    props.to_csv(out, index=False)
    log.info(f"wrote {out} rows={len(props)}")

    # Optional: wide debug snapshot
    wide = _wide_over_under(props)
    wide_out = Path(out).with_name("props_raw_wide.csv")
    wide.to_csv(wide_out, index=False)
    log.info(f"wrote {wide_out} rows={len(wide)}")

# ------------------------- CLI ----------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", default="draftkings,fanduel,betmgm,caesars")
    ap.add_argument("--markets", default="player_pass_yds,player_rec_yds,player_rush_yds,player_receptions,player_anytime_td")
    ap.add_argument("--region", default=REGION_DEFAULT)
    ap.add_argument("--date", default="")  # accepted for engine compatibility
    ap.add_argument("--out", default="outputs/props_raw.csv")
    ap.add_argument("--out_game", default="outputs/odds_game.csv")
    args = ap.parse_args()

    fetch_odds(
        books=[b.strip() for b in args.books.split(",") if b.strip()],
        markets=[m.strip() for m in args.markets.split(",") if m.strip()],
        region=args.region,
        date=args.date,
        out=args.out,
        out_game=args.out_game,
    )
