#!/usr/bin/env python3
from __future__ import annotations

import os, sys, time, argparse, logging
from pathlib import Path
from typing import List, Tuple, Optional, Any, Dict

import requests
import pandas as pd

# ------------------------- CONFIG -------------------------

SPORT = "americanfootball_nfl"
BASE  = "https://api.the-odds-api.com/v4"
REGION_DEFAULT = "us"
TIMEOUT_S = 25
BACKOFF_S = [0.6, 1.2, 2.0, 3.5, 5.0]
GAME_MARKETS = ["h2h", "spreads", "totals"]  # bulk-only

# US bookmaker keys (trim list you rely on; add more once verified on docs)
US_BOOK_KEYS = {
    "draftkings", "fanduel", "betmgm", "caesars",
}

# Canonicalize your aliases → vendor short keys for per-event props
MARKET_ALIASES: Dict[str, str] = {
    # passing yards
    "player_passing_yards": "player_pass_yds",
    "player_passing_yds":   "player_pass_yds",
    "player_pass_yds":      "player_pass_yds",
    "passing_yards":        "player_pass_yds",

    # receiving yards
    "player_receiving_yards": "player_reception_yds",
    "player_receiving_yds":   "player_reception_yds",
    "player_reception_yds":   "player_reception_yds",
    "player_rec_yds":         "player_reception_yds",
    "receiving_yards":        "player_reception_yds",

    # rushing yards
    "player_rushing_yards": "player_rush_yds",
    "player_rushing_yds":   "player_rush_yds",
    "player_rush_yds":      "player_rush_yds",
    "rushing_yards":        "player_rush_yds",

    # rush + rec
    "player_rush_and_receive_yards": "player_rush_reception_yds",
    "player_rush_and_receive_yds":   "player_rush_reception_yds",
    "player_rush_reception_yds":     "player_rush_reception_yds",
    "player_rush_rec_yds":           "player_rush_reception_yds",
    "rush_rec_yards":                "player_rush_reception_yds",

    # receptions
    "player_receptions": "player_receptions",
    "receptions":        "player_receptions",

    # anytime TD
    "player_anytime_td":        "player_anytime_td",
    "anytime_td":               "player_anytime_td",
    "player_anytime_touchdown": "player_anytime_td",

    # pass-through game markets
    "h2h": "h2h", "moneyline": "h2h", "ml": "h2h",
    "spreads": "spreads", "spread": "spreads",
    "totals": "totals", "total": "totals", "game_totals": "totals",
}

def _normalize_market(m: str) -> str:
    key = (m or "").strip().lower()
    return MARKET_ALIASES.get(key, key)

# Player props → per-event endpoint; game markets → bulk
BULK_ONLY_CANONICAL: set[str] = set(GAME_MARKETS)

# ------------------------- LOGGING ------------------------

log = logging.getLogger("oddsapi")
log.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("[oddsapi] %(message)s"))
log.addHandler(_handler)

# ------------------------- HTTP --------------------------

def _try_json(r: requests.Response):
    try:
        return r.json()
    except Exception:
        return {"text": r.text[:500]}

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
            book_key = (bm.get("key") or "").strip().lower()
            book_title = (bm.get("title") or "").strip()
            if books_filter and book_key not in books_filter:
                continue
            for mk in bm.get("markets", []):
                market = mk.get("key")
                for oc in mk.get("outcomes", []):
                    rows.append({
                        "event_id": eid,
                        "commence_time": commence,
                        "home_team": home,
                        "away_team": away,
                        "book": book_key,          # use key for joins/filters
                        "book_title": book_title,  # keep for readability
                        "market": market,
                        "name": oc.get("name"),
                        "price_american": oc.get("price"),
                        "point": oc.get("point"),
                    })
    return pd.DataFrame.from_records(rows)

def _normalize_player_rows(events: list, books_filter: set[str], market_key: str) -> pd.DataFrame:
    recs = []
    for ev in events or []:
        eid = ev.get("id")
        commence = ev.get("commence_time")
        for bm in ev.get("bookmakers", []):
            book_key = (bm.get("key") or "").strip().lower()
            book_title = (bm.get("title") or "").strip()
            if books_filter and book_key not in books_filter:
                continue
            for mk in bm.get("markets", []):
                if mk.get("key") != market_key: 
                    continue
                for oc in mk.get("outcomes", []):
                    side = (oc.get("name") or "").upper()
                    if market_key == "player_anytime_td":
                        if side == "YES": side = "OVER"
                        if side == "NO":  side = "UNDER"
                    player = oc.get("description") or oc.get("participant")
                    recs.append({
                        "event_id": eid,
                        "commence_time": commence,
                        "book": book_key,          # key
                        "book_title": book_title,  # title
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
    key = ["event_id", "commence_time", "book", "market", "player", "line"]
    over  = (df[df["side"] == "OVER"]
             .groupby(key, as_index=False)["price_american"]
             .first()
             .rename(columns={"price_american": "over_odds"}))
    under = (df[df["side"] == "UNDER"]
             .groupby(key, as_index=False)["price_american"]
             .first()
             .rename(columns={"price_american": "under_odds"}))
    return over.merge(under, on=key, how="outer")

# ------------------------- CORE FETCHERS ------------------

def _fetch_events_by_h2h(api_key: str, region: str, books: set[str]) -> list:
    url = f"{BASE}/sports/{SPORT}/odds"
    params = {"apiKey": api_key, "regions": region, "markets": "h2h", "oddsFormat": "american"}
    if books:
        params["bookmakers"] = ",".join(sorted(books))
    status, js, headers = _get(url, params)
    log.info(f"events status={status} limit={_lim(headers)}")
    if status != 200 or not isinstance(js, list):
        log.info(f"failed to fetch events: {js}")
        return []
    if books:
        filtered = []
        for ev in js:
            keep = any(
                (bm.get("key") or "").strip().lower() in books
                for bm in ev.get("bookmakers", [])
            )
            if keep:
                filtered.append(ev)
        return filtered
    
    return js

def _fetch_game_odds(api_key: str, region: str, books: set[str]) -> pd.DataFrame:
    url = f"{BASE}/sports/{SPORT}/odds"
    params = {"apiKey": api_key, "regions": region, "markets": ",".join(GAME_MARKETS), "oddsFormat": "american"}
    if books:
        params["bookmakers"] = ",".join(sorted(books))
    status, js, headers = _get(url, params)
    log.info(f"game-odds status={status} limit={_lim(headers)}")
    if status != 200 or not isinstance(js, list):
        log.info(f"failed to fetch game odds: {js}")
        return pd.DataFrame()
    return _normalize_game_rows(js, books)

# Try alternate keys if 422; these are small synonym lists
ALT_KEYS = {
    "player_reception_yds": ["player_receiving_yds", "player_receiving_yards"],
    "player_pass_yds":      ["player_passing_yds", "player_passing_yards"],
    "player_rush_yds":      ["player_rushing_yds", "player_rushing_yards"],
    "player_rush_reception_yds": ["player_rush_and_receive_yards", "player_rush_and_receive_yds"],
    "player_receptions":    ["receptions"],
    "player_anytime_td":    ["anytime_td"],
}

def _fetch_market_for_events(api_key: str, region: str, books: set[str],
                             event_ids: list[str], market_key: str) -> pd.DataFrame:
    mk = _normalize_market(market_key)
    frames: List[pd.DataFrame] = []

    tried = [mk] + ALT_KEYS.get(mk, [])
    for mk_try in tried:
        got_any = False
        for eid in event_ids:
            url = f"{BASE}/sports/{SPORT}/events/{eid}/odds"
            params = {"apiKey": api_key, "regions": region, "markets": mk_try, "oddsFormat": "american"}
            if books:
                params["bookmakers"] = ",".join(sorted(books))
            status, js, headers = _get(url, params)
            log.info(f"{mk_try} eid={eid} status={status} limit={_lim(headers)}")

            if status == 200 and isinstance(js, dict):
                bm_count = len(js.get("bookmakers", []))
                if bm_count == 0 and books:
                    log.info(f"{mk_try} eid={eid}: 0 bookmakers after filter → retry w/o filter")
                    params2 = dict(params)
                    params2.pop("bookmakers", None)
                    status2, js2, _ = _get(url, params2)
                    if status2 == 200 and isinstance(js2, dict):
                        df = _normalize_player_rows([js2], set(), mk_try)
                    else:
                        df = pd.DataFrame()
                else:
                    df = _normalize_player_rows([js], books, mk_try)
                if not df.empty:
                    frames.append(df)
                    got_any = True

            elif status == 422:
                # try next synonym
                continue
            elif status in (401, 403, 404):
                log.info(f"skip market={mk_try} for eid={eid}: {js}")
            else:
                log.info(f"market fetch error eid={eid} market={mk_try}: {js}")

        if got_any:
            break  # stop at first synonym that yields data

    # Dual-region fallback: if region=us yields nothing, try us2 once
    if not frames and region == "us":
        log.info(f"{mk} → no data from region=us; retrying region=us2 for same events...")
        frames_us2: List[pd.DataFrame] = []
        for eid in event_ids:
            url = f"{BASE}/sports/{SPORT}/events/{eid}/odds"
            params = {"apiKey": api_key, "regions": "us2", "markets": mk, "oddsFormat": "american"}
            if books:
                params["bookmakers"] = ",".join(sorted(books))
            status, js, headers = _get(url, params)
            log.info(f"{mk} eid={eid} (us2) status={status} limit={_lim(headers)}")
            if status == 200 and isinstance(js, dict):
                df = _normalize_player_rows([js], books, mk)
                if df.empty and books:
                    # try without filters in us2 too
                    params2 = dict(params)
                    params2.pop("bookmakers", None)
                    status2, js2, _ = _get(url, params2)
                    if status2 == 200 and isinstance(js2, dict):
                        df = _normalize_player_rows([js2], set(), mk)
                if not df.empty:
                    frames_us2.append(df)
        if frames_us2:
            merged = pd.concat(frames_us2, ignore_index=True)
            frames.append(merged)
            log.info(f"{mk}: merged {len(merged)} rows from region=us2 fallback.")
        else:
            log.info(f"{mk}: region=us2 also empty.")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def _fetch_bulk_market(api_key: str, region: str, books: set[str], market_key: str) -> pd.DataFrame:
    # bulk for game markets only
    mk = _normalize_market(market_key)
    url = f"{BASE}/sports/{SPORT}/odds"
    params = {"apiKey": api_key, "regions": region, "markets": mk, "oddsFormat": "american"}
    if books:
        params["bookmakers"] = ",".join(sorted(books))
    status, js, headers = _get(url, params)
    log.info(f"bulk {mk} status={status} limit={_lim(headers)}")
    if status != 200 or not isinstance(js, list):
        log.info(f"bulk fetch failed market={mk}: {js}")
        return pd.DataFrame()
    return _normalize_player_rows(js, books, mk)

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
    api_key = os.getenv("ODDS_API_KEY", "").strip()
    if not api_key:
        log.info("ERROR: ODDS_API_KEY not set")
        sys.exit(2)

    books_set = {b.strip().lower().replace(" ", "_") for b in books if b.strip()}
    # Validate US bookmaker keys
    if region == "us" and books_set:
        bad = [b for b in books_set if b not in US_BOOK_KEYS]
        if bad:
            log.info(f"unknown/retired bookmaker key(s) for region=us: {bad} → removing from filter")
            books_set = {b for b in books_set if b in US_BOOK_KEYS}

    markets = [m.strip() for m in markets if m.strip()]
    normalized_markets: List[str] = [_normalize_market(m) for m in markets]

    Path("outputs").mkdir(parents=True, exist_ok=True)

    # events for event-odds
    events = _fetch_events_by_h2h(api_key, region, books_set)
    event_ids = [e.get("id") for e in events if e.get("id")]

    # bulk game odds snapshot
    game_df = _fetch_game_odds(api_key, region, books_set)
    if not game_df.empty:
        game_df.to_csv(out_game, index=False)
        log.info(f"wrote {out_game} rows={len(game_df)}")
    else:
        pd.DataFrame().to_csv(out_game, index=False)
        log.info(f"wrote empty {out_game}")

    # per-market fetch
    frames: List[pd.DataFrame] = []
    for mk in normalized_markets:
        if mk in GAME_MARKETS:
            log.info(f"=== MARKET {mk} (bulk) ===")
            df = _fetch_bulk_market(api_key, region, books_set, mk)
        else:
            if not event_ids:
                log.info(f"no events → skip player market {mk}")
                continue
            log.info(f"=== MARKET {mk} (per-event) ===")
            df = _fetch_market_for_events(api_key, region, books_set, event_ids, mk)
        if not df.empty:
            frames.append(df)

    if not frames:
        # leave empty—your downstream wants to crash if props are empty
        pd.DataFrame(columns=["event_id","commence_time","book","market","player","side","line","price_american"]).to_csv(out, index=False)
        log.info(f"wrote empty {out}")
        return

    props = pd.concat(frames, ignore_index=True)
    props.to_csv(out, index=False)
    log.info(f"wrote {out} rows={len(props)}")

    wide = _wide_over_under(props)
    wide_out = Path(out).with_name("props_raw_wide.csv")
    wide.to_csv(wide_out, index=False)
    log.info(f"wrote {wide_out} rows={len(wide)}")

# ------------------------- CLI ----------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # Accept both names; allow empty (no filter) with nargs='?' / const=''
    ap.add_argument("--books", "--bookmakers", dest="books",
                    default="draftkings,fanduel,betmgm,caesars",
                    nargs="?", const="")
    ap.add_argument("--markets", default="player_pass_yds,player_reception_yds,player_rush_yds,player_receptions,player_rush_reception_yds,player_anytime_td")
    ap.add_argument("--region", default=REGION_DEFAULT)
    ap.add_argument("--date", nargs="?", default="", const="")
    ap.add_argument("--out", default="outputs/props_raw.csv")
    ap.add_argument("--out_game", default="outputs/odds_game.csv")
    args = ap.parse_args()

    fetch_odds(
        books=[b.strip() for b in (args.books or "").split(",") if b.strip()],
        markets=[m.strip() for m in args.markets.split(",") if m.strip()],
        region=args.region,
        date=args.date,
        out=args.out,
        out_game=args.out_game,
    )
