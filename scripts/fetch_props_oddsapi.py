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

# v4 market alias map (keeps older names working)
MARKET_ALIASES = {
    # receiving yards synonyms
    "player_rec_yds": "player_reception_yds",
    "player_receiving_yards": "player_reception_yds",
    "player_receiving_yds": "player_reception_yds",
    "player_reception_yds": "player_reception_yds",
    # rush+rec
    "player_rush_rec_yds": "player_rush_reception_yds",
    "rush_rec": "player_rush_reception_yds",
}
def _normalize_market(m: str) -> str:
    return MARKET_ALIASES.get(m.strip(), m.strip())

# some v4 NFL player markets 422 on the per-event endpoint; fetch them at sport-level
BULK_ONLY_MARKETS = {
    "player_reception_yds",
}

# ------------------------- LOGGING ------------------------

log = logging.getLogger("oddsapi")
log.setLevel(logging.INFO)
h = logging.StreamHandler(sys.stdout)
h.setFormatter(logging.Formatter("[oddsapi] %(message)s"))
log.addHandler(h)

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
                    if market_key == "player_anytime_td":
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
    key = ["event_id", "commence_time", "book", "market", "player", "line"]
    over  = (df[df["side"] == "OVER"]
             .groupby(key, as_index=False)["price_american"]
             .first().rename(columns={"price_american": "over_odds"}))
    under = (df[df["side"] == "UNDER"]
             .groupby(key, as_index=False)["price_american"]
             .first().rename(columns={"price_american": "under_odds"}))
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
                (bm.get("title") or "").strip().lower().replace(" ", "_") in books
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

def _fetch_market_for_events(api_key: str, region: str, books: set[str],
                             event_ids: list[str], market_key: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for eid in event_ids:
        url = f"{BASE}/sports/{SPORT}/events/{eid}/odds"
        params = {"apiKey": api_key, "regions": region, "markets": market_key, "oddsFormat": "american"}
        if books:
            params["bookmakers"] = ",".join(sorted(books))
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

# sport-level bulk fetch for markets that 422 on per-event endpoint (e.g., receiving yards)
def _fetch_bulk_market(api_key: str, region: str, books: set[str], market_key: str) -> pd.DataFrame:
    url = f"{BASE}/sports/{SPORT}/odds"
    params = {"apiKey": api_key, "regions": region, "markets": market_key, "oddsFormat": "american"}
    if books:
        params["bookmakers"] = ",".join(sorted(books))
    status, js, headers = _get(url, params)
    log.info(f"bulk {market_key} status={status} limit={_lim(headers)}")
    if status != 200 or not isinstance(js, list):
        log.info(f"bulk fetch failed market={market_key}: {js}")
        return pd.DataFrame()
    return _normalize_player_rows(js, books, market_key)

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
    markets = [m.strip() for m in markets if m.strip()]

    # Resolve aliases & drop any non-player markets if they slipped in
    normalized_markets: List[str] = []
    for mk in markets:
        mk = _normalize_market(mk)
        if mk in GAME_MARKETS:
            log.info(f"skip non-player market in --markets: {mk}")
            continue
        resolved = MARKET_ALIASES.get(mk, mk)
        if resolved != mk:
            log.info(f"alias: {mk} → {resolved}")
        normalized_markets.append(resolved)
    markets = normalized_markets

    Path("outputs").mkdir(parents=True, exist_ok=True)

    # events + game odds
    events = _fetch_events_by_h2h(api_key, region, books_set)
    event_ids = [e.get("id") for e in events if e.get("id")]

    game_df = _fetch_game_odds(api_key, region, books_set)
    if not game_df.empty:
        game_df.to_csv(out_game, index=False)
        log.info(f"wrote {out_game} rows={len(game_df)}")
    else:
        pd.DataFrame().to_csv(out_game, index=False)
        log.info(f"wrote empty {out_game}")

    # per-market player props
    frames: List[pd.DataFrame] = []
    for mk in markets:
        mk = _normalize_market(mk)
        if not event_ids and mk not in BULK_ONLY_MARKETS:
            break
        log.info(f"=== MARKET {mk} ===")
        if mk in BULK_ONLY_MARKETS:
            df = _fetch_bulk_market(api_key, region, books_set, mk)
        else:
            df = _fetch_market_for_events(api_key, region, books_set, event_ids, mk)
        if not df.empty:
            frames.append(df)

    if not frames:
        pd.DataFrame(columns=[
            "event_id","commence_time","book","market","player","side","line","price_american"
        ]).to_csv(out, index=False)
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

_MARKET_ALIASES = {
    "player_pass_yds": [
        "player_passing_yards", "player_passing_yds", "passing_yards",
    ],
    "player_rush_yds": [
        "player_rushing_yards", "player_rushing_yds", "rushing_yards",
    ],
    "player_rec_yds": [
        # receiving yards: we accept several v4/v3/book variants
        "player_receiving_yards", "player_receiving_yds",
        "player_reception_yds", "receiving_yards",
    ],
    "player_receptions": [
        "player_rec", "receptions",
    ],
    "player_rush_rec_yds": [
        "player_rush_and_receive_yards", "player_rush_and_receive_yds",
        "rushing_plus_receiving_yards", "rush_rec_yards",
    ],
    "player_anytime_td": [
        "player_anytime_td", "anytime_td", "player_anytime_touchdown",
    ],
    # team/game markets (we still pass them through to the API unchanged)
    "h2h": ["h2h","moneyline","ml"],
    "spreads": ["spread","spreads"],
    "totals": ["total","totals","game_totals"],
}

def _normalize_market(m: str) -> str:
    """
    Map a user/book/old-version market key to our canonical key.
    If it doesn't match any alias, return the lowercased input.
    """
    key = (m or "").strip().lower()
    if not key:
        return key
    for canon, variants in _MARKET_ALIASES.items():
        if key == canon or key in variants:
            return canon
    return key

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", default="draftkings,fanduel,betmgm,caesars")
    ap.add_argument("--markets", default="player_pass_yds,player_rec_yds,player_rush_yds,player_receptions,player_anytime_td")
    ap.add_argument("--region", default=REGION_DEFAULT)
    ap.add_argument("--date", nargs="?", default="", const="")  # accept --date with or without a value
    ap.add_argument("--out", default="outputs/props_raw.csv")
    ap.add_argument("--out_game", default="outputs/odds_game.csv")
    args = ap.parse_args()

    fetch_odds(
        books=[b.strip() for b in args.books.split(",") if b.strip()],
        markets=[_normalize_market(m) for m in args.markets.split(',') if m.strip()],
        region=args.region,
        date=args.date,
        out=args.out,
        out_game=args.out_game,
    )
