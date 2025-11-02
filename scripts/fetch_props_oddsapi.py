#!/usr/bin/env python3
from __future__ import annotations

import os, sys, time, argparse, logging, re
from pathlib import Path
from typing import List, Tuple, Optional, Any, Dict

import requests
import pandas as pd

from scripts.make_player_form import canonicalize_name, TEAM_NAME_TO_ABBR, _canon_team

# ------------------------- CONFIG -------------------------

SPORT = "americanfootball_nfl"
BASE  = "https://api.the-odds-api.com/v4"
REGION_DEFAULT = "us"
TIMEOUT_S = 25
BACKOFF_S = [0.6, 1.2, 2.0, 3.5, 5.0]
GAME_MARKETS = ["h2h", "spreads", "totals"]  # bulk-only

DATA_DIR = Path("data")
PROPS_ENRICHED_PATH = DATA_DIR / "props_enriched.csv"
PROPS_RAW_DATA_PATH = DATA_DIR / "props_raw.csv"
OPPONENT_MAP_PATH = DATA_DIR / "opponent_map_from_props.csv"
ROLES_PATH = Path("data/roles_ourlads.csv")

# Known book keys for US (include alias keys we might see from the API)
US_BOOK_KEYS = {
    "draftkings", "fanduel", "betmgm", "caesars",
    # aliases that can appear in API payloads
    "williamhill_us",  # Caesars
    "barstool",        # older alias sometimes used for BetRivers group
    "betrivers",
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

# Book key alias expansion: user can pass "caesars" and we'll also match "williamhill_us"
BOOK_KEY_ALIASES: Dict[str, set[str]] = {
    "draftkings": {"draftkings"},
    "fanduel": {"fanduel"},
    "betmgm": {"betmgm"},
    "caesars": {"caesars", "williamhill_us"},
    "betrivers": {"betrivers", "barstool"},
}

DROP_TOKENS = {
    "U", "CC", "T",
    "II", "III", "IV", "V", "VI", "VII",
    "Sr", "Sr.", "Jr", "Jr.", "III",
}

# ------------------------- LOGGING ------------------------

log = logging.getLogger("oddsapi")
log.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("[oddsapi] %(message)s"))
log.addHandler(_handler)

# ------------------------- HELPERS ------------------------

def _compact_player_id(full_name: Any) -> str:
    if not isinstance(full_name, str):
        return ""
    cleaned = (
        full_name.replace(".", "")
        .replace("'", "")
        .strip()
    )
    if not cleaned:
        return ""
    parts = cleaned.split()
    parts = [p for p in parts if p and p not in DROP_TOKENS]
    if not parts:
        return ""
    first = parts[0]
    last = parts[-1]
    if not first or not last:
        return ""
    key = (first[0] + last).lower()
    return key[0].upper() + key[1:]

def _expand_books_filter(books: Optional[List[str]]) -> Optional[set[str]]:
    """
    Return None to mean 'no filter'. If a list is given, expand aliases and
    return a concrete set of bookmaker keys.
    """
    if not books:
        return None
    want = {b.strip().lower().replace(" ", "_") for b in books if b.strip()}
    if not want:
        return None
    expanded = set()
    for b in want:
        expanded |= BOOK_KEY_ALIASES.get(b, {b})
    return expanded or None

def _normalize_market(m: str) -> str:
    key = (m or "").strip().lower()
    return MARKET_ALIASES.get(key, key)

def _lim(headers: dict) -> dict:
    out = {}
    for k in ("x-requests-remaining", "x-requests-used", "x-requests-apikey-remaining"):
        if k in headers:
            out[k] = headers.get(k)
    return out

def _try_json(r: requests.Response):
    try:
        return r.json()
    except Exception:
        return {"text": r.text[:500]}

# Player props → per-event endpoint; game markets → bulk
BULK_ONLY_CANONICAL: set[str] = set(GAME_MARKETS)

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

# ------------------------- NORMALIZERS --------------------

def _normalize_game_rows(events: list, books_filter: Optional[set[str]]) -> pd.DataFrame:
    rows = []
    for ev in events or []:
        eid = ev.get("id")
        commence = ev.get("commence_time")
        home = ev.get("home_team")
        away = ev.get("away_team")
        for bm in ev.get("bookmakers", []):
            book_key = (bm.get("key") or "").strip().lower()
            book_title = (bm.get("title") or "").strip()
            if books_filter is not None and book_key not in books_filter:
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

def _normalize_player_rows(events: list, books_filter: Optional[set[str]], market_key: str) -> pd.DataFrame:
    recs: List[Dict[str, Any]] = []
    for ev in events or []:
        eid = ev.get("id")
        commence = ev.get("commence_time")
        for bm in ev.get("bookmakers", []):
            book_key = (bm.get("key") or "").strip().lower()
            book_title = (bm.get("title") or "").strip()
            if books_filter is not None and book_key not in books_filter:
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


def _normalize_team_abbr(team: Any) -> str:
    raw = ("" if team is None else str(team)).strip()
    if not raw:
        return ""
    upper = raw.upper()
    if upper in TEAM_NAME_TO_ABBR:
        return TEAM_NAME_TO_ABBR[upper]
    lower = raw.lower()
    if lower in TEAM_NAME_TO_ABBR:
        return TEAM_NAME_TO_ABBR[lower]
    cleaned = re.sub(r"[^A-Z0-9 ]+", "", upper)
    if cleaned in TEAM_NAME_TO_ABBR:
        return TEAM_NAME_TO_ABBR[cleaned]
    if len(upper) == 3 and upper.isalpha():
        return _canon_team(upper)
    return _canon_team(upper)


def _load_roster_map() -> dict[str, set[str]]:
    if not ROLES_PATH.exists() or ROLES_PATH.stat().st_size == 0:
        return {}
    try:
        roles = pd.read_csv(ROLES_PATH)
    except Exception:
        return {}
    need = {"team", "player"}
    if roles.empty or not need.issubset(set(roles.columns)):
        return {}
    roles = roles[list(need)].copy()
    roles["team"] = roles["team"].apply(_canon_team)
    roles["player_canonical"] = roles["player"].apply(canonicalize_name)
    roster: dict[str, set[str]] = {}
    for team, group in roles.dropna(subset=["team"]).groupby("team"):
        names = {nm for nm in group["player_canonical"].astype(str) if nm}
        if names:
            roster[team] = names
    return roster


def _infer_player_team(row: pd.Series, roster: dict[str, set[str]]) -> str:
    name = row.get("player_canonical", "")
    if not isinstance(name, str) or not name:
        return ""
    home = row.get("home_team_abbr", "")
    away = row.get("away_team_abbr", "")
    for code in (home, away):
        if code and code in roster and name in roster[code]:
            return code
    return ""


def _infer_opponent(team: str, home: str, away: str) -> str:
    team = (team or "").strip().upper()
    home = (home or "").strip().upper()
    away = (away or "").strip().upper()
    if team and home and away:
        if team == home:
            return away
        if team == away:
            return home
    return ""


def _write_props_enriched(props: pd.DataFrame, out_game: str) -> pd.DataFrame:
    Path(PROPS_ENRICHED_PATH.parent).mkdir(parents=True, exist_ok=True)
    event_cols = ["event_id", "commence_time", "home_team", "away_team", "kickoff_ts"]
    try:
        game_board = pd.read_csv(out_game)
    except Exception:
        game_board = pd.DataFrame(columns=event_cols)
    if not game_board.empty:
        game_board.columns = [c.lower() for c in game_board.columns]
        if "kickoff_ts" not in game_board.columns and "commence_time" in game_board.columns:
            game_board["kickoff_ts"] = game_board["commence_time"]
        if "home_team" in game_board.columns:
            game_board["home_team_abbr"] = game_board["home_team"].apply(_normalize_team_abbr)
        else:
            game_board["home_team_abbr"] = ""
        if "away_team" in game_board.columns:
            game_board["away_team_abbr"] = game_board["away_team"].apply(_normalize_team_abbr)
        else:
            game_board["away_team_abbr"] = ""
        event_info = (
            game_board[
                [c for c in ["event_id", "kickoff_ts", "home_team_abbr", "away_team_abbr"] if c in game_board.columns]
            ]
            .drop_duplicates(subset=["event_id"])
        )
    else:
        event_info = pd.DataFrame(columns=["event_id", "kickoff_ts", "home_team_abbr", "away_team_abbr"])

    columns = [
        "event_id",
        "player",
        "player_canonical",
        "market",
        "side",
        "line",
        "price_american",
        "book",
        "book_title",
        "commence_time",
    ]
    if props.empty:
        empty = pd.DataFrame(columns=[
            "event_id",
            "player_name_raw",
            "player_canonical",
            "player_market",
            "stat_type",
            "line",
            "odds",
            "kickoff_ts",
            "player_team_abbr",
            "opponent_team_abbr",
            "home_team_abbr",
            "away_team_abbr",
        ])
        empty.to_csv(PROPS_ENRICHED_PATH, index=False)
        log.info(f"wrote empty {PROPS_ENRICHED_PATH}")
        return empty

    working = props[[c for c in columns if c in props.columns]].copy()
    working["player_name_raw"] = working.get("player", "")
    working["player_market"] = working.get("market", "")
    working["stat_type"] = working.get("side", "")
    working["odds"] = working.get("price_american", "")

    if "player_canonical" not in working.columns:
        working["player_canonical"] = working.get("player", "").apply(canonicalize_name)

    enriched = working.merge(event_info, on="event_id", how="left")

    roster_map = _load_roster_map()
    if roster_map:
        enriched["player_team_abbr"] = enriched.apply(
            lambda row: _infer_player_team(row, roster_map), axis=1
        )
    else:
        enriched["player_team_abbr"] = ""

    if "home_team_abbr" in enriched.columns:
        enriched["home_team_abbr"] = enriched["home_team_abbr"].apply(_normalize_team_abbr)
    else:
        enriched["home_team_abbr"] = ""
    if "away_team_abbr" in enriched.columns:
        enriched["away_team_abbr"] = enriched["away_team_abbr"].apply(_normalize_team_abbr)
    else:
        enriched["away_team_abbr"] = ""

    fallback_series = pd.Series([""] * len(enriched)) if len(enriched) else pd.Series([], dtype=object)
    enriched["opponent_team_abbr"] = [
        _infer_opponent(team, home, away)
        for team, home, away in zip(
            enriched.get("player_team_abbr", fallback_series),
            enriched.get("home_team_abbr", fallback_series),
            enriched.get("away_team_abbr", fallback_series),
        )
    ]

    if "kickoff_ts" not in enriched.columns:
        enriched["kickoff_ts"] = enriched.get("commence_time")
    else:
        enriched["kickoff_ts"] = enriched["kickoff_ts"].fillna(enriched.get("commence_time"))

    cols_out = [
        "event_id",
        "kickoff_ts",
        "player_name_raw",
        "player_canonical",
        "player_market",
        "stat_type",
        "line",
        "odds",
        "book",
        "book_title",
        "player_team_abbr",
        "opponent_team_abbr",
        "home_team_abbr",
        "away_team_abbr",
    ]
    extra_cols = [c for c in enriched.columns if c not in cols_out]
    props_enriched = enriched[cols_out + extra_cols]
    props_enriched.to_csv(PROPS_ENRICHED_PATH, index=False)
    log.info(f"wrote {PROPS_ENRICHED_PATH} rows={len(props_enriched)}")
    return props_enriched


def _build_opponent_map(enriched: Optional[pd.DataFrame]) -> pd.DataFrame:
    cols = ["player", "team", "opponent", "event_id"]
    if enriched is None or enriched.empty:
        return pd.DataFrame(columns=cols)

    working = enriched.copy()
    if "player_canonical" not in working.columns:
        working["player_canonical"] = working.get("player", "")

    team_series = working.get("player_team_abbr")
    if team_series is None:
        team_series = working.get("team")
    working["team"] = (
        team_series.fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    opp_series = working.get("opponent_team_abbr")
    if opp_series is None:
        fallback = [
            _infer_opponent(team, home, away)
            for team, home, away in zip(
                working.get("team", pd.Series([], dtype=object)),
                working.get("home_team_abbr", pd.Series([], dtype=object)),
                working.get("away_team_abbr", pd.Series([], dtype=object)),
            )
        ]
        opp_series = pd.Series(fallback, index=working.index)
    working["opponent"] = (
        opp_series.fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    working["player_compact"] = working["player_canonical"].apply(_compact_player_id)

    out = (
        working[["player_compact", "team", "opponent", "event_id"]]
        .rename(columns={"player_compact": "player"})
    )
    out = out[out["player"].astype(str).str.strip().ne("")]
    out = out.drop_duplicates(subset=["player", "team", "event_id"])
    out = out.sort_values(["team", "player", "event_id"], kind="mergesort")

    missing_team = out["team"].astype(str).str.strip().eq("")
    if missing_team.any():
        log.warning(
            "opponent map missing team for %d players", int(missing_team.sum())
        )
    missing_opp = out["opponent"].astype(str).str.strip().eq("")
    if missing_opp.any():
        log.warning(
            "opponent map missing opponent for %d players", int(missing_opp.sum())
        )
    return out


def _write_data_exports(raw_props: pd.DataFrame, enriched: Optional[pd.DataFrame]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    props_copy = raw_props.copy()
    if not props_copy.empty and enriched is not None and not enriched.empty:
        join_cols = [c for c in ["event_id", "player"] if c in props_copy.columns and c in enriched.columns]
        extra_cols = [
            c
            for c in [
                "event_id",
                "player",
                "player_team_abbr",
                "opponent_team_abbr",
                "home_team_abbr",
                "away_team_abbr",
            ]
            if c in enriched.columns
        ]
        if join_cols:
            props_copy = props_copy.merge(
                enriched[extra_cols].drop_duplicates(subset=join_cols),
                on=join_cols,
                how="left",
            )
    if "player_team_abbr" in props_copy.columns:
        props_copy.rename(columns={"player_team_abbr": "team"}, inplace=True)
    if "opponent_team_abbr" in props_copy.columns:
        props_copy.rename(columns={"opponent_team_abbr": "opponent"}, inplace=True)

    props_copy.to_csv(PROPS_RAW_DATA_PATH, index=False)
    log.info(f"wrote {PROPS_RAW_DATA_PATH} rows={len(props_copy)}")

    opponent_map = _build_opponent_map(enriched)
    opponent_map.to_csv(OPPONENT_MAP_PATH, index=False)
    log.info(f"wrote {OPPONENT_MAP_PATH} rows={len(opponent_map)}")

# ------------------------- CORE FETCHERS ------------------

def _fetch_events_by_h2h(api_key: str, region: str, books: Optional[set[str]]) -> list:
    url = f"{BASE}/sports/{SPORT}/odds"
    params = {"apiKey": api_key, "regions": region, "markets": "h2h", "oddsFormat": "american"}
    if books is not None:
        params["bookmakers"] = ",".join(sorted(books))
    status, js, headers = _get(url, params)
    log.info(f"events status={status} limit={_lim(headers)}")
    if status != 200 or not isinstance(js, list):
        log.info(f"failed to fetch events: {js}")
        return []
    if books is not None:
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

def _fetch_game_odds(api_key: str, region: str, books: Optional[set[str]]) -> pd.DataFrame:
    url = f"{BASE}/sports/{SPORT}/odds"
    params = {"apiKey": api_key, "regions": region, "markets": ",".join(GAME_MARKETS), "oddsFormat": "american"}
    if books is not None:
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

def _fetch_market_for_events(api_key: str, region: str, books: Optional[set[str]],
                             event_ids: list[str], market_key: str) -> pd.DataFrame:
    mk = _normalize_market(market_key)
    frames: List[pd.DataFrame] = []

    tried = [mk] + ALT_KEYS.get(mk, [])
    for mk_try in tried:
        got_any = False
        for eid in event_ids:
            url = f"{BASE}/sports/{SPORT}/events/{eid}/odds"
            params = {"apiKey": api_key, "regions": region, "markets": mk_try, "oddsFormat": "american"}
            if books is not None:
                params["bookmakers"] = ",".join(sorted(books))
            status, js, headers = _get(url, params)
            log.info(f"{mk_try} eid={eid} status={status} limit={_lim(headers)}")

            if status == 200 and isinstance(js, dict):
                bm_count = len(js.get("bookmakers", []))
                if bm_count == 0 and books is not None:
                    log.info(f"{mk_try} eid={eid}: 0 bookmakers after filter → retry w/o filter")
                    params2 = dict(params)
                    params2.pop("bookmakers", None)
                    status2, js2, _ = _get(url, params2)
                    if status2 == 200 and isinstance(js2, dict):
                        df = _normalize_player_rows([js2], None, mk_try)
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
            if books is not None:
                params["bookmakers"] = ",".join(sorted(books))
            status, js, headers = _get(url, params)
            log.info(f"{mk} eid={eid} (us2) status={status} limit={_lim(headers)}")
            if status == 200 and isinstance(js, dict):
                df = _normalize_player_rows([js], books, mk)
                if df.empty and books is not None:
                    # try without filters in us2 too
                    params2 = dict(params)
                    params2.pop("bookmakers", None)
                    status2, js2, _ = _get(url, params2)
                    if status2 == 200 and isinstance(js2, dict):
                        df = _normalize_player_rows([js2], None, mk)
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

def _fetch_bulk_market(api_key: str, region: str, books: Optional[set[str]], market_key: str) -> pd.DataFrame:
    # bulk for game markets only
    mk = _normalize_market(market_key)
    url = f"{BASE}/sports/{SPORT}/odds"
    params = {"apiKey": api_key, "regions": region, "markets": mk, "oddsFormat": "american"}
    if books is not None:
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
    books: Optional[List[str]],
    markets: List[str],
    region: str = REGION_DEFAULT,
    date: str = "",
    season: str | int | None = None,
    out: str = "outputs/props_raw.csv",
    out_game: str = "outputs/odds_game.csv",
) -> None:
    api_key = os.getenv("ODDS_API_KEY", "").strip()
    if not api_key:
        log.info("ERROR: ODDS_API_KEY not set")
        sys.exit(2)

    log.info(
        "fetching Odds API props board (season=%s, date=%s)",
        "" if season is None else season,
        date,
    )

    # Expand aliases first, then validate
    books_set: Optional[set[str]] = _expand_books_filter(books)
    if region == "us" and books_set:
        bad = [b for b in books_set if b not in US_BOOK_KEYS]
        if bad:
            log.info(f"unknown/retired bookmaker key(s) for region=us: {bad} → removing from filter")
            books_set = {b for b in books_set if b in US_BOOK_KEYS} or None

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
    if "player" in props.columns:
        props["player_canonical"] = props["player"].apply(canonicalize_name)
    else:
        props["player_canonical"] = ""
    props.to_csv(out, index=False)
    log.info(f"wrote {out} rows={len(props)}")

    wide = _wide_over_under(props)
    if not wide.empty and "player" in wide.columns:
        wide["player_canonical"] = wide["player"].apply(canonicalize_name)
    wide_out = Path(out).with_name("props_raw_wide.csv")
    wide.to_csv(wide_out, index=False)
    log.info(f"wrote {wide_out} rows={len(wide)}")

    enriched: Optional[pd.DataFrame] = None
    try:
        enriched = _write_props_enriched(props, out_game)
    except Exception as err:
        log.info(f"failed to write props_enriched.csv: {err}")
    try:
        _write_data_exports(props, enriched)
    except Exception as err:
        log.info(f"failed to write data exports: {err}")

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
    ap.add_argument("--season", default="")
    ap.add_argument("--out", default="outputs/props_raw.csv")
    ap.add_argument("--out_game", default="outputs/odds_game.csv")
    args = ap.parse_args()

    # IMPORTANT: interpret "" as ALL (None), not an empty list
    raw_books = (args.books or "").strip()
    books_list: Optional[List[str]] = None if raw_books == "" else [b.strip() for b in raw_books.split(",") if b.strip()]

    fetch_odds(
        books=books_list,
        markets=[m.strip() for m in args.markets.split(",") if m.strip()],
        region=args.region,
        date=args.date,
        season=args.season,
        out=args.out,
        out_game=args.out_game,
    )
