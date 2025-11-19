#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from zoneinfo import ZoneInfo

import pandas as pd
import requests

from scripts.lib.io_utils import safe_concat, write_atomic
from scripts.lib.logging_utils import get_logger
from scripts.lib.time_windows import compute_slate_window

from scripts._opponent_map import CANON_TEAM_ABBR, canon_team
from scripts.utils.canonical_names import (
    canonicalize_player_name_safe,
    build_roles_map_from_csv,
    norm_key,
)

# Root paths for this repo
ROOT_DIR = Path(__file__).resolve().parents[1]

# Normalize data/ and outputs/ to live under the repo root,
# consistent with make_team_form and the other builders.
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

# Ensure the directories exist before we try to read/write
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional: allow a DATA_DIR override, but keep it inside the repo if relative.
data_dir_env = os.environ.get("DATA_DIR")
if data_dir_env:
    candidate = Path(data_dir_env)
    if not candidate.is_absolute():
        candidate = ROOT_DIR / candidate
    # Only override if the path is valid; otherwise stick with ROOT_DIR / "data"
    candidate.mkdir(parents=True, exist_ok=True)
    DATA_DIR = candidate

logger = logging.getLogger(__name__)

CANON_SET = set(CANON_TEAM_ABBR.values())

# ------------------------- RUNTIME CONFIG -----------------

LOGGER = get_logger("fetch_props")

PREFERRED_BOOKS = [
    b.strip()
    for b in os.getenv("BOOKS_PREF", "draftkings,fanduel").split(",")
    if b.strip()
]
ALLOW_FALLBACK = os.getenv("ALLOW_FALLBACK", "1") == "1"  # if no offers from preferred, try all
MISSING_POLICY = os.getenv("MISSING_POLICY", "warn")  # "warn" or "ignore" (never "fail")

MARKETS = [
    # keep your existing list; include only what you actually fetch
    "player_pass_yds",
    "player_pass_tds",
    "player_rush_yds",
    "player_rush_tds",
    "player_reception_yds",
    "player_receptions",
    "player_rush_reception_yds",
    "player_anytime_td",
    # etc…
]

# ------------------------- CONFIG -------------------------

SPORT = "americanfootball_nfl"
BASE  = "https://api.the-odds-api.com/v4"
REGION_DEFAULT = "us"
TIMEOUT_S = 25
BACKOFF_S = [0.6, 1.2, 2.0, 3.5, 5.0]
GAME_MARKETS = ["h2h", "spreads", "totals"]  # bulk-only
CENTRAL_TZ = ZoneInfo("America/Chicago")

PROPS_ENRICHED_PATH = DATA_DIR / "props_enriched.csv"
PROPS_RAW_DATA_PATH = DATA_DIR / "props_raw.csv"
OPPONENT_MAP_PATH = DATA_DIR / "opponent_map_from_props.csv"
ODDS_GAME_DATA_PATH = DATA_DIR / "odds_game.csv"
NAME_MAP_PATH = DATA_DIR / "player_name_map_from_props.csv"
PLAYER_NAME_LOG_PATH = OUTPUT_DIR / "player_name_map_from_props.csv"

# Optional override for the roles CSV path used by fetch_props.
_ROLES_CSV_OVERRIDE: Path | None = None

TEAM_WEEK_MAP_CSV = os.environ.get(
    "TEAM_WEEK_MAP_CSV",
    "data/team_week_map.csv",
)


def set_roles_csv_override(path: str | Path | None) -> None:
    """Set or clear the module-level override for the roles CSV path."""

    global _ROLES_CSV_OVERRIDE
    if path is None:
        _ROLES_CSV_OVERRIDE = None
    else:
        _ROLES_CSV_OVERRIDE = Path(path)


def _locate_roles_csv(explicit_path: str | None = None) -> Path:
    """
    Decide which roles_ourlads.csv to use.

    Precedence:
      1. Explicit path passed into the function.
      2. Module-level override `_ROLES_CSV_OVERRIDE` (if not None).
      3. Default search in outputs/roles_ourlads.csv then data/roles_ourlads.csv.
    """

    candidates: list[Path] = []

    if explicit_path:
        candidates.append(Path(explicit_path))

    if _ROLES_CSV_OVERRIDE is not None:
        candidates.append(_ROLES_CSV_OVERRIDE)

    candidates.append(OUTPUT_DIR / "roles_ourlads.csv")
    candidates.append(DATA_DIR / "roles_ourlads.csv")

    seen: set[str] = set()
    filtered_candidates: list[Path] = []
    for candidate in candidates:
        candidate_str = str(candidate.resolve()) if candidate.is_absolute() else str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        filtered_candidates.append(candidate)

    for p in filtered_candidates:
        try:
            df = pd.read_csv(p)
        except FileNotFoundError:
            continue
        except pd.errors.EmptyDataError:
            continue
        except OSError:
            continue
        if df.empty:
            continue
        logging.info(
            "[fetch_props] Using roles_ourlads.csv from %s (%d rows)",
            p,
            len(df),
        )
        return p

    msg = (
        "[fetch_props] ERROR: Could not locate a non-empty roles_ourlads.csv. "
        "Tried (in order): "
        + ", ".join(str(c) for c in filtered_candidates)
    )
    logging.error(msg)
    raise FileNotFoundError(msg)


def _canonical_name_only(raw_value: Any) -> str:
    name, _ = canonicalize_player_name_safe(raw_value)
    return name

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

log = LOGGER

STATE_DIR = Path("previous_state")
STATE_FILE = STATE_DIR / "props_state.json"

_FETCH_API_KEY: str = ""
_FETCH_REGION: str = REGION_DEFAULT


def _load_state() -> dict:
    try:
        if STATE_FILE.exists():
            with STATE_FILE.open("r", encoding="utf-8") as f:
                state = json.load(f)
            if not isinstance(state, dict):
                log.warning("props_state.json is not a dict; resetting to {}")
                return {}
            log.info("Restored props state with keys: %s", list(state.keys()))
            return state
        log.info("No previous props state found; starting fresh.")
        return {}
    except Exception as e:  # pragma: no cover - defensive I/O handling
        log.warning("Failed to read props_state.json (%s); starting fresh.", e)
        return {}


def _save_state(state: dict) -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with STATE_FILE.open("w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        log.info("Saved props state to %s", STATE_FILE)
    except Exception as e:  # pragma: no cover - defensive I/O handling
        log.warning("Failed to save props_state.json: %s", e)


def load_roles_df(explicit_path: str | None = None) -> pd.DataFrame:
    roles_csv_path = _locate_roles_csv(explicit_path)
    roles_df = pd.read_csv(roles_csv_path)
    logger.info("Loaded roles_ourlads CSV with shape=%s", roles_df.shape)

    if roles_df.empty:
        raise ValueError(
            f"[roles_ourlads] roles CSV {roles_csv_path} is empty; "
            "did the 'Build depth / roles (Ourlads)' step succeed and upload its artifact?"
        )

    required_cols = {"team", "player", "position", "role", "player_key"}
    missing = required_cols - set(roles_df.columns)
    if missing:
        raise RuntimeError(
            "roles_ourlads is missing expected columns: "
            f"{sorted(missing)}. "
            f"Columns present: {sorted(roles_df.columns)}"
        )

    return roles_df

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


def offers_is_empty(offers_json) -> bool:
    try:
        return not offers_json or len(offers_json.get("bookmakers", [])) == 0
    except Exception:
        return True


def fetch_market_offers(event_id: str, market: str, bookmakers: str | None):
    url = f"{BASE}/sports/{SPORT}/events/{event_id}/odds"
    params = {
        "apiKey": _FETCH_API_KEY,
        "regions": _FETCH_REGION,
        "markets": market,
        "oddsFormat": "american",
    }
    if bookmakers:
        params["bookmakers"] = bookmakers
    status, js, headers = _get(url, params)
    log.info(f"{market} eid={event_id} status={status} limit={_lim(headers)}")
    if status == 200 and isinstance(js, dict):
        return js
    return {}


def normalize_offers_to_rows(
    offers_json,
    event_id: str,
    market: str,
    *,
    source_market: str | None = None,
) -> List[dict[str, Any]]:
    if not offers_json:
        return []
    commence = offers_json.get("commence_time")
    rows: List[dict[str, Any]] = []
    for bm in offers_json.get("bookmakers", []) or []:
        book_key = (bm.get("key") or "").strip().lower()
        book_title = (bm.get("title") or "").strip()
        for mk in bm.get("markets", []) or []:
            mk_key = mk.get("key") or source_market or market
            if source_market:
                if mk_key != source_market:
                    continue
            elif mk_key != market:
                continue
            for outcome in mk.get("outcomes", []) or []:
                side = (outcome.get("name") or "").upper()
                if market == "player_anytime_td":
                    if side == "YES":
                        side = "OVER"
                    elif side == "NO":
                        side = "UNDER"
                raw_name = outcome.get("description") or outcome.get("participant")
                canonical_name, canonical_key = canonicalize_player_name_safe(raw_name)
                if not canonical_name:
                    canonical_name = (raw_name or "")
                rows.append(
                    {
                        "event_id": str(event_id),
                        "commence_time": commence,
                        "book": book_key,
                        "book_title": book_title,
                        "market": market,
                        "player": canonical_name,
                        "player_raw": raw_name,
                        "book_player_name": raw_name,
                        "canonical_player_name": canonical_name,
                        "canonical_player_key": canonical_key,
                        "side": side,
                        "line": outcome.get("point"),
                        "price_american": outcome.get("price"),
                    }
                )
    return rows


def _sanitize_params(params: Optional[dict[str, Any]]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    if not params:
        return sanitized
    for key, value in params.items():
        if isinstance(key, str) and key.lower() == "apikey":
            sanitized[key] = "***"
        else:
            sanitized[key] = value
    return sanitized


def _derive_slate_date(events: List[dict]) -> Optional[str]:
    """Infer slate date from the first commence_time in America/Chicago."""

    if not events:
        return None
    commence_values: List[datetime] = []
    for ev in events:
        commence = ev.get("commence_time")
        if not commence:
            continue
        try:
            dt = pd.to_datetime(commence, utc=True, errors="coerce")
        except Exception:
            dt = pd.NaT
        if pd.isna(dt):
            continue
        commence_values.append(dt.to_pydatetime().astimezone(CENTRAL_TZ))
    if not commence_values:
        return None
    first = min(commence_values)
    return first.date().isoformat()

def _try_json(r: requests.Response):
    try:
        return r.json()
    except Exception:
        return {"text": r.text[:500]}


TEAM_FIXES = {
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    "ARZ": "ARI",
    "LA": "LAR",
    "WSH": "WAS",
    "LVG": "LV",
    "KAN": "KC",
    "NWE": "NE",
    "NOR": "NO",
    "SFO": "SF",
    "TAM": "TB",
    "SDG": "LAC",
}


def _canon_team(value):
    if isinstance(value, pd.Series):
        return value.apply(_canon_team)

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""

    text = str(value)
    upper = text.strip().upper()
    alias = TEAM_FIXES.get(upper, upper)
    canonical = canon_team(alias)
    if not canonical:
        return ""
    canonical = canonical.upper()
    if canonical in CANON_SET:
        return canonical
    fallback = CANON_TEAM_ABBR.get(canonical, canonical)
    if fallback in CANON_SET:
        return fallback
    if fallback.upper() in CANON_SET:
        return fallback.upper()
    return canonical


def _to_int_safe(v):
    try:
        return int(v)
    except Exception:
        return None


def _ensure_team_week_map(season: int) -> Path:
    """Ensure the default team_week_map exists and is non-empty."""

    path = Path(TEAM_WEEK_MAP_CSV)

    def _is_bad(p: Path) -> bool:
        try:
            if not p.exists():
                return True
            if p.stat().st_size <= 1:
                return True
        except OSError:
            return True
        return False

    if not _is_bad(path):
        return path

    logger.warning(
        "[fetch_props] team_week_map CSV missing or tiny at %s; rebuilding via make_team_week_map.py for season=%s",
        path,
        season,
    )

    cmd = [
        sys.executable,
        "scripts/utils/make_team_week_map.py",
        "--season",
        str(season),
    ]
    env = dict(os.environ)
    env.setdefault("SEASON", str(season))
    subprocess.run(cmd, check=True, env=env)

    if _is_bad(path):
        raise RuntimeError(
            f"After rebuilding, team_week_map CSV at {path} is still missing or empty."
        )

    return path


def _load_team_week_map(
    season: int,
    schedule_path: str | None = None,
) -> pd.DataFrame:
    """
    Load the authoritative team-week map (full-season schedule).

    The schedule_path param allows overrides, but by default we read from
    TEAM_WEEK_MAP_CSV, which is also where scripts/make_team_week_map.py writes.
    """

    if schedule_path is None:
        path = _ensure_team_week_map(season)
    else:
        path = Path(schedule_path)
    abs_path = os.path.abspath(path)

    logger.info(
        "[fetch_props] loading team_week_map for season=%s from %s",
        season,
        abs_path,
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"team_week_map CSV not found at {abs_path}. "
            "Did the 'Build team-week map (authoritative opponents)' step run in this job?"
        )

    size = os.path.getsize(path)
    if size < 64:
        if schedule_path is None:
            logger.warning(
                "[fetch_props] team_week_map CSV at %s is unexpectedly small (size=%s); rebuilding and retrying.",
                abs_path,
                size,
            )
            path = _ensure_team_week_map(season)
            abs_path = os.path.abspath(path)
            size = os.path.getsize(path)
        else:
            raise RuntimeError(
                f"team_week_map CSV at {abs_path} is unexpectedly small "
                f"(size={size} bytes). Upstream schedule build may have failed."
            )

    def _read_team_week_map(current_path: str | Path) -> pd.DataFrame:
        return pd.read_csv(current_path)

    try:
        df = _read_team_week_map(path)
    except pd.errors.EmptyDataError:
        if schedule_path is not None:
            raise
        logger.warning(
            "[fetch_props] team_week_map CSV at %s raised EmptyDataError; rebuilding and retrying once.",
            abs_path,
        )
        path = _ensure_team_week_map(season)
        abs_path = os.path.abspath(path)
        df = _read_team_week_map(path)

    if "season" in df.columns:
        df = df[df["season"].astype(int) == int(season)].copy()

    for col in ("home_abbr", "away_abbr"):
        if col in df.columns:
            df[col] = df[col].map(_canon_team)

    logger.info(
        "[fetch_props] loaded team_week_map: %d rows, columns=%s",
        len(df),
        list(df.columns),
    )

    return df


def _load_roles_ourlads():
    df = load_roles_df()
    if "team" in df.columns:
        df["team"] = df["team"].map(_canon_team)
    name_col = "player"
    df["_pn_key"] = df[name_col].astype(str).str.strip().str.lower()
    return df[["_pn_key", "team"]].drop_duplicates()


def _load_player_name_map(path: Path) -> dict[str, str]:
    """Load sportsbook player name overrides mapping raw display names to canonical versions."""

    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        mapping_df = pd.read_csv(path)
    except Exception as err:
        log.info(f"player name map load failed (%s)", err)
        return {}
    if mapping_df.empty:
        return {}
    columns = {c.lower(): c for c in mapping_df.columns}
    raw_col = next(
        (columns[key] for key in ("raw_name", "book_name", "player", "source_name") if key in columns),
        None,
    )
    canon_col = next(
        (
            columns[key]
            for key in (
                "canonical_name",
                "player_canonical",
                "canonical",
                "player_name",
            )
            if key in columns
        ),
        None,
    )
    if not raw_col or not canon_col:
        return {}
    name_map: dict[str, str] = {}
    for raw_value, canon_value in zip(mapping_df[raw_col], mapping_df[canon_col]):
        if not isinstance(raw_value, str) or not isinstance(canon_value, str):
            continue
        raw_clean = raw_value.strip()
        canon_clean = canon_value.strip()
        if raw_clean and canon_clean:
            name_map[raw_clean] = canon_clean
    return name_map


def _canonicalize_player_names(
    df: pd.DataFrame,
    name_map: dict[str, str],
) -> tuple[pd.DataFrame, List[dict[str, Any]]]:
    """Apply canonical player mapping and capture unresolved names for logging."""

    if df is None or df.empty:
        return df, []

    working = df.copy()

    raw_col = None
    for candidate in ("book_player_name", "player_raw", "player"):
        if candidate in working.columns:
            raw_col = candidate
            break
    if raw_col is None:
        raw_col = "player"
        working[raw_col] = working.get("player", "")

    working["player_raw"] = working[raw_col].astype(str)

    overrides = {k.strip(): v.strip() for k, v in (name_map or {}).items() if k and v}
    name_records: List[dict[str, Any]] = []

    canonical_values: List[str] = []
    canonical_keys: List[str] = []

    for raw in working["player_raw"].astype(str):
        raw_value = (raw or "").strip()
        override = overrides.get(raw_value, raw_value)
        canonical_name, canonical_key = canonicalize_player_name_safe(override)
        canonical_name = (canonical_name or "").strip()
        canonical_key = (canonical_key or "").strip() or norm_key(canonical_name or override)
        if not canonical_name:
            canonical_name = override
        unresolved_flag = int(not canonical_name)
        canonical_values.append(canonical_name)
        canonical_keys.append(canonical_key)
        name_records.append(
            {
                "raw_name": raw_value,
                "canonical_player_name": canonical_name,
                "canonical_player_key": canonical_key,
                "canonical_name": canonical_name,
                "source": "oddsapi",
                "unresolved": unresolved_flag,
            }
        )
        if unresolved_flag:
            log.warning(
                "[ODDS-FETCH] unresolved canonical name for '%s'", raw_value
            )

    working["player"] = canonical_values
    working["player_canonical"] = canonical_values
    working["canonical_player_name"] = canonical_values
    working["canonical_player_key"] = canonical_keys
    if "book_player_name" not in working.columns:
        working["book_player_name"] = working["player_raw"]

    return working, name_records

def _normalize_teams_and_opponents(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    working = df.copy()

    def _apply(series: pd.Series) -> pd.Series:
        return series.fillna("").apply(_canon_team)

    for col in (
        "team",
        "team_abbr",
        "player_team_abbr",
        "home",
        "away",
        "home_team",
        "away_team",
        "home_team_abbr",
        "away_team_abbr",
        "opponent",
        "opponent_abbr",
        "opponent_team_abbr",
    ):
        if col in working.columns:
            working[col] = _apply(working[col].astype(str))

    if "team_abbr" in working.columns or "team" in working.columns:
        team_series = working.get("team_abbr")
        if team_series is None:
            team_series = pd.Series(pd.NA, index=working.index, dtype="object")
        if "team" in working.columns:
            team_series = team_series.fillna(working["team"])
        working["team_abbr"] = _apply(team_series.astype(str))

    if "opponent_abbr" in working.columns:
        working["opponent_abbr"] = _apply(working["opponent_abbr"].astype(str))
    elif "opponent" in working.columns:
        working["opponent_abbr"] = _apply(working["opponent"].astype(str))

    # For any residual unknowns, log once
    valid_values = CANON_SET | {"BYE", ""}
    for col in ("team", "opponent"):
        if col in working.columns:
            values = working[col].fillna("").astype(str)
            unknown = sorted({v for v in values.unique() if v and v not in valid_values})
            if unknown:
                print(
                    f"[oddsapi][warn] unknown {col} samples:",
                    unknown[:10],
                )
    return working

# Player props → per-event endpoint; game markets → bulk
BULK_ONLY_CANONICAL: set[str] = set(GAME_MARKETS)

# ------------------------- HTTP --------------------------

_SESSION = requests.Session()


def _get(url: str, params: dict, max_retries: int = 5) -> Tuple[int, Optional[Any], dict]:
    for i in range(max_retries):
        try:
            resp = _SESSION.get(url, params=params, timeout=TIMEOUT_S)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                except Exception:
                    return 200, None, resp.headers
                if data is None:
                    log.warning("Odds API returned empty payload for %s %s", url, _sanitize_params(params))
                    return 200, None, resp.headers
                if isinstance(data, (list, dict)) and not data:
                    log.warning("Odds API returned empty payload for %s %s", url, _sanitize_params(params))
                    if isinstance(data, dict):
                        data = {}
                    else:
                        data = []
                return 200, data, resp.headers
            if resp.status_code in (401, 403, 404, 422):
                return resp.status_code, _try_json(resp), resp.headers
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = BACKOFF_S[min(i, len(BACKOFF_S)-1)]
                log.info(f"HTTP {resp.status_code} → backoff {wait}s: {url}")
                time.sleep(wait)
                continue
            raise RuntimeError(
                f"Odds API {url} failed {resp.status_code}: {resp.text[:300]}"
            )
        except RuntimeError:
            raise
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
        home = _canon_team(ev.get("home_team"))
        away = _canon_team(ev.get("away_team"))
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


def _serialize_offer_rows(df: pd.DataFrame) -> List[dict[str, Any]]:
    offers: List[dict[str, Any]] = []
    if df is None or df.empty:
        return offers
    for row in df.to_dict(orient="records"):
        offers.append(
            {
                "book": row.get("book"),
                "book_title": row.get("book_title"),
                "side": row.get("side"),
                "line": row.get("line"),
                "price": row.get("price_american"),
            }
        )
    return offers


def _build_market_records(
    df: pd.DataFrame,
    *,
    include_missing: Optional[List[dict[str, Any]]] = None,
) -> List[dict[str, Any]]:
    records: List[dict[str, Any]] = []
    if df is not None and not df.empty:
        group_cols = [c for c in ["event_id", "player", "market"] if c in df.columns]
        if len(group_cols) == 3:
            for keys, group in df.groupby(group_cols):
                event_id, player, market = keys
                commence = None
                if "commence_time" in group.columns:
                    commence = group["commence_time"].iloc[0]
                fetched_val = None
                if "fetched_at" in group.columns:
                    fetched_val = group["fetched_at"].iloc[0]
                missing_flag = int(group.get("bookmaker_missing", pd.Series([0])).max())
                team_val = group.get("team_abbr")
                team_abbr = team_val.iloc[0] if isinstance(team_val, pd.Series) else None
                opp_val = group.get("opponent_abbr")
                opponent_abbr = opp_val.iloc[0] if isinstance(opp_val, pd.Series) else None
                raw_value = group["player_raw"].iloc[0] if "player_raw" in group.columns else player
                book_value = (
                    group["book_player_name"].iloc[0]
                    if "book_player_name" in group.columns
                    else raw_value
                )
                canonical_value = (
                    group["canonical_player_name"].iloc[0]
                    if "canonical_player_name" in group.columns
                    else player
                )
                canonical_key = (
                    group["canonical_player_key"].iloc[0]
                    if "canonical_player_key" in group.columns
                    else norm_key(canonical_value)
                )
                records.append(
                    {
                        "event_id": event_id,
                        "player": canonical_value,
                        "player_raw": raw_value,
                        "book_player_name": book_value,
                        "canonical_player_name": canonical_value,
                        "canonical_player_key": canonical_key,
                        "market": market,
                        "commence_time": commence,
                        "books_json": _serialize_offer_rows(group),
                        "bookmaker_missing": missing_flag,
                        "fetched_at": fetched_val,
                        "team_abbr": team_abbr,
                        "opponent_abbr": opponent_abbr,
                    }
                )
    if include_missing:
        records.extend(include_missing)
    return records


def _write_market_dumps(records: Dict[str, List[dict[str, Any]]]) -> None:
    out_dir = OUTPUT_DIR / "props_raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    for market, recs in (records or {}).items():
        market_path = out_dir / f"{market}.csv"
        df = pd.DataFrame(recs)
        if not df.empty and "books_json" in df.columns:
            df["books_json"] = df["books_json"].apply(lambda v: json.dumps(v or []))
        else:
            df = pd.DataFrame(
                columns=[
                    "event_id",
                    "player",
                    "market",
                    "commence_time",
                    "books_json",
                    "bookmaker_missing",
                    "fetched_at",
                ]
            )
        df.to_csv(market_path, index=False)

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
    alias = TEAM_FIXES.get(upper, upper)
    canonical = canon_team(alias)
    if not canonical:
        return ""
    canonical = canonical.upper()
    if canonical in CANON_SET:
        return canonical
    fallback = CANON_TEAM_ABBR.get(canonical, canonical)
    return fallback.upper() if isinstance(fallback, str) else ""


def _load_roster_map() -> dict[str, set[str]]:
    roles = load_roles_df()
    need = {"team", "player"}
    if not need.issubset(set(roles.columns)):
        return {}
    roles = roles[list(need)].copy()
    roles["team"] = roles["team"].apply(_canon_team)
    roles["player_canonical"] = roles["player"].apply(_canonical_name_only)
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
        working["player_canonical"] = working.get("player", "").apply(
            _canonical_name_only
        )

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


def _prepare_game_meta(
    game_df: Optional[pd.DataFrame], season: int | None
) -> pd.DataFrame:
    """Normalize game-level metadata for joining onto props."""

    base_cols = ["event_id", "week", "season", "game_timestamp"]
    if game_df is None or game_df.empty:
        return pd.DataFrame(columns=base_cols)

    meta = game_df.copy()
    if "event_id" not in meta.columns:
        meta["event_id"] = pd.NA
    meta["event_id"] = meta["event_id"].fillna("").astype(str)

    ts_col = next(
        (
            col
            for col in ["kickoff_ts", "commence_time", "game_timestamp"]
            if col in meta.columns
        ),
        None,
    )
    if ts_col:
        meta["game_timestamp"] = pd.to_datetime(meta[ts_col], errors="coerce")
    else:
        meta["game_timestamp"] = pd.NaT

    keep_cols = ["event_id", "game_timestamp"]
    if "week" in meta.columns:
        keep_cols.append("week")
    if "season" in meta.columns:
        keep_cols.append("season")
    meta = meta[keep_cols].drop_duplicates(subset=["event_id"])

    if "week" in meta.columns:
        meta["week"] = pd.to_numeric(meta["week"], errors="coerce")
    else:
        meta["week"] = pd.NA

    if "season" in meta.columns:
        meta["season"] = pd.to_numeric(meta["season"], errors="coerce").astype(
            "Int64"
        )
        if season is not None:
            meta["season"] = meta["season"].fillna(int(season))
    else:
        if season is not None:
            meta["season"] = pd.Series(
                [int(season)] * len(meta), index=meta.index, dtype="Int64"
            )
        else:
            meta["season"] = pd.Series([pd.NA] * len(meta), index=meta.index, dtype="Int64")

    for col in base_cols:
        if col not in meta.columns:
            meta[col] = pd.NA

    return meta[base_cols]


def _build_opponent_map(
    enriched: Optional[pd.DataFrame],
    game_df: Optional[pd.DataFrame],
    season: int | None,
) -> pd.DataFrame:
    cols = [
        "player",
        "team",
        "opponent",
        "week",
        "season",
        "game_timestamp",
        "event_id",
    ]
    if enriched is None or enriched.empty:
        return pd.DataFrame(columns=cols)

    working = enriched.copy()
    if "player_canonical" not in working.columns:
        working["player_canonical"] = working.get("player", "")

    team_series = working.get("player_team_abbr")
    if team_series is None:
        team_series = working.get("team")
    if team_series is None:
        team_series = pd.Series([""] * len(working), index=working.index)
    raw_team_series = team_series.fillna("").astype(str)
    working["team"] = raw_team_series.apply(_canon_team)

    opp_series = working.get("opponent_team_abbr")
    if opp_series is None:
        fallback = [
            _infer_opponent(
                _canon_team(team),
                _canon_team(home),
                _canon_team(away),
            )
            for team, home, away in zip(
                working.get("team", pd.Series([], dtype=object)),
                working.get("home_team_abbr", pd.Series([], dtype=object)),
                working.get("away_team_abbr", pd.Series([], dtype=object)),
            )
        ]
        opp_series = pd.Series(fallback, index=working.index)
    raw_opp_series = opp_series.fillna("").astype(str)
    working["opponent"] = raw_opp_series.apply(_canon_team)

    def _count_unresolved(series: pd.Series) -> int:
        if series is None or series.empty:
            return 0
        series_str = series.fillna("").astype(str)
        return int(
            sum(
                1
                for value in series_str
                if value and _canon_team(value) not in CANON_SET
            )
        )

    pre_team_unresolved = _count_unresolved(raw_team_series)
    pre_opp_unresolved = _count_unresolved(raw_opp_series)

    if "event_id" in working.columns:
        working["event_id"] = working["event_id"].fillna("").astype(str)
    else:
        working["event_id"] = ""

    working["player_compact"] = working["player_canonical"].apply(_compact_player_id)

    meta = _prepare_game_meta(game_df, season)
    if not meta.empty:
        working = working.merge(meta, on="event_id", how="left")
    else:
        for col in ["week", "season"]:
            if col not in working.columns:
                working[col] = pd.NA
        if "game_timestamp" not in working.columns:
            working["game_timestamp"] = pd.NaT

    if season is not None:
        season_series = working.get("season")
        if season_series is None:
            season_series = pd.Series(pd.NA, index=working.index)
        working["season"] = season_series.fillna(int(season))

    out = (
        working[
            [
                "player_compact",
                "team",
                "opponent",
                "week",
                "season",
                "game_timestamp",
                "event_id",
            ]
        ]
        .rename(columns={"player_compact": "player"})
        .copy()
    )

    out = out[out["player"].astype(str).str.strip().ne("")]
    out = out[out["team"].astype(str).str.strip().ne("")]
    out = out.drop_duplicates(subset=["player", "team", "event_id"])

    missing_team = out["team"].astype(str).str.strip().eq("")
    if missing_team.any():
        log.warning(
            "opponent map missing team for %d players", int(missing_team.sum())
        )
    missing_opp = out["opponent"].astype(str).str.strip().eq("") | out["opponent"].isna()
    missing_week = out["week"].isna()
    missing_ts = out["game_timestamp"].isna()
    missing_any = missing_opp | missing_week | missing_ts
    if missing_any.any():
        for player in sorted(out.loc[missing_any, "player"].dropna().unique()):
            log.warning("[opponent_map] missing mapping for %s", player)

    post_team_unresolved = _count_unresolved(out["team"])
    post_opp_unresolved = _count_unresolved(out["opponent"])
    log.info(
        "[opponent_map] unresolved team/opponent counts pre=%s/%s post=%s/%s",
        pre_team_unresolved,
        pre_opp_unresolved,
        post_team_unresolved,
        post_opp_unresolved,
    )

    return out.reindex(columns=cols)


def _write_props_enriched_data(
    raw_props: pd.DataFrame,
    enriched: Optional[pd.DataFrame],
    game_df: Optional[pd.DataFrame],
) -> None:
    """Write data/props_enriched.csv with deterministic opponent inference."""

    PROPS_ENRICHED_PATH.parent.mkdir(parents=True, exist_ok=True)

    base = pd.DataFrame()
    if enriched is not None and not enriched.empty:
        base = enriched.copy()
    elif raw_props is not None and not raw_props.empty:
        base = raw_props.copy()
        if "player" in base.columns and "player_name_raw" not in base.columns:
            base["player_name_raw"] = base["player"]
        if "market" in base.columns and "player_market" not in base.columns:
            base["player_market"] = base["market"]
        if "side" in base.columns and "stat_type" not in base.columns:
            base["stat_type"] = base["side"]
        if "price_american" in base.columns and "odds" not in base.columns:
            base["odds"] = base["price_american"]

    if base.empty:
        pd.DataFrame().to_csv(PROPS_ENRICHED_PATH, index=False)
        log.info(f"wrote empty {PROPS_ENRICHED_PATH}")
        return

    if "event_id" not in base.columns:
        if raw_props is not None and "event_id" in raw_props.columns:
            base["event_id"] = raw_props["event_id"]
        else:
            base["event_id"] = ""
    base["event_id"] = base["event_id"].fillna("").astype(str)

    games = pd.DataFrame()
    if game_df is not None and not game_df.empty:
        games = game_df.copy()
    else:
        try:
            games = pd.read_csv(ODDS_GAME_DATA_PATH)
        except Exception:
            games = pd.DataFrame()

    if not games.empty and "event_id" in games.columns:
        games_subset_cols = [
            c
            for c in [
                "event_id",
                "home_team",
                "away_team",
                "home_team_abbr",
                "away_team_abbr",
                "kickoff_ts",
                "commence_time",
            ]
            if c in games.columns
        ]
        games_subset = games[games_subset_cols].copy()
        games_subset["event_id"] = games_subset["event_id"].fillna("").astype(str)
        for col in ["home_team", "away_team", "home_team_abbr", "away_team_abbr"]:
            if col in games_subset.columns:
                games_subset[col] = (
                    games_subset[col]
                    .fillna("")
                    .astype(str)
                    .str.upper()
                    .str.strip()
                )
        if "kickoff_ts" not in games_subset.columns and "commence_time" in games_subset.columns:
            games_subset["kickoff_ts"] = games_subset["commence_time"]
        games_subset = games_subset.drop_duplicates(subset=["event_id"])
        base = base.merge(games_subset, on="event_id", how="left")

    team_col = next(
        (candidate for candidate in ["player_team_abbr", "team", "player_team"] if candidate in base.columns),
        None,
    )
    if team_col:
        base[team_col] = (
            base[team_col]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.strip()
        )

    if "opponent_team_abbr" not in base.columns:
        base["opponent_team_abbr"] = pd.NA

    def _opp_from_row(row: pd.Series) -> Optional[str]:
        tm = str(row.get(team_col, "")) if team_col else ""
        tm = tm.upper().strip()
        home = row.get("home_team_abbr") or row.get("home_team")
        away = row.get("away_team_abbr") or row.get("away_team")
        home = "" if home is None else str(home).upper().strip()
        away = "" if away is None else str(away).upper().strip()
        if not tm or not home or not away:
            return None
        if tm == home:
            return away
        if tm == away:
            return home
        return None

    if team_col and len(base):
        mask = base["opponent_team_abbr"].isna() | base["opponent_team_abbr"].astype(str).str.strip().eq("")
        if mask.any():
            inferred = base.loc[mask].apply(_opp_from_row, axis=1)
            base.loc[mask, "opponent_team_abbr"] = inferred

    for col in ["player_team_abbr", "opponent_team_abbr", "home_team_abbr", "away_team_abbr"]:
        if col in base.columns:
            base[col] = (
                base[col]
                .fillna("")
                .astype(str)
                .str.upper()
                .str.strip()
            )

    if "kickoff_ts" not in base.columns:
        if "commence_time" in base.columns:
            base["kickoff_ts"] = base["commence_time"]
        elif "game_timestamp" in base.columns:
            base["kickoff_ts"] = base["game_timestamp"]

    for helper in ["home_team", "away_team"]:
        abbr_col = f"{helper}_abbr"
        if helper in base.columns and abbr_col in base.columns:
            base.drop(columns=[helper], inplace=True)

    base.to_csv(PROPS_ENRICHED_PATH, index=False)
    log.info(f"wrote {PROPS_ENRICHED_PATH} rows={len(base)}")


def _write_data_exports(
    raw_props: pd.DataFrame,
    enriched: Optional[pd.DataFrame],
    game_df: Optional[pd.DataFrame],
    season: int | None,
) -> None:
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

    for col in ("team", "opponent", "home_team_abbr", "away_team_abbr"):
        if col in props_copy.columns:
            props_copy[col] = props_copy[col].apply(_canon_team)

    props_copy.to_csv(PROPS_RAW_DATA_PATH, index=False)
    log.info(f"wrote {PROPS_RAW_DATA_PATH} rows={len(props_copy)}")

    if game_df is not None and not game_df.empty:
        game_df.to_csv(ODDS_GAME_DATA_PATH, index=False)
        log.info(f"wrote {ODDS_GAME_DATA_PATH} rows={len(game_df)}")
    else:
        pd.DataFrame().to_csv(ODDS_GAME_DATA_PATH, index=False)
        log.info(f"wrote empty {ODDS_GAME_DATA_PATH}")

    opponent_map = _build_opponent_map(enriched, game_df, season)
    opponent_map.to_csv(OPPONENT_MAP_PATH, index=False)
    log.info(f"wrote {OPPONENT_MAP_PATH} rows={len(opponent_map)}")

    _write_props_enriched_data(props_copy, enriched, game_df)

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

def _fetch_market_for_events(
    api_key: str,
    region: str,
    books: Optional[set[str]],
    event_ids: list[str],
    market_key: str,
    fetched_at: str,
) -> tuple[pd.DataFrame, List[dict[str, Any]], dict[str, int]]:
    mk = _normalize_market(market_key)
    frames: List[pd.DataFrame] = []
    event_state: Dict[str, dict[str, Any]] = {
        str(eid): {"filtered": False, "fallback": pd.DataFrame()}
        for eid in event_ids
    }

    tried = [mk] + ALT_KEYS.get(mk, [])
    for mk_try in tried:
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
                    params2 = dict(params)
                    params2.pop("bookmakers", None)
                    log.info(
                        f"{mk_try} eid={eid}: 0 bookmakers after filter → retry w/o filter"
                    )
                    status2, js2, _ = _get(url, params2)
                    if status2 == 200 and isinstance(js2, dict):
                        try:
                            fallback_df = _normalize_player_rows([js2], None, mk_try)
                        except Exception as err:
                            log.info(
                                f"normalize fail eid={eid} market={mk_try} (no filter): {err}"
                            )
                            fallback_df = pd.DataFrame()
                        event_state[str(eid)]["fallback"] = fallback_df
                    continue

                try:
                    df = _normalize_player_rows([js], books, mk_try)
                except Exception as err:
                    log.info(f"normalize fail eid={eid} market={mk_try}: {err}")
                    df = pd.DataFrame()
                if not df.empty:
                    df["fetched_at"] = fetched_at
                    df["bookmaker_missing"] = 0
                    frames.append(df)
                    event_state[str(eid)]["filtered"] = True

            elif status == 422:
                continue
            elif status in (401, 403, 404):
                log.info(f"skip market={mk_try} for eid={eid}: {js}")
            else:
                log.info(f"market fetch error eid={eid} market={mk_try}: {js}")

        if frames:
            break

    # Dual-region fallback for filtered fetch only when nothing returned
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
                try:
                    df = _normalize_player_rows([js], books, mk)
                except Exception as err:
                    log.info(f"normalize fail eid={eid} market={mk} (us2): {err}")
                    df = pd.DataFrame()
                if not df.empty:
                    df["fetched_at"] = fetched_at
                    df["bookmaker_missing"] = 0
                    frames_us2.append(df)
                    event_state[str(eid)]["filtered"] = True
            elif status == 422:
                continue
        if frames_us2:
            merged = pd.concat(frames_us2, ignore_index=True)
            frames.append(merged)
            log.info(f"{mk}: merged {len(merged)} rows from region=us2 fallback.")
        else:
            log.info(f"{mk}: region=us2 also empty.")

    merged_filtered = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    stats = {
        "events": len(event_ids),
        "offers": int(len(merged_filtered)) if not merged_filtered.empty else 0,
        "missing_filtered": 0,
        "saved_no_filter": 0,
        "bookmaker_missing": 0,
    }

    missing_rows: List[dict[str, Any]] = []
    for eid, info in event_state.items():
        if info.get("filtered"):
            continue
        stats["missing_filtered"] += 1
        fallback_df: pd.DataFrame = info.get("fallback", pd.DataFrame())
        if fallback_df is not None and not fallback_df.empty:
            stats["saved_no_filter"] += fallback_df["player"].dropna().nunique()
            for player, group in fallback_df.groupby("player"):
                books_json = _serialize_offer_rows(group)
                commence = None
                if "commence_time" in group.columns:
                    commence = group["commence_time"].iloc[0]
                missing_rows.append(
                    {
                        "event_id": eid,
                        "player": player,
                        "player_raw": player,
                        "market": mk,
                        "commence_time": commence,
                        "books_json": books_json,
                        "bookmaker_missing": 1,
                        "fetched_at": fetched_at,
                    }
                )
        else:
            stats["bookmaker_missing"] += 1
            missing_rows.append(
                {
                    "event_id": eid,
                    "player": "",
                    "player_raw": "",
                    "market": mk,
                    "commence_time": None,
                    "books_json": [],
                    "bookmaker_missing": 1,
                    "fetched_at": fetched_at,
                }
            )

    return merged_filtered, missing_rows, stats

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
    try:
        return _normalize_player_rows(js, books, mk)
    except Exception as err:
        log.info(f"normalize fail market={mk} (bulk): {err}")
        return pd.DataFrame()

# ------------------------- PUBLIC ENTRY -------------------

def fetch_odds(
    *,
    books: Optional[List[str]],
    markets: List[str],
    region: str = REGION_DEFAULT,
    date: str = "",
    season: str | int | None = None,
    out: Path | str = OUTPUT_DIR / "props_raw.csv",
    out_game: Path | str = OUTPUT_DIR / "odds_game.csv",
) -> None:
    api_key = os.getenv("ODDS_API_KEY", "").strip()
    if not api_key:
        log.info("ERROR: ODDS_API_KEY not set")
        sys.exit(2)

    global _FETCH_API_KEY, _FETCH_REGION
    _FETCH_API_KEY = api_key
    _FETCH_REGION = region

    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = ROOT_DIR / out_path
    out = str(out_path)

    out_game_path = Path(out_game)
    if not out_game_path.is_absolute():
        out_game_path = ROOT_DIR / out_game_path
    out_game = str(out_game_path)

    anchor, start_local, end_local = compute_slate_window()
    LOGGER.info(
        f"Slate anchor={anchor.isoformat()} window_local=[{start_local} → {end_local}]"
    )

    season_value: Optional[int] = None
    if season:
        try:
            season_value = int(season)
        except (TypeError, ValueError):
            season_value = None
    if season_value is None:
        env_season = os.environ.get("SEASON", "").strip()
        if env_season:
            try:
                season_value = int(env_season)
            except ValueError:
                season_value = None

    week_value: Optional[int] = None
    for key in ("WEEK", "SLATE_WEEK"):
        env_week = os.environ.get(key)
        if env_week:
            try:
                week_value = int(env_week)
                break
            except ValueError:
                continue

    # Expand aliases first, then validate
    books_set: Optional[set[str]] = _expand_books_filter(books)
    if region == "us" and books_set:
        bad = [b for b in books_set if b not in US_BOOK_KEYS]
        if bad:
            log.info(f"unknown/retired bookmaker key(s) for region=us: {bad} → removing from filter")
            books_set = {b for b in books_set if b in US_BOOK_KEYS} or None

    markets = [m.strip() for m in markets if m.strip()]
    normalized_markets: List[str] = [_normalize_market(m) for m in markets]

    state: dict = _load_state()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info(
        "[props] fetch start season=%s date=%s markets=%d books=%s",
        "" if season is None else season,
        date,
        len(normalized_markets),
        ",".join(sorted(books_set)) if books_set else "all",
    )

    fetched_at = datetime.now(timezone.utc).isoformat()

    # events for event-odds
    events = _fetch_events_by_h2h(api_key, region, books_set)
    log.info(f"events pulled={len(events)}")
    if not date:
        inferred_date = _derive_slate_date(events)
        if inferred_date:
            date = inferred_date
            log.info(
                f"defaulted slate date to {date} (America/Chicago) based on first event"
            )
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
    market_records: Dict[str, List[dict[str, Any]]] = {}
    market_stats: Dict[str, dict[str, int]] = {}
    all_market_paths: List[Path] = []
    summary: List[dict[str, Any]] = []
    preferred_param = ",".join(PREFERRED_BOOKS) if PREFERRED_BOOKS else ""
    preferred_param = preferred_param or None
    event_lookup: Dict[str, dict] = {
        str(ev.get("id")): ev for ev in events if ev.get("id")
    }

    for mk in normalized_markets:
        if mk in GAME_MARKETS:
            log.info(f"=== MARKET {mk} (bulk) ===")
            df = _fetch_bulk_market(api_key, region, books_set, mk)
            missing_rows: List[dict[str, Any]] = []
            stats = {
                "events": len(event_ids),
                "offers": int(len(df)) if not df.empty else 0,
                "missing_filtered": 0,
                "saved_no_filter": 0,
                "bookmaker_missing": 0,
            }
            if not df.empty:
                frames.append(df)
            market_records[mk] = _build_market_records(df, include_missing=missing_rows)
            market_stats[mk] = stats
            log.info(
                f"market={mk} offers={stats['offers']} missing_filtered={stats['missing_filtered']} "
                f"saved_no_filter={stats['saved_no_filter']} missing_rows={len(market_records[mk])}"
            )
            continue

        market_path = OUTPUT_DIR / f"props_{mk}.csv"
        all_market_paths.append(market_path)
        rows: List[pd.DataFrame] = []
        missing_rows: List[dict[str, Any]] = []
        missing = 0
        saved_no_filter = 0

        if not event_ids:
            log.info(f"no events → skip player market {mk}")
            market_records[mk] = _build_market_records(pd.DataFrame(), include_missing=[])
            market_stats[mk] = {
                "events": 0,
                "offers": 0,
                "missing_filtered": 0,
                "saved_no_filter": 0,
                "bookmaker_missing": 0,
            }
            summary.append({"market": mk, "rows": 0, "missing_events": 0})
            level = "info"
            getattr(LOGGER, level)(
                f"market={mk} wrote_rows=0 missing_events=0 file={market_path}"
            )
            continue

        log.info(f"=== MARKET {mk} (per-event) ===")
        for eid in event_ids:
            used_key = mk
            used_fallback = False
            offers = {}
            variants = [mk] + ALT_KEYS.get(mk, [])
            for mk_try in variants:
                used_key = mk_try
                offers = fetch_market_offers(eid, mk_try, preferred_param)
                if offers_is_empty(offers) and ALLOW_FALLBACK:
                    offers_all = fetch_market_offers(eid, mk_try, None)
                    if not offers_is_empty(offers_all):
                        offers = offers_all
                        used_fallback = True
                if not offers_is_empty(offers):
                    break
            if offers_is_empty(offers):
                missing += 1
                ev = event_lookup.get(str(eid), {})
                missing_rows.append(
                    {
                        "event_id": eid,
                        "player": "",
                        "player_raw": "",
                        "market": mk,
                        "commence_time": ev.get("commence_time"),
                        "books_json": [],
                        "bookmaker_missing": 1,
                        "fetched_at": fetched_at,
                    }
                )
                time.sleep(0.05)
                continue

            normalized_rows = normalize_offers_to_rows(
                offers, eid, mk, source_market=used_key
            )
            if normalized_rows:
                df_event = pd.DataFrame(normalized_rows)
                df_event["fetched_at"] = fetched_at
                df_event["bookmaker_missing"] = 0
                rows.append(df_event)
                if used_fallback:
                    if "player" in df_event.columns:
                        saved_no_filter += df_event["player"].dropna().nunique()
                    else:
                        saved_no_filter += len(df_event)
            else:
                missing += 1
                ev = event_lookup.get(str(eid), {})
                missing_rows.append(
                    {
                        "event_id": eid,
                        "player": "",
                        "player_raw": "",
                        "market": mk,
                        "commence_time": ev.get("commence_time"),
                        "books_json": [],
                        "bookmaker_missing": 1,
                        "fetched_at": fetched_at,
                    }
                )
            time.sleep(0.05)

        df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        if not df.empty:
            if "line" in df.columns:
                df["line"] = pd.to_numeric(df["line"], errors="coerce")
            if "price_american" in df.columns:
                df["price_american"] = pd.to_numeric(df["price_american"], errors="coerce")
            safe_concat(str(market_path), df)
            frames.append(df)
        summary.append({"market": mk, "rows": len(df), "missing_events": missing})
        level = "warning" if missing > 0 and MISSING_POLICY == "warn" else "info"
        getattr(LOGGER, level)(
            f"market={mk} wrote_rows={len(df)} missing_events={missing} file={market_path}"
        )

        stats = {
            "events": len(event_ids),
            "offers": int(len(df)) if not df.empty else 0,
            "missing_filtered": missing,
            "saved_no_filter": saved_no_filter,
            "bookmaker_missing": missing,
        }
        market_records[mk] = _build_market_records(df, include_missing=missing_rows)
        market_stats[mk] = stats

    _write_market_dumps(market_records)

    if market_stats:
        total_missing_rows = sum(stats.get("bookmaker_missing", 0) for stats in market_stats.values())
        total_no_filter_saved = sum(stats.get("saved_no_filter", 0) for stats in market_stats.values())
        total_missing_filtered = sum(stats.get("missing_filtered", 0) for stats in market_stats.values())
        log.info(
            "market summary: missing_filtered=%s saved_via_no_filter=%s remaining_missing_rows=%s",
            total_missing_filtered,
            total_no_filter_saved,
            total_missing_rows,
        )

    frames_from_disk: List[pd.DataFrame] = []
    for path in all_market_paths:
        if path.exists():
            try:
                frames_from_disk.append(pd.read_csv(path))
            except Exception as e:
                LOGGER.warning(f"skip bad file {path}: {e}")

    if frames_from_disk:
        consolidated = pd.concat(frames_from_disk, ignore_index=True)
        write_atomic(str(OUTPUT_DIR / "props_raw.csv"), consolidated)
        LOGGER.info(
            f"Consolidated props_raw.csv rows={len(consolidated)} from {len(frames_from_disk)} market files"
        )
    else:
        consolidated = pd.DataFrame()
        LOGGER.warning("No market files present; props_raw.csv not written")

    if not consolidated.empty:
        props = consolidated
    else:
        props = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    name_records_all: List[dict[str, Any]] = []
    if "player" in props.columns:
        props["player_canonical"] = props["player"].apply(_canonical_name_only)
    else:
        props["player_canonical"] = ""

    # --- BEGIN: enrich props with opponent from odds_game ---
    odds_game_path = OUTPUT_DIR / "odds_game.csv"
    try:
        games = pd.read_csv(odds_game_path)
    except Exception as err:
        log.info(f"failed to read {odds_game_path}: {err}")
        games = pd.DataFrame()
    if not games.empty:
        for c in ("home_team", "away_team"):
            games[c] = games[c].apply(_canon_team)

    props["event_id"] = props["event_id"].astype(str)
    if not games.empty:
        props = props.merge(games[["event_id","home_team","away_team"]], on="event_id", how="left")
    else:
        props["home_team"] = props.get("home_team", "")
        props["away_team"] = props.get("away_team", "")

    def _infer_opp(row):
        tm = _canon_team(row.get("team", ""))
        h = _canon_team(row.get("home_team", ""))
        a = _canon_team(row.get("away_team", ""))
        if not tm or not h or not a:
            return None
        if tm == h:
            return a
        if tm == a:
            return h
        return None

    if "team" in props.columns:
        props["opp_team"] = props.apply(_infer_opp, axis=1)

    def _apply_team_fix(value: Any) -> Any:
        if isinstance(value, str):
            canon = _canon_team(value)
            if canon:
                return canon
            return value.strip().upper()
        return value

    for col in (
        "team",
        "opp_team",
        "team_abbr",
        "opponent_abbr",
        "player_team_abbr",
        "opponent_team_abbr",
        "home_team_abbr",
        "away_team_abbr",
        "home_team",
        "away_team",
    ):
        if col in props.columns:
            props[col] = props[col].apply(_apply_team_fix)

    name_map = _load_player_name_map(NAME_MAP_PATH)
    props, name_records = _canonicalize_player_names(props, name_map)
    name_records_all.extend(name_records)
    if "canonical_player_key" in props.columns:
        props["player_clean_key"] = props["canonical_player_key"]
    props = _normalize_teams_and_opponents(props)

    # --- Attach team/opponent from authoritative sources ---
    season_for_stamp = season_value if season_value is not None else _to_int_safe(os.environ.get("SEASON"))
    if season_for_stamp is None:
        try:
            season_for_stamp = _to_int_safe(season)
        except Exception:
            season_for_stamp = None

    roles = _load_roles_ourlads()
    if roles is not None:
        if "player" in props.columns:
            props["_pn_key"] = props["player"].astype(str).str.strip().str.lower()
            roles_merge = roles.rename(columns={"team": "_roles_team"})
            props = props.merge(roles_merge, on="_pn_key", how="left")
            if "_roles_team" in props.columns:
                props["_roles_team"] = props["_roles_team"].map(_canon_team)
                if "team_abbr" in props.columns:
                    props["team_abbr"] = props["team_abbr"].fillna(props["_roles_team"])
                    mask_empty = props["team_abbr"].astype(str).str.strip() == ""
                    props.loc[mask_empty, "team_abbr"] = props.loc[mask_empty, "_roles_team"]
                else:
                    props["team_abbr"] = props["_roles_team"]
                props.drop(columns=["_roles_team"], inplace=True)
        else:
            print("[fetch_props_oddsapi] WARNING: props has no 'player' column; cannot merge roles_ourlads")

    tw = _load_team_week_map(season_for_stamp) if season_for_stamp is not None else None
    if tw is not None and "team_abbr" in props.columns:
        keep = [c for c in ["season", "week", "home_abbr", "away_abbr", "event_id", "kickoff_utc"] if c in tw.columns]
        tw = tw[keep].drop_duplicates() if keep else tw.copy()

        def _mk_row(home, away, wk, eid, ko):
            home_c = _canon_team(home)
            away_c = _canon_team(away)
            return pd.DataFrame({
                "week": [wk],
                "team_abbr": [home_c],
                "opponent_abbr_tw": [away_c],
                "event_id_tw": [eid],
                "kickoff_utc_tw": [ko],
            })

        rows: list[pd.DataFrame] = []
        for _, r in tw.iterrows():
            wk = r.get("week")
            eid = r.get("event_id")
            ko = r.get("kickoff_utc")
            h, a = r.get("home_abbr"), r.get("away_abbr")
            if pd.notna(h):
                rows.append(_mk_row(h, a, wk, eid, ko))
            if pd.notna(a):
                rows.append(_mk_row(a, h, wk, eid, ko))
        if rows:
            lut = pd.concat(rows, ignore_index=True)
            merge_keys = ["team_abbr", "week"] if "week" in props.columns else ["team_abbr"]
            props = props.merge(lut, on=merge_keys, how="left")
            for col in ["opponent_abbr", "event_id", "kickoff_utc"]:
                tw_col = f"{col}_tw"
                if tw_col in props.columns:
                    if col in props.columns:
                        props[col] = props[col].fillna(props[tw_col])
                        if props[col].dtype == object:
                            empty_mask = props[col].astype(str).str.strip() == ""
                            props.loc[empty_mask, col] = props.loc[empty_mask, tw_col]
                    else:
                        props[col] = props[tw_col]
                    props.drop(columns=[tw_col], inplace=True)
            print(
                f"[fetch_props_oddsapi] Stamped opponent for {props['opponent_abbr'].notna().sum()} rows "
                f"(merge keys {merge_keys})"
            )
        else:
            print("[fetch_props_oddsapi] team_week_map produced no rows after normalization")
    else:
        print("[fetch_props_oddsapi] Skipping opponent stamping (no tw or no team_abbr)")

    if "kickoff_utc" in props.columns:
        def _to_iso_utc(val: Any):
            if pd.isna(val):
                return pd.NA
            if isinstance(val, datetime):
                dt = val if val.tzinfo else val.replace(tzinfo=timezone.utc)
            else:
                dt = pd.to_datetime(val, utc=True, errors="coerce")
                if pd.isna(dt):
                    return val
                dt = dt.to_pydatetime()
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()

        props["kickoff_utc"] = props["kickoff_utc"].apply(_to_iso_utc)

    if "market" in props.columns:
        props["market_type"] = props["market"].astype(str)
    elif "market_type" not in props.columns:
        props["market_type"] = pd.NA
    props["prop_value"] = props.get("line", pd.Series(pd.NA, index=props.index))
    props["odds"] = props.get("price_american", pd.Series(pd.NA, index=props.index))
    if "book" not in props.columns:
        props["book"] = pd.NA
    if "team_abbr" not in props.columns:
        props["team_abbr"] = pd.Series(pd.NA, index=props.index)
    if "opponent_abbr" not in props.columns:
        props["opponent_abbr"] = pd.Series(pd.NA, index=props.index)

    props["team_abbr"] = props["team_abbr"].astype(str).str.upper().str.strip()
    props["opponent_abbr"] = props["opponent_abbr"].astype(str).str.upper().str.strip().replace({"": pd.NA})

    required_cols = [
        "player",
        "team_abbr",
        "opponent_abbr",
        "market_type",
        "prop_value",
        "book",
        "odds",
    ]
    for col in required_cols:
        if col not in props.columns:
            props[col] = pd.NA
    remaining_cols = [col for col in props.columns if col not in required_cols]
    props = props[required_cols + remaining_cols]

    if "player_name" not in props.columns and "player" in props.columns:
        props["player_name"] = props["player"]
    props["player_key"] = props["player_name"].apply(
        lambda x: re.sub(r"[^a-z]", "", str(x).lower())
    )
    props["player_clean_key"] = props["player"].apply(
        lambda x: re.sub(r"[^a-z]", "", str(x).lower())
    )

    for c in ["_pn_key"]:
        if c in props.columns:
            del props[c]

    # --- Build consolidated records for props_raw.csv ---
    actual_records = _build_market_records(props, include_missing=None)
    missing_records_all: List[dict[str, Any]] = []
    for recs in market_records.values():
        missing_records_all.extend([r for r in recs if r.get("bookmaker_missing")])

    missing_df = pd.DataFrame(missing_records_all)
    if not missing_df.empty:
        missing_df, missing_name_records = _canonicalize_player_names(missing_df, name_map)
        name_records_all.extend(missing_name_records)
        missing_records_processed = missing_df.to_dict("records")
    else:
        missing_records_processed = []

    combined_records = actual_records + missing_records_processed
    final_df = pd.DataFrame(combined_records)

    if final_df.empty:
        final_df = pd.DataFrame(
            columns=[
                "event_id",
                "market",
                "book_player_name",
                "canonical_player_name",
                "canonical_player_key",
                "player_raw",
                "player_canonical",
                "team_abbr",
                "opponent_abbr",
                "bookmaker_missing",
                "offers_json",
                "source",
                "fetched_at",
            ]
        )
    else:
        canonical_series = final_df.get("canonical_player_name")
        if canonical_series is None:
            canonical_series = final_df.get("player")
        if canonical_series is None:
            canonical_series = pd.Series([""] * len(final_df), index=final_df.index)
        else:
            canonical_series = canonical_series.astype(str).fillna("")
        final_df["canonical_player_name"] = canonical_series
        final_df["player_canonical"] = final_df["canonical_player_name"]
        if "player_raw" not in final_df.columns:
            final_df["player_raw"] = final_df["canonical_player_name"]
        else:
            final_df["player_raw"] = final_df["player_raw"].astype(str).fillna("")
        if "book_player_name" not in final_df.columns:
            final_df["book_player_name"] = final_df["player_raw"]
        else:
            final_df["book_player_name"] = final_df["book_player_name"].astype(str).fillna("")
        final_df["canonical_player_key"] = final_df.get("canonical_player_key", pd.Series(pd.NA, index=final_df.index))
        final_df["canonical_player_key"] = final_df["canonical_player_key"].fillna(
            final_df["canonical_player_name"].apply(lambda x: norm_key(str(x)))
        )
        if "fetched_at" in final_df.columns:
            final_df["fetched_at"] = final_df["fetched_at"].fillna(fetched_at)
        else:
            final_df["fetched_at"] = fetched_at
        if "bookmaker_missing" in final_df.columns:
            final_df["bookmaker_missing"] = (
                final_df["bookmaker_missing"].fillna(0).astype(int)
            )
        else:
            final_df["bookmaker_missing"] = 0
        if "books_json" in final_df.columns:
            offers_series = final_df["books_json"]
        else:
            offers_series = pd.Series([[] for _ in range(len(final_df))])
        final_df["offers_json"] = offers_series.apply(
            lambda v: json.dumps(v or []) if isinstance(v, (list, tuple)) else json.dumps([])
        )
        missing_mask_final = final_df["bookmaker_missing"].astype(int) == 1
        if missing_mask_final.any():
            final_df.loc[missing_mask_final, "offers_json"] = json.dumps([])
        final_df["source"] = "oddsapi"
        if "books_json" in final_df.columns:
            final_df.drop(columns=["books_json"], inplace=True)

    # Attach team/opponent metadata to consolidated output
    if not final_df.empty:
        final_df["event_id"] = final_df["event_id"].astype(str)
        if not props.empty:
            meta_cols = [
                c
                for c in [
                    "event_id",
                    "market",
                    "player_canonical",
                    "team_abbr",
                    "opponent_abbr",
                    "home_team_abbr",
                    "away_team_abbr",
                    "commence_time",
                ]
                if c in props.columns
            ]
            meta = (
                props[meta_cols]
                .drop_duplicates(subset=["event_id", "market", "player_canonical"])
                if meta_cols
                else pd.DataFrame()
            )
            if not meta.empty:
                final_df = final_df.merge(
                    meta,
                    on=["event_id", "market", "player_canonical"],
                    how="left",
                    suffixes=("", "_meta"),
                )

        if not game_df.empty and "event_id" in final_df.columns:
            game_subset_cols = [
                c
                for c in [
                    "event_id",
                    "home_team_abbr",
                    "away_team_abbr",
                    "commence_time",
                ]
                if c in game_df.columns
            ]
            if game_subset_cols:
                game_subset = game_df[game_subset_cols].drop_duplicates("event_id")
                final_df = final_df.merge(game_subset, on="event_id", how="left")

        roster_map = _load_roster_map()
        if roster_map:
            mask_missing_team = final_df.get("team_abbr").isna() if "team_abbr" in final_df.columns else pd.Series(False)
            if "team_abbr" not in final_df.columns:
                final_df["team_abbr"] = pd.NA
                mask_missing_team = final_df["team_abbr"].isna()
            if mask_missing_team.any():
                inferred = final_df.loc[mask_missing_team].apply(
                    lambda row: _infer_player_team(row, roster_map), axis=1
                )
                final_df.loc[mask_missing_team, "team_abbr"] = inferred

        if "team_abbr" in final_df.columns:
            final_df["team_abbr"] = final_df["team_abbr"].apply(_canon_team)
        else:
            final_df["team_abbr"] = ""

        if "opponent_abbr" not in final_df.columns:
            final_df["opponent_abbr"] = ""

        def _derive_opp_from_row(row: pd.Series) -> str:
            return _infer_opponent(
                row.get("team_abbr"),
                row.get("home_team_abbr"),
                row.get("away_team_abbr"),
            )

        opp_mask = final_df["opponent_abbr"].isna() | final_df["opponent_abbr"].astype(str).str.strip().eq("")
        if opp_mask.any():
            inferred_opps = final_df.loc[opp_mask].apply(_derive_opp_from_row, axis=1)
            final_df.loc[opp_mask, "opponent_abbr"] = inferred_opps

        final_df["team_abbr"] = final_df["team_abbr"].fillna("").astype(str).str.upper()
        final_df["opponent_abbr"] = final_df["opponent_abbr"].fillna("").astype(str).str.upper()

    final_schema = [
        "event_id",
        "market",
        "book_player_name",
        "canonical_player_name",
        "canonical_player_key",
        "player_raw",
        "player_canonical",
        "team_abbr",
        "opponent_abbr",
        "bookmaker_missing",
        "offers_json",
        "source",
        "fetched_at",
    ]
    for col in final_schema:
        if col not in final_df.columns:
            final_df[col] = pd.NA
    final_df = final_df[final_schema]

    unique_players = (
        final_df["player_canonical"].dropna().nunique()
        if "player_canonical" in final_df.columns
        else 0
    )
    log.info(
        "[props] fetch finish markets=%d players=%d rows=%d",
        len(normalized_markets),
        int(unique_players),
        len(final_df),
    )

    # keep your original file AND write an enriched one for downstream robustness
    props.to_csv(OUTPUT_DIR / "props_enriched.csv", index=False)
    props.to_csv(PROPS_ENRICHED_PATH, index=False)
    # --- END: enrich props with opponent from odds_game ---

    write_atomic(out, final_df)
    log.info(f"wrote {out} rows={len(final_df)}")
    PROPS_RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(str(PROPS_RAW_DATA_PATH), final_df)
    log.info(f"wrote {PROPS_RAW_DATA_PATH} rows={len(final_df)}")

    # Write player name mapping for overrides/unresolved tracking
    PLAYER_NAME_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    NAME_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    name_map_df = pd.DataFrame(name_records_all)
    if name_map_df.empty:
        name_map_df = pd.DataFrame(
            columns=[
                "raw_name",
                "canonical_player_name",
                "canonical_player_key",
                "canonical_name",
                "source",
                "unresolved",
            ]
        )
    else:
        if "raw_name" not in name_map_df.columns and "player_raw" in name_map_df.columns:
            name_map_df["raw_name"] = name_map_df["player_raw"]
        if "canonical_player_name" not in name_map_df.columns and "player_canonical" in name_map_df.columns:
            name_map_df["canonical_player_name"] = name_map_df["player_canonical"]
        if "canonical_name" not in name_map_df.columns:
            name_map_df["canonical_name"] = name_map_df.get("canonical_player_name", "")
        name_map_df["source"] = name_map_df.get("source", "oddsapi").fillna("oddsapi")
        if "canonical_player_key" not in name_map_df.columns:
            name_map_df["canonical_player_key"] = name_map_df["canonical_player_name"].apply(lambda x: norm_key(str(x)))
        name_map_df = name_map_df.drop_duplicates(subset=["raw_name", "canonical_player_key"])

    ordered_columns = [
        "raw_name",
        "canonical_player_name",
        "canonical_player_key",
        "canonical_name",
        "source",
        "unresolved",
    ]
    extra_cols = [c for c in name_map_df.columns if c not in ordered_columns]
    name_map_df = name_map_df[ordered_columns + extra_cols]

    name_map_df.to_csv(PLAYER_NAME_LOG_PATH, index=False)
    name_map_df.to_csv(NAME_MAP_PATH, index=False)

    wide = _wide_over_under(props)
    if not wide.empty and "player" in wide.columns:
        wide["player_canonical"] = wide["player"].apply(_canonical_name_only)
    wide_out = Path(out).with_name("props_raw_wide.csv")
    wide.to_csv(wide_out, index=False)
    log.info(f"wrote {wide_out} rows={len(wide)}")

    enriched: Optional[pd.DataFrame] = None
    try:
        enriched = _write_props_enriched(props, out_game)
    except Exception as err:
        log.info(f"failed to write props_enriched.csv: {err}")
    try:
        _write_data_exports(props, enriched, game_df, season_value)
    except Exception as err:
        log.info(f"failed to write data exports: {err}")

    books_snapshot: List[str] = sorted(books_set) if books_set else []
    state["last_fetch"] = {
        "fetched_at": fetched_at,
        "date": date,
        "season": season_value,
        "markets": normalized_markets,
        "books": books_snapshot,
        "event_count": len(event_ids),
    }
    _save_state(state)

# ------------------------- CLI ----------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    logger.info("Running fetch_props_oddsapi from cwd=%s", os.getcwd())
    logger.info("Repo root resolved to %s", ROOT_DIR)

    ap = argparse.ArgumentParser()
    # Accept both names; allow empty (no filter) with nargs='?' / const=''
    ap.add_argument("--books", "--bookmakers", dest="books",
                    default="draftkings,fanduel,betmgm,caesars",
                    nargs="?", const="")
    ap.add_argument("--markets", default=",".join(MARKETS))
    ap.add_argument("--region", default=REGION_DEFAULT)
    ap.add_argument("--date", nargs="?", default="", const="")
    ap.add_argument("--season", default="")
    ap.add_argument("--out", default=str(OUTPUT_DIR / "props_raw.csv"))
    ap.add_argument("--out_game", default=str(OUTPUT_DIR / "odds_game.csv"))
    ap.add_argument(
        "--roles-csv",
        type=str,
        default=None,
        help="Optional explicit path to roles_ourlads.csv",
    )
    args = ap.parse_args()

    # IMPORTANT: interpret "" as ALL (None), not an empty list
    raw_books = (args.books or "").strip()
    books_list: Optional[List[str]] = None if raw_books == "" else [b.strip() for b in raw_books.split(",") if b.strip()]

    try:
        env_override = os.getenv("ROLES_CSV")
        roles_arg = args.roles_csv or env_override or None
        if roles_arg:
            set_roles_csv_override(roles_arg)
        roles_csv_path = _locate_roles_csv(args.roles_csv)
        logging.info("[fetch_props] Final roles CSV path resolved to %s", roles_csv_path)
        resolved_roles_path = Path(roles_csv_path)
        default_targets = [
            ROOT_DIR / "roles_ourlads.csv",
            DATA_DIR / "roles_ourlads.csv",
            OUTPUT_DIR / "roles_ourlads.csv",
        ]
        for target in default_targets:
            if target.parent != Path('.'):
                target.parent.mkdir(parents=True, exist_ok=True)
            try:
                if resolved_roles_path.resolve() == target.resolve():
                    continue
            except OSError:
                # If resolve fails for any reason, still attempt to copy
                pass
            try:
                shutil.copyfile(resolved_roles_path, target)
                logging.info(
                    "[fetch_props] Mirrored roles CSV to %s", target
                )
            except OSError as copy_exc:
                logging.warning(
                    "[fetch_props] Unable to mirror roles CSV to %s: %s",
                    target,
                    copy_exc,
                )
        roles_map = build_roles_map_from_csv(roles_csv_path)
        print(
            f"[fetch_props_oddsapi] INFO: roles_ourlads.csv loaded from {roles_csv_path} entries={len(roles_map)}"
        )
    except (FileNotFoundError, ValueError):
        print(
            "[fetch_props_oddsapi] WARNING: proceeding without roles_ourlads.csv; "
            "player name canonicalization will fall back to raw names."
        )

    fetch_odds(
        books=books_list,
        markets=[m.strip() for m in args.markets.split(",") if m.strip()],
        region=args.region,
        date=args.date,
        season=args.season,
        out=args.out,
        out_game=args.out_game,
    )
