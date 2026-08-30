#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the current NFL injury context.

2026 production contract:
- Primary live source: ESPN's public league-wide NFL injury JSON.
- nflverse/nfl_data_py are historical fallbacks only for seasons <= 2024.
- NEVER silently keep an old injuries.csv when the current fetch fails.
- On live-source failure, write an explicit unavailable schema + source manifest
  so downstream code can fail softly without treating stale injuries as current.

Writes:
  data/injuries.csv
  data/injuries_source_status.json
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime, timezone

import pandas as pd
import requests

DATA_DIR = "data"
OUT = os.path.join(DATA_DIR, "injuries.csv")
STATUS_OUT = os.path.join(DATA_DIR, "injuries_source_status.json")
ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
COLS = [
    "player", "team", "status", "practice_status", "body_part", "designation",
    "source", "report_available",
]


def _normalize_team_value(value) -> str:
    aliases = {
        "WSH": "WAS", "WDC": "WAS", "JAC": "JAX", "ARZ": "ARI",
        "LA": "LAR", "LVR": "LV", "OAK": "LV", "SFO": "SF",
        "TAM": "TB", "GBP": "GB", "KAN": "KC",
    }
    s = str(value or "").upper().strip()
    return aliases.get(s, s)


def _normalize_player_value(value) -> str:
    s = str(value or "").replace(".", "").strip()
    return pd.Series([s]).str.replace(r"\s+(JR|SR|II|III|IV|V)\.?$", "", regex=True).str.replace(r"\s+", " ", regex=True).iloc[0].strip()


def _empty(source: str = "unavailable") -> pd.DataFrame:
    return pd.DataFrame(columns=COLS).assign(source=pd.Series(dtype="string"), report_available=pd.Series(dtype="Int64"))


def _first(mapping, names, default=""):
    if not isinstance(mapping, dict):
        return default
    for name in names:
        value = mapping.get(name)
        if value is not None and str(value).strip() not in {"", "nan", "None"}:
            return value
    return default


def load_injuries_espn() -> pd.DataFrame:
    """Load the current league-wide ESPN injury report without authentication."""
    headers = {
        "User-Agent": "imtiredofthis/2.0 (NFL injury context)",
        "Accept": "application/json",
    }
    r = requests.get(ESPN_URL, headers=headers, timeout=30)
    r.raise_for_status()
    js = r.json()
    groups = js.get("injuries", []) if isinstance(js, dict) else []
    rows = []
    for group in groups:
        team_obj = group.get("team", {}) if isinstance(group, dict) else {}
        team = _normalize_team_value(_first(team_obj, ["abbreviation", "shortDisplayName", "name"]))
        for item in group.get("injuries", []) if isinstance(group, dict) else []:
            athlete = item.get("athlete", {}) if isinstance(item, dict) else {}
            details = item.get("details", {}) if isinstance(item, dict) else {}
            player = _normalize_player_value(_first(athlete, ["fullName", "displayName", "shortName"]))
            if not player or not team:
                continue
            status = _first(item, ["status", "type", "description"], "")
            if isinstance(status, dict):
                status = _first(status, ["name", "description", "abbreviation"], "")
            detail_type = _first(details, ["type", "detail", "fantasyStatus", "returnDate"], "")
            practice = _first(details, ["practiceStatus", "practice", "practiceParticipation"], "")
            body = _first(details, ["location", "bodyPart", "injury", "detail"], "")
            designation = _first(details, ["fantasyStatus", "type"], status)
            rows.append({
                "player": player,
                "team": team,
                "status": str(status or detail_type).strip().title(),
                "practice_status": str(practice).strip().title(),
                "body_part": str(body).strip(),
                "designation": str(designation).strip().title(),
                "source": "espn_live_injuries",
                "report_available": 1,
            })
    out = pd.DataFrame(rows, columns=COLS)
    if out.empty:
        raise RuntimeError("ESPN injury endpoint returned zero usable player rows")
    return out.drop_duplicates(["player", "team"], keep="last").reset_index(drop=True)


def _to_pandas(value) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if hasattr(value, "to_pandas"):
        return value.to_pandas()
    return pd.DataFrame(value)


def _normalize_historical(raw: pd.DataFrame, source: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return _empty(source)
    x = raw.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    name_col = next((c for c in ["full_name", "player_name", "player", "name"] if c in x.columns), None)
    team_col = next((c for c in ["team", "team_abbr", "team_abbreviation", "club"] if c in x.columns), None)
    if not name_col or not team_col:
        return _empty(source)
    status_col = next((c for c in ["report_status", "game_status", "injury_status", "status"] if c in x.columns), None)
    practice_col = next((c for c in ["practice_status", "practice_participation"] if c in x.columns), None)
    body_col = next((c for c in ["report_primary_injury", "primary_injury", "injury", "injury_type"] if c in x.columns), None)
    out = pd.DataFrame({
        "player": x[name_col].map(_normalize_player_value),
        "team": x[team_col].map(_normalize_team_value),
        "status": x[status_col].astype(str).str.title() if status_col else "",
        "practice_status": x[practice_col].astype(str).str.title() if practice_col else "",
        "body_part": x[body_col].astype(str) if body_col else "",
    })
    out["designation"] = out["status"]
    out["source"] = source
    out["report_available"] = 1
    out = out.loc[out["player"].ne("") & out["team"].ne("")].copy()
    return out[COLS].drop_duplicates(["player", "team"], keep="last").reset_index(drop=True)


def load_historical_nflverse(season: int) -> tuple[pd.DataFrame, str]:
    errors = []
    try:
        import nflreadpy as nflv
        raw = nflv.load_injuries(seasons=[int(season)])
        out = _normalize_historical(_to_pandas(raw), "nflverse_historical")
        if not out.empty:
            return out, "nflverse_historical"
        errors.append("nflreadpy returned zero usable rows")
    except Exception as exc:
        errors.append(f"nflreadpy: {exc}")
    try:
        import nfl_data_py as nfld
        raw = nfld.import_injuries([int(season)])  # type: ignore[attr-defined]
        out = _normalize_historical(_to_pandas(raw), "nfl_data_py_historical")
        if not out.empty:
            return out, "nfl_data_py_historical"
        errors.append("nfl_data_py returned zero usable rows")
    except Exception as exc:
        errors.append(f"nfl_data_py: {exc}")
    raise RuntimeError(" | ".join(errors))


def _write_status(*, season: int, source: str, available: bool, rows: int, errors: list[str]) -> None:
    payload = {
        "season": int(season),
        "source": source,
        "available": bool(available),
        "rows": int(rows),
        "stale_cache_reused": False,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "errors": errors,
    }
    with open(STATUS_OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    warnings.simplefilter("ignore")
    os.makedirs(DATA_DIR, exist_ok=True)
    season = int(os.environ.get("SEASON", str(datetime.now(timezone.utc).year)))
    errors: list[str] = []

    try:
        df = load_injuries_espn()
        source = "espn_live_injuries"
    except Exception as exc:
        errors.append(f"espn_live_injuries: {exc}")
        df = _empty()
        source = "unavailable"

    # The nflverse injury feed is a historical fallback only. It is not allowed
    # to masquerade as a current 2025+ live feed.
    if df.empty and season <= 2024:
        try:
            df, source = load_historical_nflverse(season)
        except Exception as exc:
            errors.append(f"historical_fallback: {exc}")

    if not df.empty:
        df["player"] = df["player"].map(_normalize_player_value)
        df["team"] = df["team"].map(_normalize_team_value)
        df = df.loc[df["player"].ne("") & df["team"].ne("")].copy()
        df = df[COLS].drop_duplicates(["player", "team"], keep="last")
        df.to_csv(OUT, index=False)
        _write_status(season=season, source=source, available=True, rows=len(df), errors=errors)
        print(f"[injuries] wrote {len(df)} current rows from {source} -> {OUT}")
        return 0

    # Fail softly but explicitly: never preserve stale prior-week/current-season data.
    unavailable = _empty()
    unavailable.to_csv(OUT, index=False)
    _write_status(season=season, source="unavailable", available=False, rows=0, errors=errors)
    print("[injuries] CURRENT INJURY SOURCE UNAVAILABLE; wrote explicit empty context (stale cache NOT reused)", file=sys.stderr)
    for err in errors:
        print(f"[injuries] {err}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
