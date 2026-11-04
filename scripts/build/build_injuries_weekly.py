#!/usr/bin/env python3
"""Build current-season NFL injury data with explicit provider-state semantics."""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd
import requests
from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._opponent_map import canon_team
from scripts.runtime_context import resolve_season, resolve_week

HDRS = {"User-Agent": "Mozilla/5.0 (+github.com/dkaps6/imtiredofthis)"}
DEFAULT_OUT = Path("data") / "injuries.csv"
STATUS_OUT = Path("data") / "injuries_source_status.json"
SCHEMA = [
    "player", "team", "season", "week", "status", "practice_status",
    "body_part", "designation", "report_date", "source", "report_available",
]


def _empty_injuries() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in SCHEMA})


def _to_pandas(value) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if hasattr(value, "to_pandas"):
        try:
            return value.to_pandas()
        except Exception:
            pass
    if hasattr(value, "to_dicts"):
        return pd.DataFrame(value.to_dicts())
    return pd.DataFrame(value)


def _first(df: pd.DataFrame, names: list[str], default=pd.NA) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series(default, index=df.index)


def normalize_nflverse_injuries(raw: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Normalize nflverse injury rows to the repository injury contract."""
    if raw is None or raw.empty:
        return _empty_injuries()

    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "season" in df.columns:
        df = df.loc[pd.to_numeric(df["season"], errors="coerce").eq(int(season))].copy()
    if df.empty:
        return _empty_injuries()

    # nflverse injury data is weekly. Prefer the requested active week when it
    # exists, but never silently substitute a previous week's report.
    week_col = next((c for c in ("week", "report_week") if c in df.columns), None)
    if week_col is not None:
        wk = pd.to_numeric(df[week_col], errors="coerce")
        active = df.loc[wk.eq(int(week))].copy()
        if active.empty:
            return _empty_injuries()
        df = active

    out = pd.DataFrame(index=df.index)
    out["player"] = _first(df, ["full_name", "player_name", "player", "name"]).astype("string").str.strip()
    out["team"] = _first(df, ["team", "team_abbr", "team_abbreviation", "club"]).map(canon_team)
    out["season"] = int(season)
    out["week"] = int(week)
    out["status"] = _first(df, ["report_status", "game_status", "status"]).astype("string").str.strip()
    out["practice_status"] = _first(df, ["practice_status", "practice_participation"]).astype("string").str.strip()
    injury = _first(df, ["report_primary_injury", "primary_injury", "injury", "injury_type"]).astype("string").str.strip()
    out["body_part"] = injury.replace({"": pd.NA, "<NA>": pd.NA}).fillna("Unknown")
    out["designation"] = out["status"]
    out["report_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out["source"] = "nflverse"
    out["report_available"] = 1

    out = out.loc[out["player"].notna() & out["player"].ne("")].copy()
    out = out.loc[out["team"].notna() & out["team"].ne("")].copy()
    return out[SCHEMA].drop_duplicates(["player", "team"], keep="last").reset_index(drop=True)


def load_nflverse_injuries(season: int, week: int) -> pd.DataFrame:
    import nflreadpy as nfl

    raw = nfl.load_injuries(seasons=[int(season)])
    return normalize_nflverse_injuries(_to_pandas(raw), int(season), int(week))


def fetch_injuries_html(url: str = "https://www.nfl.com/injuries/") -> str:
    response = requests.get(url, headers=HDRS, timeout=45)
    response.raise_for_status()
    return response.text


def detect_current_week_from_html(html: str) -> int:
    match = re.search(r"(?:WEEK|Week)\s+(\d+)", html)
    if not match:
        raise RuntimeError("Unable to detect current week from NFL.com injuries page.")
    return int(match.group(1))


def extract_injury_tables(soup: BeautifulSoup) -> List[pd.DataFrame]:
    tables: list[pd.DataFrame] = []
    for table in soup.find_all("table"):
        try:
            parsed = pd.read_html(io.StringIO(str(table)))[0]
        except ValueError:
            continue
        cols = {str(c).strip().lower() for c in parsed.columns}
        if any("player" in c for c in cols) and any("status" in c for c in cols):
            tables.append(parsed)
    return tables


def normalize_nflcom_dataframe(df: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_injuries()
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if "player" in key:
            rename[col] = "player"
        elif "team" in key or "club" in key:
            rename[col] = "team_raw"
        elif "game status" in key or key == "status":
            rename[col] = "status"
        elif "practice" in key:
            rename[col] = "practice_status"
        elif "injur" in key:
            rename[col] = "injury"
    x = df.rename(columns=rename).copy()
    if "player" not in x.columns:
        return _empty_injuries()
    out = pd.DataFrame(index=x.index)
    out["player"] = x["player"].astype("string").str.strip()
    out["team"] = x.get("team_raw", pd.Series(pd.NA, index=x.index)).map(canon_team)
    out["season"] = int(season)
    out["week"] = int(week)
    out["status"] = x.get("status", pd.Series(pd.NA, index=x.index)).astype("string").str.strip()
    out["practice_status"] = x.get("practice_status", pd.Series(pd.NA, index=x.index)).astype("string").str.strip()
    out["body_part"] = x.get("injury", pd.Series("Unknown", index=x.index)).astype("string").str.strip().replace("", "Unknown")
    out["designation"] = out["status"]
    out["report_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out["source"] = "nfl.com"
    out["report_available"] = 1
    out = out.loc[out["player"].notna() & out["player"].ne("") & out["team"].notna() & out["team"].ne("")]
    return out[SCHEMA].drop_duplicates(["player", "team"], keep="last").reset_index(drop=True)


def load_nflcom_fallback(season: int, week: int) -> pd.DataFrame:
    html = fetch_injuries_html()
    detected = detect_current_week_from_html(html)
    if detected != int(week):
        return _empty_injuries()
    tables = extract_injury_tables(BeautifulSoup(html, "html.parser"))
    if not tables:
        return _empty_injuries()
    return normalize_nflcom_dataframe(pd.concat(tables, ignore_index=True), season, week)


def build_injuries_with_status(season: int, week: int) -> tuple[pd.DataFrame, dict]:
    """Return injury rows plus an explicit provider-health state.

    An empty official report is valid before weekly reports exist, but it is not
    equivalent to a provider outage. At least one provider must have completed a
    successful check before ``no_official_report`` can be emitted.
    """
    errors: list[str] = []
    successful_checks: list[str] = []

    try:
        df = load_nflverse_injuries(season, week)
        successful_checks.append("nflverse")
        if not df.empty:
            status = {
                "season": int(season), "week": int(week), "state": "official_report",
                "source": "nflverse", "rows": int(len(df)),
                "successful_checks": successful_checks, "provider_errors": errors,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            return df, status
    except Exception as exc:
        errors.append(f"nflverse={type(exc).__name__}: {exc}")

    try:
        df = load_nflcom_fallback(season, week)
        successful_checks.append("nfl.com")
        if not df.empty:
            status = {
                "season": int(season), "week": int(week), "state": "official_report",
                "source": "nfl.com", "rows": int(len(df)),
                "successful_checks": successful_checks, "provider_errors": errors,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            return df, status
    except Exception as exc:
        errors.append(f"nfl.com={type(exc).__name__}: {exc}")

    if successful_checks:
        state = "no_official_report"
        source = "no_official_report"
    else:
        state = "provider_outage"
        source = "provider_outage"

    status = {
        "season": int(season), "week": int(week), "state": state,
        "source": source, "rows": 0,
        "successful_checks": successful_checks, "provider_errors": errors,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    print(
        f"[injuries_v3] season={season} week={week} state={state}; "
        f"successful_checks={successful_checks or ['none']} provider_errors={errors or ['none']}"
    )
    return _empty_injuries(), status


def build_injuries(season: int, week: int) -> tuple[pd.DataFrame, str]:
    """Backwards-compatible two-value API used by existing tests/callers."""
    df, status = build_injuries_with_status(season, week)
    return df, str(status["source"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_csv", nargs="?", default=str(DEFAULT_OUT))
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None)
    args = parser.parse_args()

    season = int(args.season) if args.season is not None else int(resolve_season())
    week = int(args.week) if args.week is not None else int(resolve_week())
    df, status = build_injuries_with_status(season, week)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    STATUS_OUT.parent.mkdir(parents=True, exist_ok=True)
    STATUS_OUT.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[injuries_v3] wrote {len(df)} rows -> {out}; season={season} week={week} "
        f"state={status['state']} source={status['source']} status={STATUS_OUT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
