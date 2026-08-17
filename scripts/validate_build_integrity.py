"""Core build validation used by make_metrics and CI.

Validation is season-aware and uses the same artifact contracts as the rest of
the production pipeline.  It never assumes season 2025 and never references the
obsolete data/opponent_map.csv path.
"""
from __future__ import annotations

import os
from datetime import datetime

import pandas as pd

from scripts.artifact_contracts import CONTRACTS
from scripts.runtime_context import resolve_season

DATA_PATH = "data"
LOG_PATH = os.path.join(DATA_PATH, "logs")
os.makedirs(LOG_PATH, exist_ok=True)
LOG_FILE = os.path.join(LOG_PATH, "validate_build.log")

CORE_AT_METRICS = (
    "roles_ourlads",
    "team_week_map",
    "opponent_map",
    "props_raw",
    "team_form",
    "player_game_logs",
    "player_form",
    "player_form_consensus",
    "metrics_ready",
)


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[VALIDATE] {ts} | {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _validate_contract(key: str) -> bool:
    contract = CONTRACTS[key]
    path = contract.path
    if not path.exists():
        log(f"❌ Missing file: {path}")
        return False
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        log(f"❌ Could not read {path}: {exc}")
        return False
    if len(df) < contract.min_rows:
        log(f"❌ {path} has {len(df)} rows; expected >= {contract.min_rows}")
        return False
    missing = [c for c in contract.required_columns if c not in df.columns]
    if missing:
        log(f"❌ {path} missing required columns: {missing}")
        return False
    log(f"✅ {path} validated ({len(df)} rows, {len(df.columns)} cols)")
    return True


def _validate_season(path, season: int, *, require_active_rows: bool) -> bool:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        log(f"❌ Could not read {path} for season validation: {exc}")
        return False
    if "season" not in df.columns:
        return True
    s = pd.to_numeric(df["season"], errors="coerce")
    if require_active_rows and not s.eq(int(season)).any():
        log(f"❌ {path} contains no rows for active season {season}")
        return False
    return True


def _validate_opponents() -> bool:
    path = CONTRACTS["team_week_map"].path
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    if not {"season", "week", "team", "opponent"}.issubset(df.columns):
        return False
    season = resolve_season()
    s = pd.to_numeric(df["season"], errors="coerce")
    scoped = df.loc[s.eq(season)].copy()
    scoped["team"] = scoped["team"].astype(str).str.upper().str.strip()
    scoped["opponent"] = scoped["opponent"].astype(str).str.upper().str.strip()
    games = scoped.loc[~scoped["opponent"].isin(["", "NAN", "NONE", "BYE"])]
    lookup = {(int(r.week), r.team): r.opponent for r in games.itertuples() if pd.notna(r.week)}
    bad = []
    for (week, team), opp in lookup.items():
        reverse = lookup.get((week, opp))
        if reverse != team:
            bad.append((week, team, opp, reverse))
    if bad:
        log(f"❌ team_week_map has {len(bad)} non-bidirectional matchup rows; sample={bad[:5]}")
        return False
    return True


def run_core_validation() -> bool:
    log("=" * 60)
    log("STARTING BUILD VALIDATION")
    log("=" * 60)
    passed = True
    for key in CORE_AT_METRICS:
        passed = _validate_contract(key) and passed

    season = resolve_season()
    # Current-slate tables must carry active-season rows. Historical game logs
    # intentionally may contain both prior and current seasons.
    for key in ("team_form", "player_form", "player_form_consensus", "metrics_ready"):
        passed = _validate_season(CONTRACTS[key].path, season, require_active_rows=True) and passed

    passed = _validate_opponents() and passed

    mm = CONTRACTS["metrics_ready"].path
    if mm.exists():
        try:
            metrics = pd.read_csv(mm)
            if "player" in metrics.columns:
                missing = int(metrics["player"].isna().sum())
                if missing:
                    log(f"❌ {missing} missing player names in metrics_ready.csv")
                    passed = False
        except Exception as exc:
            log(f"❌ Unable to inspect metrics_ready player names: {exc}")
            passed = False

    # Optional enrichments are informative, not contradictory hard failures.
    for key, contract in CONTRACTS.items():
        if contract.required or key in CORE_AT_METRICS:
            continue
        if not contract.path.exists() or contract.path.stat().st_size == 0:
            log(f"⚠️ Optional artifact missing/empty: {contract.path}")

    log("=" * 60)
    log("✅ VALIDATION PASSED SUCCESSFULLY" if passed else "❌ VALIDATION FOUND ISSUES — BUILD WILL FAIL")
    log("=" * 60)
    return passed


if __name__ == "__main__":
    raise SystemExit(0 if run_core_validation() else 1)
