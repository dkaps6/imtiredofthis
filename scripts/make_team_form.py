#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build metrics/team_form.csv with 2025-only rows.
Adds:
- strict 2025 clamp
- plays_est from neutral pace
- PROE (approx if true PROE not available)
Keeps all other columns if they already exist (pass-through).
"""

import argparse
import os
import sys
from typing import Optional

import pandas as pd

TEAM_CODES_2025 = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAC", "KC",
    "LV", "LAC", "LAR", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
]

# Optional: nfl_data_py as a fallback when you want to compute base fields locally.
try:
    import nfl_data_py as nfl
except Exception:
    nfl = None


OUT_DIR = "metrics"
OUT_FILE = os.path.join(OUT_DIR, "team_form.csv")
# Keep a compatibility copy in data/ so downstream joins (make_metrics, engine snapshot)
# continue to find the table at the historical location.
DATA_DIR = "data"
DATA_OUT_FILE = os.path.join(DATA_DIR, "team_form.csv")


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _read_existing_inputs() -> Optional[pd.DataFrame]:
    """
    If you already generate a base team metrics CSV elsewhere (e.g., providers/enrich),
    load it here and enrich; otherwise return None and fall back to building minimal columns.
    """
    candidates = [
        "metrics/team_form_base.csv",
        "metrics/team_form_raw.csv",
        "data/team_form.csv",  # legacy path
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                df.columns = [c.strip() for c in df.columns]
                return df
            except Exception:
                pass
    return None


def _fallback_from_nfl_data_py(season: int) -> pd.DataFrame:
    """
    Minimal build if you don’t already have an upstream file.
    Produces a row per team-season with generic placeholders where needed.
    """
    if nfl is None:
        base = pd.DataFrame({"team": TEAM_CODES_2025})
        base["season"] = int(season)
        for col in [
            "def_pressure_rate", "def_sack_rate",
            "def_pass_epa", "def_rush_epa",
            "pace_neutral",
            "light_box_rate", "heavy_box_rate",
            "ay_per_att",
            "pass_rate_neutral",
        ]:
            base[col] = pd.NA
        base["pace_neutral"] = 27.5
        base["pass_rate_neutral"] = 0.55
        print("[make_team_form] ⚠️ nfl_data_py unavailable; using static team shell")
        return base

    teams = pd.DataFrame()
    schedule = pd.DataFrame()

    try:
        teams = nfl.import_team_desc()
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[make_team_form] ⚠️ import_team_desc failed: {exc}")

    try:
        schedule = nfl.import_schedules([season])
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[make_team_form] ⚠️ import_schedules failed: {exc}")

    # Derive simple team list for the season
    teams_played = []
    if not schedule.empty:
        teams_played = pd.unique(
            pd.concat([schedule["home_team"], schedule["away_team"]], ignore_index=True)
        ).tolist()
    elif not teams.empty and "team_abbr" in teams.columns:
        teams_played = teams["team_abbr"].dropna().astype(str).str.upper().unique().tolist()
    if not teams_played:
        teams_played = TEAM_CODES_2025
        print("[make_team_form] ⚠️ using static 2025 team list fallback")

    base = pd.DataFrame({"team": teams_played})
    base["season"] = int(season)

    # These next bits would be replaced by your richer pipelines;
    # we keep placeholders to avoid breaking merges downstream.
    base["def_pressure_rate"] = pd.NA
    base["def_sack_rate"] = pd.NA
    base["def_pass_epa"] = pd.NA
    base["def_rush_epa"] = pd.NA
    base["pace_neutral"] = 27.5  # league-ish default seconds/snap; refined later if you have it
    base["light_box_rate"] = pd.NA
    base["heavy_box_rate"] = pd.NA
    base["ay_per_att"] = pd.NA
    base["pass_rate_neutral"] = 0.55  # neutral pass rate default

    return base


def _add_plays_est_and_proe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- plays_est from neutral pace ---
    league_secs_per_snap = 27.5
    base_team_plays = 77.5
    pace_col = None
    for cand in ("pace_neutral", "pace", "neutral_pace_secs_per_snap"):
        if cand in df.columns:
            pace_col = cand
            break
    if pace_col is not None:
        df["plays_est"] = (
            base_team_plays * (league_secs_per_snap / df[pace_col].astype(float).clip(lower=20, upper=40))
        ).round(1)
    else:
        df["plays_est"] = base_team_plays

    # --- PROE (do not overwrite if already present) ---
    if "proe" not in df.columns:
        neutral_pass_col = None
        for cand in ("pass_rate_neutral", "neutral_pass_rate", "pr_neutral"):
            if cand in df.columns:
                neutral_pass_col = cand
                break

        if neutral_pass_col is not None:
            league_mean = float(pd.to_numeric(df[neutral_pass_col], errors="coerce").mean())
            df["proe"] = (pd.to_numeric(df[neutral_pass_col], errors="coerce") - league_mean).round(4)
        else:
            df["proe"] = 0.0

    return df


def _zscore_cols(df: pd.DataFrame, cols) -> pd.DataFrame:
    """
    If you rely on *_z columns later (e.g., pricing), create them when raw columns are present.
    We never remove existing columns—only add.
    """
    df = df.copy()
    for c in cols:
        if c in df.columns:
            series = pd.to_numeric(df[c], errors="coerce")
            mu = series.mean()
            sd = series.std(ddof=0)
            if sd and sd > 0:
                df[f"{c}_z"] = (series - mu) / sd
            else:
                df[f"{c}_z"] = 0.0
    return df


def build_team_form(season: int, strict: bool):
    _ensure_dir(OUT_DIR)
    _ensure_dir(DATA_DIR)

    df = _read_existing_inputs()
    if df is None:
        df = _fallback_from_nfl_data_py(season)

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Hard clamp to 2025 (season param still passed for clarity)
    if "season" in df.columns:
        df = df[df["season"].astype(int) == 2025].copy()
    else:
        df["season"] = int(2025)

    # If team column uses a different name, try to standardize
    if "team" not in df.columns:
        for alt in ("club_code", "abbr", "team_abbr", "team_name"):
            if alt in df.columns:
                df = df.rename(columns={alt: "team"})
                break

    df = _add_plays_est_and_proe(df)

    # Add z-scores for downstream usage if raw present
    z_cols = [
        "def_pressure_rate",
        "def_sack_rate",
        "def_pass_epa",
        "def_rush_epa",
        "light_box_rate",
        "heavy_box_rate",
        "ay_per_att",
        "pace_neutral",
    ]
    df = _zscore_cols(df, z_cols)

    # Persist (do not drop any extra columns user already had)
    df.to_csv(OUT_FILE, index=False)
    df.to_csv(DATA_OUT_FILE, index=False)
    print(
        f"[make_team_form] wrote {OUT_FILE} and {DATA_OUT_FILE} rows={len(df)}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--strict", action="store_true", help="Kept for compatibility; always clamps to 2025.")
    args = parser.parse_args()

    try:
        build_team_form(season=args.season, strict=args.strict)
    except Exception as e:
        print(f"[make_team_form] ERROR: {e}", file=sys.stderr)
        # keep CI green-ish; re-raise to signal failure if you're running strict jobs
        raise


if __name__ == "__main__":
    main()
