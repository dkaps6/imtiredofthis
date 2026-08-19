"""Recover defensible historical team-level defensive context from nflverse PBP.

Migration 13 intentionally recovers only fields directly supported by historical
play-by-play. `defenders_in_box` is observed on completed plays, so each team-week
observation can safely be stored and later lagged by historical_context.py before
a target week. nflverse PBP does not provide a trustworthy man/zone coverage label,
so coverage scheme remains explicitly unavailable rather than inferred from results.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.utils.pbp import get_pbp


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def build_box_observations(seasons: list[int]) -> pd.DataFrame:
    rows: list[dict] = []
    for season in sorted(set(int(s) for s in seasons)):
        pbp = _lower(get_pbp(season, min_rows=1))
        if "season_type" in pbp.columns:
            reg = pbp.loc[pbp["season_type"].astype(str).str.upper().eq("REG")].copy()
            if not reg.empty:
                pbp = reg
        required = {"week", "defteam", "rush_attempt", "defenders_in_box"}
        missing = required - set(pbp.columns)
        if missing:
            print(f"[historical_defense] season={season} box unavailable missing={sorted(missing)}")
            continue
        pbp["defteam"] = pbp["defteam"].map(canon_team)
        pbp["rush_attempt"] = pd.to_numeric(pbp["rush_attempt"], errors="coerce")
        pbp["defenders_in_box"] = pd.to_numeric(pbp["defenders_in_box"], errors="coerce")
        rush = pbp.loc[
            pbp["rush_attempt"].fillna(0).eq(1)
            & pbp["defteam"].ne("")
            & pbp["defenders_in_box"].notna()
        ].copy()
        for (week, team), g in rush.groupby(["week", "defteam"]):
            box = g["defenders_in_box"]
            rows.append({
                "season": int(season),
                "week": int(week),
                "team": canon_team(team),
                "box_snap_count": int(box.notna().sum()),
                "avg_defenders_in_box": float(box.mean()),
                "light_box_rate": float(box.le(6).mean()),
                "heavy_box_rate": float(box.ge(8).mean()),
                "box_source": "nflverse_pbp_defenders_in_box",
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["season", "week", "team"]).reset_index(drop=True)
        if out.duplicated(["season", "week", "team"]).any():
            raise RuntimeError("historical box observations contain duplicate team-weeks")
    return out


def enrich_team_weekly(team_weekly: pd.DataFrame, box: pd.DataFrame) -> pd.DataFrame:
    base = _lower(team_weekly)
    if not {"season", "week", "team"}.issubset(base.columns):
        raise RuntimeError("team_weekly_history missing season/week/team")
    base["team"] = base["team"].map(canon_team)
    for c in ["light_box_rate", "heavy_box_rate", "avg_defenders_in_box", "box_snap_count", "box_source"]:
        if c in base.columns:
            base = base.drop(columns=[c])
    if box.empty:
        base["light_box_rate"] = np.nan
        base["heavy_box_rate"] = np.nan
        base["avg_defenders_in_box"] = np.nan
        base["box_snap_count"] = np.nan
        base["box_source"] = ""
        return base
    return base.merge(box, on=["season", "week", "team"], how="left", validate="one_to_one")


def audit_frame(team_weekly: pd.DataFrame) -> pd.DataFrame:
    total = int(len(team_weekly))
    light = pd.to_numeric(team_weekly.get("light_box_rate"), errors="coerce")
    heavy = pd.to_numeric(team_weekly.get("heavy_box_rate"), errors="coerce")
    available = int((light.notna() & heavy.notna()).sum())
    return pd.DataFrame([
        {"feature": "box_rates", "source": "nflverse_pbp_defenders_in_box", "available_team_weeks": available, "total_team_weeks": total, "coverage": available / total if total else np.nan, "status": "recovered" if available else "unavailable"},
        {"feature": "coverage_scheme", "source": "nflverse_pbp", "available_team_weeks": 0, "total_team_weeks": total, "coverage": 0.0 if total else np.nan, "status": "unsupported_no_trustworthy_man_zone_label"},
    ])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--team-weekly", type=Path, required=True)
    p.add_argument("--seasons", default="2024,2025")
    p.add_argument("--audit", type=Path, default=Path("data/backtests/historical_defense_enrichment_audit.csv"))
    args = p.parse_args()
    if not args.team_weekly.exists() or args.team_weekly.stat().st_size == 0:
        raise RuntimeError(f"missing team weekly history: {args.team_weekly}")
    seasons = [int(x.strip()) for x in str(args.seasons).split(",") if x.strip()]
    base = pd.read_csv(args.team_weekly)
    box = build_box_observations(seasons)
    enriched = enrich_team_weekly(base, box)
    enriched.to_csv(args.team_weekly, index=False)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    audit = audit_frame(enriched)
    audit.to_csv(args.audit, index=False)
    print(f"[historical_defense] team_weeks={len(enriched)} box_rows={len(box)}")
    print(audit.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
