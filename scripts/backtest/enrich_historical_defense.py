"""Recover historical defensive box and coverage context from nflverse participation.

Participation carries the tracking/charting fields that normal PBP does not. We
join participation back to PBP at play grain, aggregate completed team-week
observations, then rely on historical_context.py to use only weeks strictly before
each target week. No target-week result is used as a pregame feature.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team


def _to_pandas(value) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if hasattr(value, "to_pandas"):
        return value.to_pandas()
    return pd.DataFrame(value)


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _join_keys(participation: pd.DataFrame, pbp: pd.DataFrame) -> list[str]:
    p, b = _lower(participation), _lower(pbp)
    for keys in (["nflverse_game_id", "play_id"], ["old_game_id", "play_id"], ["game_id", "play_id"]):
        if all(k in p.columns and k in b.columns for k in keys):
            return list(keys)
    raise RuntimeError("participation/PBP have no shared game+play join key")


def _clean_label(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper().replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA})


def _man_zone_flags(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    label = _clean_label(series)
    man = label.str.contains("MAN", na=False)
    zone = label.str.contains("ZONE", na=False)
    # Some releases use simple categorical letters/words; preserve explicit labels only.
    man |= label.isin(["M"])
    zone |= label.isin(["Z"])
    return man, zone


def load_joined_participation(seasons: list[int]) -> pd.DataFrame:
    import nflreadpy as nfl

    p = _lower(_to_pandas(nfl.load_participation(seasons=seasons)))
    b = _lower(_to_pandas(nfl.load_pbp(seasons=seasons)))
    keys = _join_keys(p, b)
    required_pbp = [c for c in [*keys, "season", "week", "season_type", "defteam", "rush_attempt"] if c in b.columns]
    if not {"week", "defteam"}.issubset(required_pbp):
        raise RuntimeError("PBP missing week/defteam required for defensive reconstruction")
    if "season" not in b.columns:
        raise RuntimeError("PBP missing season required for defensive reconstruction")
    right = b[required_pbp].drop_duplicates(keys)
    joined = p.merge(right, on=keys, how="inner", suffixes=("", "_pbp"), validate="one_to_one")
    if joined.empty:
        raise RuntimeError("participation/PBP join produced zero rows")
    if "season_type" in joined.columns:
        reg = joined.loc[joined["season_type"].astype(str).str.upper().eq("REG")].copy()
        if not reg.empty:
            joined = reg
    joined["team"] = joined["defteam"].map(canon_team)
    joined = joined.loc[joined["team"].ne("")].copy()
    return joined


def build_defensive_observations_from_joined(joined: pd.DataFrame) -> pd.DataFrame:
    x = _lower(joined)
    required = {"season", "week", "team"}
    if not required.issubset(x.columns):
        raise RuntimeError(f"joined participation missing {sorted(required - set(x.columns))}")
    rows: list[dict] = []
    for (season, week, team), g in x.groupby(["season", "week", "team"]):
        rec: dict = {"season": int(season), "week": int(week), "team": canon_team(team)}

        box = pd.to_numeric(g.get("defenders_in_box"), errors="coerce") if "defenders_in_box" in g.columns else pd.Series(np.nan, index=g.index)
        rush = pd.to_numeric(g.get("rush_attempt"), errors="coerce").fillna(0).eq(1) if "rush_attempt" in g.columns else pd.Series(False, index=g.index)
        box_rush = box.loc[rush & box.notna()]
        rec["box_snap_count"] = int(len(box_rush))
        rec["avg_defenders_in_box"] = float(box_rush.mean()) if len(box_rush) else np.nan
        rec["light_box_rate"] = float(box_rush.le(6).mean()) if len(box_rush) else np.nan
        rec["heavy_box_rate"] = float(box_rush.ge(8).mean()) if len(box_rush) else np.nan
        rec["box_source"] = "nflverse_participation"

        mz = g.get("defense_man_zone_type", pd.Series(pd.NA, index=g.index, dtype="string"))
        man, zone = _man_zone_flags(mz)
        labeled = man | zone
        rec["coverage_snap_count"] = int(labeled.sum())
        rec["coverage_man_rate"] = float(man.loc[labeled].mean()) if labeled.any() else np.nan
        rec["coverage_zone_rate"] = float(zone.loc[labeled].mean()) if labeled.any() else np.nan
        rec["coverage_source"] = "nflverse_participation"

        cov = _clean_label(g.get("defense_coverage_type", pd.Series(pd.NA, index=g.index, dtype="string")))
        cov_valid = cov.notna()
        rec["coverage_family_snap_count"] = int(cov_valid.sum())
        if cov_valid.any():
            for family in ["COVER_0", "COVER_1", "COVER_2", "COVER_3", "COVER_4", "COVER_6", "COVER_9", "2_MAN"]:
                aliases = {
                    "COVER_0": ["COVER 0", "COVER-0", "COVER_0", "0"],
                    "COVER_1": ["COVER 1", "COVER-1", "COVER_1", "1"],
                    "COVER_2": ["COVER 2", "COVER-2", "COVER_2", "2"],
                    "COVER_3": ["COVER 3", "COVER-3", "COVER_3", "3"],
                    "COVER_4": ["COVER 4", "COVER-4", "COVER_4", "4"],
                    "COVER_6": ["COVER 6", "COVER-6", "COVER_6", "6"],
                    "COVER_9": ["COVER 9", "COVER-9", "COVER_9", "9"],
                    "2_MAN": ["2 MAN", "2-MAN", "2_MAN", "COVER 2 MAN"],
                }[family]
                rec[f"{family.lower()}_rate"] = float(cov.loc[cov_valid].isin(aliases).mean())
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values(["season", "week", "team"]).reset_index(drop=True)
    if out.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("historical defensive observations contain duplicate team-weeks")
    return out


def build_defensive_observations(seasons: list[int]) -> pd.DataFrame:
    return build_defensive_observations_from_joined(load_joined_participation(seasons))


def enrich_team_weekly(team_weekly: pd.DataFrame, defense: pd.DataFrame) -> pd.DataFrame:
    base = _lower(team_weekly)
    if not {"season", "week", "team"}.issubset(base.columns):
        raise RuntimeError("team_weekly_history missing season/week/team")
    base["team"] = base["team"].map(canon_team)
    enrichment_cols = [c for c in defense.columns if c not in {"season", "week", "team"}]
    base = base.drop(columns=[c for c in enrichment_cols if c in base.columns], errors="ignore")
    if defense.empty:
        return base
    return base.merge(defense, on=["season", "week", "team"], how="left", validate="one_to_one")


def audit_frame(team_weekly: pd.DataFrame) -> pd.DataFrame:
    x = _lower(team_weekly); total = int(len(x))
    light = pd.to_numeric(x.get("light_box_rate"), errors="coerce")
    heavy = pd.to_numeric(x.get("heavy_box_rate"), errors="coerce")
    man = pd.to_numeric(x.get("coverage_man_rate"), errors="coerce")
    zone = pd.to_numeric(x.get("coverage_zone_rate"), errors="coerce")
    box_n = int((light.notna() & heavy.notna()).sum())
    cov_n = int((man.notna() & zone.notna()).sum())
    return pd.DataFrame([
        {"feature":"box_rates","source":"nflverse_participation","available_team_weeks":box_n,"total_team_weeks":total,"coverage":box_n/total if total else np.nan,"status":"recovered" if box_n else "unavailable"},
        {"feature":"coverage_scheme","source":"nflverse_participation","available_team_weeks":cov_n,"total_team_weeks":total,"coverage":cov_n/total if total else np.nan,"status":"recovered" if cov_n else "unavailable"},
    ])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--team-weekly", type=Path, required=True)
    p.add_argument("--seasons", default="2024,2025")
    p.add_argument("--audit", type=Path, default=Path("data/backtests/historical_defense_enrichment_audit.csv"))
    p.add_argument("--observations", type=Path, default=Path("data/backtests/historical_defense_observations.csv"))
    args = p.parse_args()
    if not args.team_weekly.exists() or args.team_weekly.stat().st_size == 0:
        raise RuntimeError(f"missing team weekly history: {args.team_weekly}")
    seasons = [int(v.strip()) for v in args.seasons.split(",") if v.strip()]
    defense = build_defensive_observations(seasons)
    base = pd.read_csv(args.team_weekly)
    enriched = enrich_team_weekly(base, defense)
    enriched.to_csv(args.team_weekly, index=False)
    args.observations.parent.mkdir(parents=True, exist_ok=True)
    defense.to_csv(args.observations, index=False)
    audit = audit_frame(enriched)
    audit.to_csv(args.audit, index=False)
    print(f"[historical_defense] team_weeks={len(enriched)} participation_team_weeks={len(defense)}")
    print(audit.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
