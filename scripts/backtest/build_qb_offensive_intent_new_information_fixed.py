#!/usr/bin/env python3
"""Execution-safe M67 source builder.

Changes no M67 feature definition, family, model, gate, or historical boundary.
It only works around nflreadpy injury loading/alignment issues by loading injury
reports one season at a time, attaching requested-season provenance, and using
that provenance when constructing team-week injury features.
"""
from __future__ import annotations

import pandas as pd

import scripts.backtest.build_qb_offensive_intent_new_information as m


def load_sources_fixed(seasons: list[int]):
    import nflreadpy as nfl

    pbp = m.lower(m.to_pd(nfl.load_pbp(seasons=seasons)))
    part = m.lower(m.to_pd(nfl.load_participation(seasons=seasons)))

    injury_frames: list[pd.DataFrame] = []
    injury_errors: list[str] = []
    for season in sorted(set(int(s) for s in seasons)):
        try:
            q = m.lower(m.to_pd(nfl.load_injuries(seasons=[season])))
            if not q.empty:
                # The season requested from the season-scoped nflverse asset is
                # authoritative provenance for M67 alignment.  Preserve the raw
                # value for diagnostics, but do not let provider/schema quirks
                # silently relabel an entire target season.
                q["_provider_season_raw"] = q.get("season", pd.Series(pd.NA, index=q.index))
                q["_requested_season"] = int(season)
                raw_vals = sorted(pd.to_numeric(q["_provider_season_raw"], errors="coerce").dropna().astype(int).unique().tolist())
                injury_frames.append(q)
                print(f"[M67 fixed] injury requested_season={season} rows={len(q)} raw_seasons={raw_vals}")
            else:
                injury_errors.append(f"{season}: empty")
        except Exception as exc:
            injury_errors.append(f"{season}: {exc}")

    inj = pd.concat(injury_frames, ignore_index=True, sort=False) if injury_frames else pd.DataFrame()
    injury_ok = bool(injury_frames)
    if injury_errors:
        print("[M67 fixed] injury load notes: " + " | ".join(injury_errors))

    manifest = [
        {
            "family": "pbp_intent",
            "source": "nflverse_pbp",
            "status": "recovered" if len(pbp) else "unavailable",
            "live_2026_capability": "live_in_season",
            "notes": "rolling prior-game intent; no target-game plays used",
        },
        {
            "family": "formation_personnel_continuity",
            "source": "nflverse_participation",
            "status": "recovered" if len(part) else "unavailable",
            "live_2026_capability": "historical_only_postseason_release_2023plus",
            "notes": "diagnostic until equivalent live source is acquired",
        },
        {
            "family": "injury_availability",
            "source": "nflverse_injuries",
            "status": "recovered" if injury_ok else "unavailable",
            "live_2026_capability": "live_pregame",
            "notes": "official report status/practice fields by offensive position; season-scoped loader with requested-season provenance",
        },
        {
            "family": "actual_playcaller",
            "source": "not_in_current_nflverse_stack",
            "status": "not_recovered_m67",
            "live_2026_capability": "requires_new_source",
            "notes": "preserved as next new-information family",
        },
        {
            "family": "true_playoff_leverage",
            "source": "not_built_m67",
            "status": "not_recovered_m67",
            "live_2026_capability": "requires_new_derived_source",
            "notes": "preserved after offensive-intent audit",
        },
    ]
    return pbp, part, inj, manifest


def injury_team_week_fixed(inj: pd.DataFrame) -> pd.DataFrame:
    """Frozen M67 injury aggregation with execution-safe season attribution."""
    if inj.empty:
        return m.injury_team_week(inj)
    x = inj.copy()
    if "_requested_season" in x.columns:
        # Preserve the requested season from the season-specific provider asset.
        # All downstream feature definitions remain exactly those in the frozen
        # M67 injury_team_week implementation.
        x["season"] = pd.to_numeric(x["_requested_season"], errors="coerce").astype("Int64")
    out = _ORIGINAL_INJURY_TEAM_WEEK(x)
    if len(out):
        counts = out.groupby("season").size().to_dict()
        weeks = out.groupby("season")["week"].agg(["min", "max"]).to_dict("index")
        print(f"[M67 fixed] injury team-week rows by season={counts} week_ranges={weeks}")
    return out


_ORIGINAL_INJURY_TEAM_WEEK = m.injury_team_week
m.load_sources = load_sources_fixed
m.injury_team_week = injury_team_week_fixed


if __name__ == "__main__":
    raise SystemExit(m.main())
