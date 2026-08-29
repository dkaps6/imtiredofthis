#!/usr/bin/env python3
"""Execution-safe M67 source builder.

Changes no M67 feature definition, family, model, gate, or historical boundary.
It only works around nflreadpy injury loading/alignment issues by loading injury
reports one season at a time, attaching requested-season provenance, and using
that provenance when constructing team-week injury features.
"""
from __future__ import annotations

import numpy as np
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
                q["_provider_season_raw"] = q.get("season", pd.Series(pd.NA, index=q.index))
                q["_requested_season"] = int(season)
                raw_vals = sorted(pd.to_numeric(q["_provider_season_raw"], errors="coerce").dropna().astype(int).unique().tolist())
                st = sorted(q.get("season_type", pd.Series("", index=q.index)).astype(str).dropna().unique().tolist())
                injury_frames.append(q)
                print(f"[M67 fixed] injury requested_season={season} rows={len(q)} raw_seasons={raw_vals} season_types={st}")
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
            "notes": "official report status/practice fields by offensive position; season-scoped loader with requested-season provenance; target weeks 1-18",
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
    """Frozen M67 injury definitions with robust season/week attribution.

    nflverse injury assets have inconsistent historical season_type encoding.
    M67's scored universe is regular-season weeks 1-18, so use the explicit
    requested-season provenance plus week boundary rather than provider
    season_type. No injury feature definition is changed.
    """
    cols = ["season", "week", "team"] + m.AVAILABILITY_FEATURES
    if inj.empty:
        return pd.DataFrame(columns=cols)

    x = inj.copy()
    if "_requested_season" in x.columns:
        x["season"] = pd.to_numeric(x["_requested_season"], errors="coerce")
    else:
        x["season"] = pd.to_numeric(x.get("season"), errors="coerce")
    x["week"] = pd.to_numeric(x.get("week"), errors="coerce")
    x = x[x["season"].notna() & x["week"].between(1, 18, inclusive="both")].copy()
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)
    x["team"] = x["team"].map(m.canon_team)
    x = x[x["team"].ne("")].copy()

    pos = m.text(x.get("position", pd.Series("", index=x.index))).str.upper()
    status = m.text(x.get("report_status", x.get("status", pd.Series("", index=x.index)))).str.upper()
    practice = m.text(x.get("practice_status", pd.Series("", index=x.index))).str.upper()
    x["_outd"] = status.str.contains("OUT|DOUBTFUL", regex=True)
    x["_q"] = status.str.contains("QUESTIONABLE", regex=True)
    x["_dnp"] = practice.str.contains("DNP|DID NOT", regex=True)
    x["_limited"] = practice.str.contains("LIMITED", regex=True)
    x["_ol"] = pos.isin(["C", "G", "T", "OL", "OG", "OT", "LT", "RT", "LG", "RG"])
    x["_rb"] = pos.isin(["RB", "FB"])
    x["_wrte"] = pos.isin(["WR", "TE"])
    x["_skill"] = x["_rb"] | x["_wrte"]

    rows: list[dict] = []
    for (season, week, team), g in x.groupby(["season", "week", "team"], sort=True):
        rows.append({
            "season": int(season),
            "week": int(week),
            "team": m.canon_team(team),
            "availability_report_rows": int(len(g)),
            "availability_out_doubtful_total": int(g["_outd"].sum()),
            "availability_questionable_total": int(g["_q"].sum()),
            "availability_dnp_total": int(g["_dnp"].sum()),
            "availability_limited_total": int(g["_limited"].sum()),
            "availability_ol_out_doubtful": int((g["_outd"] & g["_ol"]).sum()),
            "availability_ol_questionable": int((g["_q"] & g["_ol"]).sum()),
            "availability_rb_out_doubtful": int((g["_outd"] & g["_rb"]).sum()),
            "availability_wrte_out_doubtful": int((g["_outd"] & g["_wrte"]).sum()),
            "availability_skill_out_doubtful": int((g["_outd"] & g["_skill"]).sum()),
        })

    out = pd.DataFrame(rows, columns=cols)
    if len(out):
        counts = out.groupby("season").size().to_dict()
        weeks = out.groupby("season")["week"].agg(["min", "max"]).to_dict("index")
        print(f"[M67 fixed] injury team-week rows by season={counts} week_ranges={weeks}")
        for required in (2024, 2025):
            if required in set(pd.to_numeric(inj.get("_requested_season", pd.Series(dtype=float)), errors="coerce").dropna().astype(int)) and counts.get(required, 0) == 0:
                raise RuntimeError(f"M67 injury aggregation unexpectedly produced zero team-weeks for {required}")
    return out


m.load_sources = load_sources_fixed
m.injury_team_week = injury_team_week_fixed


if __name__ == "__main__":
    raise SystemExit(m.main())
