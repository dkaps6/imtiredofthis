#!/usr/bin/env python3
"""PRISM RB/TE discrete depth-role transition V1 source audit.

Source/lineage only. No outcomes, no model fitting, no sportsbook inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest.historical_inputs import (
    ALLOWED_ROSTER_STATUS,
    _lower,
    _to_pandas,
    build_schedule_history,
)

VERSION = "PRISM_RB_TE_DEPTH_ROLE_TRANSITION_V1_SOURCE_AUDIT"
HIST_SEASONS = (2024, 2025)
PROSPECTIVE_SEASON = 2026
ROOMS = {
    "RB": {"RB", "HB", "FB"},
    "TE": {"TE"},
}
PLAYER_FLOORS = {"RB": 100, "TE": 80}
TRANSITION_FLOORS = {"RB": 100, "TE": 60}


def key(v: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(v or "").lower())


def num(s: Any) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def sha256_df(x: pd.DataFrame) -> str:
    payload = x.to_csv(index=False, float_format="%.12f", lineterminator="\n").encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def load_sources(season: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    import nflreadpy as nfl

    roster_raw = _lower(_to_pandas(nfl.load_rosters_weekly(int(season))))
    depth_error = ""
    try:
        depth_raw = _lower(_to_pandas(nfl.load_depth_charts(int(season))))
    except Exception as exc:
        depth_raw = pd.DataFrame()
        depth_error = f"{type(exc).__name__}: {exc}"

    meta = {
        "season": int(season),
        "roster_rows_raw": int(len(roster_raw)),
        "depth_rows_raw": int(len(depth_raw)),
        "depth_load_error": depth_error,
        "depth_has_season": bool("season" in depth_raw.columns),
        "depth_has_week": bool("week" in depth_raw.columns),
        "depth_columns": sorted(map(str, depth_raw.columns)),
        "roster_columns": sorted(map(str, roster_raw.columns)),
    }
    if "week" in depth_raw.columns:
        wk = num(depth_raw["week"])
        meta["depth_latest_week"] = int(wk.max()) if wk.notna().any() else None
        meta["depth_distinct_weeks"] = int(wk.nunique())
    else:
        meta["depth_latest_week"] = None
        meta["depth_distinct_weeks"] = 0
    return roster_raw, depth_raw, meta


def roster_frame(roster: pd.DataFrame, season: int) -> pd.DataFrame:
    req = {"season", "week", "team", "position"}
    missing = req - set(roster.columns)
    if missing:
        raise RuntimeError(f"roster {season} missing {sorted(missing)}")
    name_col = "full_name" if "full_name" in roster.columns else "football_name" if "football_name" in roster.columns else None
    if not name_col:
        raise RuntimeError(f"roster {season} missing full_name/football_name")

    x = roster.loc[num(roster["season"]).eq(int(season))].copy()
    x["week"] = num(x["week"])
    x = x.loc[x["week"].between(1, 18, inclusive="both")].copy()
    x["team"] = x["team"].map(canon_team)
    x["position"] = x["position"].astype(str).str.upper().str.strip()
    if "status" in x.columns:
        status = x["status"].astype(str).str.upper().str.strip()
        keep = status.isin(ALLOWED_ROSTER_STATUS)
        if keep.any():
            x = x.loc[keep].copy()
    x["player"] = x[name_col].astype(str).str.strip()
    x["player_join"] = x["player"].map(key)
    return x


def depth_frame(depth: pd.DataFrame, season: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    if depth.empty or not {"season", "week"}.issubset(depth.columns):
        return pd.DataFrame(), {
            "week_tagged": False,
            "raw_duplicate_team_week_player_rows": 0,
        }

    team_col = "club_code" if "club_code" in depth.columns else "team" if "team" in depth.columns else None
    name_col = "full_name" if "full_name" in depth.columns else "football_name" if "football_name" in depth.columns else None
    if not team_col or not name_col:
        return pd.DataFrame(), {
            "week_tagged": True,
            "raw_duplicate_team_week_player_rows": 0,
            "missing_team_or_name_column": True,
        }

    d = depth.loc[num(depth["season"]).eq(int(season))].copy()
    d["week"] = num(d["week"])
    d = d.loc[d["week"].between(1, 18, inclusive="both")].copy()
    d["team"] = d[team_col].map(canon_team)
    d["player"] = d[name_col].astype(str).str.strip()
    d["player_join"] = d["player"].map(key)

    if "depth_position" not in d.columns:
        d["depth_position"] = d["position"] if "position" in d.columns else ""
    if "depth_team" not in d.columns:
        d["depth_team"] = np.nan

    d["depth_position"] = d["depth_position"].astype(str).str.upper().str.strip()
    d["depth_rank"] = num(d["depth_team"])
    d["depth_role"] = d["depth_position"].fillna("").astype(str)
    has_rank = d["depth_rank"].notna()
    d.loc[has_rank, "depth_role"] = (
        d.loc[has_rank, "depth_position"].fillna("").astype(str)
        + d.loc[has_rank, "depth_rank"].astype(int).astype(str)
    )
    d["depth_role"] = d["depth_role"].replace({"NAN": "", "NONE": ""})

    dup = int(d.duplicated(["team", "week", "player_join"], keep=False).sum())
    if dup:
        # Exact duplicate semantic rows are harmless; conflicting role rows fail later.
        role_n = d.groupby(["team", "week", "player_join"], dropna=False)["depth_role"].nunique(dropna=False)
        conflict_keys = role_n[role_n.gt(1)]
        if len(conflict_keys):
            d["_conflict"] = d.set_index(["team", "week", "player_join"]).index.isin(conflict_keys.index)
        else:
            d["_conflict"] = False
    else:
        d["_conflict"] = False

    d = d.sort_values(["team", "week", "player_join", "depth_role"]).drop_duplicates(
        ["team", "week", "player_join"], keep="first"
    )
    return d[[
        "team", "week", "player_join", "depth_position", "depth_rank",
        "depth_role", "_conflict"
    ]].reset_index(drop=True), {
        "week_tagged": True,
        "raw_duplicate_team_week_player_rows": dup,
    }


def audit_family(
    season: int,
    family: str,
    roster: pd.DataFrame,
    depth: pd.DataFrame,
    schedule: pd.DataFrame,
    source_meta: dict[str, Any],
    depth_meta: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pos = ROOMS[family]
    r = roster.loc[roster["position"].isin(pos)].copy()
    sched = schedule.loc[num(schedule["season"]).eq(int(season))].copy()
    expected_tw = set(zip(sched["week"].astype(int), sched["team"].astype(str)))
    roster_tw = set(zip(r["week"].dropna().astype(int), r.loc[r["week"].notna(), "team"].astype(str)))
    roster_tw_cov = len(expected_tw & roster_tw) / len(expected_tw) if expected_tw else 0.0

    if depth.empty:
        x = r.copy()
        x["depth_role"] = np.nan
        x["depth_rank"] = np.nan
        x["_conflict"] = False
    else:
        x = r.merge(
            depth,
            on=["team", "week", "player_join"],
            how="left",
            validate="many_to_one",
        )

    x = x.sort_values(["team", "player_join", "week"]).reset_index(drop=True)
    x["prior_depth_role"] = x.groupby(["team", "player_join"])["depth_role"].shift(1)
    x["prior_depth_rank"] = x.groupby(["team", "player_join"])["depth_rank"].shift(1)
    x["prior_week"] = x.groupby(["team", "player_join"])["week"].shift(1)
    x["chronology_violation"] = (
        x["prior_week"].notna()
        & x["week"].notna()
        & x["prior_week"].ge(x["week"])
    )
    x["role_transition"] = (
        x["depth_role"].fillna("").ne("")
        & x["prior_depth_role"].fillna("").ne("")
        & x["depth_role"].ne(x["prior_depth_role"])
    )
    x["rank_transition"] = (
        x["depth_rank"].notna()
        & x["prior_depth_rank"].notna()
        & x["depth_rank"].ne(x["prior_depth_rank"])
    )
    x["rank_delta"] = x["depth_rank"] - x["prior_depth_rank"]

    n = len(x)
    matched = x["depth_role"].fillna("").ne("")
    numeric_rank = x["depth_rank"].notna()
    wk3 = x["week"].ge(3)
    prior_supported = x["prior_depth_role"].fillna("").ne("")
    support_rows = x.loc[wk3]
    prior_cov = float(prior_supported.loc[wk3].mean()) if len(support_rows) else 0.0
    prior_players = int(x.loc[prior_supported, "player_join"].nunique())
    transitions = int(x["role_transition"].sum())
    chronology = int(x["chronology_violation"].sum())
    conflicts = int(x.get("_conflict", pd.Series(False, index=x.index)).fillna(False).sum())

    gates = {
        "roster_teamweek_coverage_ge_095": roster_tw_cov >= 0.95,
        "depth_explicit_season_week": bool(source_meta["depth_has_season"] and source_meta["depth_has_week"]),
        "roster_depth_identity_coverage_ge_090": (float(matched.mean()) if n else 0.0) >= 0.90,
        "nonblank_depth_role_coverage_ge_085": (float(matched.mean()) if n else 0.0) >= 0.85,
        "numeric_depth_rank_coverage_ge_080": (float(numeric_rank.mean()) if n else 0.0) >= 0.80,
        "prior_same_team_role_state_week3plus_ge_080": prior_cov >= 0.80,
        "distinct_players_with_prior_support_floor": prior_players >= PLAYER_FLOORS[family],
        "role_transition_rows_floor": transitions >= TRANSITION_FLOORS[family],
        "chronology_violations_zero": chronology == 0,
        "conflicting_team_week_player_roles_zero": conflicts == 0,
    }
    ready = all(gates.values())
    result = {
        "season": int(season),
        "family": family,
        "roster_player_games": int(n),
        "scheduled_team_weeks": int(len(expected_tw)),
        "roster_team_weeks": int(len(roster_tw)),
        "roster_teamweek_coverage": float(roster_tw_cov),
        "depth_week_tagged": bool(depth_meta.get("week_tagged", False)),
        "depth_identity_rows": int(matched.sum()),
        "depth_identity_coverage": float(matched.mean()) if n else 0.0,
        "depth_role_rows": int(matched.sum()),
        "depth_role_coverage": float(matched.mean()) if n else 0.0,
        "numeric_depth_rank_rows": int(numeric_rank.sum()),
        "numeric_depth_rank_coverage": float(numeric_rank.mean()) if n else 0.0,
        "week3plus_rows": int(wk3.sum()),
        "prior_role_supported_week3plus_rows": int((wk3 & prior_supported).sum()),
        "prior_role_coverage_week3plus": prior_cov,
        "distinct_players_with_prior_support": prior_players,
        "role_transition_player_games": transitions,
        "rank_transition_player_games": int(x["rank_transition"].sum()),
        "chronology_violations": chronology,
        "conflicting_role_rows": conflicts,
        "gates": gates,
        "ready": bool(ready),
        "disposition": (
            f"{family}_DEPTH_ROLE_TRANSITION_SOURCE_READY"
            if ready else f"{family}_DEPTH_ROLE_TRANSITION_SOURCE_BLOCKED"
        ),
    }
    x["season"] = int(season)
    x["family"] = family
    return x, result


def prospective_probe(
    roster: pd.DataFrame,
    depth: pd.DataFrame,
    source_meta: dict[str, Any],
    depth_meta: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "season": PROSPECTIVE_SEASON,
        "depth_load_error": source_meta.get("depth_load_error", ""),
        "depth_has_season": bool(source_meta.get("depth_has_season")),
        "depth_has_week": bool(source_meta.get("depth_has_week")),
        "depth_latest_week": source_meta.get("depth_latest_week"),
        "week_tagged": bool(depth_meta.get("week_tagged", False)),
    }
    if depth.empty or not result["week_tagged"]:
        result.update({
            "probe_week": None,
            "RB_identity_coverage": 0.0,
            "RB_role_coverage": 0.0,
            "TE_identity_coverage": 0.0,
            "TE_role_coverage": 0.0,
            "prospective_parity_ready": False,
            "reason": "2026 depth source lacks eligible explicit week-tagged rows",
        })
        return result

    common = sorted(set(num(roster["week"]).dropna().astype(int)) & set(num(depth["week"]).dropna().astype(int)))
    common = [w for w in common if 1 <= w <= 18]
    if not common:
        result.update({
            "probe_week": None,
            "RB_identity_coverage": 0.0,
            "RB_role_coverage": 0.0,
            "TE_identity_coverage": 0.0,
            "TE_role_coverage": 0.0,
            "prospective_parity_ready": False,
            "reason": "no common week-tagged roster/depth week in 2026",
        })
        return result

    w = max(common)
    result["probe_week"] = int(w)
    ready = True
    for fam, positions in ROOMS.items():
        rr = roster.loc[roster["week"].eq(w) & roster["position"].isin(positions)].copy()
        mm = rr.merge(
            depth.loc[depth["week"].eq(w)],
            on=["team", "week", "player_join"],
            how="left",
            validate="many_to_one",
        )
        cov = float(mm["depth_role"].fillna("").ne("").mean()) if len(mm) else 0.0
        rank_cov = float(mm["depth_rank"].notna().mean()) if len(mm) else 0.0
        result[f"{fam}_rows"] = int(len(mm))
        result[f"{fam}_identity_coverage"] = cov
        result[f"{fam}_role_coverage"] = cov
        result[f"{fam}_rank_coverage"] = rank_cov
        ready &= bool(len(mm) > 0 and cov >= 0.90 and rank_cov >= 0.80)

    # Week-tag presence alone does not prove an archived snapshot existed before each
    # already-completed 2026 game. This probe is prospective-readiness only.
    latest = source_meta.get("depth_latest_week")
    ready &= bool(latest is not None and int(latest) >= int(w))
    result["prospective_parity_ready"] = bool(ready)
    result["reason"] = (
        "same explicit week-tagged role/rank semantics available on latest common 2026 source week"
        if ready else
        "2026 week-tagged source exists but coverage/rank parity gate failed"
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    schedule = build_schedule_history(HIST_SEASONS)
    all_detail: list[pd.DataFrame] = []
    family_results: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []

    loaded: dict[int, tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]] = {}
    for season in (*HIST_SEASONS, PROSPECTIVE_SEASON):
        roster_raw, depth_raw, meta = load_sources(season)
        roster = roster_frame(roster_raw, season)
        depth, depth_meta = depth_frame(depth_raw, season)
        loaded[season] = (roster, depth, meta, depth_meta)
        source_rows.append({
            "season": season,
            "roster_rows": int(len(roster)),
            "depth_rows": int(len(depth)),
            "depth_has_season": meta["depth_has_season"],
            "depth_has_week": meta["depth_has_week"],
            "depth_latest_week": meta["depth_latest_week"],
            "depth_distinct_weeks": meta["depth_distinct_weeks"],
            "depth_load_error": meta["depth_load_error"],
            "raw_duplicate_team_week_player_rows": int(depth_meta.get("raw_duplicate_team_week_player_rows", 0)),
        })

    for season in HIST_SEASONS:
        roster, depth, meta, depth_meta = loaded[season]
        for family in ("RB", "TE"):
            detail, result = audit_family(
                season, family, roster, depth, schedule, meta, depth_meta
            )
            all_detail.append(detail)
            family_results.append(result)

    roster26, depth26, meta26, dmeta26 = loaded[PROSPECTIVE_SEASON]
    probe = prospective_probe(roster26, depth26, meta26, dmeta26)

    hist_ready = {
        fam: all(
            r["ready"] for r in family_results if r["family"] == fam
        )
        for fam in ("RB", "TE")
    }
    ready_families = [fam for fam, ok in hist_ready.items() if ok and probe["prospective_parity_ready"]]
    if len(ready_families) == 2:
        disposition = "PRISM_RB_TE_DEPTH_ROLE_TRANSITION_SOURCE_READY_ALL"
    elif len(ready_families) == 1:
        disposition = "PRISM_RB_TE_DEPTH_ROLE_TRANSITION_SOURCE_READY_PARTIAL"
    else:
        disposition = "PRISM_RB_TE_DEPTH_ROLE_TRANSITION_SOURCE_BLOCKED"

    detail = pd.concat(all_detail, ignore_index=True) if all_detail else pd.DataFrame()
    family_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "gates"} | {
            **{f"gate__{g}": ok for g, ok in r["gates"].items()}
        }
        for r in family_results
    ])
    source_df = pd.DataFrame(source_rows)

    integrity = {
        "target_game_outcomes_used": 0,
        "target_game_snaps_used_as_source_state": 0,
        "sportsbook_inputs_used": 0,
        "oddsapi_calls": 0,
        "predictive_models_fit": 0,
        "thresholds_searched": 0,
        "production_changed": False,
        "raw_recent_share_used": False,
        "vacancy_transfer_used": False,
    }
    result = {
        "version": VERSION,
        "status": "SOURCE_AUDIT_COMPLETE",
        "disposition": disposition,
        "historical_seasons": list(HIST_SEASONS),
        "prospective_probe_season": PROSPECTIVE_SEASON,
        "historical_family_ready": hist_ready,
        "ready_families_with_prospective_parity": ready_families,
        "family_results": family_results,
        "prospective_probe": probe,
        "integrity": integrity,
        "boundary": (
            "source/lineage only; no outcome scoring. A READY source would still "
            "require a separate frozen predictive mechanism before any projection change."
        ),
    }

    detail.to_csv(args.out_dir / "depth_role_transition_source_detail.csv", index=False)
    family_df.to_csv(args.out_dir / "depth_role_transition_family_summary.csv", index=False)
    source_df.to_csv(args.out_dir / "depth_role_transition_source_schema.csv", index=False)
    result["detail_sha256"] = sha256_df(detail)
    result["family_summary_sha256"] = sha256_df(family_df)
    result["source_schema_sha256"] = sha256_df(source_df)
    (args.out_dir / "depth_role_transition_source_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
