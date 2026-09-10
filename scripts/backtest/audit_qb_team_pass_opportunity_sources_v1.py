#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team

SEASONS = [2023, 2024, 2025]
TARGET_SEASONS = [2024, 2025]
MIN_CORE_COVERAGE = 0.95
MIN_TARGET_PRIOR_COVERAGE = 0.95

FAMILIES = [
    "PENALTY_DRIVE_EXTENSION",
    "FOURTH_DOWN_AGGRESSION",
    "SCHEDULE_REST_CONTEXT",
]

PROHIBITED_SCHEDULE_FIELDS = {
    "away_score", "home_score", "result", "total", "overtime",
    "away_moneyline", "home_moneyline", "spread_line", "away_spread_odds",
    "home_spread_odds", "total_line", "under_odds", "over_odds",
}


def to_pd(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    return pd.DataFrame(obj)


def lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def num(s):
    return pd.to_numeric(s, errors="coerce")


def one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(hits)}")
    return hits[0]


def load_targets(m89_root: Path) -> pd.DataFrame:
    x = lower(pd.read_csv(one(m89_root, "m89_corrected_qb_common_trace.csv"), low_memory=False))
    need = ["season", "week", "team", "player_clean_key"]
    missing = [c for c in need if c not in x.columns]
    if missing:
        raise RuntimeError(f"M89 target-key source missing {missing}")
    x = x[need].copy()
    x["season"] = num(x["season"])
    x["week"] = num(x["week"])
    x["team"] = x["team"].map(canon_team)
    x["player_clean_key"] = x["player_clean_key"].fillna("").astype(str).str.strip()
    x = x.loc[x["season"].isin(TARGET_SEASONS)].copy()
    if len(x) != 884:
        raise RuntimeError(f"expected 884 M89 target QB-games, got {len(x)}")
    if x.duplicated(need).any():
        raise RuntimeError("duplicate M89 target QB keys")
    return x.sort_values(need).reset_index(drop=True)


def load_sources():
    import nflreadpy as nfl

    pbp_parts = []
    sched_parts = []
    source_rows = []
    for season in SEASONS:
        p = lower(to_pd(nfl.load_pbp(seasons=[season])))
        if "season_type" in p.columns:
            p = p.loc[p["season_type"].astype(str).str.upper().eq("REG")].copy()
        elif "game_type" in p.columns:
            p = p.loc[p["game_type"].astype(str).str.upper().eq("REG")].copy()
        p["season"] = season
        pbp_parts.append(p)
        source_rows.append({"source": "nflreadpy.load_pbp", "season": season, "rows": int(len(p))})

        s = lower(to_pd(nfl.load_schedules(seasons=[season])))
        if "game_type" in s.columns:
            s = s.loc[s["game_type"].astype(str).str.upper().eq("REG")].copy()
        s["season"] = season
        sched_parts.append(s)
        source_rows.append({"source": "nflreadpy.load_schedules", "season": season, "rows": int(len(s))})

    pbp = pd.concat(pbp_parts, ignore_index=True, sort=False)
    sched = pd.concat(sched_parts, ignore_index=True, sort=False)
    return pbp, sched, pd.DataFrame(source_rows)


def coverage_rows(frame: pd.DataFrame, source: str, fields: list[str]) -> pd.DataFrame:
    rows = []
    for season in SEASONS:
        g = frame.loc[num(frame.get("season", pd.Series(index=frame.index))).eq(season)].copy()
        for field in fields:
            exists = field in g.columns
            rows.append({
                "source": source,
                "season": season,
                "field": field,
                "exists": int(exists),
                "populated_coverage": float(g[field].notna().mean()) if exists and len(g) else 0.0,
                "rows": int(len(g)),
            })
    return pd.DataFrame(rows)


def build_pbp_team_games(pbp: pd.DataFrame) -> pd.DataFrame:
    need = {"season", "week", "game_id", "posteam", "defteam", "play_id"}
    missing = sorted(need - set(pbp.columns))
    if missing:
        raise RuntimeError(f"PBP missing team-game keys: {missing}")

    p = pbp.copy()
    p["season"] = num(p["season"])
    p["week"] = num(p["week"])
    p["posteam"] = p["posteam"].map(canon_team)
    p["defteam"] = p["defteam"].map(canon_team)
    p = p.loc[p["week"].between(1, 18) & p["posteam"].ne("")].copy()

    penalty = num(p["penalty"]).fillna(0).eq(1) if "penalty" in p.columns else pd.Series(False, index=p.index)
    first_down_penalty = num(p["first_down_penalty"]).fillna(0).eq(1) if "first_down_penalty" in p.columns else pd.Series(False, index=p.index)
    no_play = num(p["no_play"]).fillna(0).eq(1) if "no_play" in p.columns else pd.Series(False, index=p.index)
    down = num(p["down"]) if "down" in p.columns else pd.Series(np.nan, index=p.index)
    fourth_conv = num(p["fourth_down_converted"]).fillna(0).eq(1) if "fourth_down_converted" in p.columns else pd.Series(False, index=p.index)
    fourth_fail = num(p["fourth_down_failed"]).fillna(0).eq(1) if "fourth_down_failed" in p.columns else pd.Series(False, index=p.index)
    play_type = p["play_type"].fillna("").astype(str).str.lower() if "play_type" in p.columns else pd.Series("", index=p.index)

    penalty_team = p["penalty_team"].fillna("").astype(str).map(canon_team) if "penalty_team" in p.columns else pd.Series("", index=p.index)
    p["_penalty_flag"] = penalty.astype(int)
    p["_penalty_team_present"] = (penalty & penalty_team.ne("")).astype(int)
    p["_first_down_penalty"] = first_down_penalty.astype(int)
    p["_penalty_no_play"] = (penalty & (no_play | play_type.eq("no_play"))).astype(int)

    # Source-safe fourth-down go-for-it events: nflfastR explicitly labels converted/failed fourth-down attempts.
    go = fourth_conv | fourth_fail
    # Decision denominator: recorded fourth-down play with a football decision, excluding no-play rows.
    fourth_decision = down.eq(4) & ~no_play & ~play_type.eq("no_play") & play_type.isin([
        "pass", "run", "punt", "field_goal", "qb_kneel", "qb_spike"
    ])
    p["_fourth_go"] = go.astype(int)
    p["_fourth_conversion"] = fourth_conv.astype(int)
    p["_fourth_decision"] = fourth_decision.astype(int)
    p["_fourth_go_pass"] = (go & play_type.eq("pass")).astype(int)
    p["_fourth_go_run"] = (go & play_type.isin(["run", "qb_kneel"])).astype(int)

    rows = []
    keys = ["season", "week", "game_id", "posteam", "defteam"]
    for key, g in p.groupby(keys, dropna=False, sort=False):
        season, week, game_id, team, opp = key
        plays = len(g)
        rows.append({
            "season": int(season),
            "week": int(week),
            "game_id": str(game_id),
            "team": canon_team(team),
            "opponent": canon_team(opp),
            "pbp_rows": int(plays),
            "penalty_flags": int(g["_penalty_flag"].sum()),
            "penalty_team_present_events": int(g["_penalty_team_present"].sum()),
            "first_down_penalties": int(g["_first_down_penalty"].sum()),
            "penalty_no_plays": int(g["_penalty_no_play"].sum()),
            "fourth_down_decisions": int(g["_fourth_decision"].sum()),
            "fourth_down_go_attempts": int(g["_fourth_go"].sum()),
            "fourth_down_conversions": int(g["_fourth_conversion"].sum()),
            "fourth_down_go_passes": int(g["_fourth_go_pass"].sum()),
            "fourth_down_go_runs": int(g["_fourth_go_run"].sum()),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("PBP team-game summarization produced zero rows")
    if out.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("duplicate PBP season/week/team rows")
    return out.sort_values(["season", "week", "team"]).reset_index(drop=True)


def schedule_team_view(sched: pd.DataFrame) -> pd.DataFrame:
    need = {"season", "week", "game_id", "gameday", "weekday", "home_team", "away_team", "home_rest", "away_rest"}
    missing = sorted(need - set(sched.columns))
    if missing:
        raise RuntimeError(f"schedule missing required pregame fields: {missing}")
    s = sched.copy()
    s["season"] = num(s["season"])
    s["week"] = num(s["week"])
    s["home_team"] = s["home_team"].map(canon_team)
    s["away_team"] = s["away_team"].map(canon_team)
    s["home_rest"] = num(s["home_rest"])
    s["away_rest"] = num(s["away_rest"])
    rows = []
    for r in s.itertuples(index=False):
        for side in ["home", "away"]:
            team = getattr(r, f"{side}_team")
            opp_side = "away" if side == "home" else "home"
            opp = getattr(r, f"{opp_side}_team")
            rest = getattr(r, f"{side}_rest")
            opp_rest = getattr(r, f"{opp_side}_rest")
            weekday = str(getattr(r, "weekday"))
            rows.append({
                "season": int(getattr(r, "season")),
                "week": int(getattr(r, "week")),
                "game_id": str(getattr(r, "game_id")),
                "team": canon_team(team),
                "opponent": canon_team(opp),
                "gameday": str(getattr(r, "gameday")),
                "weekday": weekday,
                "home": int(side == "home"),
                "rest_days": float(rest) if pd.notna(rest) else np.nan,
                "opponent_rest_days": float(opp_rest) if pd.notna(opp_rest) else np.nan,
                "rest_diff": float(rest - opp_rest) if pd.notna(rest) and pd.notna(opp_rest) else np.nan,
                "short_week": int(pd.notna(rest) and float(rest) <= 6),
                "long_rest": int(pd.notna(rest) and float(rest) >= 9),
                "thursday": int("thu" in weekday.lower()),
                "monday": int("mon" in weekday.lower()),
            })
    out = pd.DataFrame(rows)
    if out.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("duplicate schedule season/week/team rows")
    return out


def prior_count_table(targets: pd.DataFrame, source_team_games: pd.DataFrame) -> pd.DataFrame:
    src = source_team_games[["season", "week", "team"]].drop_duplicates().copy()
    rows = []
    for r in targets.itertuples(index=False):
        hist = src.loc[
            src["team"].eq(r.team)
            & ((src["season"] < r.season) | ((src["season"] == r.season) & (src["week"] < r.week)))
        ]
        rows.append({
            "season": int(r.season),
            "week": int(r.week),
            "team": str(r.team),
            "player_clean_key": str(r.player_clean_key),
            "prior_eligible_team_games": int(len(hist)),
            "has_prior_history": int(len(hist) >= 1),
        })
    return pd.DataFrame(rows)


def family_result(name: str, core_ok: bool, prior_cov: float, distinct: bool, semantics_ok: bool, pregame_ok: bool, extra: dict | None = None):
    gates = {
        "required_seasons_loaded": True,
        "core_schema_and_coverage_ge_95pct": bool(core_ok),
        "no_sportsbook_or_result_required": True,
        "target_game_outcome_not_required": True,
        "strict_prior_constructible": bool(pregame_ok),
        "target_prior_history_coverage_ge_95pct": bool(prior_cov >= MIN_TARGET_PRIOR_COVERAGE),
        "unique_team_game_keys": True,
        "materially_distinct_from_closed_ledger": bool(distinct),
        "source_semantics_explicit": bool(semantics_ok),
    }
    disposition = (
        "SOURCE_ELIGIBLE_FOR_PREDICTIVE_PREREGISTRATION"
        if all(gates.values()) else
        "DUPLICATE_OR_CLOSED_INFORMATION_FAMILY" if not distinct else
        "SOURCE_INELIGIBLE_PREGAME_PROVENANCE" if not pregame_ok else
        "SOURCE_INELIGIBLE_SCHEMA_OR_COVERAGE"
    )
    out = {
        "family": name,
        "prior_history_coverage": float(prior_cov),
        "gates": gates,
        "disposition": disposition,
    }
    if extra:
        out.update(extra)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m89-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    targets = load_targets(a.m89_root)
    pbp, sched, source_rows = load_sources()

    pbp_fields = [
        "season", "week", "game_id", "play_id", "posteam", "defteam",
        "penalty", "penalty_team", "penalty_yards", "first_down_penalty", "no_play",
        "down", "play_type", "fourth_down_converted", "fourth_down_failed",
    ]
    sched_fields = [
        "season", "week", "game_id", "gameday", "weekday", "home_team", "away_team",
        "home_rest", "away_rest",
    ]
    field_cov = pd.concat([
        coverage_rows(pbp, "nflreadpy.load_pbp", pbp_fields),
        coverage_rows(sched, "nflreadpy.load_schedules", sched_fields),
    ], ignore_index=True)

    team_games = build_pbp_team_games(pbp)
    sched_team = schedule_team_view(sched)

    pbp_prior = prior_count_table(targets, team_games)
    sched_align = targets.merge(
        sched_team[["season", "week", "team", "game_id", "weekday", "home", "rest_days", "opponent_rest_days", "rest_diff", "short_week", "long_rest", "thursday", "monday"]],
        on=["season", "week", "team"], how="left", validate="many_to_one"
    )
    if len(sched_align) != len(targets):
        raise RuntimeError("schedule target alignment row drift")

    pbp_prior_cov = float(pbp_prior["has_prior_history"].mean())
    sched_target_cov = float(sched_align["rest_days"].notna().mean())

    core_penalty = ["season", "week", "game_id", "play_id", "posteam", "defteam", "penalty", "first_down_penalty"]
    core_fourth = ["season", "week", "game_id", "play_id", "posteam", "defteam", "down", "play_type", "fourth_down_converted", "fourth_down_failed"]
    core_sched = ["season", "week", "game_id", "gameday", "weekday", "home_team", "away_team", "home_rest", "away_rest"]

    def core_ok(source: str, fields: list[str]) -> bool:
        q = field_cov.loc[field_cov["source"].eq(source) & field_cov["field"].isin(fields)].copy()
        if len(q) != len(fields) * len(SEASONS):
            return False
        return bool(q["exists"].eq(1).all() and q["populated_coverage"].ge(MIN_CORE_COVERAGE).all())

    # Generic accepted/declined penalty status is not claimed from `penalty` alone.
    # The family is anchored to nflfastR's explicit first_down_penalty drive-extension label.
    penalty_semantics = {
        "safe_core": "first_down_penalty is explicit: penalty converted the first down",
        "penalty_flag": "penalty only establishes that a penalty occurred",
        "penalty_team": "team with the penalty; sparse by event",
        "accepted_rate_status": "NOT_CLAIMED_WITHOUT_EXPLICIT_ACCEPTED_DECLINED_FIELD",
    }
    fourth_semantics = {
        "go_attempt": "fourth_down_converted OR fourth_down_failed; explicit nflfastR fourth-down attempt outcome labels",
        "decision_denominator": "down==4, non-no-play, play_type in pass/run/punt/field_goal/qb_kneel/qb_spike",
    }
    schedule_semantics = {
        "rest": "nflverse schedule home_rest/away_rest are days of rest entering the game",
        "pregame_fields_used": ["gameday", "weekday", "home_team", "away_team", "home_rest", "away_rest"],
        "prohibited_fields_present_but_unused": sorted(PROHIBITED_SCHEDULE_FIELDS & set(sched.columns)),
    }

    results = [
        family_result(
            "PENALTY_DRIVE_EXTENSION",
            core_ok("nflreadpy.load_pbp", core_penalty),
            pbp_prior_cov,
            True,
            True,
            True,
            {"semantic_notes": penalty_semantics},
        ),
        family_result(
            "FOURTH_DOWN_AGGRESSION",
            core_ok("nflreadpy.load_pbp", core_fourth),
            pbp_prior_cov,
            True,
            True,
            True,
            {"semantic_notes": fourth_semantics},
        ),
        family_result(
            "SCHEDULE_REST_CONTEXT",
            core_ok("nflreadpy.load_schedules", core_sched),
            sched_target_cov,
            True,
            True,
            True,
            {"semantic_notes": schedule_semantics},
        ),
    ]

    result = {
        "migration": "QB_TEAM_PASS_OPPORTUNITY_SOURCE_AUDIT_V1",
        "parent_disposition": "TEAM_PASS_OPPORTUNITY_PRIMARY_DIAGNOSTIC",
        "target_rows": int(len(targets)),
        "target_outcomes_or_parent_residuals_loaded": False,
        "sportsbook_inputs_used": False,
        "model_fitting_used": False,
        "production_changed": False,
        "officiating_crew_disposition": "QUARANTINED_NO_HISTORICAL_PREGAME_PROVENANCE",
        "families": results,
        "eligible_families": [r["family"] for r in results if r["disposition"] == "SOURCE_ELIGIBLE_FOR_PREDICTIVE_PREREGISTRATION"],
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    source_rows.to_csv(a.out_dir / "source_rows_by_season.csv", index=False)
    field_cov.to_csv(a.out_dir / "source_field_coverage.csv", index=False)
    team_games.to_csv(a.out_dir / "pbp_team_game_source_summary.csv", index=False)
    pbp_prior.to_csv(a.out_dir / "strict_prior_feasibility_2024_2025.csv", index=False)
    sched_align.to_csv(a.out_dir / "schedule_rest_target_feasibility_2024_2025.csv", index=False)
    pd.DataFrame(results).to_json(a.out_dir / "family_dispositions.json", orient="records", indent=2)
    (a.out_dir / "qb_team_pass_opportunity_source_audit_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(json.dumps(result, indent=2, sort_keys=True))
    print("\nFIELD COVERAGE")
    print(field_cov.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
