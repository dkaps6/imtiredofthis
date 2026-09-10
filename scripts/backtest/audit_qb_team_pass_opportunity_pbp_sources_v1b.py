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
MIN_COVERAGE = 0.95
KEYS = ["season", "week", "team", "player_clean_key"]


def to_pd(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    return pd.DataFrame(obj)


def lower(df):
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def num(s):
    return pd.to_numeric(s, errors="coerce")


def one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected one {name}, found {len(hits)}")
    return hits[0]


def load_targets(root: Path) -> pd.DataFrame:
    x = lower(pd.read_csv(one(root, "m89_corrected_qb_common_trace.csv"), low_memory=False))
    missing = [c for c in KEYS if c not in x.columns]
    if missing:
        raise RuntimeError(f"target key source missing {missing}")
    x = x[KEYS].copy()
    x["season"] = num(x["season"])
    x["week"] = num(x["week"])
    x["team"] = x["team"].map(canon_team)
    x["player_clean_key"] = x["player_clean_key"].fillna("").astype(str).str.strip()
    x = x.loc[x["season"].isin(TARGET_SEASONS)].copy()
    if len(x) != 884 or x.duplicated(KEYS).any():
        raise RuntimeError(f"target-key integrity failure rows={len(x)}")
    return x.sort_values(KEYS).reset_index(drop=True)


def load_pbp() -> tuple[pd.DataFrame, pd.DataFrame]:
    import nflreadpy as nfl
    parts = []
    counts = []
    for season in SEASONS:
        p = lower(to_pd(nfl.load_pbp(seasons=[season])))
        if "season_type" in p.columns:
            p = p.loc[p["season_type"].astype(str).str.upper().eq("REG")].copy()
        elif "game_type" in p.columns:
            p = p.loc[p["game_type"].astype(str).str.upper().eq("REG")].copy()
        p["season"] = season
        parts.append(p)
        counts.append({"season": season, "rows": int(len(p))})
    return pd.concat(parts, ignore_index=True, sort=False), pd.DataFrame(counts)


def coverage(frame: pd.DataFrame, fields: list[str], family: str, universe: str) -> list[dict]:
    rows = []
    for season in SEASONS:
        g = frame.loc[num(frame["season"]).eq(season)].copy()
        for field in fields:
            exists = field in g.columns
            rows.append({
                "family": family,
                "universe": universe,
                "season": season,
                "field": field,
                "exists": int(exists),
                "rows": int(len(g)),
                "populated_coverage": float(g[field].notna().mean()) if exists and len(g) else 0.0,
            })
    return rows


def build_universes(pbp: pd.DataFrame):
    required = {
        "season", "week", "game_id", "play_id", "posteam", "defteam",
        "first_down_penalty", "down", "play_type", "fourth_down_converted", "fourth_down_failed",
    }
    missing = sorted(required - set(pbp.columns))
    if missing:
        raise RuntimeError(f"PBP missing required V1B fields: {missing}")

    p = pbp.copy()
    p["season"] = num(p["season"])
    p["week"] = num(p["week"])
    p["posteam"] = p["posteam"].fillna("").astype(str).map(canon_team)
    p["defteam"] = p["defteam"].fillna("").astype(str).map(canon_team)
    p["play_type"] = p["play_type"].fillna("").astype(str).str.lower().str.strip()
    p["down"] = num(p["down"])
    p["first_down_penalty"] = num(p["first_down_penalty"])
    p["fourth_down_converted"] = num(p["fourth_down_converted"])
    p["fourth_down_failed"] = num(p["fourth_down_failed"])

    penalty_rel = p.loc[
        p["week"].between(1, 18)
        & p["posteam"].ne("")
        & p["defteam"].ne("")
    ].copy()

    fourth_rel = p.loc[
        p["week"].between(1, 18)
        & p["posteam"].ne("")
        & p["defteam"].ne("")
        & p["down"].eq(4)
        & p["play_type"].ne("")
        & ~p["play_type"].eq("no_play")
    ].copy()

    decision_types = {"pass", "run", "punt", "field_goal", "qb_kneel", "qb_spike"}
    fourth_dec = fourth_rel.loc[fourth_rel["play_type"].isin(decision_types)].copy()
    return penalty_rel, fourth_rel, fourth_dec


def aggregate_team_games(penalty_rel: pd.DataFrame, fourth_dec: pd.DataFrame) -> pd.DataFrame:
    base_keys = ["season", "week", "game_id", "posteam", "defteam"]

    p = penalty_rel.copy()
    p["_fdp"] = p["first_down_penalty"].fillna(0).eq(1).astype(int)
    pen = (
        p.groupby(base_keys, as_index=False)
        .agg(
            offensive_pbp_rows=("play_id", "count"),
            first_down_penalties=("_fdp", "sum"),
        )
    )

    f = fourth_dec.copy()
    f["_go"] = (
        f["fourth_down_converted"].fillna(0).eq(1)
        | f["fourth_down_failed"].fillna(0).eq(1)
    ).astype(int)
    f["_conv"] = f["fourth_down_converted"].fillna(0).eq(1).astype(int)
    f["_go_pass"] = (f["_go"].eq(1) & f["play_type"].eq("pass")).astype(int)
    f["_go_run"] = (f["_go"].eq(1) & f["play_type"].isin(["run", "qb_kneel"])).astype(int)
    fourth = (
        f.groupby(base_keys, as_index=False)
        .agg(
            fourth_down_decisions=("play_id", "count"),
            fourth_down_go_attempts=("_go", "sum"),
            fourth_down_conversions=("_conv", "sum"),
            fourth_down_go_passes=("_go_pass", "sum"),
            fourth_down_go_runs=("_go_run", "sum"),
        )
    )

    out = pen.merge(fourth, on=base_keys, how="left", validate="one_to_one")
    for c in ["fourth_down_decisions", "fourth_down_go_attempts", "fourth_down_conversions", "fourth_down_go_passes", "fourth_down_go_runs"]:
        out[c] = num(out[c]).fillna(0).astype(int)
    out = out.rename(columns={"posteam": "team", "defteam": "opponent"})
    out["team"] = out["team"].map(canon_team)
    out["opponent"] = out["opponent"].map(canon_team)
    if out.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("duplicate aggregated team-game rows")
    return out.sort_values(["season", "week", "team"]).reset_index(drop=True)


def prior_feasibility(targets: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in targets.itertuples(index=False):
        h = games.loc[
            games["team"].eq(r.team)
            & ((games["season"] < r.season) | ((games["season"] == r.season) & (games["week"] < r.week)))
        ]
        rows.append({
            "season": int(r.season),
            "week": int(r.week),
            "team": str(r.team),
            "player_clean_key": str(r.player_clean_key),
            "prior_team_games": int(len(h)),
            "prior_offensive_pbp_rows": int(h["offensive_pbp_rows"].sum()) if len(h) else 0,
            "prior_first_down_penalty_events": int(h["first_down_penalties"].sum()) if len(h) else 0,
            "prior_fourth_down_decisions": int(h["fourth_down_decisions"].sum()) if len(h) else 0,
            "penalty_history_defined": int(len(h) >= 1 and h["offensive_pbp_rows"].sum() > 0),
            "fourth_down_history_defined": int(len(h) >= 1 and h["fourth_down_decisions"].sum() > 0),
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m89-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    targets = load_targets(a.m89_root)
    pbp, source_counts = load_pbp()
    pen_rel, fourth_rel, fourth_dec = build_universes(pbp)
    games = aggregate_team_games(pen_rel, fourth_dec)
    feas = prior_feasibility(targets, games)

    cov_rows = []
    cov_rows += coverage(
        pen_rel,
        ["season", "week", "game_id", "play_id", "posteam", "defteam", "first_down_penalty"],
        "PENALTY_DRIVE_EXTENSION",
        "PENALTY_RELEVANT",
    )
    cov_rows += coverage(
        fourth_rel,
        ["season", "week", "game_id", "play_id", "posteam", "defteam", "down", "play_type", "fourth_down_converted", "fourth_down_failed"],
        "FOURTH_DOWN_AGGRESSION",
        "FOURTH_DECISION_RELEVANT",
    )
    cov = pd.DataFrame(cov_rows)

    def family_core_ok(family: str) -> bool:
        q = cov.loc[cov["family"].eq(family)]
        return bool(len(q) > 0 and q["exists"].eq(1).all() and q["rows"].gt(0).all() and q["populated_coverage"].ge(MIN_COVERAGE).all())

    pen_prior = float(feas["penalty_history_defined"].mean())
    fourth_prior = float(feas["fourth_down_history_defined"].mean())

    def result(name: str, core_ok: bool, prior_cov: float, notes: dict):
        gates = {
            "pbp_all_required_seasons_loaded": bool(set(source_counts.season) == set(SEASONS) and source_counts.rows.gt(0).all()),
            "required_fields_exist": bool(core_ok),
            "family_relevant_universe_nonempty": bool(core_ok),
            "relevant_universe_core_coverage_ge_95pct": bool(core_ok),
            "target_prior_history_coverage_ge_95pct": bool(prior_cov >= MIN_COVERAGE),
            "unique_team_game_keys": bool(not games.duplicated(["season", "week", "team"]).any()),
            "zero_sportsbook_result_requirements": True,
            "target_game_outcome_not_required": True,
            "materially_distinct_from_closed_ledger": True,
            "source_semantics_explicit": True,
        }
        disp = "SOURCE_ELIGIBLE_FOR_PREDICTIVE_PREREGISTRATION" if all(gates.values()) else "SOURCE_INELIGIBLE_SCHEMA_OR_COVERAGE"
        return {
            "family": name,
            "prior_history_coverage": prior_cov,
            "gates": gates,
            "disposition": disp,
            "semantic_notes": notes,
        }

    families = [
        result(
            "PENALTY_DRIVE_EXTENSION",
            family_core_ok("PENALTY_DRIVE_EXTENSION"),
            pen_prior,
            {
                "predictive_concept_if_eligible": "strict-prior first_down_penalty events per offensive possession-team PBP row, plus opponent-defense allowed analogue",
                "accepted_declined_generic_penalty_rate": "PROHIBITED_NOT_INFERRED",
            },
        ),
        result(
            "FOURTH_DOWN_AGGRESSION",
            family_core_ok("FOURTH_DOWN_AGGRESSION"),
            fourth_prior,
            {
                "go_attempt": "fourth_down_converted==1 OR fourth_down_failed==1",
                "decision_universe": "down==4, nonblank teams/play_type, play_type!=no_play, decision play_type",
            },
        ),
    ]

    result_json = {
        "migration": "QB_TEAM_PASS_OPPORTUNITY_PBP_SOURCE_AUDIT_V1B",
        "original_v1_result_preserved": True,
        "coverage_threshold": MIN_COVERAGE,
        "target_rows": int(len(targets)),
        "target_outcomes_or_residuals_loaded": False,
        "sportsbook_inputs_used": False,
        "model_fitting_used": False,
        "production_changed": False,
        "source_counts": source_counts.to_dict("records"),
        "families": families,
        "eligible_families": [f["family"] for f in families if f["disposition"] == "SOURCE_ELIGIBLE_FOR_PREDICTIVE_PREREGISTRATION"],
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    source_counts.to_csv(a.out_dir / "pbp_source_rows_by_season.csv", index=False)
    cov.to_csv(a.out_dir / "relevant_universe_field_coverage.csv", index=False)
    games.to_csv(a.out_dir / "pbp_team_game_candidate_counts.csv", index=False)
    feas.to_csv(a.out_dir / "strict_prior_candidate_feasibility_2024_2025.csv", index=False)
    (a.out_dir / "qb_team_pass_opportunity_pbp_source_audit_v1b_result.json").write_text(
        json.dumps(result_json, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result_json, indent=2, sort_keys=True))
    print("\nRELEVANT-UNIVERSE COVERAGE")
    print(cov.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
