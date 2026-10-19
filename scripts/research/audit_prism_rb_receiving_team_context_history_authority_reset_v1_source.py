#!/usr/bin/env python3
"""Source-only audit for the PRISM RB receiving team-context history-authority reset V1.

No predictive outcome column is read. The audit establishes exact raw-ID lineage,
first-post-transition market eligibility, and availability of the frozen
Stage-1 role-prior counterfactual before any predictive scoring is allowed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team

VERSION = "PRISM_RB_RECEIVING_TEAM_CONTEXT_HISTORY_AUTHORITY_RESET_V1_SOURCE"
FREEZE_COMMIT = "821d43059c1feabb3c7de617698e4913dfee0e6b"
TEAM_CONTEXT_RUN = 39182076127
TEAM_CONTEXT_ARTIFACT = 12156204035
TEAM_CONTEXT_DIGEST = "sha256:abc2a5b9d3ea2a4d683e56626ea2d33495fa1e7cad2c36c56c01071ccc0339c8"
AUTHORITY_RUN = 39099474655
AUTHORITY_ARTIFACT = 12125534698
AUTHORITY_DIGEST = "sha256:3d1f8043dc7a2ded0ed7fd9705b359d31a094a272ecce6958823ce660c2898e4"
STAGE1_RUN = 38907459996
STAGE1_ARTIFACT = 12049120604
STAGE1_DIGEST = "sha256:41b1acd3f1b8d77abb53980db160ece32f8fc40351939d4827248b9caad0c64b"

SEASONS = (2024, 2025)
TRANSITION_KINDS = {"OFFSEASON_TEAM_CHANGE", "INSEASON_TEAM_CHANGE"}
TOL = 1e-12


def text(v: Any) -> str:
    if v is None or pd.isna(v):
        return ""
    s = str(v).strip()
    return "" if s.upper() in {"", "NAN", "NONE", "<NA>"} else s


def as_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    return (
        s.astype("string")
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
        .astype(bool)
    )


def read_columns(path: Path, label: str, cols: list[str]) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    header = pd.read_csv(path, nrows=0)
    mapping = {str(c).strip().lower(): c for c in header.columns}
    missing = [c for c in cols if c not in mapping]
    if missing:
        raise RuntimeError(f"{label} missing columns {missing}")
    x = pd.read_csv(path, usecols=[mapping[c] for c in cols], low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def prepare_events(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [
        "season", "week", "team", "stable_id", "family",
        "transition_kind", "prior_team_support_ge4",
    ]
    x = read_columns(path, "team-context transitions", cols)
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x["team"] = x["team"].map(canon_team)
    x["stable_id"] = x["stable_id"].map(text)
    x["family"] = x["family"].astype("string").fillna("").str.upper().str.strip()
    x["transition_kind"] = (
        x["transition_kind"].astype("string").fillna("").str.strip()
    )
    x["prior_team_support_ge4"] = as_bool(x["prior_team_support_ge4"])
    full = x.loc[
        x["season"].isin(SEASONS)
        & x["family"].eq("RB")
        & x["transition_kind"].ne("")
        & x["stable_id"].ne("")
    ].copy()
    full["season"] = full["season"].astype(int)
    full["week"] = full["week"].astype(int)

    eligible = full.loc[
        full["transition_kind"].isin(TRANSITION_KINDS)
        & full["prior_team_support_ge4"]
    ].copy()
    eligible = eligible.sort_values(
        ["season", "stable_id", "week", "team"], kind="mergesort"
    ).reset_index(drop=True)
    eligible["transition_id"] = [
        f"{int(s)}:{int(w)}:{t}:{pid}:{kind}"
        for s, w, t, pid, kind in eligible[
            ["season", "week", "team", "stable_id", "transition_kind"]
        ].itertuples(index=False, name=None)
    ]
    if eligible["transition_id"].duplicated().any():
        raise RuntimeError("duplicate eligible transition_id")
    return full.reset_index(drop=True), eligible


def prepare_states(path: Path) -> pd.DataFrame:
    cols = ["season", "week", "team", "stable_id", "family"]
    x = read_columns(path, "team-context states", cols)
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x["team"] = x["team"].map(canon_team)
    x["stable_id"] = x["stable_id"].map(text)
    x["family"] = x["family"].astype("string").fillna("").str.upper().str.strip()
    x = x.loc[
        x["season"].isin(SEASONS)
        & x["family"].eq("RB")
        & x["stable_id"].ne("")
    ].copy()
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)
    if x.duplicated(["season", "week", "team", "stable_id"]).any():
        raise RuntimeError("duplicate exact team-context state identity")
    return x.reset_index(drop=True)


def prepare_authority(path: Path) -> pd.DataFrame:
    # Deliberately exclude y, N, CRPS, AE and every predictive outcome/score column.
    cols = [
        "season", "week", "game_key", "team", "cell", "player_id",
        "stage1_p_active", "p_effective", "player_prior_positive_q_rows",
        "hierarchical_expected_benefit", "own_authority_disabled",
    ]
    x = read_columns(path, "qualified RB authority", cols)
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x["team"] = x["team"].map(canon_team)
    x["player_id"] = x["player_id"].map(text)
    x["cell"] = x["cell"].astype("string").fillna("").str.strip()
    x["game_key"] = x["game_key"].map(text)
    for c in (
        "stage1_p_active", "p_effective", "player_prior_positive_q_rows",
        "hierarchical_expected_benefit",
    ):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x["own_authority_disabled"] = as_bool(x["own_authority_disabled"])
    x = x.loc[
        x["season"].isin(SEASONS)
        & x["cell"].eq("RB_RECEIVING")
        & x["player_id"].ne("")
    ].copy()
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)
    keys = ["season", "week", "team", "player_id"]
    if x.duplicated(keys).any():
        raise RuntimeError("duplicate qualified RB_RECEIVING authority identity")
    return x.sort_values(
        ["season", "week", "team", "player_id"], kind="mergesort"
    ).reset_index(drop=True)


def prepare_stage1(path: Path) -> pd.DataFrame:
    # Deliberately exclude Stage-1 outcome y and all name fields.
    cols = [
        "season", "week", "team", "family", "player_id",
        "p_baseline", "p_candidate", "n_player_prior",
    ]
    x = read_columns(path, "Stage-1 mechanism rows", cols)
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x["team"] = x["team"].map(canon_team)
    x["player_id"] = x["player_id"].map(text)
    x["family"] = x["family"].astype("string").fillna("").str.upper().str.strip()
    for c in ("p_baseline", "p_candidate", "n_player_prior"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.loc[
        x["season"].isin(SEASONS)
        & x["family"].eq("RB")
        & x["player_id"].ne("")
    ].copy()
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)
    keys = ["season", "week", "team", "player_id"]
    if x.duplicated(keys).any():
        raise RuntimeError("duplicate Stage-1 exact RB identity")
    return x.reset_index(drop=True)


def authority_state_link_rate(authority: pd.DataFrame, states: pd.DataFrame) -> tuple[int, int, float]:
    state_keys = set(
        tuple(v)
        for v in states[["season", "week", "team", "stable_id"]]
        .itertuples(index=False, name=None)
    )
    attempts = int(len(authority))
    hits = int(sum(
        (int(s), int(w), str(t), str(pid)) in state_keys
        for s, w, t, pid in authority[
            ["season", "week", "team", "player_id"]
        ].itertuples(index=False, name=None)
    ))
    return hits, attempts, (hits / attempts if attempts else 0.0)


def select_first_market_rows(
    full_events: pd.DataFrame,
    eligible: pd.DataFrame,
    authority: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, Any]] = []
    exclusions: dict[str, int] = {"NO_CANONICAL_RB_RECEIVING_BASELINE": 0}

    by_player: dict[tuple[int, str], pd.DataFrame] = {
        (int(season), str(pid)): g.sort_values(["week", "team"], kind="mergesort")
        for (season, pid), g in full_events.groupby(["season", "stable_id"], sort=False)
    }

    for r in eligible.itertuples(index=False):
        later = by_player.get((int(r.season), str(r.stable_id)))
        next_week: int | None = None
        if later is not None:
            qlater = later.loc[later["week"].gt(int(r.week))]
            if len(qlater):
                next_week = int(qlater["week"].min())

        q = authority.loc[
            authority["season"].eq(int(r.season))
            & authority["player_id"].eq(str(r.stable_id))
            & authority["team"].eq(str(r.team))
            & authority["week"].ge(int(r.week))
        ].copy()
        if next_week is not None:
            q = q.loc[q["week"].lt(next_week)].copy()

        if q.empty:
            exclusions["NO_CANONICAL_RB_RECEIVING_BASELINE"] += 1
            continue

        q = q.sort_values(["week", "game_key"], kind="mergesort")
        first = q.iloc[0]
        rows.append({
            "transition_id": str(r.transition_id),
            "season": int(r.season),
            "transition_week": int(r.week),
            "team": str(r.team),
            "stable_id": str(r.stable_id),
            "transition_kind": str(r.transition_kind),
            "next_transition_week": next_week,
            "target_week": int(first["week"]),
            "game_key": str(first["game_key"]),
            "stage1_p_active": float(first["stage1_p_active"]),
            "baseline_p_effective": float(first["p_effective"]),
            "player_prior_positive_q_rows": int(first["player_prior_positive_q_rows"]),
            "hierarchical_expected_benefit": float(first["hierarchical_expected_benefit"]),
            "own_authority_disabled": bool(first["own_authority_disabled"]),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out, exclusions
    if out["transition_id"].duplicated().any():
        raise RuntimeError("multiple selected rows for one transition")
    if (out["target_week"] < out["transition_week"]).any():
        raise RuntimeError("selected market row precedes transition")
    with_next = out["next_transition_week"].notna()
    if (
        out.loc[with_next, "target_week"]
        >= out.loc[with_next, "next_transition_week"]
    ).any():
        raise RuntimeError("selected market row crosses later transition boundary")
    return out.reset_index(drop=True), exclusions


def attach_reset_source(selected: pd.DataFrame, stage1: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return selected.copy()

    s = stage1.rename(columns={
        "week": "target_week",
        "player_id": "stable_id",
        "p_candidate": "stage1_p_candidate",
        "p_baseline": "stage1_p_baseline",
        "n_player_prior": "stage1_player_prior_rows",
    })[
        [
            "season", "target_week", "team", "stable_id",
            "stage1_p_baseline", "stage1_p_candidate", "stage1_player_prior_rows",
        ]
    ].copy()

    out = selected.merge(
        s,
        on=["season", "target_week", "team", "stable_id"],
        how="left",
        validate="many_to_one",
    )
    out["stage1_exact_mechanism_row"] = out["stage1_p_candidate"].notna()

    matched = out["stage1_exact_mechanism_row"]
    diff = (
        out.loc[matched, "stage1_p_candidate"]
        - out.loc[matched, "stage1_p_active"]
    ).abs()
    out["stage1_candidate_matches_current"] = True
    out.loc[matched, "stage1_candidate_matches_current"] = diff.le(TOL)

    out["reset_p_active"] = out["stage1_p_baseline"].where(
        matched, out["stage1_p_active"]
    )
    out["actionable_joint_history_reset"] = (
        out["reset_p_active"] - out["baseline_p_effective"]
    ).abs().gt(TOL)
    return out


def audit(
    full_events: pd.DataFrame,
    eligible: pd.DataFrame,
    states: pd.DataFrame,
    authority: pd.DataFrame,
    stage1: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    link_hits, link_attempts, link_rate = authority_state_link_rate(authority, states)
    selected, exclusions = select_first_market_rows(full_events, eligible, authority)
    rows = attach_reset_source(selected, stage1)

    per_season: dict[str, dict[str, Any]] = {}
    for season in SEASONS:
        g = rows.loc[rows["season"].eq(season)].copy() if len(rows) else rows
        act = g.loc[g.get("actionable_joint_history_reset", False)].copy() if len(g) else g
        per_season[str(season)] = {
            "selected_market_rows": int(len(g)),
            "distinct_stable_ids": int(g["stable_id"].nunique()) if len(g) else 0,
            "stage1_exact_mechanism_rows": int(g["stage1_exact_mechanism_row"].sum()) if len(g) else 0,
            "own_stage2_authority_disabled_rows": int(g["own_authority_disabled"].sum()) if len(g) else 0,
            "actionable_reset_rows": int(len(act)),
            "actionable_reset_stable_ids": int(act["stable_id"].nunique()) if len(act) else 0,
            "transition_kind_counts": {
                str(k): int(v)
                for k, v in g["transition_kind"].value_counts().sort_index().items()
            } if len(g) else {},
        }

    reset_open = bool(
        len(rows)
        and np.isfinite(rows["reset_p_active"]).all()
        and rows["reset_p_active"].gt(0).all()
        and rows["reset_p_active"].lt(1).all()
    )
    exact_stage1 = bool(
        len(rows)
        and rows.loc[
            rows["stage1_exact_mechanism_row"],
            "stage1_candidate_matches_current",
        ].all()
    )

    gates = {
        "immutable_parent_contracts_verified_by_workflow": True,
        "eligible_transition_provider_ids_nonblank": bool(
            len(eligible) and eligible["stable_id"].ne("").all()
        ),
        "qualified_rb_receiving_authority_exact_state_link_rate_ge_099": link_rate >= 0.99,
        "selected_transition_rows_unique": bool(
            len(rows) and not rows["transition_id"].duplicated().any()
        ),
        "selected_rows_chronology_valid": bool(
            len(rows)
            and rows["target_week"].ge(rows["transition_week"]).all()
            and (
                rows["next_transition_week"].isna()
                | rows["target_week"].lt(rows["next_transition_week"])
            ).all()
        ),
        "stage1_exact_rows_match_current_p_active": exact_stage1,
        "reset_probability_finite_open_interval_all": reset_open,
        "development_2024_selected_rows_ge_30": per_season["2024"]["selected_market_rows"] >= 30,
        "development_2024_distinct_ids_ge_25": per_season["2024"]["distinct_stable_ids"] >= 25,
        "confirmation_2025_selected_rows_ge_30": per_season["2025"]["selected_market_rows"] >= 30,
        "confirmation_2025_distinct_ids_ge_25": per_season["2025"]["distinct_stable_ids"] >= 25,
        "development_2024_actionable_rows_ge_10": per_season["2024"]["actionable_reset_rows"] >= 10,
        "development_2024_actionable_ids_ge_10": per_season["2024"]["actionable_reset_stable_ids"] >= 10,
        "confirmation_2025_actionable_rows_ge_10": per_season["2025"]["actionable_reset_rows"] >= 10,
        "confirmation_2025_actionable_ids_ge_10": per_season["2025"]["actionable_reset_stable_ids"] >= 10,
        "names_read_or_used_zero": True,
        "predictive_outcomes_read_or_scored_zero": True,
        "target_game_opportunity_counts_read_zero": True,
        "sportsbook_inputs_zero": True,
        "oddsapi_calls_zero": True,
        "coefficients_fit_zero": True,
        "thresholds_searched_zero": True,
        "production_changed_false": True,
    }
    ready = bool(all(gates.values()))

    result = {
        "version": VERSION,
        "freeze_commit": FREEZE_COMMIT,
        "team_context_parent": {
            "run": TEAM_CONTEXT_RUN,
            "artifact": TEAM_CONTEXT_ARTIFACT,
            "digest": TEAM_CONTEXT_DIGEST,
        },
        "qualified_authority_parent": {
            "run": AUTHORITY_RUN,
            "artifact": AUTHORITY_ARTIFACT,
            "digest": AUTHORITY_DIGEST,
        },
        "stage1_parent": {
            "run": STAGE1_RUN,
            "artifact": STAGE1_ARTIFACT,
            "digest": STAGE1_DIGEST,
        },
        "disposition": (
            "PRISM_RB_RECEIVING_TEAM_CONTEXT_HISTORY_AUTHORITY_RESET_V1_SOURCE_READY"
            if ready else
            "PRISM_RB_RECEIVING_TEAM_CONTEXT_HISTORY_AUTHORITY_RESET_V1_SOURCE_BLOCKED"
        ),
        "source_ready": ready,
        "eligible_team_context_transitions": int(len(eligible)),
        "selected_first_market_rows": int(len(rows)),
        "source_exclusions": exclusions,
        "authority_state_link_hits": link_hits,
        "authority_state_link_attempts": link_attempts,
        "authority_state_link_rate": float(link_rate),
        "per_season": per_season,
        "source_gates": gates,
        "names_read_or_used": 0,
        "predictive_outcomes_read_or_scored": 0,
        "target_game_opportunity_counts_read": 0,
        "sportsbook_inputs_used": 0,
        "oddsapi_calls": 0,
        "coefficients_fit": 0,
        "thresholds_searched": 0,
        "production_changed": False,
    }
    return rows, result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--team-context-events", type=Path, required=True)
    ap.add_argument("--team-context-states", type=Path, required=True)
    ap.add_argument("--qualified-authority-rows", type=Path, required=True)
    ap.add_argument("--stage1-mechanism-rows", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    full_events, eligible = prepare_events(args.team_context_events)
    states = prepare_states(args.team_context_states)
    authority = prepare_authority(args.qualified_authority_rows)
    stage1 = prepare_stage1(args.stage1_mechanism_rows)
    rows, result = audit(full_events, eligible, states, authority, stage1)

    rows.to_csv(
        args.out_dir / "rb_receiving_team_context_history_authority_reset_source_rows.csv",
        index=False,
    )
    (args.out_dir / "rb_receiving_team_context_history_authority_reset_source_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
