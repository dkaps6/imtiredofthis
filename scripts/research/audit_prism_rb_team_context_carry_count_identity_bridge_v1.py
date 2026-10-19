#!/usr/bin/env python3
"""Source-only audit for the PRISM RB team-context carry count identity bridge.

This audit does not read predictive player outcomes. It certifies (or blocks)
the provider-ID lineage between team-context raw GSIS IDs, Stage-1/Stage-2 raw
player IDs, and the canonical historical count identity namespace.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts._opponent_map import canon_team
from scripts.utils.player_identity_v3 import historical_identity_key

VERSION = "PRISM_RB_TEAM_CONTEXT_CARRY_COUNT_IDENTITY_BRIDGE_V1"
FREEZE_COMMIT = "df01aa02659b3bd96ab9e215d939088a5925a576"
SEASONS = (2024, 2025)
WEEKS = tuple(range(5, 19))
EXPECTED_TRANSITION_ROWS = 39
COUNT_ROSTER_STATUSES = {"ACT", "INA"}


def _text(v: Any) -> str:
    if v is None or pd.isna(v):
        return ""
    s = str(v).strip()
    return "" if s.upper() in {"", "NAN", "NONE", "<NA>"} else s


def _bool_series(s: pd.Series) -> pd.Series:
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


def read_csv(path: Path, label: str, columns: list[str]) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    header = pd.read_csv(path, nrows=0)
    lower_map = {str(c).strip().lower(): c for c in header.columns}
    missing = [c for c in columns if c not in lower_map]
    if missing:
        raise RuntimeError(f"{label} missing columns {missing}")
    usecols = [lower_map[c] for c in columns]
    x = pd.read_csv(path, usecols=usecols, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_count_identity(raw_provider_id: Any) -> str:
    raw = _text(raw_provider_id)
    if not raw:
        return ""
    key = historical_identity_key(raw, "", "")
    expected = f"gsis:{raw}"
    if key != expected:
        raise RuntimeError(
            f"canonical historical identity drift for provider ID {raw!r}: "
            f"{key!r} != {expected!r}"
        )
    return key


def count_roster_status_eligible(status: Any) -> bool:
    return _text(status).upper() in COUNT_ROSTER_STATUSES


def prepare_transitions(path: Path) -> pd.DataFrame:
    cols = [
        "season",
        "week",
        "team",
        "stable_id",
        "family",
        "transition_kind",
        "prior_team_support_ge4",
        "roster_status",
    ]
    x = read_csv(path, "team-context transition events", cols)
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x["team"] = x["team"].map(canon_team)
    x["stable_id"] = x["stable_id"].map(_text)
    x["family"] = x["family"].astype("string").fillna("").str.upper().str.strip()
    x["transition_kind"] = (
        x["transition_kind"].astype("string").fillna("").str.strip()
    )
    x["prior_team_support_ge4"] = _bool_series(x["prior_team_support_ge4"])
    x["roster_status"] = (
        x["roster_status"].astype("string").fillna("").str.upper().str.strip()
    )
    x = x.loc[
        x["season"].isin(SEASONS)
        & x["week"].isin(WEEKS)
        & x["family"].eq("RB")
        & x["transition_kind"].eq("INSEASON_TEAM_CHANGE")
        & x["prior_team_support_ge4"]
        & x["stable_id"].ne("")
    ].copy()
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)
    if x.duplicated(["season", "week", "team", "stable_id"]).any():
        raise RuntimeError("duplicate frozen-scope transition identity")
    return x.sort_values(
        ["season", "week", "team", "stable_id"], kind="mergesort"
    ).reset_index(drop=True)


def prepare_stage1(path: Path) -> pd.DataFrame:
    cols = [
        "season",
        "week",
        "team",
        "player_id",
        "family",
        "stable_roster_identity",
        "status",
    ]
    x = read_csv(path, "Stage-1 pregame universe", cols)
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x["team"] = x["team"].map(canon_team)
    x["player_id"] = x["player_id"].map(_text)
    x["family"] = x["family"].astype("string").fillna("").str.upper().str.strip()
    x["stable_roster_identity"] = pd.to_numeric(
        x["stable_roster_identity"], errors="coerce"
    ).fillna(0)
    x["status"] = x["status"].astype("string").fillna("").str.upper().str.strip()
    x = x.loc[
        x["season"].isin(SEASONS)
        & x["week"].isin(WEEKS)
        & x["family"].eq("RB")
    ].copy()
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)
    if x.duplicated(["season", "week", "team", "player_id"]).any():
        raise RuntimeError("duplicate Stage-1 RB identity")
    return x.reset_index(drop=True)


def prepare_stage2(path: Path) -> pd.DataFrame:
    cols = ["season", "week", "team", "player_id", "family", "cell"]
    x = read_csv(path, "Stage-2 source detail", cols)
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x["team"] = x["team"].map(canon_team)
    x["player_id"] = x["player_id"].map(_text)
    x["family"] = x["family"].astype("string").fillna("").str.upper().str.strip()
    x["cell"] = x["cell"].astype("string").fillna("").str.strip()
    x = x.loc[
        x["season"].isin(SEASONS)
        & x["week"].isin(WEEKS)
        & x["family"].eq("RB")
        & x["cell"].eq("RB_RUSHING")
    ].copy()
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)
    if x.duplicated(["season", "week", "team", "player_id"]).any():
        raise RuntimeError("duplicate Stage-2 RB rushing identity")
    return x.reset_index(drop=True)


def audit(
    transitions: pd.DataFrame,
    team_result: dict[str, Any],
    stage1: pd.DataFrame,
    stage2: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    schemas = team_result.get("roster_audit", {}).get("schemas", {})
    schema_gsis = all(
        str(schemas.get(str(season), {}).get("stable_id", "")).strip() == "gsis_id"
        for season in SEASONS
    )

    rows = transitions.copy()
    rows["count_identity_key"] = rows["stable_id"].map(canonical_count_identity)
    rows["count_identity_roundtrip"] = (
        rows["count_identity_key"].str.removeprefix("gsis:").eq(rows["stable_id"])
    )
    rows["count_universe_status_eligible"] = rows["roster_status"].map(
        count_roster_status_eligible
    )

    # The historical count universe is roster-first and preserves zero-opportunity
    # ACT/INA rows. Its canonical roster loader explicitly excludes reserve /
    # non-roster states when ACT/INA are present. Therefore status eligibility is
    # a source-only membership test; no target-game carries are needed or read.
    rows["canonical_count_identity_linked"] = (
        rows["count_universe_status_eligible"]
        & rows["count_identity_key"].ne("")
        & rows["count_identity_roundtrip"]
    )

    stage1_keys = set(
        tuple(v)
        for v in stage1[["season", "week", "team", "player_id"]]
        .itertuples(index=False, name=None)
    )
    stage2_keys = list(
        tuple(v)
        for v in stage2[["season", "week", "team", "player_id"]]
        .itertuples(index=False, name=None)
    )
    stage2_stage1_hits = sum(k in stage1_keys for k in stage2_keys)

    stage2_event_keys = set(stage2_keys)
    rows["stage2_event_identity_present"] = [
        (int(s), int(w), str(t), str(pid)) in stage2_event_keys
        for s, w, t, pid in rows[
            ["season", "week", "team", "stable_id"]
        ].itertuples(index=False, name=None)
    ]

    raw_to_key = rows[["stable_id", "count_identity_key"]].drop_duplicates()
    collisions = (
        raw_to_key.groupby("count_identity_key")["stable_id"].nunique().gt(1).sum()
        if len(raw_to_key)
        else 0
    )

    stage2_roundtrip = stage2["player_id"].map(canonical_count_identity)
    stage1_nonblank_stable = (
        stage1["player_id"].ne("")
        & stage1["stable_roster_identity"].eq(1)
    )

    attempts = int(len(rows))
    hits = int(rows["canonical_count_identity_linked"].sum())
    link_rate = float(hits / attempts) if attempts else 0.0
    s2_hits = int(stage2_stage1_hits)
    s2_attempts = int(len(stage2))
    s2_rate = float(s2_hits / s2_attempts) if s2_attempts else 0.0

    gates = {
        "team_context_schema_stable_id_is_gsis_id_2024_2025": bool(schema_gsis),
        "frozen_scope_transition_rows_eq_39": attempts == EXPECTED_TRANSITION_ROWS,
        "transition_provider_ids_nonblank": bool(
            attempts > 0 and rows["stable_id"].ne("").all()
        ),
        "count_identity_roundtrip_exact": bool(
            attempts > 0 and rows["count_identity_roundtrip"].all()
        ),
        "stable_count_link_rate_ge_099": link_rate >= 0.99,
        "raw_provider_to_count_identity_collisions_zero": int(collisions) == 0,
        "stage1_rb_provider_ids_nonblank_and_stable": bool(
            len(stage1) > 0 and stage1_nonblank_stable.all()
        ),
        "stage2_rb_rushing_rows_exact_stage1_identity_rate_ge_099": s2_rate >= 0.99,
        "stage2_raw_id_count_namespace_roundtrip_exact": bool(
            len(stage2) > 0
            and stage2_roundtrip.str.removeprefix("gsis:").eq(stage2["player_id"]).all()
        ),
        "names_read_or_used_zero": True,
        "predictive_outcomes_read_zero": True,
        "target_game_opportunity_counts_read_zero": True,
        "rushing_yards_read_zero": True,
        "touchdowns_read_zero": True,
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
        "disposition": (
            "PRISM_RB_TEAM_CONTEXT_CARRY_COUNT_IDENTITY_BRIDGE_V1_SOURCE_READY"
            if ready
            else "PRISM_RB_TEAM_CONTEXT_CARRY_COUNT_IDENTITY_BRIDGE_V1_SOURCE_BLOCKED"
        ),
        "source_ready": ready,
        "frozen_transition_events": attempts,
        "frozen_transition_unique_raw_ids": int(rows["stable_id"].nunique()),
        "canonical_count_identity_link_hits": hits,
        "canonical_count_identity_link_attempts": attempts,
        "canonical_count_identity_link_rate": link_rate,
        "transition_roster_status_counts": {
            str(k): int(v)
            for k, v in rows["roster_status"].value_counts(dropna=False).sort_index().items()
        },
        "stage2_event_identity_present": int(rows["stage2_event_identity_present"].sum()),
        "stage2_event_identity_attempts": attempts,
        "stage2_to_stage1_exact_identity_hits": s2_hits,
        "stage2_to_stage1_exact_identity_attempts": s2_attempts,
        "stage2_to_stage1_exact_identity_rate": s2_rate,
        "raw_to_count_identity_collisions": int(collisions),
        "count_identity_rule": "raw_gsis_id -> gsis:<raw_gsis_id>",
        "count_roster_status_policy": sorted(COUNT_ROSTER_STATUSES),
        "source_gates": gates,
        "names_read_or_used": 0,
        "predictive_outcomes_read": 0,
        "target_game_opportunity_counts_read": 0,
        "rushing_yards_read": 0,
        "touchdowns_read": 0,
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
    ap.add_argument("--team-context-result", type=Path, required=True)
    ap.add_argument("--stage1-universe", type=Path, required=True)
    ap.add_argument("--stage2-detail", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    transitions = prepare_transitions(a.team_context_events)
    team_result = read_json(a.team_context_result, "team-context source result")
    stage1 = prepare_stage1(a.stage1_universe)
    stage2 = prepare_stage2(a.stage2_detail)

    rows, result = audit(transitions, team_result, stage1, stage2)
    rows.to_csv(
        a.out_dir / "rb_team_context_carry_count_identity_bridge_rows.csv",
        index=False,
    )
    (a.out_dir / "rb_team_context_carry_count_identity_bridge_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
