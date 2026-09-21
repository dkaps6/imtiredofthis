#!/usr/bin/env python3
"""Reconstruct certified completed 2026 RB history from frozen pregame sources.

V1 implements the Week-1 source contract using the 107-player football-only
P3 context from the canonical production promotion artifact. Projection lineage
is frozen and validated before realized rushing yards are loaded. Realized
outcomes are attached only for completed games via GSIS/roster identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.operations.grade_market_track_record_gsis_v1 import (
    build_alias_index,
    load_actual_stats_unfiltered,
    load_roster_identity,
    resolve_gsis,
)
from scripts.research.rb_pd2_forward_shadow_v1 import (
    ELIGIBLE_POSITIONS,
    RUSH_YARDS_MC_WEIGHT,
    RUSH_YARDS_ML_WEIGHT,
    RUSH_YARDS_STATE_WEIGHT,
)
from scripts.utils.canonical_names import canonicalize_player_name_safe

WEEK1_SOURCE_RUN_ID = "33993595929"
WEEK1_SOURCE_ARTIFACT_ID = "9977398453"
WEEK1_SOURCE_ARTIFACT_DIGEST = "e97c9534823e5f29ab1e3c51aa794de2e6d47bc0ca999b18a89349d872fd4ddd"
WEEK1_SOURCE_GIT_SHA = "1daba60e3a6cd53319ba3223a1fc5f7e7184222e"
WEEK1_EXPECTED_ROWS = 107
WEEK1_VERSION = "RB_P3_SYNTHESIS_V1"
WEEK1_ROUTE = "WEEK1_STACK_OVERRIDE"
ATOL = 1e-8


def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canon_key(value: Any) -> str:
    _, key = canonicalize_player_name_safe(value)
    return str(key or "").strip()


def build_week1_projection_frame(context: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate and freeze the football-only Week-1 parent before outcomes exist."""
    x = context.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    required = {
        "season", "week", "event_id", "player", "player_clean_key", "team",
        "opponent", "position", "stack_yards", "rb_synthesis_proj",
        "rb_synthesis_route", "rb_synthesis_version", "rb_synthesis_applied",
        "football_only_no_odds", "sportsbook_inputs_used",
        "rush_yards_ensemble_weight_mc", "rush_yards_ensemble_weight_ml",
        "rush_yards_ensemble_weight_state",
    }
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"Week-1 P3 context missing columns: {sorted(missing)}")

    if len(x) != WEEK1_EXPECTED_ROWS:
        raise RuntimeError(f"Week-1 P3 context row count drift: {len(x)} != {WEEK1_EXPECTED_ROWS}")
    if not _num(x["season"]).eq(2026).all() or not _num(x["week"]).eq(1).all():
        raise RuntimeError("Week-1 P3 context season/week drift")
    pos = x["position"].fillna("").astype(str).str.upper().str.strip()
    if not pos.isin(ELIGIBLE_POSITIONS).all():
        raise RuntimeError("Week-1 P3 context contains out-of-scope position")
    if not x["rb_synthesis_version"].astype(str).eq(WEEK1_VERSION).all():
        raise RuntimeError("Week-1 P3 version drift")
    if not x["rb_synthesis_route"].astype(str).eq(WEEK1_ROUTE).all():
        raise RuntimeError("Week-1 P3 route drift")
    if not _num(x["rb_synthesis_applied"]).eq(1).all():
        raise RuntimeError("Week-1 P3 contains unapplied row")
    if not _num(x["football_only_no_odds"]).eq(1).all():
        raise RuntimeError("Week-1 P3 football-only flag drift")
    if not _num(x["sportsbook_inputs_used"]).eq(0).all():
        raise RuntimeError("Week-1 P3 sportsbook contamination flag nonzero")

    for col, want in [
        ("rush_yards_ensemble_weight_mc", RUSH_YARDS_MC_WEIGHT),
        ("rush_yards_ensemble_weight_ml", RUSH_YARDS_ML_WEIGHT),
        ("rush_yards_ensemble_weight_state", RUSH_YARDS_STATE_WEIGHT),
    ]:
        got = _num(x[col])
        if got.isna().any() or not np.allclose(got.to_numpy(float), want, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"Week-1 frozen rush_yards ensemble weight drift: {col}")

    p3 = _num(x["rb_synthesis_proj"])
    stack = _num(x["stack_yards"])
    if not np.isfinite(p3).all() or not np.isfinite(stack).all():
        raise RuntimeError("Week-1 P3 projection contains non-finite value")
    parity = (p3 - stack).abs()
    max_diff = float(parity.max()) if len(parity) else float("nan")
    if max_diff > ATOL:
        raise RuntimeError(f"Week-1 P3/STACK1 parity failed max_abs_diff={max_diff}")

    x["team"] = x["team"].map(canon_team)
    x["opponent"] = x["opponent"].map(canon_team)
    x["player_clean_key"] = x["player_clean_key"].map(_canon_key)
    if x["player_clean_key"].eq("").any():
        raise RuntimeError("Week-1 P3 context contains blank canonical player key")
    identity = ["season", "week", "team", "player_clean_key"]
    if x.duplicated(identity).any():
        raise RuntimeError("Week-1 P3 context duplicate football identity")

    out = pd.DataFrame({
        "season": 2026,
        "week": 1,
        "event_id": x["event_id"].astype(str),
        "player": x["player"].astype(str),
        "player_clean_key": x["player_clean_key"].astype(str),
        "team": x["team"].astype(str),
        "opponent": x["opponent"].astype(str),
        "position": pos,
        "projection_mean": p3.astype(float),
        "pregame_lineage_certified": True,
        "projection_lineage": (
            f"2026_W1_P3_STACK1|run={WEEK1_SOURCE_RUN_ID}|"
            f"artifact={WEEK1_SOURCE_ARTIFACT_ID}|sha={WEEK1_SOURCE_GIT_SHA}"
        ),
        "week1_p3_stack1_parity_pass": True,
    })
    audit = {
        "projection_rows": int(len(out)),
        "unique_players": int(out["player_clean_key"].nunique()),
        "max_p3_stack1_abs_diff": max_diff,
        "sportsbook_inputs_used": 0,
        "source_run_id": WEEK1_SOURCE_RUN_ID,
        "source_artifact_id": WEEK1_SOURCE_ARTIFACT_ID,
        "source_artifact_digest": WEEK1_SOURCE_ARTIFACT_DIGEST,
        "source_git_sha": WEEK1_SOURCE_GIT_SHA,
    }
    return out, audit


def attach_verified_actuals(
    projections: pd.DataFrame,
    actual: pd.DataFrame,
    roster: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Attach completed outcomes after the projection frame has been frozen."""
    p = projections.copy()
    a = actual.copy()
    r = roster.copy()
    idx = build_alias_index(a, r)

    actual_by_id = (
        a[["season", "week", "gsis_id", "rush_yards"]]
        .drop_duplicates(["season", "week", "gsis_id"], keep="last")
        .set_index(["season", "week", "gsis_id"])["rush_yards"]
        .to_dict()
    )
    roster_keys = set(zip(r["season"], r["week"], r["team"], r["gsis_id"]))

    rows = []
    exclusions = []
    stats_n = zero_n = 0
    for rec in p.to_dict("records"):
        gsis, status = resolve_gsis(str(rec["player_clean_key"]), str(rec["team"]), idx)
        if status != "RESOLVED_GSIS" or not gsis:
            exclusions.append({
                "player_clean_key": rec["player_clean_key"],
                "team": rec["team"],
                "reason": status,
            })
            continue

        key = (int(rec["season"]), int(rec["week"]), str(gsis))
        if key in actual_by_id:
            value = float(actual_by_id[key])
            source = "stats_table"
            stats_n += 1
        elif (int(rec["season"]), int(rec["week"]), str(rec["team"]), str(gsis)) in roster_keys:
            value = 0.0
            source = "roster_confirmed_verified_zero"
            zero_n += 1
        else:
            exclusions.append({
                "player_clean_key": rec["player_clean_key"],
                "team": rec["team"],
                "reason": "NO_VERIFIED_COMPLETED_OUTCOME",
            })
            continue

        if not np.isfinite(value):
            raise RuntimeError("non-finite verified rushing outcome")
        rec["actual_rush_yards"] = value
        rec["actual_source"] = source
        rec["gsis_id"] = str(gsis)
        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("zero Week-1 completed-history rows survived verified actual join")
    if out.duplicated(["season", "week", "team", "player_clean_key"]).any():
        raise RuntimeError("duplicate Week-1 completed-history identity after actual join")

    audit = {
        "verified_rows": int(len(out)),
        "verified_stats_table": int(stats_n),
        "verified_roster_zero": int(zero_n),
        "excluded_rows": int(len(exclusions)),
        "exclusions": exclusions,
    }
    return out, audit


def build_week1_history(context_path: Path, out_dir: Path) -> tuple[pd.DataFrame, dict]:
    if not context_path.exists() or context_path.stat().st_size == 0:
        raise RuntimeError(f"missing Week-1 P3 context: {context_path}")
    context = pd.read_csv(context_path, low_memory=False)
    projections, proj_audit = build_week1_projection_frame(context)

    # Outcome data are loaded only after projection lineage is frozen above.
    actual = load_actual_stats_unfiltered(2026, [1])
    roster = load_roster_identity(2026, [1])
    completed, actual_audit = attach_verified_actuals(projections, actual, roster)

    audit = {
        "version": "RB_PD2_COMPLETED_2026_HISTORY_W1_V1",
        "disposition": "WEEK1_COMPLETED_HISTORY_CERTIFIED",
        "context_path": str(context_path),
        "context_sha256": _file_sha256(context_path),
        "projection_audit": proj_audit,
        "actual_audit": actual_audit,
        "outcomes_loaded_after_projection_freeze": True,
        "prospective_confirmation_observations_created": 0,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    completed.to_csv(out_dir / "rb_pd2_completed_2026_week1_history.csv", index=False)
    (out_dir / "rb_pd2_completed_2026_week1_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return completed, audit


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--week1-context", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    out, audit = build_week1_history(a.week1_context, a.out_dir)
    print(f"[rb-pd2-2026-history] Week1 certified rows={len(out)}")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
