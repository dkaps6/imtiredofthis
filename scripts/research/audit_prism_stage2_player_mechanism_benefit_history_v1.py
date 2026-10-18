#!/usr/bin/env python3
"""Source-only audit for PRISM player-specific Stage-2 mechanism-benefit history.

Frozen contract:
docs/research/PRISM_STAGE2_PLAYER_MECHANISM_BENEFIT_HISTORY_V1_SOURCE_PLAN.md

Consumes the immutable row-level output of the already-scored Stage-2 bridge.
It constructs strictly-prior same-player history state only. It does not score
a personalization rule, fit coefficients, search thresholds, use 2026 outcomes,
or mutate production.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VERSION = "PRISM_STAGE2_PLAYER_MECHANISM_BENEFIT_HISTORY_V1_SOURCE_AUDIT"
FREEZE_COMMIT = "2a7d0f4a33b4669207f34bd98ee44128de7522c2"
PARENT_RUN = 38957575645
PARENT_HEAD = "eba5787c0b06aa5f975008965df7d023b3525ba7"
PARENT_ARTIFACT = 12070774378
PARENT_DIGEST = "sha256:e07edbad1bf97a4917ba95c58156c8941448e7b2668bd0eae2e9bcad5d4945b0"
QUALIFIED_CELLS = ("RB_RUSHING", "RB_RECEIVING", "TE_RECEIVING")
TOL = 1e-10

GATES = {
    "RB_RUSHING": {"rows_ge3": 900, "players_ge3": 90, "players_ge6": 60, "teams": 28},
    "RB_RECEIVING": {"rows_ge3": 900, "players_ge3": 90, "players_ge6": 60, "teams": 28},
    "TE_RECEIVING": {"rows_ge3": 800, "players_ge3": 75, "players_ge6": 50, "teams": 28},
}


def num(s: Any) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def support_tier(n: int) -> str:
    if n <= 0:
        return "TIER0_0"
    if n <= 2:
        return "TIER1_1_2"
    if n <= 5:
        return "TIER2_3_5"
    if n <= 9:
        return "TIER3_6_9"
    return "TIER4_10_PLUS"


def load_parent(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing parent Stage-2 row file: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    required = {
        "season", "week", "game_key", "team", "cell", "family", "player_id",
        "player_name_key", "stage1_p_active", "authority_q",
        "baseline_crps", "candidate_crps", "delta_crps",
        "baseline_ae", "candidate_ae", "delta_ae",
    }
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"parent rows missing columns: {sorted(missing)}")
    forbidden = [
        c for c in x.columns
        if any(t in c for t in (
            "sportsbook", "bookmaker", "prop_line", "market_line",
            "over_odds", "under_odds", "moneyline", "closing_line",
            "no_vig", "implied_prob",
        ))
    ]
    if forbidden:
        raise RuntimeError(f"forbidden sportsbook columns present: {forbidden}")

    x["season"] = num(x["season"]).astype("Int64")
    x["week"] = num(x["week"]).astype("Int64")
    for c in ("stage1_p_active", "authority_q", "baseline_crps", "candidate_crps",
              "delta_crps", "baseline_ae", "candidate_ae", "delta_ae"):
        x[c] = num(x[c])
    for c in ("game_key", "team", "cell", "family", "player_id", "player_name_key"):
        x[c] = x[c].astype("string").fillna("").str.strip()

    x = x.loc[x["cell"].isin(QUALIFIED_CELLS)].copy()
    if x.empty:
        raise RuntimeError("zero qualified Stage-2 rows after filter")
    if not x["season"].isin([2024, 2025]).all():
        bad = sorted(set(x.loc[~x["season"].isin([2024, 2025]), "season"].dropna().astype(int)))
        raise RuntimeError(f"unexpected seasons in parent rows: {bad}")
    if x["player_id"].eq("").any():
        raise RuntimeError("blank player_id in qualified parent rows")
    if x["team"].eq("").any() or x["game_key"].eq("").any():
        raise RuntimeError("blank team/game identity in qualified parent rows")
    if x.duplicated(["season", "week", "team", "cell", "player_id"]).any():
        raise RuntimeError("duplicate player/cell/week identity")
    if x[["baseline_crps", "candidate_crps", "delta_crps", "authority_q", "stage1_p_active"]].isna().any(axis=None):
        raise RuntimeError("non-finite required Stage-2 values")
    crps_gap = (x["baseline_crps"] - x["candidate_crps"] - x["delta_crps"]).abs()
    if float(crps_gap.max()) > TOL:
        raise RuntimeError(f"parent delta_crps identity drift max={float(crps_gap.max())}")
    ae_gap = (x["baseline_ae"] - x["candidate_ae"] - x["delta_ae"]).abs()
    if float(ae_gap.max()) > TOL:
        raise RuntimeError(f"parent delta_ae identity drift max={float(ae_gap.max())}")
    return x


def add_room_rank(x: pd.DataFrame) -> pd.DataFrame:
    z = x.copy()
    # Purely pregame/source-derived inventory. This is not selected using benefit.
    z["authority_rank"] = (
        z.groupby(["season", "week", "team", "cell"], sort=False)["authority_q"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    z["authority_rank_state"] = np.select(
        [
            z["authority_q"].le(TOL),
            z["authority_rank"].eq(1),
            z["authority_rank"].eq(2),
        ],
        ["ZERO", "RANK1", "RANK2"],
        default="RANK3_PLUS",
    )
    return z


def build_strict_prior_state(x: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    z = add_room_rank(x)
    z = z.sort_values(["cell", "player_id", "season", "week", "game_key", "team"]).reset_index(drop=True)

    prior_n = np.zeros(len(z), dtype=int)
    prior_mean = np.full(len(z), np.nan)
    prior_median = np.full(len(z), np.nan)
    prior_positive_share = np.full(len(z), np.nan)
    chronology_violations = 0

    for (_, _), idx in z.groupby(["cell", "player_id"], sort=False).groups.items():
        locs = list(idx)
        history: list[float] = []
        previous_key: tuple[int, int] | None = None
        for pos in locs:
            season = int(z.at[pos, "season"])
            week = int(z.at[pos, "week"])
            key = (season, week)
            if previous_key is not None and key <= previous_key:
                chronology_violations += 1
            prior_n[pos] = len(history)
            if history:
                arr = np.asarray(history, dtype=float)
                prior_mean[pos] = float(arr.mean())
                prior_median[pos] = float(np.median(arr))
                prior_positive_share[pos] = float((arr > 0).mean())
            # Append only after current-row state is captured.
            history.append(float(z.at[pos, "delta_crps"]))
            previous_key = key

    z["prior_mechanism_rows"] = prior_n
    z["prior_mean_delta_crps"] = prior_mean
    z["prior_median_delta_crps"] = prior_median
    z["prior_positive_share_delta_crps"] = prior_positive_share
    z["support_tier"] = [support_tier(int(n)) for n in prior_n]
    return z, chronology_violations


def summarize_cell(z: pd.DataFrame, cell: str) -> dict:
    g = z.loc[z["cell"].eq(cell) & z["season"].eq(2025)].copy()
    if g.empty:
        raise RuntimeError(f"zero 2025 target rows for {cell}")
    ge3 = g.loc[g["prior_mechanism_rows"].ge(3)]
    player_max = g.groupby("player_id")["prior_mechanism_rows"].max()
    spanning = (
        z.loc[z["cell"].eq(cell)]
        .groupby("player_id")["season"].nunique()
        .ge(2)
    )
    tier_counts = g["support_tier"].value_counts().to_dict()
    rank_finite = g["authority_rank"].notna().mean()
    stable_identity_rate = g["player_id"].ne("").mean()
    req = GATES[cell]
    gate = {
        "rows_with_ge3_prior": int(len(ge3)) >= req["rows_ge3"],
        "players_with_ge3_prior": int(ge3["player_id"].nunique()) >= req["players_ge3"],
        "players_with_ge6_prior": int((player_max >= 6).sum()) >= req["players_ge6"],
        "teams": int(g["team"].nunique()) >= req["teams"],
        "stable_player_identity_ge_99pct": float(stable_identity_rate) >= 0.99,
        "authority_rank_derivable": float(rank_finite) >= 0.99,
    }
    return {
        "cell": cell,
        "target_season": 2025,
        "target_rows": int(len(g)),
        "distinct_players": int(g["player_id"].nunique()),
        "rows_with_ge3_prior": int(len(ge3)),
        "players_with_ge3_prior": int(ge3["player_id"].nunique()),
        "players_with_ge6_prior": int((player_max >= 6).sum()),
        "players_with_ge10_prior": int((player_max >= 10).sum()),
        "players_spanning_2024_2025": int(spanning.sum()),
        "teams": int(g["team"].nunique()),
        "prior_history_median": float(g["prior_mechanism_rows"].median()),
        "prior_history_p25": float(g["prior_mechanism_rows"].quantile(0.25)),
        "prior_history_p75": float(g["prior_mechanism_rows"].quantile(0.75)),
        "tier_counts": {str(k): int(v) for k, v in tier_counts.items()},
        "stable_player_identity_rate": float(stable_identity_rate),
        "authority_rank_derivable_rate": float(rank_finite),
        "source_gates": gate,
        "source_ready": bool(all(gate.values())),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-rows", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        parent = load_parent(args.parent_rows)
        state, chronology_violations = build_strict_prior_state(parent)
        if chronology_violations:
            raise RuntimeError(f"chronology violations={chronology_violations}")

        dupes = int(state.duplicated(["season", "week", "team", "cell", "player_id"]).sum())
        if dupes:
            raise RuntimeError(f"duplicate identities after state build={dupes}")

        summaries = {cell: summarize_cell(state, cell) for cell in QUALIFIED_CELLS}
        dispositions = {
            cell: (
                f"{cell}_STAGE2_PLAYER_BENEFIT_HISTORY_SOURCE_READY"
                if rec["source_ready"]
                else f"{cell}_STAGE2_PLAYER_BENEFIT_HISTORY_SOURCE_BLOCKED"
            )
            for cell, rec in summaries.items()
        }
        ready_n = sum(rec["source_ready"] for rec in summaries.values())
        overall = (
            "PRISM_STAGE2_PLAYER_MECHANISM_BENEFIT_HISTORY_SOURCE_READY_ALL"
            if ready_n == len(QUALIFIED_CELLS)
            else "PRISM_STAGE2_PLAYER_MECHANISM_BENEFIT_HISTORY_SOURCE_READY_PARTIAL"
            if ready_n
            else "PRISM_STAGE2_PLAYER_MECHANISM_BENEFIT_HISTORY_SOURCE_BLOCKED"
        )

        # State artifact excludes the current target-row benefit. It contains
        # strictly-prior summaries only, suitable for a later frozen experiment.
        state_out = state[
            [
                "season", "week", "game_key", "team", "cell", "family",
                "player_id", "player_name_key", "stage1_p_active", "authority_q",
                "authority_rank", "authority_rank_state",
                "prior_mechanism_rows", "support_tier",
                "prior_mean_delta_crps", "prior_median_delta_crps",
                "prior_positive_share_delta_crps",
            ]
        ].copy()
        state_out.to_csv(args.out_dir / "stage2_player_mechanism_benefit_strict_prior_state.csv", index=False)

        support_rows = []
        for cell, rec in summaries.items():
            row = {k: v for k, v in rec.items() if k not in {"tier_counts", "source_gates"}}
            for k, v in rec["tier_counts"].items():
                row[f"tier_{k}"] = v
            for k, v in rec["source_gates"].items():
                row[f"gate_{k}"] = v
            support_rows.append(row)
        pd.DataFrame(support_rows).to_csv(
            args.out_dir / "stage2_player_mechanism_benefit_source_support.csv", index=False
        )

        result = {
            "version": VERSION,
            "freeze_commit": FREEZE_COMMIT,
            "parent_run": PARENT_RUN,
            "parent_head": PARENT_HEAD,
            "parent_artifact": PARENT_ARTIFACT,
            "parent_digest": PARENT_DIGEST,
            "qualified_cells": list(QUALIFIED_CELLS),
            "cell_summaries": summaries,
            "cell_dispositions": dispositions,
            "disposition": overall,
            "chronology_violations": 0,
            "duplicate_identities": 0,
            "personalization_rule_scored": False,
            "coefficients_fit": 0,
            "threshold_searches": 0,
            "outcomes_2026_used": 0,
            "week6_outcomes_used": 0,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "production_changed": False,
        }
    except Exception as exc:
        result = {
            "version": VERSION,
            "freeze_commit": FREEZE_COMMIT,
            "parent_run": PARENT_RUN,
            "parent_head": PARENT_HEAD,
            "parent_artifact": PARENT_ARTIFACT,
            "parent_digest": PARENT_DIGEST,
            "disposition": "PRISM_STAGE2_PLAYER_MECHANISM_BENEFIT_HISTORY_SOURCE_INTEGRITY_FAILURE",
            "error": str(exc),
            "personalization_rule_scored": False,
            "coefficients_fit": 0,
            "threshold_searches": 0,
            "outcomes_2026_used": 0,
            "week6_outcomes_used": 0,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "production_changed": False,
        }
        (args.out_dir / "stage2_player_mechanism_benefit_source_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        raise

    (args.out_dir / "stage2_player_mechanism_benefit_source_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
