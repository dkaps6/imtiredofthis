#!/usr/bin/env python3
"""Build a future PRISM Stage-2 market-translation trace.

Mechanics only. This module translates an already-locked coherent Stage-2
opportunity row into market-specific football means while holding the existing
pregame efficiency authority fixed.

It explicitly refuses Week 6, target-game outcomes, and sportsbook predictors.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VERSION = "PRISM_FUTURE_STAGE2_MARKET_TRANSLATION_TRACE_V1"
ELIGIBLE_CELLS = {"RB_RUSHING", "RB_RECEIVING", "TE_RECEIVING"}
TOL = 1e-12

STAGE2_REQUIRED = {
    "season", "week", "event_id", "team", "cell", "family", "player",
    "player_clean_key", "player_id", "p_active", "stage1_probability_source",
    "canonical_q", "candidate_expected_q", "share_shift",
    "team_opportunity_mean", "baseline_expected_opportunities",
    "candidate_expected_opportunities", "expected_opportunity_shift",
    "availability_authority", "final_availability_state",
}

UNIVERSE_REQUIRED = {
    "event_id", "team", "player_clean_key", "player_id",
    "rules_ypc", "rules_ypt", "rules_catch_rate",
}

FORBIDDEN_COLUMN_TOKENS = (
    "sportsbook",
    "bookmaker",
    "american_odds",
    "decimal_odds",
    "over_odds",
    "under_odds",
    "no_vig",
    "market_implied",
    "prop_line",
    "fair_probability",
    "fair_prob",
)

OUTCOME_PREFIXES = (
    "actual_",
    "target_game_actual",
    "realized_",
    "settled_",
    "outcome_",
)


def num(v: Any) -> pd.Series:
    return pd.to_numeric(v, errors="coerce")


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing/empty {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def audit_columns(frame: pd.DataFrame, label: str) -> None:
    bad_market = [
        c for c in frame.columns
        if any(tok in c for tok in FORBIDDEN_COLUMN_TOKENS)
    ]
    if bad_market:
        raise RuntimeError(f"{label} contains forbidden sportsbook/market columns: {bad_market}")

    bad_outcome = [
        c for c in frame.columns
        if c in {"actual", "result", "settled", "target_outcome"}
        or any(c.startswith(prefix) for prefix in OUTCOME_PREFIXES)
    ]
    if bad_outcome:
        raise RuntimeError(f"{label} contains forbidden target-outcome columns: {bad_outcome}")


def validate_inputs(stage2: pd.DataFrame, universe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_columns(stage2, "Stage-2 lock")
    audit_columns(universe, "football universe")

    missing = STAGE2_REQUIRED - set(stage2.columns)
    if missing:
        raise RuntimeError(f"Stage-2 lock missing {sorted(missing)}")
    missing = UNIVERSE_REQUIRED - set(universe.columns)
    if missing:
        raise RuntimeError(f"football universe missing {sorted(missing)}")

    s = stage2.copy()
    u = universe.copy()

    s["season"] = num(s["season"])
    s["week"] = num(s["week"])
    s = s.loc[s["cell"].isin(ELIGIBLE_CELLS)].copy()
    if s.empty:
        raise RuntimeError("zero eligible Stage-2 rows")

    if s["season"].isna().any() or s["week"].isna().any():
        raise RuntimeError("Stage-2 season/week missing")
    if ((s["season"].eq(2026)) & (s["week"].le(6))).any():
        raise RuntimeError("Week-6-or-earlier 2026 backfill is forbidden for future translation mechanics")

    target_pairs = s[["season", "week"]].drop_duplicates()
    if len(target_pairs) != 1:
        raise RuntimeError(f"translation lock must contain one target season/week, got {target_pairs.to_dict('records')}")

    key = ["event_id", "team", "cell", "player_id"]
    if s.duplicated(key).any():
        raise RuntimeError("duplicate Stage-2 player-cell identity")

    for c in (
        "p_active", "canonical_q", "candidate_expected_q",
        "team_opportunity_mean", "baseline_expected_opportunities",
        "candidate_expected_opportunities", "expected_opportunity_shift",
    ):
        s[c] = num(s[c])
        if s[c].isna().any() or (~np.isfinite(s[c])).any():
            raise RuntimeError(f"non-finite Stage-2 numeric field {c}")

    for c in ("p_active", "canonical_q", "candidate_expected_q"):
        if ((s[c] < -TOL) | (s[c] > 1.0 + TOL)).any():
            raise RuntimeError(f"Stage-2 probability/share outside [0,1]: {c}")

    for c in ("team_opportunity_mean", "baseline_expected_opportunities", "candidate_expected_opportunities"):
        if (s[c] < -TOL).any():
            raise RuntimeError(f"negative Stage-2 opportunity field {c}")

    share_gap = (
        s["candidate_expected_q"] - s["canonical_q"] - num(s["share_shift"])
    ).abs()
    if float(share_gap.max()) > TOL:
        raise RuntimeError(f"Stage-2 share-shift identity drift: {share_gap.max()}")

    baseline_mean_gap = (
        s["team_opportunity_mean"] * s["canonical_q"]
        - s["baseline_expected_opportunities"]
    ).abs()
    if float(baseline_mean_gap.max()) > TOL:
        raise RuntimeError(
            f"Stage-2 baseline opportunity identity drift: {baseline_mean_gap.max()}"
        )

    candidate_mean_gap = (
        s["team_opportunity_mean"] * s["candidate_expected_q"]
        - s["candidate_expected_opportunities"]
    ).abs()
    if float(candidate_mean_gap.max()) > TOL:
        raise RuntimeError(
            f"Stage-2 candidate opportunity identity drift: {candidate_mean_gap.max()}"
        )

    shift_gap = (
        s["candidate_expected_opportunities"]
        - s["baseline_expected_opportunities"]
        - s["expected_opportunity_shift"]
    ).abs()
    if float(shift_gap.max()) > TOL:
        raise RuntimeError(f"Stage-2 opportunity-shift identity drift: {shift_gap.max()}")

    ukey = ["event_id", "team", "player_id"]
    if u.duplicated(ukey).any():
        raise RuntimeError("duplicate football-universe event/team/player identity")

    for c in ("rules_ypc", "rules_ypt", "rules_catch_rate"):
        u[c] = num(u[c])

    return s, u


def _translation_rows(row: pd.Series) -> list[dict]:
    common = {
        "season": int(row["season"]),
        "week": int(row["week"]),
        "event_id": str(row["event_id"]),
        "team": str(row["team"]),
        "cell": str(row["cell"]),
        "family": str(row["family"]),
        "player": str(row["player"]),
        "player_clean_key": str(row["player_clean_key"]),
        "player_id": str(row["player_id"]),
        "p_active": float(row["p_active"]),
        "stage1_probability_source": str(row["stage1_probability_source"]),
        "canonical_q": float(row["canonical_q"]),
        "candidate_expected_q": float(row["candidate_expected_q"]),
        "share_shift": float(row["share_shift"]),
        "team_opportunity_mean": float(row["team_opportunity_mean"]),
        "baseline_expected_opportunities": float(row["baseline_expected_opportunities"]),
        "candidate_expected_opportunities": float(row["candidate_expected_opportunities"]),
        "expected_opportunity_shift": float(row["expected_opportunity_shift"]),
        "availability_authority": str(row["availability_authority"]),
        "final_availability_state": str(row["final_availability_state"]),
    }

    out: list[dict] = []

    def add(market: str, eff_field: str, eff_value: float) -> None:
        if not np.isfinite(eff_value):
            raise RuntimeError(
                f"missing/nonfinite {eff_field} for {row['event_id']} {row['player_id']} {row['cell']}"
            )
        if market == "rush_yards" and eff_value <= 0:
            raise RuntimeError("rules_ypc must be >0")
        if market == "receptions" and not (0.0 <= eff_value <= 1.0):
            raise RuntimeError("rules_catch_rate must be in [0,1]")
        if market == "rec_yards" and eff_value < 0:
            raise RuntimeError("rules_ypt must be >=0")

        b = float(row["baseline_expected_opportunities"]) * eff_value
        c = float(row["candidate_expected_opportunities"]) * eff_value
        shift = c - b
        expected_shift = float(row["expected_opportunity_shift"]) * eff_value
        gap = abs(shift - expected_shift)
        if gap > TOL:
            raise RuntimeError(
                f"market translation shift identity drift {market}: {gap}"
            )
        out.append({
            **common,
            "market": market,
            "efficiency_field": eff_field,
            "efficiency_value": eff_value,
            "baseline_football_mean": b,
            "candidate_football_mean": c,
            "candidate_minus_baseline_mean": shift,
            "shift_identity_gap": gap,
            "efficiency_changed_by_translation": False,
        })

    cell = str(row["cell"])
    if cell == "RB_RUSHING":
        add("rush_attempts", "IDENTITY_1", 1.0)
        add("rush_yards", "rules_ypc", float(row["rules_ypc"]))
    elif cell in {"RB_RECEIVING", "TE_RECEIVING"}:
        add("targets", "IDENTITY_1", 1.0)
        add("receptions", "rules_catch_rate", float(row["rules_catch_rate"]))
        add("rec_yards", "rules_ypt", float(row["rules_ypt"]))
    else:
        raise RuntimeError(f"unexpected eligible cell {cell}")

    return out


def build(stage2: pd.DataFrame, universe: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    s, u = validate_inputs(stage2, universe)

    joined = s.merge(
        u[[
            "event_id", "team", "player_id", "player_clean_key",
            "rules_ypc", "rules_ypt", "rules_catch_rate",
        ]],
        on=["event_id", "team", "player_id"],
        how="left",
        suffixes=("", "_universe"),
        validate="many_to_one",
    )

    if joined["player_clean_key_universe"].isna().any():
        bad = joined.loc[
            joined["player_clean_key_universe"].isna(),
            ["event_id", "team", "cell", "player", "player_id"],
        ].to_dict("records")
        raise RuntimeError(f"missing football-universe identity: {bad[:10]}")

    mismatch = (
        joined["player_clean_key"].astype(str)
        != joined["player_clean_key_universe"].astype(str)
    )
    if mismatch.any():
        bad = joined.loc[
            mismatch,
            ["event_id", "team", "cell", "player_clean_key", "player_clean_key_universe", "player_id"],
        ].to_dict("records")
        raise RuntimeError(f"player identity mismatch: {bad[:10]}")

    rows: list[dict] = []
    for _, r in joined.iterrows():
        rows.extend(_translation_rows(r))

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("zero market-translation rows")

    expected_rows = (
        int(s["cell"].eq("RB_RUSHING").sum()) * 2
        + int(s["cell"].isin({"RB_RECEIVING", "TE_RECEIVING"}).sum()) * 3
    )
    if len(out) != expected_rows:
        raise RuntimeError(f"translation row-count drift {len(out)} != {expected_rows}")

    max_gap = float(num(out["shift_identity_gap"]).max())
    if max_gap > TOL:
        raise RuntimeError(f"max market shift identity gap {max_gap}")

    by_market = (
        out.groupby(["cell", "market"], as_index=False)
        .agg(
            rows=("player_id", "size"),
            players=("player_id", "nunique"),
            mean_abs_market_shift=("candidate_minus_baseline_mean", lambda z: float(num(z).abs().mean())),
            max_abs_market_shift=("candidate_minus_baseline_mean", lambda z: float(num(z).abs().max())),
        )
        .to_dict("records")
    )

    season = int(s["season"].iloc[0])
    week = int(s["week"].iloc[0])
    result = {
        "version": VERSION,
        "status": "PRISM_FUTURE_STAGE2_MARKET_TRANSLATION_TRACE_BUILT",
        "target_season": season,
        "target_week": week,
        "stage2_rows": int(len(s)),
        "translation_rows": int(len(out)),
        "players": int(out["player_id"].nunique()),
        "eligible_cells": sorted(s["cell"].unique().tolist()),
        "market_summary": by_market,
        "max_shift_identity_gap": max_gap,
        "efficiency_fields_changed": 0,
        "sportsbook_inputs_used": 0,
        "target_outcomes_read": 0,
        "week6_backfill": False,
        "oddsapi_calls": 0,
        "models_fit": 0,
        "threshold_searches": 0,
        "production_changed": False,
        "scientific_lock": False,
        "boundary": (
            "mechanical market translation of an already-coherent Stage-2 opportunity "
            "lock with existing player-specific efficiency held fixed"
        ),
    }
    return out, result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2-rows", type=Path, required=True)
    ap.add_argument("--football-universe", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows, result = build(
        read_csv(args.stage2_rows, "Stage-2 lock rows"),
        read_csv(args.football_universe, "football universe"),
    )

    rows.to_csv(
        args.out_dir / "prism_future_stage2_market_translation_rows.csv",
        index=False,
        float_format="%.12f",
    )
    (args.out_dir / "prism_future_stage2_market_translation_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
