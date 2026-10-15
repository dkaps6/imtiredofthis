#!/usr/bin/env python3
"""Freeze prospective Week-6 WR efficiency-trust labels.

Research only. Reuses the already-frozen W1-W5 WR-R15 pregame reconstruction
from the opportunity-trust lock and adds only settled receiving-yard labels
from the no-odds Week-6 Full Slate artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "WR_PLAYER_EFFICIENCY_DIFFICULTY_TRUST_WEEK6_PREGAME_LOCK_V1"
STATUS = "WEEK6_WR_EFFICIENCY_TRUST_PREGAME_LOCK_FROZEN"
SEASON = 2026
WEEK = 6
EXPECTED_COHORT = 152
MIN_PRIOR = 4
PRIOR_WINDOW = 8
TOL = 1e-9


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing/empty {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def num(s):
    return pd.to_numeric(s, errors="coerce")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-result", type=Path, required=True)
    ap.add_argument("--opportunity-lock-result", type=Path, required=True)
    ap.add_argument("--opportunity-lock-rows", type=Path, required=True)
    ap.add_argument("--opportunity-history", type=Path, required=True)
    ap.add_argument("--player-logs", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    parent = json.loads(args.parent_result.read_text())
    opp_result = json.loads(args.opportunity_lock_result.read_text())

    if parent.get("disposition") != (
        "PASS_CATCHER_EFFICIENCY_DIFFICULTY_TRUST_SELECTIVITY_PARTIAL"
    ):
        raise RuntimeError("historical parent overall disposition drift")
    if parent.get("wr", {}).get("disposition") != (
        "WR_PLAYER_EFFICIENCY_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED"
    ):
        raise RuntimeError("WR historical efficiency-trust parent not qualified")
    if parent.get("te", {}).get("disposition") != (
        "NO_ACTIONABLE_TE_PLAYER_EFFICIENCY_DIFFICULTY_TRUST_SELECTIVITY"
    ):
        raise RuntimeError("TE historical efficiency-trust disposition drift")
    for key in (
        "sportsbook_inputs_used",
        "oddsapi_calls",
        "outcomes_2026_read",
        "predictive_models_fit",
        "projection_distribution_values_changed",
    ):
        if int(parent.get(key, 1)) != 0:
            raise RuntimeError(f"historical parent leakage: {key}")
    if bool(parent.get("production_changed", True)):
        raise RuntimeError("historical parent production mutation")

    if opp_result.get("status") != "WEEK6_WR_TRUST_PREGAME_LOCK_FROZEN":
        raise RuntimeError("Week6 WR opportunity-trust lock status drift")
    if (
        int(opp_result.get("week6_outcomes_read", 1)) != 0
        or int(opp_result.get("sportsbook_inputs_used", 1)) != 0
        or int(opp_result.get("oddsapi_calls", 1)) != 0
        or bool(opp_result.get("production_changed", True))
    ):
        raise RuntimeError("Week6 WR opportunity-trust parent boundary failed")

    cohort = read_csv(args.opportunity_lock_rows, "Week6 WR opportunity lock")
    history = read_csv(args.opportunity_history, "Week6 WR strict-prior history")
    logs = read_csv(args.player_logs, "no-odds Week6 player logs")

    if (
        len(cohort) != EXPECTED_COHORT
        or int(cohort["player_clean_key"].nunique()) != EXPECTED_COHORT
    ):
        raise RuntimeError("Week6 WR cohort drift")
    if cohort["player_clean_key"].fillna("").astype(str).str.strip().eq("").any():
        raise RuntimeError("blank Week6 WR identity")

    if int(num(history["week"]).max()) >= WEEK:
        raise RuntimeError("Week6-or-later row in frozen WR history")
    hist_keys = ["season", "week", "team", "player_clean_key"]
    if history.duplicated(hist_keys).any():
        raise RuntimeError("duplicate frozen WR history identity")

    season = num(logs["season"])
    week = num(logs["week"])
    bad = sorted(
        week.loc[season.eq(SEASON) & week.ge(WEEK)]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    if bad:
        raise RuntimeError(
            f"Week6-or-later outcomes present; refuse prospective lock: {bad}"
        )

    need = {
        "season",
        "week",
        "team",
        "player_clean_key",
        "targets",
        "receiving_yards",
    }
    missing = need - set(logs.columns)
    if missing:
        raise RuntimeError(f"player logs missing {sorted(missing)}")

    current = logs.loc[
        season.eq(SEASON) & week.between(1, WEEK - 1)
    ].copy()
    current["targets"] = num(current["targets"]).fillna(0.0)
    current["receiving_yards"] = num(current["receiving_yards"]).fillna(0.0)
    actual = (
        current.groupby(hist_keys, as_index=False)[["targets", "receiving_yards"]]
        .sum()
        .rename(
            columns={
                "targets": "actual_targets_log",
                "receiving_yards": "actual_rec_yards",
            }
        )
    )

    h = history.merge(
        actual,
        on=hist_keys,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    zero_stat_missing = h["_merge"].eq("left_only")
    bad_missing = zero_stat_missing & num(h["actual_targets"]).abs().gt(TOL)
    if bad_missing.any():
        raise RuntimeError("missing log row with nonzero frozen actual targets")

    h["actual_targets_log"] = num(h["actual_targets_log"]).fillna(0.0)
    h["actual_rec_yards"] = num(h["actual_rec_yards"]).fillna(0.0)

    target_gap = (
        num(h["actual_targets"]) - h["actual_targets_log"]
    ).abs()
    if float(target_gap.max()) > TOL:
        raise RuntimeError(
            f"frozen actual-target parity failed: {float(target_gap.max())}"
        )

    h["efficiency_error"] = (
        num(h["actual_targets"]) * num(h["pred_ypt"])
        - h["actual_rec_yards"]
    )
    h["total_error_reconstructed"] = (
        num(h["pred_rec_yards"]) - h["actual_rec_yards"]
    )
    decomp_gap = (
        num(h["opportunity_error"])
        + h["efficiency_error"]
        - h["total_error_reconstructed"]
    ).abs()

    if float(decomp_gap.max()) > 1e-8:
        raise RuntimeError(
            f"WR efficiency decomposition failed: {float(decomp_gap.max())}"
        )
    if not np.isfinite(h["efficiency_error"].to_numpy(float)).all():
        raise RuntimeError("nonfinite reconstructed WR efficiency errors")

    rows = []
    for r in cohort.itertuples(index=False):
        player_key = str(r.player_clean_key)
        q = (
            h.loc[h["player_clean_key"].astype(str).eq(player_key)]
            .sort_values(["season", "week"])
            .tail(PRIOR_WINDOW)
        )
        supported = len(q) >= MIN_PRIOR
        mae = (
            float(np.mean(np.abs(q["efficiency_error"].to_numpy(float))))
            if supported
            else np.nan
        )
        rows.append(
            {
                "season": SEASON,
                "week": WEEK,
                "event_id": str(r.event_id),
                "team": str(r.team),
                "opponent": str(r.opponent),
                "player": str(r.player),
                "player_clean_key": player_key,
                "prior_games": int(len(q)),
                "player_efficiency_mae": mae,
                "supported": bool(supported),
                "history_min_week": int(q.iloc[0]["week"]) if len(q) else np.nan,
                "history_max_week": int(q.iloc[-1]["week"]) if len(q) else np.nan,
                "opportunity_player_mae": getattr(r, "player_mae", np.nan),
                "opportunity_trust_fragility_score": getattr(
                    r, "trust_fragility_score", np.nan
                ),
                "opportunity_trust_bucket": str(
                    getattr(r, "trust_bucket", "UNSUPPORTED")
                ),
            }
        )

    out = pd.DataFrame(rows)
    supported = out.loc[out["supported"]].copy()
    if supported.empty:
        raise RuntimeError("zero supported Week6 WR efficiency-trust rows")

    pool = float(supported["player_efficiency_mae"].median())
    if not np.isfinite(pool) or pool <= 0:
        raise RuntimeError("invalid Week6 WR efficiency pool MAE")

    supported["week_pool_efficiency_mae"] = pool
    supported["efficiency_difficulty_ratio_vs_pool"] = (
        supported["player_efficiency_mae"] / pool
    )
    supported["efficiency_fragility_score"] = supported[
        "player_efficiency_mae"
    ].rank(method="average", pct=True)
    supported["efficiency_trust_bucket"] = "MIDDLE"
    supported.loc[
        supported["efficiency_fragility_score"].le(0.25),
        "efficiency_trust_bucket",
    ] = "HIGH_EFFICIENCY_TRUST"
    supported.loc[
        supported["efficiency_fragility_score"].ge(0.75),
        "efficiency_trust_bucket",
    ] = "LOW_EFFICIENCY_TRUST"

    out = out.merge(
        supported[
            [
                "player_clean_key",
                "week_pool_efficiency_mae",
                "efficiency_difficulty_ratio_vs_pool",
                "efficiency_fragility_score",
                "efficiency_trust_bucket",
            ]
        ],
        on="player_clean_key",
        how="left",
        validate="one_to_one",
    )
    out["efficiency_trust_bucket"] = out[
        "efficiency_trust_bucket"
    ].fillna("UNSUPPORTED")
    out = out.sort_values(["team", "player_clean_key"]).reset_index(drop=True)

    payload = out.to_csv(
        index=False, float_format="%.12f", lineterminator="\n"
    ).encode()
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()

    result = {
        "version": VERSION,
        "status": STATUS,
        "historical_parent_run": 38826418315,
        "season": SEASON,
        "week": WEEK,
        "current_wr_cohort": int(len(out)),
        "supported_wr": int(out["supported"].sum()),
        "unsupported_wr": int((~out["supported"]).sum()),
        "efficiency_trust_bucket_counts": {
            str(k): int(v)
            for k, v in out["efficiency_trust_bucket"].value_counts().to_dict().items()
        },
        "week_pool_efficiency_mae": pool,
        "reconstructed_2026_rows": int(len(h)),
        "history_source_max_week_2026": int(num(h["week"]).max()),
        "actual_target_parity_max_gap": float(target_gap.max()),
        "efficiency_decomposition_max_gap": float(decomp_gap.max()),
        "missing_zero_stat_player_weeks_filled_zero": int(zero_stat_missing.sum()),
        "row_csv_sha256": digest,
        "prior_window": PRIOR_WINDOW,
        "min_prior_games": MIN_PRIOR,
        "sportsbook_inputs_used": 0,
        "oddsapi_calls": 0,
        "week6_outcomes_read": 0,
        "models_fit": 0,
        "projection_distribution_values_changed": 0,
        "production_changed": False,
        "opportunity_efficiency_scores_combined": False,
    }

    out.to_csv(
        args.out_dir / "wr_week6_efficiency_trust_lock.csv",
        index=False,
        float_format="%.12f",
    )
    h.to_csv(
        args.out_dir / "wr_week6_efficiency_history.csv",
        index=False,
        float_format="%.12f",
    )
    (
        args.out_dir / "wr_week6_efficiency_trust_lock.json"
    ).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
