#!/usr/bin/env python3
"""RB-R14B mechanical correction for PBP coverage accounting.

The frozen R14 football hypothesis and signal thresholds are unchanged. Before any
R14 science was executed, review caught that the draft coverage denominator used
all NFL receiver targets, which would mechanically penalize an RB-only mapper for
WR/TE targets. R14B measures PBP coverage against canonical weekly RB/FB targets
instead. No football feature, signal threshold, target mean, or production value is
changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest import diagnose_rb_r14_pbp_efficiency_role_v1 as r14
from scripts.backtest.audit_rb_receiving_identity_v1 import _load_logs


def _coverage(logs: pd.DataFrame, games: pd.DataFrame, seasons: list[int]) -> dict:
    exp = logs.loc[
        logs.position_family.isin({"RB", "FB"})
        & pd.to_numeric(logs.season, errors="coerce").isin(seasons)
    ].copy()
    exp["targets"] = pd.to_numeric(exp.targets, errors="coerce").fillna(0.0)
    exp = exp.loc[exp.targets.gt(0)].copy()
    exp = exp.groupby(["season", "week", "team", "player_clean_key"], as_index=False).agg(
        weekly_targets=("targets", "sum")
    )
    obs = games.groupby(["season", "week", "team", "player_clean_key"], as_index=False).agg(
        pbp_targets=("targets", "sum")
    )
    z = exp.merge(obs, on=["season", "week", "team", "player_clean_key"], how="left")
    z["pbp_targets"] = pd.to_numeric(z.pbp_targets, errors="coerce").fillna(0.0)
    z["matched_target_units"] = np.minimum(z.weekly_targets, z.pbp_targets)
    expected = float(z.weekly_targets.sum())
    matched = float(z.matched_target_units.sum())
    return {
        "expected_rb_target_player_games": int(len(z)),
        "pbp_present_rb_target_player_games": int(z.pbp_targets.gt(0).sum()),
        "rb_player_game_coverage_rate": float(z.pbp_targets.gt(0).mean()) if len(z) else np.nan,
        "expected_weekly_rb_targets": expected,
        "matched_pbp_rb_target_units": matched,
        "rb_weekly_target_coverage_rate": float(matched / expected) if expected > 0 else np.nan,
        "exact_target_count_game_rate": float(np.isclose(z.weekly_targets, z.pbp_targets, atol=1e-9).mean()) if len(z) else np.nan,
        "mean_abs_target_count_gap": float((z.weekly_targets - z.pbp_targets).abs().mean()) if len(z) else np.nan,
        "coverage_denominator": "canonical weekly RB/FB target units only",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--pbp-start", type=int, default=2018)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    q = pd.read_csv(a.predictions, low_memory=False)
    req = {"season", "week", "team", "player_clean_key", "identity_bucket", "actual_targets", "actual_rec_yards", "frozen_ypt", "state_probability"}
    missing = sorted(req - set(q.columns))
    if missing:
        raise RuntimeError(f"R14B predictions missing columns: {missing}")

    through = int(pd.to_numeric(q.season, errors="coerce").max())
    seasons = list(range(int(a.pbp_start), through + 1))
    logs = _load_logs(seasons)
    games, draft_mapping_audit = r14._pbp_rb_games(logs, seasons)
    if games.empty:
        raise RuntimeError("R14B produced zero RB PBP receiving games")
    coverage = _coverage(logs, games, seasons)

    x, time_audit = r14._attach(q, games)
    x["actual_targets"] = pd.to_numeric(x.actual_targets, errors="coerce")
    x["actual_rec_yards"] = pd.to_numeric(x.actual_rec_yards, errors="coerce")
    x["actual_ypt"] = np.where(x.actual_targets.gt(0), x.actual_rec_yards / x.actual_targets, np.nan)

    rows = []
    for sb, g0 in [("COMBINED", x)] + [(str(int(s)), g) for s, g in x.groupby("season")]:
        for pop, g1 in [
            ("ALL_RB", g0),
            ("TOP20_IDENTITY", g0.loc[g0.identity_bucket.eq("TOP20")]),
            ("REST80_IDENTITY", g0.loc[g0.identity_bucket.eq("REST80")]),
        ]:
            g = g1.loc[g1.actual_targets.ge(3)].copy()
            for f in r14.FEATURES:
                if f in g.columns:
                    row = r14._row(g, f, sb, pop)
                    if row is not None:
                        rows.append(row)
    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError("R14B produced zero feature summaries")

    combined = summary.loc[
        summary.season_bucket.eq("COMBINED") & summary.population.eq("ALL_RB")
    ].copy()
    primary = combined.loc[combined.feature.isin(r14.PRIMARY)].copy()
    primary["positive_seasons"] = 0
    for i, row in primary.iterrows():
        positive = 0
        for season in (2023, 2024, 2025):
            z = summary.loc[
                summary.season_bucket.eq(str(season))
                & summary.population.eq("ALL_RB")
                & summary.feature.eq(row.feature)
            ]
            positive += int(
                len(z) == 1
                and pd.notna(z.iloc[0].spearman_actual_ypt)
                and float(z.iloc[0].spearman_actual_ypt) > 0
            )
        primary.loc[i, "positive_seasons"] = positive

    mapping_ok = bool(
        pd.notna(coverage["rb_weekly_target_coverage_rate"])
        and coverage["rb_weekly_target_coverage_rate"] >= r14.MIN_PBP_MAPPING_RATE
    )
    signal = (
        pd.to_numeric(primary.spearman_actual_ypt, errors="coerce").ge(r14.MIN_PRIMARY_SPEARMAN)
        & pd.to_numeric(primary.high8_auc, errors="coerce").ge(r14.MIN_PRIMARY_HIGH8_AUC)
        & pd.to_numeric(primary.positive_seasons, errors="coerce").ge(r14.MIN_POSITIVE_SEASONS)
    )
    strict_ok = (
        time_audit["strict_prior_player_time_violations"] == 0
        and time_audit["strict_prior_same_team_time_violations"] == 0
    )
    supported = bool(mapping_ok and strict_ok and signal.any())

    best = None
    if len(primary):
        b = primary.sort_values(["spearman_actual_ypt", "high8_auc"], ascending=False).iloc[0]
        best = {
            "feature": str(b.feature),
            "spearman_actual_ypt": float(b.spearman_actual_ypt),
            "high8_auc": float(b.high8_auc),
            "high10_auc": float(b.high10_auc),
            "positive_seasons": int(b.positive_seasons),
        }

    result = {
        "diagnostic": "RB_R14_PBP_EFFICIENCY_ROLE_V1B",
        "disposition": "RB_R14_PBP_EFFICIENCY_SIGNAL_SUPPORTED_DIAGNOSTIC_ONLY" if supported else "RB_R14_PBP_EFFICIENCY_SIGNAL_NOT_SUPPORTED_DIAGNOSTIC_ONLY",
        "pbp_efficiency_signal_supported": supported,
        "r13_status": "RB_R13_EFFICIENCY_HISTORY_SIGNAL_NOT_SUPPORTED_DIAGNOSTIC_ONLY_UNCHANGED",
        "best_primary_signal": best,
        "gates": {
            "correct_rb_target_coverage": mapping_ok,
            "strict_prior_integrity": strict_ok,
            "frozen_primary_signal_gate": bool(signal.any()),
        },
        "thresholds": {
            "min_primary_spearman": r14.MIN_PRIMARY_SPEARMAN,
            "min_primary_high8_auc": r14.MIN_PRIMARY_HIGH8_AUC,
            "min_positive_seasons": r14.MIN_POSITIVE_SEASONS,
            "min_pbp_mapping_rate": r14.MIN_PBP_MAPPING_RATE,
        },
        "corrected_pbp_coverage_audit": coverage,
        "draft_mapping_audit_retained_for_lineage": draft_mapping_audit,
        "strict_prior_audit": time_audit,
        "mechanical_correction_only": True,
        "football_features_changed_from_r14": 0,
        "science_thresholds_changed_from_r14": 0,
        "sportsbook_inputs_added": 0,
        "production_parameters_changed": 0,
        "governance_note": "2023-2025 are research-visible diagnostic seasons; support only authorizes a separately frozen candidate/OOS design.",
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    x.to_csv(a.out_dir / "rb_r14b_pbp_efficiency_casebook.csv", index=False)
    summary.to_csv(a.out_dir / "rb_r14b_pbp_efficiency_feature_summary.csv", index=False)
    primary.to_csv(a.out_dir / "rb_r14b_primary_summary.csv", index=False)
    (a.out_dir / "rb_r14b_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("\n=== primary PBP signals ===\n", primary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
