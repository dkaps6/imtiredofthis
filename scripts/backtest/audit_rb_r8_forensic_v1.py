#!/usr/bin/env python3
"""Diagnostic-only forensic audit of the failed RB-R8 receiving-identity candidate.

R8 remains a scientific FAIL. This script does not tune, refit, or change R8.
It asks *why* the identity signal improved aggregate targets/yards and severe tails
while failing the fresh p90/cat30/top20-MAE gates.

Questions are frozen before inspecting this forensic output:
1. Are harmful errors concentrated in the largest R8 receiving-yard adjustments?
2. Did R8 often move in the correct direction but overshoot the realized outcome?
3. Are failures concentrated among sparse-history or same-team-history cases?
4. Does the top receiving-identity group need reliability shrinkage rather than a
   wholesale within-room rerank?

Current-game outcomes are labels only. All history features are strict-prior.
Sportsbook data is neither loaded nor used. No production change is authorized.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest import audit_rb_receiving_identity_v1 as ident

BASE = "M38_EXPLICIT_BASELINE"
CAND = "RB_R8_RECEIVING_IDENTITY"
RB = {"RB", "FB"}


def _num(x):
    return pd.to_numeric(x, errors="coerce")


def _paired(pred: pd.DataFrame) -> pd.DataFrame:
    x = pred.loc[pred.position_family.isin(RB)].copy()
    keys = [
        "season", "week", "event_id", "team", "player_clean_key", "player",
        "position_family", "actual_targets", "actual_rec_yards",
        "prior_rb_room_share", "train_season",
    ]
    b = x.loc[x.variant.eq(BASE), keys + ["entitlement_tgt_share", "pred_targets", "mc_rec_yards"]].rename(columns={
        "entitlement_tgt_share":"baseline_entitlement", "pred_targets":"baseline_targets", "mc_rec_yards":"baseline_rec_yards"
    })
    c = x.loc[x.variant.eq(CAND), keys + ["entitlement_tgt_share", "pred_targets", "mc_rec_yards"]].rename(columns={
        "entitlement_tgt_share":"candidate_entitlement", "pred_targets":"candidate_targets", "mc_rec_yards":"candidate_rec_yards"
    })
    z = b.merge(c, on=keys, how="inner", validate="one_to_one")
    z["delta_entitlement"] = z.candidate_entitlement - z.baseline_entitlement
    z["delta_targets"] = z.candidate_targets - z.baseline_targets
    z["delta_rec_yards"] = z.candidate_rec_yards - z.baseline_rec_yards
    z["abs_rec_yard_move"] = z.delta_rec_yards.abs()
    z["baseline_error"] = (z.baseline_rec_yards - z.actual_rec_yards).abs()
    z["candidate_error"] = (z.candidate_rec_yards - z.actual_rec_yards).abs()
    z["mae_gain"] = z.baseline_error - z.candidate_error
    z["baseline_signed_error"] = z.baseline_rec_yards - z.actual_rec_yards
    z["candidate_signed_error"] = z.candidate_rec_yards - z.actual_rec_yards
    desired = z.actual_rec_yards - z.baseline_rec_yards
    z["movement_correct_direction"] = np.sign(z.delta_rec_yards).eq(np.sign(desired)) | z.delta_rec_yards.eq(0)
    z["crossed_actual"] = (z.baseline_signed_error * z.candidate_signed_error).lt(0)
    z["harmful"] = z.mae_gain.lt(0)
    z["helpful"] = z.mae_gain.gt(0)
    z["cat30_baseline"] = z.baseline_error.ge(30)
    z["cat30_candidate"] = z.candidate_error.ge(30)
    z["cat50_baseline"] = z.baseline_error.ge(50)
    z["cat50_candidate"] = z.candidate_error.ge(50)
    z["identity_pct"] = z.groupby(["season", "week"])["prior_rb_room_share"].rank(pct=True, method="average")
    z["identity_bucket"] = np.where(z.identity_pct.gt(.80), "TOP20", "REST80")
    return z


def _attach_history(z: pd.DataFrame, history_start: int) -> pd.DataFrame:
    through = int(z.season.max())
    logs = ident._add_rb_room_share(ident._load_logs(list(range(history_start, through + 1))))
    rb = logs.loc[logs.position_family.isin(RB)].copy()
    states = ident._build_states(rb)
    prev = ident._previous_season_features(rb)
    q = z[["season", "week", "team", "player_clean_key"]].copy()
    feat = ident._snapshot_queries(q, states, prev)
    keep = [c for c in [
        "prior_games", "same_team_prior_games", "prev_season_games",
        "prior_targets_pg", "prior_receptions_pg", "prior_target_share", "prior_rb_room_share",
        "last8_targets_pg", "last8_receptions_pg", "last8_target_share", "last8_rb_room_share",
        "same_team_prior_targets_pg", "same_team_prior_rb_room_share",
        "prev_season_targets_pg", "prev_season_receptions_pg", "prev_season_target_share", "prev_season_rb_room_share",
    ] if c in feat.columns]
    out = z.copy()
    for c in keep:
        # snapshot query preserves row order
        out[c + "_snapshot"] = _num(feat[c]).to_numpy()
    pg = _num(out.get("prior_games_snapshot", np.nan))
    st = _num(out.get("same_team_prior_games_snapshot", np.nan))
    out["prior_games"] = pg
    out["same_team_prior_games"] = st
    out["history_bin"] = pd.cut(pg.fillna(0), [-1, 0, 3, 7, 15, np.inf], labels=["0", "1-3", "4-7", "8-15", "16+"])
    out["same_team_history_bin"] = pd.cut(st.fillna(0), [-1, 0, 3, 7, 15, np.inf], labels=["0", "1-3", "4-7", "8-15", "16+"])
    out["team_continuity"] = np.where(st.fillna(0).ge(1), "HAS_SAME_TEAM_HISTORY", "NO_SAME_TEAM_HISTORY")
    return out


def _summary(g: pd.DataFrame) -> dict:
    return {
        "n": int(len(g)),
        "baseline_mae": float(g.baseline_error.mean()) if len(g) else np.nan,
        "candidate_mae": float(g.candidate_error.mean()) if len(g) else np.nan,
        "mae_gain": float(g.mae_gain.mean()) if len(g) else np.nan,
        "mean_abs_move": float(g.abs_rec_yard_move.mean()) if len(g) else np.nan,
        "median_abs_move": float(g.abs_rec_yard_move.median()) if len(g) else np.nan,
        "harm_rate": float(g.harmful.mean()) if len(g) else np.nan,
        "correct_direction_rate": float(g.movement_correct_direction.mean()) if len(g) else np.nan,
        "cross_actual_rate": float(g.crossed_actual.mean()) if len(g) else np.nan,
        "baseline_bias": float(g.baseline_signed_error.mean()) if len(g) else np.nan,
        "candidate_bias": float(g.candidate_signed_error.mean()) if len(g) else np.nan,
        "baseline_cat30_rate": float(g.cat30_baseline.mean()) if len(g) else np.nan,
        "candidate_cat30_rate": float(g.cat30_candidate.mean()) if len(g) else np.nan,
        "baseline_cat50_rate": float(g.cat50_baseline.mean()) if len(g) else np.nan,
        "candidate_cat50_rate": float(g.cat50_candidate.mean()) if len(g) else np.nan,
    }


def _group_table(z: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in z.groupby(cols, observed=False, dropna=False):
        if not isinstance(key, tuple): key = (key,)
        row = dict(zip(cols, key)); row.update(_summary(g)); rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--history-start", type=int, default=2015)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(a.predictions)
    z = _attach_history(_paired(pred), a.history_start)

    # Magnitude buckets are descriptive quantiles within each season so the audit
    # does not impose an arbitrary yard cutoff after seeing R8's result.
    z["movement_quintile"] = z.groupby("season")["abs_rec_yard_move"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=["Q1_SMALLEST","Q2","Q3","Q4","Q5_LARGEST"])
    )
    z["movement_direction"] = np.select(
        [z.delta_rec_yards.gt(1e-9), z.delta_rec_yards.lt(-1e-9)],
        ["BOOSTED", "SUPPRESSED"], default="UNCHANGED"
    )

    movement = _group_table(z, ["season", "movement_quintile"])
    identity = _group_table(z, ["season", "identity_bucket", "movement_direction"])
    history = _group_table(z, ["season", "history_bin", "identity_bucket"])
    continuity = _group_table(z, ["season", "team_continuity", "identity_bucket"])

    case = z.sort_values(["season", "abs_rec_yard_move"], ascending=[True, False]).copy()
    case_cols = [
        "season","week","team","player","player_clean_key","identity_bucket","prior_rb_room_share",
        "prior_games","same_team_prior_games","baseline_targets","candidate_targets","actual_targets",
        "baseline_rec_yards","candidate_rec_yards","actual_rec_yards","delta_rec_yards","abs_rec_yard_move",
        "baseline_error","candidate_error","mae_gain","movement_correct_direction","crossed_actual",
        "cat30_baseline","cat30_candidate","cat50_baseline","cat50_candidate"
    ]
    case[case_cols].to_csv(a.out_dir / "rb_r8_forensic_casebook.csv", index=False)
    z.to_csv(a.out_dir / "rb_r8_forensic_paired.csv", index=False)
    movement.to_csv(a.out_dir / "rb_r8_forensic_movement_quintiles.csv", index=False)
    identity.to_csv(a.out_dir / "rb_r8_forensic_identity_direction.csv", index=False)
    history.to_csv(a.out_dir / "rb_r8_forensic_history_reliability.csv", index=False)
    continuity.to_csv(a.out_dir / "rb_r8_forensic_team_continuity.csv", index=False)

    fresh = z.loc[z.season.eq(2018)].copy()
    fresh_q5 = fresh.loc[fresh.movement_quintile.eq("Q5_LARGEST")]
    fresh_rest = fresh.loc[~fresh.movement_quintile.eq("Q5_LARGEST")]
    fresh_top = fresh.loc[fresh.identity_bucket.eq("TOP20")]
    fresh_top_boost = fresh_top.loc[fresh_top.movement_direction.eq("BOOSTED")]
    result = {
        "diagnostic": "RB_R8_FORENSIC_V1",
        "disposition": "DIAGNOSTIC_ONLY_R8_REMAINS_SCIENTIFIC_FAIL",
        "sportsbook_inputs_used": 0,
        "production_parameters_changed": 0,
        "questions_frozen_before_forensic_output": True,
        "fresh_2018": _summary(fresh),
        "fresh_largest_move_quintile": _summary(fresh_q5),
        "fresh_other_four_move_quintiles": _summary(fresh_rest),
        "fresh_top20_identity": _summary(fresh_top),
        "fresh_top20_boosted": _summary(fresh_top_boost),
        "replication_2019": _summary(z.loc[z.season.eq(2019)]),
        "interpretation_contract": [
            "R8 cannot be rescued or reclassified from this diagnostic.",
            "If harm concentrates in Q5 while smaller moves are beneficial, a new separately frozen reliability/shrinkage hypothesis is justified.",
            "If sparse/same-team history concentrates harm, history-count reliability may be used as a new hypothesis, not a post-hoc R8 tweak.",
            "Any R9 candidate must be frozen and evaluated separately; no R8 threshold or coefficient tuning is authorized."
        ]
    }
    with open(a.out_dir / "rb_r8_forensic_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print("\n=== movement quintiles ===")
    print(movement.to_string(index=False))
    print("\n=== history reliability ===")
    print(history.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
