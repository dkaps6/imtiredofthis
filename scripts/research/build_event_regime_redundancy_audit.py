#!/usr/bin/env python3
"""Outcome-free redundancy/qualification audit for episodic football regime events.

This script evaluates whether event/regime signals are reconstructible from the
canonical pregame PlayerForm opportunity state. It never reads target-game
outcomes, betting results, sportsbook fields, or post-kickoff information.

Binary events use an out-of-time deterministic logistic classifier trained on
2019-2023 and evaluated on 2024-2025. Continuous returning-opportunity overlap
descriptors use the same out-of-time reconstructibility R2 convention as the
role/room redundancy audit.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.research.build_role_room_redundancy_audit import (
    KEY,
    build_production_opportunity_state,
    _holdout_r2,
)

BINARY_EVENTS = [
    "team_change_flag",
    "joint_player_room_transition_flag",
    "target_room_churn_flag",
    "rush_room_churn_flag",
]
CONTINUOUS_EVENTS = [
    "returning_target_opportunity_overlap",
    "returning_rush_opportunity_overlap",
]
CORE_POSITIONS = {"QB", "RB", "WR", "TE"}
PROD_INPUTS = [
    "prod_tgt_prior_share", "prod_tgt_prior_games",
    "prod_tgt_current_share", "prod_tgt_current_games",
    "prod_tgt_playerform_blend",
    "prod_rush_prior_share", "prod_rush_prior_games",
    "prod_rush_current_share", "prod_rush_current_games",
    "prod_rush_playerform_blend",
]

MECHANISMS = {
    "team_change_flag": "role-regime uncertainty / stale-history detection",
    "joint_player_room_transition_flag": "stale-history detection / role-regime uncertainty",
    "target_room_churn_flag": "personnel continuity uncertainty",
    "rush_room_churn_flag": "personnel continuity uncertainty",
    "returning_target_opportunity_overlap": "personnel continuity uncertainty",
    "returning_rush_opportunity_overlap": "personnel continuity uncertainty",
}


def _binary_reconstructibility(df: pd.DataFrame, target: str) -> dict:
    z = df[["season", target, *PROD_INPUTS]].copy()
    z[target] = z[target].astype(bool).astype(int)
    train = z[z["season"] <= 2023].copy()
    test = z[z["season"] >= 2024].copy()
    pos_train = int(train[target].sum())
    pos_test = int(test[target].sum())
    if len(train) < 500 or len(test) < 200 or pos_train < 25 or pos_test < 10:
        return {
            "balanced_accuracy": np.nan, "precision": np.nan, "recall": np.nan,
            "f1": np.nan, "train_rows": len(train), "holdout_rows": len(test),
            "train_positive": pos_train, "holdout_positive": pos_test,
        }
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced", solver="lbfgs", max_iter=1000,
            random_state=0,
        )),
    ])
    pipe.fit(train[PROD_INPUTS], train[target])
    pred = pipe.predict(test[PROD_INPUTS])
    y = test[target].to_numpy(int)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "train_rows": len(train), "holdout_rows": len(test),
        "train_positive": pos_train, "holdout_positive": pos_test,
    }


def audit_event_redundancy(history: pd.DataFrame, detail: pd.DataFrame, event_summary: pd.DataFrame) -> pd.DataFrame:
    state = build_production_opportunity_state(history)
    d = detail.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]
    e = event_summary.copy()
    e.columns = [str(c).strip().lower() for c in e.columns]

    aliases = {
        "returning_target_opportunity_overlap": "prior_tgt_share_game_returning_overlap",
        "returning_rush_opportunity_overlap": "prior_rush_share_game_returning_overlap",
    }
    for friendly, canonical in aliases.items():
        if friendly not in d.columns and canonical in d.columns:
            d[friendly] = d[canonical]

    required = {*KEY, "position", "known_context_flag", *BINARY_EVENTS, *CONTINUOUS_EVENTS}
    miss = required - set(d.columns)
    if miss:
        raise RuntimeError(f"detail missing columns: {sorted(miss)}")
    if d.duplicated(KEY).any():
        raise RuntimeError("detail has duplicate canonical player-game keys")

    d = d.drop(columns=[c for c in d.columns if c.startswith("prod_")], errors="ignore")
    x = d.merge(state, on=KEY, how="left", validate="one_to_one")
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["known_context_flag"] = x["known_context_flag"].astype(bool)

    stable_id_coverage = (
        float(pd.to_numeric(x["stable_identity_flag"], errors="coerce").fillna(0).mean())
        if "stable_identity_flag" in x.columns else 1.0
    )
    duplicate_count = int(x.duplicated(KEY).sum())

    rows = []
    for pos in sorted(CORE_POSITIONS):
        px = x[(x["position"] == pos) & x["known_context_flag"]].copy()
        base_n = int((x["position"] == pos).sum())
        known_coverage = float(len(px) / base_n) if base_n else np.nan
        for event in BINARY_EVENTS:
            q = e[(e["event_name"] == event) & (e["position"] == pos)]
            if q.empty:
                continue
            qrow = q.iloc[0]
            metrics = _binary_reconstructibility(px, event)
            highly_reconstructible = (
                np.isfinite(metrics["balanced_accuracy"]) and
                np.isfinite(metrics["f1"]) and
                metrics["balanced_accuracy"] >= 0.90 and
                metrics["f1"] >= 0.80
            )
            temporal_precision_pass = (
                pd.notna(qrow.get("onset_fraction_of_positive")) and
                float(qrow["onset_fraction_of_positive"]) >= 0.80
            )
            support_pass = bool(qrow.get("support_gate_250", False))
            coverage_pass = bool(qrow.get("known_coverage_gate_080", False))
            integrity_pass = stable_id_coverage >= 0.99 and duplicate_count == 0
            if not integrity_pass:
                disposition = "REJECTED_INTEGRITY"
            elif not coverage_pass or not support_pass:
                disposition = "ENGINEERING_READY_SOURCE_THIN"
            elif not temporal_precision_pass:
                disposition = "DESCRIPTIVE_ONLY"
            elif highly_reconstructible:
                disposition = "DESCRIPTIVE_ONLY"
            else:
                disposition = "READY_FOR_FROZEN_EXPERIMENT"
            rows.append({
                "event_name": event,
                "position": pos,
                "event_type": "binary",
                "known_rows": int(qrow["known_rows"]),
                "positive_events": int(qrow["positive_events"]),
                "prevalence": float(qrow["prevalence"]),
                "seasons_with_positive": int(qrow["seasons_with_positive"]),
                "max_season_share": float(qrow["max_season_share"]) if pd.notna(qrow["max_season_share"]) else np.nan,
                "onset_fraction_of_positive": float(qrow["onset_fraction_of_positive"]) if pd.notna(qrow["onset_fraction_of_positive"]) else np.nan,
                "known_coverage": known_coverage,
                "stable_id_coverage": stable_id_coverage,
                "duplicate_key_count": duplicate_count,
                **metrics,
                "highly_reconstructible": bool(highly_reconstructible),
                "integrity_gate": bool(integrity_pass),
                "coverage_gate": bool(coverage_pass),
                "support_gate": bool(support_pass),
                "temporal_precision_gate": bool(temporal_precision_pass),
                "mechanism": MECHANISMS[event],
                "qualification_disposition": disposition,
                "outcomes_read": False,
                "sportsbook_read": False,
            })

        for event in CONTINUOUS_EVENTS:
            vals = pd.to_numeric(px[event], errors="coerce")
            coverage = float(vals.notna().mean()) if len(vals) else np.nan
            r2, ntr, nte = _holdout_r2(px.assign(**{event: vals}), event, PROD_INPUTS)
            if np.isfinite(r2) and r2 >= 0.90:
                reconstruct = "HIGHLY_RECONSTRUCTIBLE_REDUNDANT"
            elif np.isfinite(r2) and r2 >= 0.75:
                reconstruct = "PARTIALLY_RECONSTRUCTIBLE_REVIEW"
            else:
                reconstruct = "INCREMENTAL_INFORMATION_SURVIVES_REDUNDANCY_GATE"
            integrity_pass = stable_id_coverage >= 0.99 and duplicate_count == 0
            if not integrity_pass:
                disposition = "REJECTED_INTEGRITY"
            elif coverage < 0.80:
                disposition = "ENGINEERING_READY_SOURCE_THIN"
            elif reconstruct != "INCREMENTAL_INFORMATION_SURVIVES_REDUNDANCY_GATE":
                disposition = "DESCRIPTIVE_ONLY"
            else:
                disposition = "READY_FOR_FROZEN_EXPERIMENT"
            rows.append({
                "event_name": event,
                "position": pos,
                "event_type": "continuous_descriptor",
                "known_rows": int(vals.notna().sum()),
                "positive_events": np.nan,
                "prevalence": np.nan,
                "seasons_with_positive": np.nan,
                "max_season_share": np.nan,
                "onset_fraction_of_positive": np.nan,
                "known_coverage": coverage,
                "stable_id_coverage": stable_id_coverage,
                "duplicate_key_count": duplicate_count,
                "balanced_accuracy": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "train_rows": ntr,
                "holdout_rows": nte,
                "train_positive": np.nan,
                "holdout_positive": np.nan,
                "holdout_reconstructibility_r2": r2,
                "redundancy_disposition": reconstruct,
                "highly_reconstructible": bool(reconstruct == "HIGHLY_RECONSTRUCTIBLE_REDUNDANT"),
                "integrity_gate": bool(integrity_pass),
                "coverage_gate": bool(coverage >= 0.80),
                "support_gate": bool(ntr >= 500 and nte >= 200),
                "temporal_precision_gate": True,
                "mechanism": MECHANISMS[event],
                "qualification_disposition": disposition,
                "outcomes_read": False,
                "sportsbook_read": False,
            })
    return pd.DataFrame(rows).sort_values(["qualification_disposition", "event_name", "position"]).reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--history", type=Path, required=True)
    p.add_argument("--detail", type=Path, required=True)
    p.add_argument("--event-summary", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    out = audit_event_redundancy(
        pd.read_csv(a.history),
        pd.read_csv(a.detail),
        pd.read_csv(a.event_summary),
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    cols = [
        c for c in [
            "event_name","position","event_type","known_rows","positive_events","prevalence",
            "balanced_accuracy","precision","recall","f1","holdout_reconstructibility_r2",
            "integrity_gate","coverage_gate","support_gate","temporal_precision_gate",
            "qualification_disposition"
        ] if c in out.columns
    ]
    print(out[cols].to_string(index=False))
    print(f"[event_regime_redundancy] rows={len(out)} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
