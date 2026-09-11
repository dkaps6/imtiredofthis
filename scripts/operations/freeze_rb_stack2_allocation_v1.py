#!/usr/bin/env python3
"""Materialize the frozen STACK2 allocation learner and verify historical parity.

This is productionization of already-evaluated science, not a new model search.
It intentionally reuses the exact STACK2 feature construction and learner
hyperparameters from scripts/backtest/evaluate_rb_stack2_enriched_allocation.py.

Outputs are suitable for an immutable production artifact freeze only if the
archived 2025 allocation prediction parity gate passes.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from scripts.backtest import evaluate_rb_stack2_enriched_allocation as s2

SEED = 17
PARITY_TOL = 1e-10
EXPECTED_EVAL_ROWS = 1393
EXPECTED_TRAIN_ROWS = 2102

MODEL_KWARGS = {
    "loss": "squared_error",
    "learning_rate": 0.05,
    "max_iter": 160,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 30,
    "l2_regularization": 1.0,
    "random_state": SEED,
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_training_matrix(train: pd.DataFrame) -> pd.DataFrame:
    cols = ["season", "week", "team", "name_key", "actual_share", *s2.FULL]
    out = train.reindex(columns=cols).copy()
    out = out.sort_values(["season", "week", "team", "name_key"], kind="mergesort").reset_index(drop=True)
    return out


def train_frozen_model(train: pd.DataFrame) -> HistGradientBoostingRegressor:
    X = train.reindex(columns=s2.FULL).apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(train["actual_share"], errors="coerce")
    ok = y.notna()
    X = X.loc[ok]
    y = y.loc[ok].clip(0, 1)
    model = HistGradientBoostingRegressor(**MODEL_KWARGS)
    model.fit(X, y)
    return model


def predict(model: HistGradientBoostingRegressor, test: pd.DataFrame) -> np.ndarray:
    X = test.reindex(columns=s2.FULL).apply(pd.to_numeric, errors="coerce")
    return np.clip(model.predict(X), 0, 1)


def find_casebook(root: Path) -> Path:
    hits = list(root.rglob("stack2_2025_casebook.csv"))
    if len(hits) != 1:
        raise RuntimeError(f"Expected exactly one archived STACK2 casebook under {root}, found {len(hits)}")
    return hits[0]


def roster_2026_contract() -> tuple[pd.DataFrame, dict]:
    try:
        rr = s2.load_rosters([2026])
    except Exception as exc:
        return pd.DataFrame(), {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "rows": 0,
            "weeks": [],
        }
    fields = ["status_raw", "years_exp", "rookie_year", "entry_year", "draft_number"]
    coverage = {}
    for c in fields:
        if c not in rr.columns:
            coverage[c] = 0.0
        else:
            v = rr[c]
            if c == "status_raw":
                coverage[c] = float(v.fillna("").astype(str).str.strip().ne("").mean()) if len(v) else 0.0
            else:
                coverage[c] = float(pd.to_numeric(v, errors="coerce").notna().mean()) if len(v) else 0.0
    weeks = sorted(pd.to_numeric(rr.get("week"), errors="coerce").dropna().astype(int).unique().tolist()) if len(rr) else []
    return rr, {
        "available": bool(len(rr) > 0),
        "rows": int(len(rr)),
        "teams": int(rr["team"].nunique()) if len(rr) and "team" in rr else 0,
        "weeks": weeks,
        "field_coverage": coverage,
        "source": "nflreadpy.load_rosters_weekly(2026)",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m94c-root", type=Path, required=True)
    ap.add_argument("--stack1-root", type=Path, required=True)
    ap.add_argument("--archived-stack2-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import nflreadpy as nfl

    m94 = s2.one(args.m94c_root, "m94c_2025_rb_trace.csv")
    stack1 = s2.one(args.stack1_root, "stack1_2025_rb_trace.csv")
    archived = pd.read_csv(find_casebook(args.archived_stack2_root))

    # Rebuild only the historically frozen feature state used by the authoritative
    # 2024-fit / 2025-holdout STACK2 experiment.
    logs = s2.load_weekly_logs([2023, 2024, 2025])
    rosters = s2.load_rosters([2024, 2025])
    sched = s2.lower(s2.pdx(nfl.load_schedules(seasons=[2024, 2025])))
    depth = s2.depth_tables([2024, 2025], sched)
    snaps = s2.load_snaps([2023, 2024, 2025])
    injuries = s2.load_injuries([2024, 2025])

    train = s2.finalize_features(s2.build_training(rosters, logs, depth, snaps, injuries))
    test = s2.finalize_features(s2.build_eval(m94, rosters, logs, depth, snaps, injuries))

    matrix = canonical_training_matrix(train)
    matrix_bytes = matrix.to_csv(index=False, lineterminator="\n").encode("utf-8")
    matrix_path = args.out_dir / "rb_stack2_frozen_training_matrix_v1.csv"
    matrix_path.write_bytes(matrix_bytes)

    model = train_frozen_model(train)
    test["alloc_full_score_rebuilt"] = predict(model, test)
    s2.normalize_team_scores(test, "alloc_full_score_rebuilt", "alloc_full_share_rebuilt")

    keys = ["season", "week", "team", "name_key"]
    a = archived.copy()
    a["team"] = a["team"].map(s2.tm)
    a["name_key"] = a["name_key"].map(s2.nk)
    a = a[keys + ["alloc_full_score", "alloc_full_share"]].drop_duplicates(keys)
    p = test[keys + ["alloc_full_score_rebuilt", "alloc_full_share_rebuilt"]].drop_duplicates(keys)
    parity = a.merge(p, on=keys, how="outer", indicator=True, validate="one_to_one")
    parity["score_abs_diff"] = (
        pd.to_numeric(parity["alloc_full_score"], errors="coerce")
        - pd.to_numeric(parity["alloc_full_score_rebuilt"], errors="coerce")
    ).abs()
    parity["share_abs_diff"] = (
        pd.to_numeric(parity["alloc_full_share"], errors="coerce")
        - pd.to_numeric(parity["alloc_full_share_rebuilt"], errors="coerce")
    ).abs()
    parity.to_csv(args.out_dir / "rb_stack2_2025_prediction_parity_v1.csv", index=False)

    matched = parity["_merge"].eq("both")
    max_score = float(parity.loc[matched, "score_abs_diff"].max()) if matched.any() else float("inf")
    max_share = float(parity.loc[matched, "share_abs_diff"].max()) if matched.any() else float("inf")
    parity_pass = bool(
        len(parity) == EXPECTED_EVAL_ROWS
        and matched.all()
        and max_score <= PARITY_TOL
        and max_share <= PARITY_TOL
        and len(matrix) == EXPECTED_TRAIN_ROWS
    )

    model_bytes = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    (args.out_dir / "rb_stack2_allocation_model_v1.pkl.b64").write_text(
        base64.b64encode(model_bytes).decode("ascii") + "\n", encoding="utf-8"
    )

    roster26, roster_meta = roster_2026_contract()
    if not roster26.empty:
        roster26.to_csv(args.out_dir / "rb_stack2_2026_roster_metadata_audit.csv", index=False)
    (args.out_dir / "rb_stack2_2026_roster_metadata_contract.json").write_text(
        json.dumps(roster_meta, indent=2, sort_keys=True), encoding="utf-8"
    )

    result = {
        "disposition": "RB_STACK2_FROZEN_ALLOCATION_PARITY_PASS" if parity_pass else "RB_STACK2_FROZEN_ALLOCATION_PARITY_FAIL",
        "scientific_search_performed": False,
        "sportsbook_inputs_used": False,
        "training_season": 2024,
        "holdout_season": 2025,
        "training_rows": int(len(matrix)),
        "expected_training_rows": EXPECTED_TRAIN_ROWS,
        "holdout_rows": int(len(parity)),
        "expected_holdout_rows": EXPECTED_EVAL_ROWS,
        "parity_tolerance": PARITY_TOL,
        "max_alloc_full_score_abs_diff": max_score,
        "max_alloc_full_share_abs_diff": max_share,
        "full_feature_contract": list(s2.FULL),
        "learner": "sklearn.ensemble.HistGradientBoostingRegressor",
        "learner_kwargs": MODEL_KWARGS,
        "sklearn_version": sklearn.__version__,
        "training_matrix_sha256": sha256_bytes(matrix_bytes),
        "serialized_model_sha256": sha256_bytes(model_bytes),
        "roster_2026_contract": roster_meta,
    }
    (args.out_dir / "rb_stack2_frozen_allocation_result_v1.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))

    if not parity_pass:
        raise RuntimeError("Archived STACK2 2025 allocation prediction parity failed; do not freeze production artifact")
    if not roster_meta.get("available"):
        raise RuntimeError("2026 weekly roster metadata source unavailable; season-long live contract not yet satisfied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
