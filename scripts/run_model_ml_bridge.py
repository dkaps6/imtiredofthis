#!/usr/bin/env python3
"""Train leakage-safe ML v2 models and materialize current-slate diagnostics."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.modeling.ml_v2 import LOGS, CONSENSUS, MODEL_PATH, build_and_train, save_bundle

OUT = Path("data/model_ml_diagnostics.csv")


def main() -> int:
    if not LOGS.exists() or LOGS.stat().st_size == 0:
        raise RuntimeError(f"ML v2 source missing: {LOGS}")
    if not CONSENSUS.exists() or CONSENSUS.stat().st_size == 0:
        raise RuntimeError(f"ML v2 source missing: {CONSENSUS}")
    logs = pd.read_csv(LOGS)
    consensus = pd.read_csv(CONSENSUS)
    if consensus.empty:
        raise RuntimeError("ML v2 consensus source contains 0 rows")
    season = int(pd.to_numeric(consensus["season"], errors="coerce").dropna().iloc[0])
    week = int(pd.to_numeric(consensus["week"], errors="coerce").dropna().iloc[0])

    bundle, pred = build_and_train(logs, consensus, season, week)
    if pred.empty:
        raise RuntimeError("ML v2 produced 0 current-slate predictions")
    if pred.duplicated(["team", "player_clean_key"]).any():
        raise RuntimeError("ML v2 produced duplicate player/team rows")
    target_cols = [c for c in pred.columns if c.startswith("ml_") and c not in {"ml_available", "ml_method", "ml_training_cutoff", "ml_targets_available"}]
    projected = int(pred[target_cols].notna().any(axis=1).sum())
    if projected == 0:
        raise RuntimeError("ML v2 produced no finite target projections")

    save_bundle(bundle, MODEL_PATH)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(OUT, index=False)
    print(f"[ml_v2] train_rows={bundle.train_rows}")
    print(f"[ml_v2] trained_targets={sorted(bundle.models)}")
    print(f"[ml_v2] active_players={len(pred)} players_with_projection={projected}")
    print(f"[ml_v2] wrote model -> {MODEL_PATH}")
    print(f"[ml_v2] wrote diagnostics -> {OUT}")
    print("[ml_v2] leakage-safe: training examples use lagged history only; sportsbook lines are not features")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
