#!/usr/bin/env python3
"""Train state v2 and materialize current-slate diagnostics without live odds."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.modeling.state_v2 import LOGS, CONSENSUS, MODEL_PATH, build_state_predictions, save_bundle
from scripts.runtime_context import resolve_season, resolve_week

OUT = Path("data/model_state_diagnostics.csv")


def main() -> int:
    if not LOGS.exists() or LOGS.stat().st_size == 0:
        raise RuntimeError(f"State v2 source missing: {LOGS}")
    if not CONSENSUS.exists() or CONSENSUS.stat().st_size == 0:
        raise RuntimeError(f"State v2 source missing: {CONSENSUS}")
    season, week = int(resolve_season()), int(resolve_week())
    logs = pd.read_csv(LOGS)
    consensus = pd.read_csv(CONSENSUS)
    bundle, pred = build_state_predictions(logs, consensus, season, week)
    if pred.empty:
        raise RuntimeError("State v2 produced 0 current-player rows")
    if int(pd.to_numeric(pred["state_available"], errors="coerce").fillna(0).sum()) == 0:
        raise RuntimeError("State v2 produced 0 available player predictions")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(OUT, index=False)
    save_bundle(bundle, MODEL_PATH)
    print(f"[state_v2] specs={len(bundle.specs)} players={len(pred)} available={int(pred['state_available'].sum())}")
    print(f"[state_v2] wrote {OUT} and {MODEL_PATH}")
    print("[state_v2] first-order transition signal only; not yet blended into model_proj")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
