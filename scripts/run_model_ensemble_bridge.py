#!/usr/bin/env python3
"""Materialize canonical ensemble calibration status.

Until the walk-forward backtest produces out-of-sample component predictions,
this bridge intentionally records MC-only fallback status rather than fabricating
weights. When data/backtests/component_predictions.csv exists, market-specific
weights are fitted and persisted for production pricing.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.modeling.ensemble_v2 import CALIBRATION_PATH, WEIGHTS_PATH, fit_market_weights, save_weights

DATA = Path("data")
OUT = DATA / "model_ensemble_diagnostics.csv"
SUPPORTED_MARKETS = ("pass_yards", "rush_yards", "rec_yards", "receptions", "rush_att", "rush_rec_yards", "anytime_td")


def main() -> int:
    rows = []
    if CALIBRATION_PATH.exists() and CALIBRATION_PATH.stat().st_size > 0:
        calibration = pd.read_csv(CALIBRATION_PATH)
        weights = fit_market_weights(calibration)
        if not weights.empty:
            save_weights(weights)
        fitted = {str(r.market).lower(): r for _, r in weights.iterrows()} if not weights.empty else {}
        for market in SUPPORTED_MARKETS:
            r = fitted.get(market)
            if r is None:
                rows.append({"market": market, "ensemble_status": "uncalibrated_mc_only", "mc_weight": 1.0, "ml_weight": 0.0, "state_weight": 0.0, "calibration_rows": 0, "method": "mc_fallback_insufficient_oos_rows"})
            else:
                rows.append({"market": market, "ensemble_status": "calibrated", "mc_weight": float(r.mc_weight), "ml_weight": float(r.ml_weight), "state_weight": float(r.state_weight), "calibration_rows": int(r.calibration_rows), "method": str(r.method)})
    else:
        for market in SUPPORTED_MARKETS:
            rows.append({"market": market, "ensemble_status": "uncalibrated_mc_only", "mc_weight": 1.0, "ml_weight": 0.0, "state_weight": 0.0, "calibration_rows": 0, "method": "mc_fallback_awaiting_walk_forward"})

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"[ensemble_v2] wrote {len(out)} market rows -> {OUT}")
    print("[ensemble_v2] status:", out["ensemble_status"].value_counts().to_dict())
    if WEIGHTS_PATH.exists() and WEIGHTS_PATH.stat().st_size > 0:
        print(f"[ensemble_v2] calibrated weights -> {WEIGHTS_PATH}")
    else:
        print("[ensemble_v2] no calibrated weights yet; production remains explicit MC-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
