#!/usr/bin/env python3
"""Materialize canonical ensemble calibration status for production.

Committed promoted weights are production artifacts and take precedence over
ad-hoc fitting. Historical OOS component predictions may fill markets that do
not yet have a promoted artifact, but they never overwrite an explicitly
promoted market. Markets without either source remain explicit MC-only.

The M89/M90 QB passing-yards synthesis requires the calibrated pass-yards base
ensemble. If the promoted QB model exists but the pass-yards weight artifact is
missing, this bridge fails closed rather than allowing an MC-only base to feed a
model that was validated on a different architecture.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.modeling.ensemble_v2 import (
    CALIBRATION_PATH,
    WEIGHTS_PATH,
    fit_market_weights,
    load_weights,
    save_weights,
)

DATA = Path("data")
OUT = DATA / "model_ensemble_diagnostics.csv"
QB_SYNTHESIS = Path("model/qb_pass_synthesis_v1.json")
SUPPORTED_MARKETS = (
    "pass_yards",
    "rush_yards",
    "rec_yards",
    "receptions",
    "rush_att",
    "rush_rec_yards",
    "anytime_td",
)


def _normalise_weights(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    required = {"market", "mc_weight", "ml_weight", "state_weight"}
    missing = required - set(out.columns)
    if missing:
        raise RuntimeError(f"promoted ensemble weights missing columns: {sorted(missing)}")
    out["market"] = out["market"].astype(str).str.lower().str.strip()
    for c in ("mc_weight", "ml_weight", "state_weight", "calibration_rows"):
        if c not in out.columns:
            out[c] = 0
        out[c] = pd.to_numeric(out[c], errors="coerce")
    sums = out[["mc_weight", "ml_weight", "state_weight"]].sum(axis=1)
    bad = (
        out[["mc_weight", "ml_weight", "state_weight"]].isna().any(axis=1)
        | out[["mc_weight", "ml_weight", "state_weight"]].lt(0).any(axis=1)
        | ~sums.sub(1.0).abs().le(1e-6)
    )
    if bad.any():
        raise RuntimeError(
            f"invalid promoted ensemble weights: {out.loc[bad, ['market','mc_weight','ml_weight','state_weight']].to_dict('records')}"
        )
    if out["market"].duplicated().any():
        raise RuntimeError("promoted ensemble weights contain duplicate markets")
    if "method" not in out.columns:
        out["method"] = "promoted_oos_weights"
    return out


def main() -> int:
    promoted = _normalise_weights(load_weights())
    promoted_markets = set(promoted["market"].tolist()) if not promoted.empty else set()

    fitted = pd.DataFrame()
    if CALIBRATION_PATH.exists() and CALIBRATION_PATH.stat().st_size > 0:
        calibration = pd.read_csv(CALIBRATION_PATH)
        fitted = fit_market_weights(calibration)
        if not fitted.empty:
            fitted["market"] = fitted["market"].astype(str).str.lower().str.strip()
            # Historical calibration can populate a market that has no promoted
            # artifact, but never overwrite a frozen production promotion.
            fitted = fitted.loc[~fitted["market"].isin(promoted_markets)].copy()

    combined = pd.concat([promoted, fitted], ignore_index=True, sort=False)
    if not combined.empty:
        combined = _normalise_weights(combined)
        save_weights(combined)

    by_market = {
        str(r.market).lower(): r
        for _, r in combined.iterrows()
    } if not combined.empty else {}

    rows = []
    for market in SUPPORTED_MARKETS:
        r = by_market.get(market)
        if r is None:
            rows.append({
                "market": market,
                "ensemble_status": "uncalibrated_mc_only",
                "mc_weight": 1.0,
                "ml_weight": 0.0,
                "state_weight": 0.0,
                "calibration_rows": 0,
                "method": "mc_fallback_no_promoted_oos_weights",
            })
            continue
        rows.append({
            "market": market,
            "ensemble_status": "calibrated",
            "mc_weight": float(r.mc_weight),
            "ml_weight": float(r.ml_weight),
            "state_weight": float(r.state_weight),
            "calibration_rows": int(pd.to_numeric(pd.Series([r.get('calibration_rows', 0)]), errors="coerce").fillna(0).iloc[0]),
            "method": str(r.get("method", "promoted_oos_weights")),
        })

    out = pd.DataFrame(rows)
    if QB_SYNTHESIS.exists():
        qb = out.loc[out["market"].eq("pass_yards")]
        if qb.empty or not qb["ensemble_status"].eq("calibrated").all():
            raise RuntimeError(
                "promoted QB synthesis exists but calibrated pass_yards ensemble weights are unavailable"
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"[ensemble_v2] wrote {len(out)} market rows -> {OUT}")
    print("[ensemble_v2] status:", out["ensemble_status"].value_counts().to_dict())
    if WEIGHTS_PATH.exists() and WEIGHTS_PATH.stat().st_size > 0:
        print(f"[ensemble_v2] production weights -> {WEIGHTS_PATH}; markets={sorted(by_market)}")
    else:
        print("[ensemble_v2] no calibrated weights; production remains explicit MC-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
