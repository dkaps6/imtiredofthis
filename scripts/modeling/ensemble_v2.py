"""Evidence-weighted ensemble for canonical player projections.

This replaces the legacy fixed 25/25/25/25 voting system. The canonical
ensemble combines independent projection signals only after their walk-forward
performance has been measured.

Independent components currently are:
- ``mc_proj``: Bayesian + empirical rules + joint Monte Carlo projection,
- ``ml_proj``: supervised ML v2 projection,
- ``state_proj``: first-order state-transition v2 projection.

Bayesian v2 is intentionally NOT treated as a fourth independent projection:
it is already part of the Monte Carlo baseline. Counting it again would double
count the same information.

Weights are market-specific, non-negative, sum to one, and are fitted only from
historical out-of-sample component predictions. Sportsbook lines/odds are never
features or weight targets. If calibrated weights do not exist, production
explicitly falls back to Monte Carlo only instead of inventing equal weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

DATA = Path("data")
WEIGHTS_PATH = DATA / "model_ensemble_weights.csv"
CALIBRATION_PATH = DATA / "backtests" / "component_predictions.csv"
COMPONENTS = ("mc_proj", "ml_proj", "state_proj")
MIN_CALIBRATION_ROWS = 40


@dataclass(frozen=True)
class EnsembleWeights:
    market: str
    mc_weight: float
    ml_weight: float
    state_weight: float
    calibration_rows: int
    method: str = "nonnegative_oos_linear_blend_v2"

    def as_dict(self) -> Dict[str, float]:
        return {
            "mc_proj": float(self.mc_weight),
            "ml_proj": float(self.ml_weight),
            "state_proj": float(self.state_weight),
        }


def _normalise_nonnegative(coefs: np.ndarray) -> np.ndarray:
    w = np.clip(np.asarray(coefs, dtype=float), 0.0, None)
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return w / total


def fit_market_weights(calibration: pd.DataFrame, min_rows: int = MIN_CALIBRATION_ROWS) -> pd.DataFrame:
    """Fit market-specific non-negative weights from OOS component predictions.

    Required columns: market, actual, mc_proj, ml_proj, state_proj. Rows missing
    any component are excluded from fitting that market so weights are learned
    from directly comparable predictions.
    """
    if calibration is None or calibration.empty:
        raise RuntimeError("Ensemble calibration requires non-empty OOS component predictions")
    df = calibration.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"market", "actual", *COMPONENTS}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"ensemble calibration missing columns: {sorted(missing)}")

    rows = []
    for market, part in df.groupby(df["market"].astype(str).str.lower(), dropna=False):
        usable = part.copy()
        for col in ("actual", *COMPONENTS):
            usable[col] = pd.to_numeric(usable[col], errors="coerce")
        usable = usable.dropna(subset=["actual", *COMPONENTS])
        n = len(usable)
        if n < int(min_rows):
            continue
        X = usable[list(COMPONENTS)].to_numpy(dtype=float)
        y = usable["actual"].to_numpy(dtype=float)
        reg = LinearRegression(fit_intercept=False, positive=True)
        reg.fit(X, y)
        w = _normalise_nonnegative(reg.coef_)
        pred = X @ w
        rows.append({
            "market": str(market),
            "mc_weight": float(w[0]),
            "ml_weight": float(w[1]),
            "state_weight": float(w[2]),
            "calibration_rows": int(n),
            "calibration_mae": float(np.mean(np.abs(pred - y))),
            "calibration_rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
            "method": "nonnegative_oos_linear_blend_v2",
        })
    return pd.DataFrame(rows)


def save_weights(weights: pd.DataFrame, path: Path = WEIGHTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    weights.to_csv(path, index=False)


def load_weights(path: Path = WEIGHTS_PATH) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    out = pd.read_csv(path)
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _weight_map(weights: pd.DataFrame) -> dict[str, EnsembleWeights]:
    if weights is None or weights.empty:
        return {}
    out: dict[str, EnsembleWeights] = {}
    for _, r in weights.iterrows():
        market = str(r.get("market", "")).lower().strip()
        if not market:
            continue
        vals = _normalise_nonnegative(np.array([
            pd.to_numeric(pd.Series([r.get("mc_weight")]), errors="coerce").fillna(0).iloc[0],
            pd.to_numeric(pd.Series([r.get("ml_weight")]), errors="coerce").fillna(0).iloc[0],
            pd.to_numeric(pd.Series([r.get("state_weight")]), errors="coerce").fillna(0).iloc[0],
        ]))
        out[market] = EnsembleWeights(
            market=market,
            mc_weight=float(vals[0]), ml_weight=float(vals[1]), state_weight=float(vals[2]),
            calibration_rows=int(pd.to_numeric(pd.Series([r.get("calibration_rows", 0)]), errors="coerce").fillna(0).iloc[0]),
            method=str(r.get("method", "nonnegative_oos_linear_blend_v2")),
        )
    return out


def apply_ensemble(frame: pd.DataFrame, weights: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach final ensemble projection with explicit uncalibrated fallback.

    The function never invents equal weights. If a market lacks calibrated
    weights, ``ensemble_proj`` equals ``mc_proj`` and the status is explicit.
    """
    if frame is None or frame.empty:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    out = frame.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    if "market" not in out.columns or "mc_proj" not in out.columns:
        raise RuntimeError("ensemble requires market and mc_proj")
    wmap = _weight_map(load_weights() if weights is None else weights)

    projections, statuses, methods = [], [], []
    wmcs, wmls, wstates, calib_rows = [], [], [], []
    for _, row in out.iterrows():
        market = str(row.get("market", "")).lower().strip()
        mc = pd.to_numeric(pd.Series([row.get("mc_proj")]), errors="coerce").iloc[0]
        ml = pd.to_numeric(pd.Series([row.get("ml_proj")]), errors="coerce").iloc[0]
        state = pd.to_numeric(pd.Series([row.get("state_proj")]), errors="coerce").iloc[0]
        spec = wmap.get(market)
        if spec is None or not np.isfinite(mc):
            projections.append(float(mc) if np.isfinite(mc) else np.nan)
            statuses.append("uncalibrated_mc_only")
            methods.append("mc_fallback_no_oos_weights")
            wmcs.append(1.0); wmls.append(0.0); wstates.append(0.0); calib_rows.append(0)
            continue

        vals = {"mc_proj": mc, "ml_proj": ml, "state_proj": state}
        raw_w = spec.as_dict()
        available = {k: v for k, v in vals.items() if np.isfinite(v) and raw_w.get(k, 0.0) > 0}
        if not available:
            projections.append(float(mc))
            statuses.append("uncalibrated_mc_only")
            methods.append("mc_fallback_missing_components")
            wmcs.append(1.0); wmls.append(0.0); wstates.append(0.0); calib_rows.append(spec.calibration_rows)
            continue
        total_w = float(sum(raw_w[k] for k in available))
        use_w = {k: raw_w[k] / total_w for k in available}
        projections.append(float(sum(use_w[k] * vals[k] for k in available)))
        statuses.append("calibrated")
        methods.append(spec.method)
        wmcs.append(float(use_w.get("mc_proj", 0.0)))
        wmls.append(float(use_w.get("ml_proj", 0.0)))
        wstates.append(float(use_w.get("state_proj", 0.0)))
        calib_rows.append(spec.calibration_rows)

    out["ensemble_proj"] = projections
    out["ensemble_status"] = statuses
    out["ensemble_method"] = methods
    out["ensemble_weight_mc"] = wmcs
    out["ensemble_weight_ml"] = wmls
    out["ensemble_weight_state"] = wstates
    out["ensemble_calibration_rows"] = calib_rows
    return out
