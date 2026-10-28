"""Mean-preserving dispersion calibration for the Monte Carlo pricing distribution.

Production prices from the empirical MC distribution
(`run_pricing_v2.py`: `p_over = np.mean(adjusted_outcomes > line)`). Measured on live
2026 boards, those draws are under-dispersed by a market-specific factor, which makes
the probabilities badly overconfident: `fair_prob` 0.061 against a realised 0.447, and
0.844 against 0.538, with Brier 0.2800 versus the market's 0.2491.

This module widens each player's draws about **their own mean**, leaving the mean
exactly where the promoted syntheses put it.

Validated: `docs/production/MC_DISPERSION_RECALIBRATION_PREREGISTRATION_V1.md`,
frozen before scoring; `k` derived from 2026 weeks 1/2/6 and validated on week 7
re-simulated from the archived fixture. Brier 0.2713 -> 0.2555, mean absolute
calibration gap 0.1692 -> 0.0935, sd(z) 1.767 -> 1.186, worst relative mean shift
0.00000.

NOT a structural fix. The underlying cause is that `simulation_v2.simulate()` draws
outcome noise conditional on point-estimate `ypt` / `ypc` / `catch_rate` / target share
with no sampling of their own estimation uncertainty. That remains open.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

VERSION = "MC_DISPERSION_CALIBRATION_V1"
CALIBRATION_PATH = Path("data/mc_dispersion_calibration_v1.json")

# A market with no calibrated k is left untouched at exactly this value.
NEUTRAL_K = 1.0
# Guard against a corrupted artifact silently distorting the board.
MIN_K, MAX_K = 0.5, 4.0


def load_calibration(path: Path | None = None) -> dict:
    """Load k per market. Fail SOFT but VISIBLE: a missing artifact degrades to today's
    behaviour (k=1.0 everywhere) with an explicit state, rather than failing the slate.
    """
    p = Path(path) if path is not None else CALIBRATION_PATH
    if not p.exists():
        return {"state": "unavailable", "version": VERSION, "k": {},
                "reason": f"{p.as_posix()} not found"}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"state": "unreadable", "version": VERSION, "k": {},
                "reason": f"{type(exc).__name__}: {exc}"}
    if str(raw.get("version")) != VERSION:
        return {"state": "version_mismatch", "version": VERSION, "k": {},
                "reason": f"artifact version {raw.get('version')!r} != {VERSION!r}"}
    out: dict[str, float] = {}
    rejected: dict[str, str] = {}
    for market, spec in (raw.get("markets") or {}).items():
        try:
            k = float(spec["k"])
        except Exception:
            rejected[str(market)] = "unparseable k"
            continue
        if not np.isfinite(k) or not (MIN_K <= k <= MAX_K):
            rejected[str(market)] = f"k={k} outside [{MIN_K}, {MAX_K}]"
            continue
        out[str(market)] = k
    return {"state": "current", "version": VERSION, "k": out, "rejected": rejected,
            "source": p.as_posix(),
            "derived_from_weeks": (raw.get("derivation") or {}).get("derived_from_weeks")}


def apply_dispersion_calibration(draws, market: str, calibration: dict | None):
    """Widen `draws` about their own mean without moving it.

    Returns (draws, meta). The transform is widen -> clip to the non-negative domain ->
    multiplicatively restore the original mean. The multiplicative restore preserves the
    zero floor and restores the mean exactly, so the realised dispersion ends slightly
    below k x sd_base; that is the cost of respecting the domain and is reported in
    `meta`, not hidden. Widen-then-clip alone was rejected in validation because it
    lifts a low-mean player's mean by up to 27.8%.
    """
    arr = np.asarray(draws, dtype=float)
    meta = {"mc_dispersion_version": VERSION, "mc_dispersion_applied": 0,
            "mc_dispersion_k": NEUTRAL_K, "mc_dispersion_state": "not_applied",
            "mc_dispersion_sd_before": float("nan"), "mc_dispersion_sd_after": float("nan")}
    if calibration is None:
        meta["mc_dispersion_state"] = "no_calibration_object"
        return arr, meta
    state = str(calibration.get("state", "unavailable"))
    k = float((calibration.get("k") or {}).get(str(market), NEUTRAL_K))
    if state != "current":
        meta["mc_dispersion_state"] = state
        return arr, meta
    if str(market) not in (calibration.get("k") or {}):
        meta["mc_dispersion_state"] = "market_not_calibrated"
        return arr, meta
    if arr.size < 2 or not np.all(np.isfinite(arr)):
        meta["mc_dispersion_state"] = "draws_unusable"
        return arr, meta

    mu = float(arr.mean())
    sd_before = float(arr.std(ddof=1))
    meta["mc_dispersion_k"] = k
    meta["mc_dispersion_sd_before"] = sd_before
    if abs(k - NEUTRAL_K) < 1e-12:
        meta.update({"mc_dispersion_state": "neutral_k", "mc_dispersion_sd_after": sd_before})
        return arr, meta

    wide = np.clip(mu + k * (arr - mu), 0.0, None)
    m = float(wide.mean())
    if m <= 0.0 or not np.isfinite(m):
        # cannot restore the mean without dividing by ~zero; leave the draws alone
        meta["mc_dispersion_state"] = "mean_restore_unsafe"
        meta["mc_dispersion_sd_after"] = sd_before
        return arr, meta
    wide = wide * (mu / m)
    meta.update({"mc_dispersion_applied": 1, "mc_dispersion_state": "applied",
                 "mc_dispersion_sd_after": float(wide.std(ddof=1))})
    return wide, meta
