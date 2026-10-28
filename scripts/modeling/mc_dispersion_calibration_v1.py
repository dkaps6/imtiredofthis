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

# Promotion gate. While the artifact is a CANDIDATE the calibration is inert in
# production: it is labelled on every row but never changes a price. Flip the artifact's
# `promotion_state` to "PROMOTED" only after an exact production-path baseline-parity
# replay certifies k (GPT audit 6472971619, section 3).
PROMOTED_STATE = "PROMOTED"
CANDIDATE_STATE = "CANDIDATE_PENDING_PARITY_REPLAY"

# INDEPENDENT PROMOTION AUTHORITY -- deliberately in code, not in the artifact.
#
# Promotion state stored only inside the calibration file cannot make that file
# required: if the file is missing or unreadable, the loader has nothing left to tell it
# the calibration was supposed to be there, so it would return "unavailable" and price
# on silently. This flag is the independent authority. While it is False the calibration
# is optional and absence is harmless; once it is True, live pricing REQUIRES a complete,
# readable, correct-version artifact with usable k and fails closed otherwise.
# Flip to True only together with promotion_state=PROMOTED in the artifact.
CALIBRATION_REQUIRED_FOR_LIVE_PRICING = False

# Once promoted, a missing or invalid artifact must FAIL CLOSED rather than silently
# reverting live pricing to the known under-dispersed distribution (AGENTS.md section 4:
# required production model artifacts fail closed). While the artifact is a candidate,
# absence is harmless because the calibration is inert anyway.
class DispersionCalibrationUnavailable(RuntimeError):
    """Raised when a PROMOTED calibration cannot be loaded for live pricing."""
# Guard against a corrupted artifact silently distorting the board.
MIN_K, MAX_K = 0.5, 4.0


def _maybe_fail_closed(result: dict, live_pricing: bool) -> dict:
    """Raise if live pricing REQUIRES the calibration and we could not load it.

    The decision uses the in-code authority, because the artifact that would carry
    `promotion_state` is exactly the thing we failed to read.
    """
    if live_pricing and CALIBRATION_REQUIRED_FOR_LIVE_PRICING:
        raise DispersionCalibrationUnavailable(
            f"dispersion calibration is required for live pricing but could not be "
            f"loaded: state={result.get('state')} reason={result.get('reason')}"
        )
    return result


def load_calibration(path: Path | None = None, *, live_pricing: bool = False) -> dict:
    """Load k per market.

    `live_pricing=True` is the production call. If the artifact says it is PROMOTED but
    cannot be loaded or validated, this raises rather than silently reverting the board
    to the known under-dispersed distribution. A CANDIDATE artifact never raises: the
    calibration is inert, so its absence changes nothing.
    """
    p = Path(path) if path is not None else CALIBRATION_PATH
    if not p.exists():
        return _maybe_fail_closed(
            {"state": "unavailable", "version": VERSION, "k": {},
             "promotion_state": None, "reason": f"{p.as_posix()} not found"},
            live_pricing)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return _maybe_fail_closed(
            {"state": "unreadable", "version": VERSION, "k": {},
             "promotion_state": None, "reason": f"{type(exc).__name__}: {exc}"},
            live_pricing)
    if str(raw.get("version")) != VERSION:
        return _maybe_fail_closed(
            {"state": "version_mismatch", "version": VERSION, "k": {},
             "promotion_state": str(raw.get("promotion_state") or ""),
             "reason": f"artifact version {raw.get('version')!r} != {VERSION!r}"},
            live_pricing)
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
    promotion_state = str(raw.get("promotion_state") or CANDIDATE_STATE)
    cal = {"state": "current", "version": VERSION, "k": out, "rejected": rejected,
           "source": p.as_posix(), "promotion_state": promotion_state,
           "derived_from_weeks": (raw.get("derivation") or {}).get("derived_from_weeks")}
    # BOTH authorities must agree before a single price moves. The artifact alone must
    # not be able to put the calibration live, and the code flag alone must not either.
    artifact_promoted = promotion_state == PROMOTED_STATE
    cal["artifact_promoted"] = artifact_promoted
    cal["code_requires_calibration"] = CALIBRATION_REQUIRED_FOR_LIVE_PRICING
    if not (artifact_promoted and CALIBRATION_REQUIRED_FOR_LIVE_PRICING):
        # inert: keep the labels, drop the multipliers
        cal["k_candidate"] = out
        cal["k"] = {}
        cal["state"] = "candidate_not_promoted"
    if live_pricing and artifact_promoted != CALIBRATION_REQUIRED_FOR_LIVE_PRICING:
        raise DispersionCalibrationUnavailable(
            f"promotion authorities disagree: {p.as_posix()} says "
            f"{promotion_state!r} but CALIBRATION_REQUIRED_FOR_LIVE_PRICING is "
            f"{CALIBRATION_REQUIRED_FOR_LIVE_PRICING}; refusing to price until they match"
        )
    if live_pricing and CALIBRATION_REQUIRED_FOR_LIVE_PRICING and not out:
        raise DispersionCalibrationUnavailable(
            f"{p.as_posix()} is PROMOTED and required but yielded no usable k "
            f"(rejected={rejected}); refusing to price with the uncalibrated distribution"
        )
    return cal


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
