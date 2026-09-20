#!/usr/bin/env python3
"""Observational capture of the production RB rush-yards empirical draw array.

Implements section 7 of
`docs/research/RB_PD2_FORWARD_SHADOW_CONFIRMATION_V1_PLAN.md` and nothing
else. This module captures the exact production `adjusted_outcomes` array for
eligible RB/HB/FB `rush_yards` rows so the frozen shadow candidate can be built
*outside* the production process.

It deliberately does NOT compute the difficulty score, the width multiplier or
the candidate distribution. Those need the reconstructed generic-ensemble parent
history (plan section 3), which does not exist inside the pricing loop; pulling
it in would put historical reconstruction on the certified production path,
which is precisely what section 7's isolation requirement exists to prevent.

Contract with production (`scripts/run_pricing_v2.py`):
- gated by an opt-in environment flag, default OFF;
- when OFF, the only cost is one boolean check per priced row;
- the draw array is copied at capture, so the shadow can never alias or mutate
  production memory;
- nothing here writes to `outputs/props_priced_clean.csv`, the workbook, or any
  production artifact.

Failure policy: when the flag is ON, capture errors raise rather than being
swallowed. The flag is OFF in production, so production is never exposed to
them; and a silent shadow data loss would quietly corrupt the confirmation
study, which is a worse error than a loud failure in a research run.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

FLAG = "RB_PD2_SHADOW_CAPTURE"
DEFAULT_OUT = Path("data/research/rb_pd2_shadow/baseline_capture.jsonl")

ELIGIBLE_POSITIONS = {"RB", "HB", "FB"}
ELIGIBLE_MARKET = "rush_yards"

_BUFFER: list[dict] = []


def capture_enabled() -> bool:
    """True only when the research flag is explicitly turned on."""
    return str(os.getenv(FLAG, "")).strip().lower() in {"1", "true", "yes", "on"}


def _digest(draws: np.ndarray) -> str:
    """Stable digest of the exact draw array, for lock-artifact reproducibility."""
    return hashlib.sha256(np.ascontiguousarray(draws, dtype=np.float64).tobytes()).hexdigest()


def _summary(draws: np.ndarray) -> dict:
    q05, q10, q50, q90, q95 = (float(v) for v in np.quantile(draws, [0.05, 0.10, 0.50, 0.90, 0.95]))
    return {
        "draw_count": int(draws.size),
        "draw_digest_sha256": _digest(draws),
        "mean": float(np.mean(draws)),
        "sd": float(np.std(draws, ddof=1)) if draws.size > 1 else 0.0,
        "q05": q05, "q10": q10, "q50": q50, "q90": q90, "q95": q95,
    }


def is_eligible(position: str, market: str) -> bool:
    return str(position or "").upper().strip() in ELIGIBLE_POSITIONS and str(market) == ELIGIBLE_MARKET


def capture(
    *,
    row,
    adjusted_outcomes,
    target_mean: float,
    market: str,
    position: str,
    season: int,
    week: int,
) -> bool:
    """Record one baseline row. Returns True if captured, False if not eligible.

    `adjusted_outcomes` is copied immediately and never written back.
    """
    if not is_eligible(position, market):
        return False

    draws = np.array(adjusted_outcomes, dtype=float, copy=True)
    if draws.ndim != 1 or draws.size == 0:
        raise RuntimeError(f"shadow capture: invalid draw array shape {draws.shape}")
    if not np.isfinite(draws).all():
        raise RuntimeError("shadow capture: non-finite draw encountered")

    record = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "season": int(season),
        "week": int(week),
        "event_id": str(row.get("event_id") or ""),
        "game_id": str(row.get("game_id") or ""),
        "team": str(row.get("team") or "").upper().strip(),
        "opponent": str(row.get("opponent") or "").upper().strip(),
        "player": str(row.get("player") or ""),
        "player_clean_key": str(row.get("player_clean_key") or ""),
        "position": str(position).upper().strip(),
        "market": ELIGIBLE_MARKET,
        "target_mean": float(target_mean),
        "baseline": _summary(draws),
        # Explicit section 9 integrity flags, asserted at the point of capture.
        "sportsbook_inputs_used_in_candidate": False,
        "production_output_mutated": False,
        "outcome_present_at_lock": False,
    }
    _BUFFER.append(record)
    return True


def pending() -> int:
    return len(_BUFFER)


def flush(out_path: Path | None = None) -> dict:
    """Write buffered captures to the research artifact and clear the buffer."""
    path = Path(out_path) if out_path is not None else DEFAULT_OUT
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = len(_BUFFER)
    with path.open("a", encoding="utf-8") as f:
        for record in _BUFFER:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    _BUFFER.clear()
    return {"rows_written": rows, "path": str(path)}


def reset() -> None:
    """Drop buffered captures without writing. Test support only."""
    _BUFFER.clear()
