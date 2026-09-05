"""Promoted RB rushing-yards synthesis contract (P3).

Lineage
-------
RB-STACK1 run 33535308110 supplies the calibrated full-stack rushing
components. RB-STACK2 run 33538770934 supplies enriched RB opportunity.
RB-STACK3 run 33539468967 froze the deterministic P3 composition:

* Week 1: use the STACK1 full-stack rushing-yard projection unchanged.
* Weeks 2-18: enriched RB carries x STACK1 implied yards per carry.

This module intentionally contains no sportsbook inputs. Sportsbook lines and
odds are downstream pricing/benchmark data and may never construct P3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RB_RUSH_SYNTHESIS_VERSION = "RB_P3_SYNTHESIS_V1"
WEEK1_ROUTE = "WEEK1_STACK_OVERRIDE"
WEEKS2_18_ROUTE = "WEEKS2_18_ENRICHED_OPP_STACK_EFF"


def _finite(value, default=np.nan) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else float(default)
    except Exception:
        return float(default)


def compose_p3_row(
    *,
    week: int,
    stack_att: float,
    stack_yards: float,
    enriched_att: float | None = None,
    m94c_implied_ypc: float | None = None,
) -> dict[str, float | str | int]:
    """Compose one frozen P3 RB rushing-yards projection.

    The exact STACK2 efficiency contract uses ``stack_yards / stack_att`` when
    STACK1 carries exceed 0.20.  Only if that ratio is unavailable does it use
    the M94C implied-YPC fallback.  The authoritative 2025 STACK2 casebook used
    the fallback on zero rows, but the live contract remains explicit and
    fail-closed when neither efficiency source is available.
    """
    week = int(week)
    if week < 1 or week > 18:
        raise RuntimeError(f"RB P3 requires regular-season week 1-18, got {week}")

    s_att = _finite(stack_att)
    s_yards = _finite(stack_yards)
    if not np.isfinite(s_yards):
        raise RuntimeError("RB P3 stack_yards is missing/non-finite")

    if week == 1:
        return {
            "rb_synthesis_proj": float(s_yards),
            "rb_synthesis_route": WEEK1_ROUTE,
            "rb_synthesis_version": RB_RUSH_SYNTHESIS_VERSION,
            "rb_synthesis_applied": 1,
            "rb_stack_implied_ypc": float(s_yards / s_att) if np.isfinite(s_att) and s_att > 0.20 else np.nan,
            "rb_ypc_fallback_used": 0,
        }

    e_att = _finite(enriched_att)
    if not np.isfinite(e_att) or e_att < 0:
        raise RuntimeError("RB P3 Weeks 2-18 requires finite non-negative enriched_att")

    fallback_used = 0
    if np.isfinite(s_att) and s_att > 0.20:
        ypc = float(s_yards / s_att)
    else:
        ypc = _finite(m94c_implied_ypc)
        fallback_used = 1
        if not np.isfinite(ypc):
            raise RuntimeError(
                "RB P3 cannot construct STACK1 implied efficiency: "
                "stack_att <= 0.20 and M94C implied-YPC fallback is unavailable"
            )

    if ypc < 0:
        raise RuntimeError(f"RB P3 implied YPC must be non-negative, got {ypc}")
    proj = float(e_att * ypc)
    if not np.isfinite(proj):
        raise RuntimeError("RB P3 produced non-finite rushing-yard projection")

    return {
        "rb_synthesis_proj": proj,
        "rb_synthesis_route": WEEKS2_18_ROUTE,
        "rb_synthesis_version": RB_RUSH_SYNTHESIS_VERSION,
        "rb_synthesis_applied": 1,
        "rb_stack_implied_ypc": ypc,
        "rb_ypc_fallback_used": fallback_used,
    }


def apply_p3(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen P3 composition to a player-level RB context frame."""
    required = {"week", "stack_att", "stack_yards"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"RB P3 context missing required columns: {missing}")
    if frame.empty:
        raise RuntimeError("RB P3 context is empty")

    rows = []
    for _, row in frame.iterrows():
        result = compose_p3_row(
            week=int(row["week"]),
            stack_att=row["stack_att"],
            stack_yards=row["stack_yards"],
            enriched_att=row.get("enriched_att"),
            m94c_implied_ypc=row.get("m94c_implied_ypc"),
        )
        rows.append(result)

    out = frame.copy().reset_index(drop=True)
    add = pd.DataFrame(rows)
    for col in add.columns:
        out[col] = add[col].values
    return out
