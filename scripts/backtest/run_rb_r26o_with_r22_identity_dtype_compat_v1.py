#!/usr/bin/env python3
"""Mechanical dtype-only compatibility wrapper for frozen R26O evaluator.

The frozen R26O evaluator and protected R22 production adapter are not modified.
This wrapper patches only R22's re-exported `_attach_identity` seam so pandas
as-of join keys use the same plain object dtype on current and historical frames.
Identity strings, row counts, non-key values, football features, R9 values, and
R22 science remain unchanged.
"""
from __future__ import annotations

import pandas as pd

import scripts.backtest.evaluate_rb_r26o_2026_week1_receptions_shadow_integration_v1 as r26o
import scripts.modeling.rb_receiving_tail_production_adapter_v1 as r22

_ORIGINAL_ATTACH_IDENTITY = r22.r8._attach_identity


def _object_key_copy(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ("player_clean_key", "team"):
        if col in out.columns:
            before = out[col].astype("string").fillna("<NA>").tolist()
            out[col] = out[col].astype(object)
            after = out[col].astype("string").fillna("<NA>").tolist()
            if before != after:
                raise RuntimeError(f"R26O dtype repair changed identity key values in {col}")
    return out


def _compat_attach_identity(
    rb: pd.DataFrame,
    season: int,
    week: int,
    states: pd.DataFrame,
    prev: pd.DataFrame,
) -> pd.DataFrame:
    rb2 = _object_key_copy(rb)
    states2 = _object_key_copy(states)
    prev2 = _object_key_copy(prev)

    if len(rb2) != len(rb) or len(states2) != len(states) or len(prev2) != len(prev):
        raise RuntimeError("R26O dtype repair changed row counts")

    for original, repaired, label in (
        (rb, rb2, "rb"),
        (states, states2, "states"),
        (prev, prev2, "prev"),
    ):
        nonkeys = [c for c in original.columns if c not in {"player_clean_key", "team"}]
        if nonkeys and not original[nonkeys].equals(repaired[nonkeys]):
            raise RuntimeError(f"R26O dtype repair changed non-key values in {label}")

    print(
        "R26O_R22_IDENTITY_DTYPE_COMPAT_PASS "
        f"rb_key_dtype={rb2['player_clean_key'].dtype if 'player_clean_key' in rb2 else 'NA'} "
        f"states_key_dtype={states2['player_clean_key'].dtype if 'player_clean_key' in states2 else 'NA'}"
    )
    return _ORIGINAL_ATTACH_IDENTITY(rb2, season, week, states2, prev2)


r22.r8._attach_identity = _compat_attach_identity


if __name__ == "__main__":
    raise SystemExit(r26o.main())
