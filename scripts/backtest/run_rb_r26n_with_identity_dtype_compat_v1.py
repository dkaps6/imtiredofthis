#!/usr/bin/env python3
"""Mechanical dtype-only compatibility wrapper for frozen R26N builder.

The frozen R26N candidate builder is not modified. This wrapper patches only the
imported attach_identity call so pandas as-of join keys use the same plain object
dtype on current and historical frames. Key values and all football/R9 features
remain unchanged.
"""
from __future__ import annotations

import pandas as pd

import scripts.backtest.build_rb_r26n_2026_week1_unmodified_r26_structural_candidate_v1 as r26n

_ORIGINAL_ATTACH_IDENTITY = r26n.attach_identity


def _object_key_copy(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ("player_clean_key", "team"):
        if col in out.columns:
            before = out[col].astype("string").fillna("<NA>").tolist()
            out[col] = out[col].astype(object)
            after = out[col].astype("string").fillna("<NA>").tolist()
            if before != after:
                raise RuntimeError(f"R26N dtype repair changed identity key values in {col}")
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
        raise RuntimeError("R26N dtype repair changed row counts")

    # Verify every non-key column remains exactly equal before delegation.
    for original, repaired, label in ((rb, rb2, "rb"), (states, states2, "states"), (prev, prev2, "prev")):
        nonkeys = [c for c in original.columns if c not in {"player_clean_key", "team"}]
        if nonkeys and not original[nonkeys].equals(repaired[nonkeys]):
            raise RuntimeError(f"R26N dtype repair changed non-key values in {label}")

    print(
        "R26N_IDENTITY_DTYPE_COMPAT_PASS "
        f"rb_key_dtype={rb2['player_clean_key'].dtype if 'player_clean_key' in rb2 else 'NA'} "
        f"states_key_dtype={states2['player_clean_key'].dtype if 'player_clean_key' in states2 else 'NA'}"
    )
    return _ORIGINAL_ATTACH_IDENTITY(rb2, season, week, states2, prev2)


r26n.attach_identity = _compat_attach_identity


if __name__ == "__main__":
    raise SystemExit(r26n.main())
