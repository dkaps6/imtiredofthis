#!/usr/bin/env python3
"""Mechanical WR-ND5 runner fixing only timestamp comparison dtypes.

The frozen ND5 evaluator remains the scientific authority. This wrapper replaces
only `_attach_depth_signals` so depth snapshot timestamps and game-date cutoffs
are compared as normalized pandas datetimes rather than a mixed
`datetime.date`/NaN object series. Signal definitions, thresholds, casebook,
parent reconstruction, and all frozen gates are unchanged.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

# Load the evaluator from this research checkout explicitly.  PYTHONPATH places
# the exact M38 parent first so its model modules remain authoritative; importing
# `scripts.backtest` normally would therefore resolve to the parent's package,
# which does not contain the research-only ND5 evaluator.
_EVALUATOR = Path(__file__).with_name("evaluate_wr_nd5_snap_depth_entitlement.py")
_spec = importlib.util.spec_from_file_location("wr_nd5_frozen_evaluator", _EVALUATOR)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"unable to load frozen ND5 evaluator: {_EVALUATOR}")
nd5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nd5)


def _attach_depth_signals(casebook: pd.DataFrame, depth: pd.DataFrame, dates: pd.DataFrame) -> pd.DataFrame:
    x = casebook.merge(dates, on=["season", "week", "team"], how="left", validate="many_to_one")
    if x["game_date"].isna().any():
        bad = x.loc[x["game_date"].isna(), ["season", "week", "team"]].drop_duplicates().head(20)
        raise RuntimeError(f"ND5 missing target game dates:\n{bad.to_string(index=False)}")

    rows = []
    for _, r in x.iterrows():
        cur_rank, cur_slot, cur_dt = nd5._depth_rank_at(depth, r["team"], r["player_id"], r["game_date"])
        prev_rank, prev_slot, prev_dt = nd5._depth_rank_at(depth, r["team"], r["player_id"], r["previous_game_date"])
        top2 = 1.0 if np.isfinite(cur_rank) and cur_rank <= 2 else 0.0 if np.isfinite(cur_rank) and cur_rank >= 3 else np.nan
        promotion = float(prev_rank - cur_rank) if np.isfinite(prev_rank) and np.isfinite(cur_rank) else np.nan
        rows.append({
            "depth_current_rank": cur_rank,
            "depth_current_slot": cur_slot,
            "depth_previous_rank": prev_rank,
            "depth_previous_slot": prev_slot,
            "depth_top2_state": top2,
            "depth_rank_promotion": promotion,
            "depth_current_snapshot_dt": cur_dt,
            "depth_previous_snapshot_dt": prev_dt,
        })

    out = pd.concat([x.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

    # Mechanical repair only: normalize both sides to comparable timestamp
    # series before validating the already-frozen strict-before-date contract.
    cur_ts = pd.to_datetime(out["depth_current_snapshot_dt"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    prev_ts = pd.to_datetime(out["depth_previous_snapshot_dt"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    game_ts = pd.to_datetime(out["game_date"], errors="coerce").dt.normalize()
    prev_game_ts = pd.to_datetime(out["previous_game_date"], errors="coerce").dt.normalize()

    out["current_depth_timestamp_violation"] = cur_ts.notna() & game_ts.notna() & cur_ts.ge(game_ts)
    out["previous_depth_timestamp_violation"] = prev_ts.notna() & prev_game_ts.notna() & prev_ts.ge(prev_game_ts)
    if int(out["current_depth_timestamp_violation"].sum()) or int(out["previous_depth_timestamp_violation"].sum()):
        raise RuntimeError("ND5 depth timestamp leakage detected")
    return out


nd5._attach_depth_signals = _attach_depth_signals

if __name__ == "__main__":
    raise SystemExit(nd5.main())
