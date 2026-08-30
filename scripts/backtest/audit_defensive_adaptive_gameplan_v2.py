#!/usr/bin/env python3
"""M83 robustness wrapper for nullable participation man/zone labels.

The frozen defensive-adaptation construction, similarity metric, k=4 rule,
source gates and predictability gates are unchanged. This wrapper only makes the
historical auxiliary participation parser NA-safe so the independent FTN audit
can complete.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import audit_defensive_adaptive_gameplan as m83


def build_part_response(j: pd.DataFrame) -> pd.DataFrame:
    if j.empty:
        return pd.DataFrame()
    x = m83.lower(j)
    mz = x.defense_man_zone_type.astype("string").str.upper().str.strip()
    man_mask = (
        mz.str.contains("MAN", na=False).fillna(False)
        | mz.eq("M").fillna(False)
    ).astype(bool)
    zone_mask = (
        mz.str.contains("ZONE", na=False).fillna(False)
        | mz.eq("Z").fillna(False)
    ).astype(bool)
    x["man_flag"] = np.where(
        man_mask.to_numpy(),
        1.0,
        np.where(zone_mask.to_numpy(), 0.0, np.nan),
    )
    x["zone_flag"] = np.where(np.isfinite(x.man_flag), 1.0 - x.man_flag, np.nan)
    x["box_num"] = pd.to_numeric(x.defenders_in_box, errors="coerce")
    rush = pd.to_numeric(x.get("rush_attempt"), errors="coerce").fillna(0).eq(1)
    x["box_rush"] = x.box_num.where(rush)
    gid = "nflverse_game_id" if "nflverse_game_id" in x.columns else ("game_id" if "game_id" in x.columns else "old_game_id")
    x["game_id_norm"] = x[gid].astype(str)
    return x.loc[x.defteam.notna() & x.posteam.notna()].groupby(
        ["season", "week", "game_id_norm", "defteam", "posteam"], as_index=False
    ).agg(
        man_rate=("man_flag", "mean"),
        zone_rate=("zone_flag", "mean"),
        avg_box=("box_rush", "mean"),
    ).rename(columns={
        "game_id_norm": "game_id",
        "defteam": "defense",
        "posteam": "offense",
    })


m83.build_part_response = build_part_response

if __name__ == "__main__":
    raise SystemExit(m83.main())
