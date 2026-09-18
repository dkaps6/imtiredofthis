#!/usr/bin/env python3
"""Join strict-prior player usage-regime and room-continuity context.

Engineering/QA only. This creates a qualification-ready player-game context table;
it does not read target outcomes or fit a projection model. The join is many-to-one
from player-game usage rows onto team-position-week room rows and fails closed on
key duplication or fanout.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PLAYER_KEY = ["season", "week", "team", "player_identity_key"]
ROOM_KEY = ["season", "week", "team", "position"]


def build_role_room_context(usage: pd.DataFrame, room: pd.DataFrame) -> pd.DataFrame:
    u = usage.copy(); r = room.copy()
    u.columns = [str(c).strip().lower() for c in u.columns]
    r.columns = [str(c).strip().lower() for c in r.columns]
    for name, df, required in (
        ("usage", u, {*PLAYER_KEY, "position", "career_games_prior", "usage_regime_coverage_flag"}),
        ("room", r, {*ROOM_KEY, "room_prior_games", "room_continuity_coverage_flag"}),
    ):
        missing = required - set(df.columns)
        if missing:
            raise RuntimeError(f"{name} missing columns: {sorted(missing)}")
    if u.duplicated(PLAYER_KEY).any():
        raise RuntimeError("usage contains duplicate canonical player-game keys")
    if r.duplicated(ROOM_KEY).any():
        raise RuntimeError("room contains duplicate canonical team-position-week keys")

    before = len(u)
    out = u.merge(r, on=ROOM_KEY, how="left", validate="many_to_one", indicator=True)
    if len(out) != before:
        raise RuntimeError("room join changed player-game row count")

    # Qualification-ready, outcome-free support/missingness diagnostics.
    out["stable_identity_flag"] = out["player_identity_key"].notna().astype(int)
    out["pregame_context_eligible_flag"] = 1
    out["usage_unknown_flag"] = out["usage_regime_coverage_flag"].ne("known").astype(int)
    out["room_unknown_flag"] = (
        out["room_continuity_coverage_flag"].fillna("missing_room").ne("known").astype(int)
    )
    out["any_context_unknown_flag"] = np.maximum(out["usage_unknown_flag"], out["room_unknown_flag"])
    out["strict_prior_support_games"] = pd.concat([
        pd.to_numeric(out["career_games_prior"], errors="coerce"),
        pd.to_numeric(out["room_prior_games"], errors="coerce"),
    ], axis=1).min(axis=1).fillna(0).astype(int)
    out["room_join_state"] = out["_merge"].map({"both":"matched", "left_only":"missing_room", "right_only":"unexpected"})
    out = out.drop(columns=["_merge"])
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--usage", type=Path, required=True)
    p.add_argument("--room", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    out = build_role_room_context(pd.read_csv(args.usage), pd.read_csv(args.room))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(
        f"[role_room_context] rows={len(out)} players={out.player_identity_key.nunique()} "
        f"room_match={(out.room_join_state == 'matched').mean():.4f} "
        f"known_both={(out.any_context_unknown_flag == 0).mean():.4f} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
