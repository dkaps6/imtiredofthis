#!/usr/bin/env python3
"""Materialize outcome-free strict-prior player usage-regime context.

Consumes the canonical historical player-game table. For each target player-game,
all features are shifted/rolling summaries of games strictly earlier than the
target row. Target-game usage never contributes to its own features.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["season", "week", "team", "player_identity_key"]
USAGE = ["tgt_share_game", "rush_share_game", "route_rate_game"]


def build_usage_regime_context(history: pd.DataFrame) -> pd.DataFrame:
    x = history.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    required = {"season", "week", "team", "player_identity_key", "position", *USAGE}
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"history missing columns: {sorted(missing)}")
    if x.duplicated(KEY).any():
        raise RuntimeError("history contains duplicate canonical player-game keys")
    x["season"] = pd.to_numeric(x["season"], errors="coerce").astype("Int64")
    x["week"] = pd.to_numeric(x["week"], errors="coerce").astype("Int64")
    x = x.sort_values(["player_identity_key", "season", "week", "team"]).reset_index(drop=True)

    g = x.groupby("player_identity_key", sort=False, group_keys=False)
    x["prior_team"] = g["team"].shift(1)
    x["same_team_as_prior_game"] = np.where(x["prior_team"].notna(), x["team"].eq(x["prior_team"]).astype(float), np.nan)
    x["team_change_prior"] = np.where(x["prior_team"].notna(), x["team"].ne(x["prior_team"]).astype(float), np.nan)
    x["career_games_prior"] = g.cumcount()

    # Number of completed prior games in the current team stint. A team change
    # resets the counter; the target row itself is never counted as prior.
    change = x["prior_team"].notna() & x["team"].ne(x["prior_team"])
    stint = change.groupby(x["player_identity_key"]).cumsum()
    x["games_with_current_team_prior"] = x.groupby([x["player_identity_key"], stint], sort=False).cumcount()

    for col in USAGE:
        vals = pd.to_numeric(x[col], errors="coerce")
        x[f"prior_{col}"] = vals.groupby(x["player_identity_key"]).shift(1)
        for window in (3, 5):
            x[f"prior{window}_{col}_mean"] = vals.groupby(x["player_identity_key"]).transform(
                lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
            )
        x[f"prior_{col}_delta_vs3"] = x[f"prior_{col}"] - x[f"prior3_{col}_mean"]

    out_cols = KEY + [
        "position", "prior_team", "same_team_as_prior_game", "team_change_prior",
        "career_games_prior", "games_with_current_team_prior",
    ]
    for col in USAGE:
        out_cols += [
            f"prior_{col}", f"prior3_{col}_mean", f"prior5_{col}_mean",
            f"prior_{col}_delta_vs3",
        ]
    out = x[out_cols].copy()
    out["usage_regime_coverage_flag"] = np.where(out["career_games_prior"].gt(0), "known", "no_prior_game")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--history", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    out = build_usage_regime_context(pd.read_csv(args.history))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[usage_regime] rows={len(out)} players={out['player_identity_key'].nunique()} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
