#!/usr/bin/env python3
"""Materialize strict-prior team/position-room continuity context.

This is descriptive pregame context, not an inheritance model. For each target
team-position-week, it summarizes only completed earlier games. It measures room
concentration and how much of the immediately prior game's opportunity belonged
to identities also observed in the game before that. Target-game participation
or opportunity is never used to construct its own features.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["season", "week", "team", "player_identity_key"]
USAGE = ["tgt_share_game", "rush_share_game"]


def _weighted_overlap(cur: pd.DataFrame, prev_ids: set[str], col: str) -> float:
    vals = pd.to_numeric(cur[col], errors="coerce").fillna(0.0).clip(lower=0.0)
    denom = float(vals.sum())
    if denom <= 0:
        return np.nan
    keep = cur["player_identity_key"].astype(str).isin(prev_ids)
    return float(vals[keep].sum() / denom)


def build_room_continuity_context(history: pd.DataFrame) -> pd.DataFrame:
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
    x["position"] = x["position"].astype(str).str.upper().str.strip()

    # Build completed-game room snapshots first. A target week only receives
    # snapshots with week < target week in the same season/team/position room.
    snapshots: dict[tuple[int, str, str], list[dict]] = {}
    for (season, week, team, pos), g in x.groupby(["season", "week", "team", "position"], sort=True):
        row = {"season": int(season), "week": int(week), "team": team, "position": pos,
               "ids": set(g["player_identity_key"].astype(str))}
        for col in USAGE:
            vals = pd.to_numeric(g[col], errors="coerce").fillna(0.0).clip(lower=0.0)
            ordered = vals.sort_values(ascending=False)
            row[f"{col}_sum"] = float(vals.sum())
            row[f"{col}_top1"] = float(ordered.iloc[0]) if len(ordered) else np.nan
            row[f"{col}_top2"] = float(ordered.iloc[:2].sum()) if len(ordered) else np.nan
            row[f"frame_{col}"] = g[["player_identity_key", col]].copy()
        snapshots.setdefault((int(season), str(team), str(pos)), []).append(row)

    targets = x[["season", "week", "team", "position"]].drop_duplicates().sort_values(
        ["season", "week", "team", "position"]
    )
    out = []
    for t in targets.itertuples(index=False):
        prior = [s for s in snapshots.get((int(t.season), str(t.team), str(t.position)), []) if s["week"] < int(t.week)]
        prior.sort(key=lambda s: s["week"])
        r = {"season": int(t.season), "week": int(t.week), "team": t.team, "position": t.position,
             "room_prior_games": len(prior)}
        if not prior:
            r["room_continuity_coverage_flag"] = "no_prior_room_game"
            for col in USAGE:
                r[f"prior_{col}_top1"] = np.nan
                r[f"prior_{col}_top2"] = np.nan
                r[f"prior_{col}_returning_overlap"] = np.nan
        else:
            last = prior[-1]
            r["room_continuity_coverage_flag"] = "known"
            prev_ids = prior[-2]["ids"] if len(prior) >= 2 else set()
            for col in USAGE:
                r[f"prior_{col}_top1"] = last[f"{col}_top1"]
                r[f"prior_{col}_top2"] = last[f"{col}_top2"]
                r[f"prior_{col}_returning_overlap"] = (
                    _weighted_overlap(last[f"frame_{col}"], prev_ids, col) if len(prior) >= 2 else np.nan
                )
        out.append(r)
    return pd.DataFrame(out)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--history", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    out = build_room_continuity_context(pd.read_csv(args.history))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[room_continuity] rows={len(out)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
