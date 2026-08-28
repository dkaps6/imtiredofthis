#!/usr/bin/env python3
"""Attach nflverse/GSIS player IDs to frozen Migration 60B QB projections.

This is a market-join utility only. It does not alter any football projection.
The historical player logs and the independent projection trace share the
season/week/team/player_clean_key identity; the resulting GSIS player_id is used
only to join the downstream free sportsbook archive safely.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team


def clean_id(v) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "<na>"}:
        return ""
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--projection-file", type=Path, required=True)
    ap.add_argument("--player-logs", type=Path, required=True)
    args = ap.parse_args()

    p = pd.read_csv(args.projection_file)
    logs = pd.read_csv(args.player_logs, low_memory=False)
    p.columns = [str(c).strip().lower() for c in p.columns]
    logs.columns = [str(c).strip().lower() for c in logs.columns]

    need_p = {"season", "week", "team", "player_clean_key"}
    need_l = need_p | {"player_id"}
    if not need_p.issubset(p.columns):
        raise RuntimeError(f"projection file missing columns: {sorted(need_p-set(p.columns))}")
    if not need_l.issubset(logs.columns):
        raise RuntimeError(f"player logs missing columns: {sorted(need_l-set(logs.columns))}")

    for x in (p, logs):
        x["season"] = pd.to_numeric(x["season"], errors="coerce").astype("Int64")
        x["week"] = pd.to_numeric(x["week"], errors="coerce").astype("Int64")
        x["team"] = x["team"].fillna("").astype(str).map(canon_team)
        x["player_clean_key"] = x["player_clean_key"].fillna("").astype(str).str.strip()
    logs["player_id"] = logs["player_id"].map(clean_id)

    keys = ["season", "week", "team", "player_clean_key"]
    ids = logs.loc[logs.player_id.ne(""), keys + ["player_id"]].drop_duplicates()
    ambiguous = ids.groupby(keys).player_id.nunique().gt(1)
    if ambiguous.any():
        bad = ambiguous[ambiguous].reset_index().head(10).to_dict(orient="records")
        raise RuntimeError(f"ambiguous GSIS IDs for projection keys: {bad}")
    ids = ids.drop_duplicates(keys)

    if "player_id" in p.columns:
        p = p.drop(columns=["player_id"])
    out = p.merge(ids, on=keys, how="left", validate="one_to_one")
    out["player_id"] = out["player_id"].map(clean_id)

    matched = int(out.player_id.ne("").sum())
    total = len(out)
    coverage = matched / total if total else 0.0
    print(f"[m60b gsis] {args.projection_file} matched={matched}/{total} coverage={coverage:.3%}")
    if coverage < 0.95:
        sample = out.loc[out.player_id.eq(""), keys].head(20).to_dict(orient="records")
        raise RuntimeError(f"GSIS projection-ID coverage below 95%; sample missing={sample}")

    out.to_csv(args.projection_file, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
