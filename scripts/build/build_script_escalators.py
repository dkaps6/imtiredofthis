#!/usr/bin/env python3
"""Compatibility wrapper for season-aware game-script features."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from scripts.build.pbp_features import build_script_escalators
from scripts.utils.pbp import get_pbp


def main(out_csv: str = "data/script_escalators.csv", season: int | None = None) -> None:
    season = int(season if season is not None else os.getenv("SEASON", "2026"))
    pbp = get_pbp(season, min_rows=1)
    if "season_type" in pbp.columns:
        reg = pbp.loc[pbp["season_type"].astype(str).str.upper().eq("REG")]
        if not reg.empty:
            pbp = reg.copy()
    out = build_script_escalators(pbp)
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"[script_escalators] season={season} rows={len(out)} -> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out_csv", nargs="?", default="data/script_escalators.csv")
    parser.add_argument("--season", type=int, default=None)
    args = parser.parse_args()
    main(args.out_csv, args.season)
