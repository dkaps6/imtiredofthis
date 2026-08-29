#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.backtest.build_historical_injuries import load_historical_injuries

COLUMNS = [
    "player", "team", "season", "week", "status", "practice_status",
    "body_part", "designation", "source", "report_available",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", default="2024,2025")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    seasons = [int(v.strip()) for v in args.seasons.split(",") if v.strip()]
    try:
        out = load_historical_injuries(seasons)
    except Exception as exc:
        print(f"[historical_injuries_optional] provider unavailable: {exc}")
        print("[historical_injuries_optional] continuing with empty compatible injury artifact")
        out = pd.DataFrame(columns=COLUMNS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[historical_injuries_optional] rows={len(out)} seasons={seasons} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
