#!/usr/bin/env python3
"""Mechanical compatibility wrapper for historical defensive enrichment.

nflverse participation is only available from 2016 onward. Older historical
backtests must therefore preserve their existing team-weekly rows and treat the
participation-derived defensive fields as unavailable rather than failing the
workflow. This wrapper does not alter candidate science or impute old outcomes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.backtest.enrich_historical_defense import (
    audit_frame,
    build_defensive_observations,
    enrich_team_weekly,
)

MIN_PARTICIPATION_SEASON = 2016
AUDIT_CONTEXT_COLS = [
    "light_box_rate",
    "heavy_box_rate",
    "coverage_man_rate",
    "coverage_zone_rate",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--team-weekly", type=Path, required=True)
    p.add_argument("--seasons", required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--observations", type=Path, required=True)
    args = p.parse_args()

    requested = [int(v.strip()) for v in args.seasons.split(",") if v.strip()]
    supported = [s for s in requested if s >= MIN_PARTICIPATION_SEASON]
    unsupported = [s for s in requested if s < MIN_PARTICIPATION_SEASON]

    if not args.team_weekly.exists() or args.team_weekly.stat().st_size == 0:
        raise RuntimeError(f"missing team weekly history: {args.team_weekly}")

    base = pd.read_csv(args.team_weekly, low_memory=False)
    if supported:
        defense = build_defensive_observations(supported)
    else:
        defense = pd.DataFrame(columns=["season", "week", "team"])

    enriched = enrich_team_weekly(base, defense)
    for col in AUDIT_CONTEXT_COLS:
        if col not in enriched.columns:
            enriched[col] = pd.Series(float("nan"), index=enriched.index, dtype=float)
    enriched.to_csv(args.team_weekly, index=False)

    args.observations.parent.mkdir(parents=True, exist_ok=True)
    defense.to_csv(args.observations, index=False)

    audit = audit_frame(enriched)
    audit["requested_seasons"] = ",".join(map(str, requested))
    audit["supported_participation_seasons"] = ",".join(map(str, supported))
    audit["unsupported_participation_seasons"] = ",".join(map(str, unsupported))
    audit["mechanical_fallback_only"] = True
    audit.to_csv(args.audit, index=False)

    print(
        "[historical_defense_supported] "
        f"requested={requested} supported={supported} unsupported={unsupported} "
        f"team_weeks={len(enriched)} participation_team_weeks={len(defense)}"
    )
    print(audit.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
