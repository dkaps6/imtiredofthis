#!/usr/bin/env python3
"""M90 season-parameterized wrapper around the frozen M89 stable-primary trace builder.

M89's helper is intentionally hard-coded to 2023 because it is part of the
historical M89 artifact. M90 needs the exact same logic for an earlier temporal
rotation, so this wrapper changes only the season constant at runtime and then
calls the frozen M89 functions unchanged.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import scripts.backtest.build_m89_2023_training_trace as m89


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--player-logs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if args.season < 2000:
        raise RuntimeError(f"invalid season: {args.season}")

    # The only behavior change versus the frozen M89 helper is the target season.
    m89.SEASON = int(args.season)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    q = m89.build_oos_ensemble(m89.load_predictions(args.predictions))
    out = m89.stable_primary_trace(q, m89.load_logs(args.player_logs))
    out.to_csv(args.out, index=False)

    print(f"[m90_rotated_trace] season={args.season} rows={len(out)} -> {args.out}")
    print(out[["week", "team", "player_clean_key", "actual_attempts", "pred_attempts", "actual_pass_yards", "ensemble_proj"]].head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
