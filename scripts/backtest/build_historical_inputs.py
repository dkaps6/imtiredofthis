"""CLI for building historical walk-forward input artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.backtest.historical_inputs import build_all_historical_inputs


def _parse_weeks(value: str) -> list[int]:
    out = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(token))
    return sorted(set(out))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=2025)
    p.add_argument("--prior-season", type=int, default=2024)
    p.add_argument("--weeks", default="1-18")
    p.add_argument("--out-dir", type=Path, default=Path("data/backtests"))
    args = p.parse_args()
    paths = build_all_historical_inputs(
        season=args.season,
        prior_season=args.prior_season,
        weeks=_parse_weeks(args.weeks),
        out_dir=args.out_dir,
    )
    print(f"[backtest_inputs] schedule={paths['schedule']}")
    print(f"[backtest_inputs] team_weekly={paths['team_weekly']}")
    print(f"[backtest_inputs] universe_dir={paths['universe_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
