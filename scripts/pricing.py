#!/usr/bin/env python3
"""Backward-compatible entry point for the independent pricing engine.

The previous implementation initialized model mean at the sportsbook line and
filtered contextual frames to 2025.  Production pricing now lives in
``scripts.pricing_v2``; this module remains so old commands/imports continue to
work without reviving the legacy behavior.
"""
from __future__ import annotations

import argparse

from scripts.pricing_v2 import price as _price
from scripts.runtime_context import resolve_season


def price(season: int, props_path=None):
    # ``props_path`` is retained for API compatibility.  pricing_v2 consumes the
    # validated metrics_ready table, which is the only supported production input.
    return _price(int(season))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--write", type=str, default="outputs")
    parser.add_argument("--props", type=str, default="")
    args = parser.parse_args()
    season = int(args.season if args.season is not None else resolve_season())
    out = price(season, props_path=args.props or None)
    # pricing_v2.price returns a DataFrame; its CLI handles writing.  Preserve
    # historical direct-call behavior by writing here too.
    from scripts.pricing_v2 import OUT
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"[pricing] wrote {OUT} rows={len(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
