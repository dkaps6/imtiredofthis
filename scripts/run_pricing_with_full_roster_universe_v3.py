#!/usr/bin/env python3
"""Canonical 2026 Week-1 Full Slate pricing entrypoint.

The former certified V3 implementation is preserved byte-for-byte in
`run_pricing_with_full_roster_universe_v3_core.py`.  The public V3 entrypoint now
routes to the R22-certified V4 stack so existing Full Slate workflow wiring does
not need a broad mechanical rewrite.
"""
from scripts.run_pricing_with_full_roster_universe_v4_production import main


if __name__ == "__main__":
    raise SystemExit(main())
