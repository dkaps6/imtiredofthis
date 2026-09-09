#!/usr/bin/env python3
"""Canonical 2026 Week-1 Full Slate pricing entrypoint.

The former certified V3 implementation is preserved byte-for-byte in
`run_pricing_with_full_roster_universe_v3_core.py`. The public V3 entrypoint now
routes to the R26-qualified V5 production stack so the existing Full Slate
workflow keeps one stable entrypoint while consuming:

- M38 + WR-R15 + TE-R5P receiving entitlement,
- QB M89/M90 + C2 distribution refinement,
- RB-P3 rushing,
- RB-R22 receiving-yard tails,
- RB-R26 Week-1 receptions refinement,
- the existing unchanged calibrated MC/ML/state ensemble,
- one final authoritative `model_proj` per priced offer.

Sportsbook data remains downstream of football generation.
"""
from scripts.run_pricing_with_full_roster_universe_v5_production import main


if __name__ == "__main__":
    raise SystemExit(main())
