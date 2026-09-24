#!/usr/bin/env python3
"""Canonical 2026 Full Slate pricing entrypoint.

The former certified V3 implementation is preserved byte-for-byte in
`run_pricing_with_full_roster_universe_v3_core.py`. The stable public entrypoint
keeps the protected V4/V5 parent chain explicit for production provenance and
binds execution to the certified V6 production wrapper.

Production execution consumes:
- M38 + WR-R15 + TE-R5P receiving entitlement,
- QB M89/M90 + C2 distribution refinement,
- RB-P3 rushing on its qualified Week-1 route,
- RB-R22 Week-1 receiving-yard tails,
- RB-R26 Week-1 receptions refinement,
- RB Rush+Receiving Conservation V2 for non-Week-1 RB rush+receiving yards,
- the existing calibrated MC/ML/state ensemble everywhere else,
- one final authoritative `model_proj` per priced offer.

RB Rush+Receiving Conservation V2 is RB-only; FB is explicitly unchanged.
Sportsbook data remains downstream of football generation.
"""
# Keep the protected V4 and V5 parents explicit in the stable public chain.
from scripts.run_pricing_with_full_roster_universe_v4_production import main
from scripts.run_pricing_with_full_roster_universe_v5_production import main as _r26_v5_main
from scripts.run_pricing_with_full_roster_universe_v6_production import main as _rb_rr_v6_main

# V6 is the promoted public authority. V5/R26 and V4 remain explicit protected
# parent dependencies rather than competing user-facing projection paths.
main = _rb_rr_v6_main


if __name__ == "__main__":
    raise SystemExit(main())
