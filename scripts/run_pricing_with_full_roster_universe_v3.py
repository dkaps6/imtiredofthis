#!/usr/bin/env python3
"""Canonical 2026 Week-1 Full Slate pricing entrypoint.

The former certified V3 implementation is preserved byte-for-byte in
`run_pricing_with_full_roster_universe_v3_core.py`. The stable public entrypoint
keeps the protected V4 parent explicit for production provenance/static-audit
compatibility, then binds execution to the qualified V5/R26 production stack.

Final Week-1 execution therefore consumes:
- M38 + WR-R15 + TE-R5P receiving entitlement,
- QB M89/M90 + C2 distribution refinement,
- RB-P3 rushing,
- RB-R22 receiving-yard tails,
- RB-R26 Week-1 receptions refinement,
- the existing unchanged calibrated MC/ML/state ensemble,
- one final authoritative `model_proj` per priced offer.

Sportsbook data remains downstream of football generation.
"""
# Keep the protected V4 parent explicit in the stable public chain. V5 imports and
# executes this same V4 parent before applying the qualified R26 receptions layer.
from scripts.run_pricing_with_full_roster_universe_v4_production import main
from scripts.run_pricing_with_full_roster_universe_v5_production import main as _r26_v5_main

# The promoted public authority is V5/R26; V4 above remains the explicit protected
# parent dependency rather than a competing user-facing projection path.
main = _r26_v5_main


if __name__ == "__main__":
    raise SystemExit(main())
