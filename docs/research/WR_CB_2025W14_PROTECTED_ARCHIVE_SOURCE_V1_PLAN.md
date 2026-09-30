# Protected 2025 Week 14 Exact Archive Source Verification V1

A source-only verification of the availability-API candidate `20251206131409` for the exact FantasyAlarm 2025 Week-14 WR/CB article. This is a **protected confirmation-season provenance task**, not model science.

Required before source acceptance:
1. recover the exact CDX timestamp+digest for that one archived record;
2. exact replay must remain on the requested Wayback timestamp and raw-body SHA-1/Base32 must match CDX digest;
3. parse only explicit factual WR↔CB pairings from the archived body/JSON-LD;
4. keep only rows whose game had not kicked off at `2025-12-06T13:14:09Z` and whose WR team/opponent schedule matches;
5. resolve WR and CB only through unique exact 2025 Week-14 rosters on the stated offense/opponent; no provider bridge, season fallback, fuzzy or manual rescue;
6. preserve only sanitized GSIS pair/alignment facts, hashes and counts.

The script contains no outcome loader and must record `confirmation_outcomes_accessed=false`, `target_game_outcomes=false`, `parameters_fit=0`. If CDX digest cannot be obtained or replay digest mismatches, it fails closed and the source remains unverified. No 2025 outcome can be examined until a later scientific design is frozen.
