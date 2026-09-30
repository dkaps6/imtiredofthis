# WR-CB Wayback 2024-2025 Exact-URL Index Expansion V1 — Access Result

Date: 2026-09-30  
Disposition: **INFRASTRUCTURE-INCONCLUSIVE. DO NOT INTERPRET ZERO CANDIDATES AS ZERO SNAPSHOTS.**

Run `36769614598` completed successfully at the workflow level after querying the 35 exact 2024-2025 article URLs with bounded parallel CDX requests. It returned **0 usable candidate timestamps**, but this is not a source finding because the public Wayback index throttled/failed the request set.

Crucial positive control:
- 2024 Week 1 returned `NETWORK_ReadTimeout` in this scan.
- The same exact URL is independently proven to have an indexed capture at `20240907005714`, and that capture has already been replayed, digest-matched and recovered into 46 strict two-sided exact-week roster-verified pregame pairings.

Therefore the mass-scan's zero-candidate count is demonstrably a **false negative due to index access**.

Observed query states included:
- `NETWORK_ReadTimeout`;
- `NETWORK_ConnectTimeout`;
- `NETWORK_ConnectionError`;
- `HTTP_503`;
- a small number of `NO_INDEX_MATCH_NOT_PROOF_OF_ABSENCE`.

No archived article bodies were fetched by this scan, no current page reacquisition, no sportsbook inputs, no target outcomes, no fitted parameters. 2025 remains protected from outcome/model use.

Next policy:
1. Do not hammer the Wayback CDX with broad concurrent scans.
2. Preserve the validated 2024-W1 archive result as the positive-control recovery method.
3. Use sparse exact-URL/timestamp discovery from public indices/searches when available, then replay only frozen candidates.
4. Prefer prospective immutable 2026+ captures for guaranteed forward provenance.
5. Never infer historical absence from CDX timeout/503/connection-error states.
