# QB Individual Mechanism Decomposition — Result

## Canonical run
- Run: `34065369449`
- Tested SHA: `f8b2f6578560c66764d2102a0c765dfe67995015`
- Artifact: `9998765968`
- Artifact SHA256: `48b83ac86218e3d8ef7b023bf7e70b840811a0cbcfd3e0e28272d6e12c9de3`
- Rows: 884 (2024-2025); QBs: 59; qualifying profiles >=8 games: 34
- Actual-yard reconciliation max error: 0
- Decomposition max identity error: `1.6e-7`
- Sportsbook/model fitting/production change: none

## Aggregate mechanism
- Final synthesis MAE: 55.060 yards
- Mean absolute attempt contribution: **50.517 yards**
- Mean absolute YPA contribution: **43.057 yards**
- Mean absolute stack adjustment: 17.610 yards
- Mean absolute synthesis adjustment: 27.410 yards

High-MAE examples and dominant mechanism:
- Russell Wilson: 82.06 MAE — YPA
- Bryce Young: 74.23 — YPA
- Joe Flacco: 73.28 — ATTEMPTS
- Kirk Cousins: 66.68 — ATTEMPTS
- Geno Smith: 66.41 — ATTEMPTS
- Baker Mayfield: 65.45 — ATTEMPTS
- Joe Burrow: 64.96 — ATTEMPTS
- Jalen Hurts: 64.18 — YPA
- Brock Purdy: 61.32 — YPA
- Matthew Stafford: 61.23 — ATTEMPTS
- Josh Allen: 59.12 — ATTEMPTS
- Justin Herbert: 58.01 — ATTEMPTS
- Jared Goff: 53.86 — ATTEMPTS
- Jordan Love: 52.68 — ATTEMPTS

## Official disposition
**`QB_INDIVIDUAL_MECHANISMS_MAPPED`**

The individual-QB error problem is not a single blanket residual-bias issue. The earlier player-bias shrinkage test failed; this decomposition instead identifies attempt-volume versus YPA/efficiency mechanisms. Any future correction must be a separately frozen generalizable pregame mechanism test. The closed generic QB mean-feature hunt remains closed.
