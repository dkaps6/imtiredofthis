# RB R24 — Entitlement + Production Efficiency Decomposition

Status: **FROZEN BEFORE OUTCOME INSPECTION**
Date: 2026-09-09
Base production authority: `main@f8417f55b04ce0e19baf260e9d532765034c47f1`
Research branch: `research-rb-r24-entitlement-production-efficiency-v1`
Parent null: RB R23 authoritative run `34332613867`, disposition `MIXED_OR_FAIL_NO_PROMOTION`

## Why R24 exists

R23 produced a clean mechanistic split: conserved within-room RB opportunity allocation improved pooled target and reception prediction, but the combined new opportunity + new shrunk receiving-efficiency mean failed receiving-yard replication, p90 protection, and RB1 role robustness. R23 is frozen as a no-promotion result and will not be retuned.

R24 asks a different question: **does the demonstrated entitlement/reception improvement remain useful when paired with the existing production receiving-efficiency mean logic instead of R23's failed new YPR component?**

This is a decomposition experiment, not a threshold rescue. R24 removes the failed efficiency mechanism entirely and preserves production efficiency unchanged.

## Protected authorities

R24 may not change:

1. RB P3 rushing authority.
2. RB R22 receiving-yard distribution/tail authority, parameters, residual pools, thresholds, seeds, routing, rank and mean-preservation contracts.
3. M89/M90 QB point-mean authority and QB C2 selector.
4. M38 + WR-R15 entitlement.
5. TE-R5P entitlement.
6. Finite team target pool and residual-mass accounting.
7. Current production RB receiving-efficiency mean formula/parameters used beneath R22.
8. Sportsbook independence upstream.

## Frozen candidate

### Control

Reproduce the current production RB receiving target, reception, and receiving-yard point means exactly.

### R24 candidate

Use only the R23 entitlement/reception mechanism that was defined before R23 outcomes:

- redistribute only the existing finite RB-room target mass;
- use strictly-prior 6-game recent and 16-game stabilizing histories with the same pre-frozen empirical-Bayes role shrinkage used by R23;
- derive receptions using R23's pre-frozen strictly-prior catch-conversion component;
- **do not use R23's new shrunk YPR receiving-efficiency component**;
- convert opportunity/receptions to receiving-yard mean using the existing production receiving-efficiency logic unchanged.

No R23 coefficient, window, prior, cap, threshold, or gate may be changed after the R23 result. No new candidate family may be added after R24 outcomes are seen.

## Confirmation population and timing

Exactly the same strict-prior historical contract and evaluation population as R23, over confirmation seasons 2023, 2024, and 2025. 2026 outcomes remain forbidden.

## Frozen metrics

For targets, receptions, and receiving yards:

- MAE
- RMSE
- bias
- Pearson/Spearman
- median, p75, p90 absolute error

For receiving yards also report 20+/30+/40+ miss rates, RB1 and RB2+ slices, season slices, low-history/new-player slices where available, and conservation/protected-authority parity.

## Frozen promotion gates

R24 is eligible for later full-stack integration only if every gate passes:

### Integrity

1. Sportsbook inputs used upstream = 0.
2. Future/2026 outcomes used = 0.
3. Strict-prior timing verified.
4. RB-room target mass conserved to numerical tolerance.
5. WR/TE entitlement unchanged exactly.
6. P3 rushing unchanged exactly.
7. R22 logic/parameters unchanged and mean-preserving around the R24 upstream mean.
8. Production receiving-efficiency logic is byte/parameter-equivalent to the baseline implementation except for the changed upstream expected target/reception inputs required by the decomposition.

### Scientific

1. Pooled target MAE improves versus production baseline and target RMSE does not worsen.
2. Pooled reception MAE improves versus baseline and reception RMSE does not worsen.
3. Pooled receiving-yard MAE improves by at least **0.5%** versus production baseline and receiving-yard RMSE does not worsen.
4. Receiving-yard MAE improves in at least **2 of 3** seasons; no season may worsen by more than **0.75%**.
5. Pooled p90 receiving-yard absolute error may not worsen by more than **1.0%**.
6. Pooled 30+ receiving-yard miss rate may not worsen by more than **0.5 percentage points**.
7. Absolute pooled receiving-yard bias may not worsen by more than **0.75 yards**; absolute reception bias may not worsen by more than **0.10 receptions**.
8. Neither RB1 nor RB2+ receiving-yard MAE may worsen by more than **0.75%**.
9. Mechanism coherence: any receiving-yard improvement must coexist with nonworse target/reception structure and exact finite opportunity conservation.

These thresholds are frozen now, before R24 outcomes are inspected. A near miss fails.

## Disposition

- **PASS:** all integrity and scientific gates pass; proceed only to a separately frozen full-stack integration test under unchanged R22.
- **MIXED / NO PROMOTION:** any frozen gate fails despite partial improvement; preserve and move on.
- **FAIL:** no robust improvement or any integrity violation; preserve null and bound the RB receiving lane before proceeding to the next program priority.

## Required artifacts

- plan/code hashes
- baseline reproduction audit
- fold predictions and pooled/season/role metrics
- conservation and protected-authority parity audits
- exact run/job/SHA/artifact lineage
- final disposition document

This plan was committed before inspecting any R24 candidate outcome.