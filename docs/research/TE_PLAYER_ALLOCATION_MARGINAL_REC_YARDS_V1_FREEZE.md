# TE Player Allocation Marginal Receiving-Yards V1 — Frozen Plan

Date: 2026-10-15
Status: FROZEN BEFORE RESULT EXPOSURE / RESEARCH ONLY

Version:
`TE_PLAYER_ALLOCATION_MARGINAL_REC_YARDS_V1`

## Question

The confirmed TE player-allocation-difficulty signal is absent from the current
TE-R5P target-allocation distribution: TE-R5P changes each named TE's mean
entitlement, then the canonical simulator allocates targets through the shared
multinomial using fixed entitlement probabilities. The existing
`rules_volatility_mult` is a team/matchup pressure modifier on yardage noise
and does not encode the named TE's own allocation reliability.

Test one narrow integration question:

> Does the already-confirmed named-TE allocation-difficulty scale improve the
> **marginal TE receiving-yard distribution** when added downstream of the
> canonical TE-R5P production-order Monte Carlo, while preserving the exact
> TE-R5P mean?

This is not a new mean model and not a new joint allocation family.

## Frozen source authorities

1. Exact recovered TE-R5P fold authority:
   - run `37782189538`
   - artifact `11552447166`
   - digest `sha256:94a4bee7011877b1d7a24708d5ba3e86ccc5d29b427bea650af574fc1f09fa24`
   - manifest disposition `WR_TE_FOLD_AUTHORITIES_RECOVERED_EXACT`
   - provider rebuild must be false.

2. Confirmed allocation-only difficulty state:
   - run `38656442803`
   - artifact `11936407975`
   - digest `sha256:7ded306cd0a049902130ccc5cab995231b192120c02679c3852757bc8df29cc0`
   - TE disposition must be `TE_PLAYER_ALLOCATION_DIFFICULTY_CONFIRMED`.

3. Historical TE-R5P production-order marginal:
   - rebuild 2024 and 2025 with the repository's current canonical historical
     production-order path;
   - 2,000 MC draws;
   - seed `42 + week`;
   - TE-R5P applied before WR-R15 exactly as the production-order replay does;
   - use the TE receiving-yard distribution arrays from that replay;
   - no sportsbook input may enter construction.

The historical baseline is the rebuilt canonical TE-R5P marginal itself.
No requirement is made to reproduce the expired September raw-array artifact.

## Population

Primary:
- TE only;
- 2024 development;
- 2025 independent confirmation;
- rows present in the confirmed allocation-only state;
- minimum 4 strictly-prior same-player allocation-error games is inherited from
  the confirmed parent;
- finite positive promoted targets;
- finite positive pregame YPT;
- finite actual receiving yards;
- exact season/week/team/player identity join to one TE-R5P receiving-yard array.

No TE1/TE2 carveout.

## Confirmed player state reused without refit

From the frozen allocation-only parent:

`player_sigma_targets = prior8_abs_alloc_error_mean * sqrt(pi/2)`

and contemporaneous pooled control:

`pool_sigma_targets = week_pool_mae * sqrt(pi/2)`.

No coefficient, cap, threshold, or alternative window is fit.

Convert allocation uncertainty to receiving-yard units using only pregame TE-R5P
football means:

`pregame_ypt = promoted_rec_yards_mean / promoted_targets_mean`

`player_sigma_yards = player_sigma_targets * pregame_ypt`

`pool_sigma_yards = pool_sigma_targets * pregame_ypt`.

The conversion uses no realized target count, receiving yards, sportsbook line,
or 2026 outcome.

## Three frozen distribution arms

For each eligible row, let `X` be the exact canonical TE-R5P receiving-yard
draw array and `mu = mean(X)`.

### A. Canonical baseline
Use `X` unchanged.

### B. Pooled allocation-width control
Add independent centered Normal noise with SD `pool_sigma_yards` to the
canonical marginal.

### C. Named-player candidate
Add independent centered Normal noise with SD `player_sigma_yards` to the
canonical marginal.

For B/C:
- deterministic stable per-row seed derived only from version +
  season/week/team/player + arm;
- exactly 2,000 noise draws;
- center the generated noise to sample mean zero before addition;
- floor resulting receiving yards at zero;
- rescale the nonnegative draw vector multiplicatively back to exact canonical
  mean `mu`;
- if exact mean cannot be restored within `1e-10`, fail closed.

This is a **marginal shadow only**. It does not alter target shares, team target
mass, teammate arrays, receptions, production means, or any joint covariance
claim.

## Primary scoring

Score actual receiving yards using the empirical draw distributions.

For each arm report:
- empirical CRPS;
- central 80% coverage and absolute nominal gap;
- central 90% coverage and absolute nominal gap.

Paired gains:
- named vs canonical;
- named vs pooled.

Dependence-aware inference:
- player-cluster bootstrap;
- game-cluster bootstrap;
- 5,000 replicates;
- fixed seeds recorded in implementation.

Also report:
- high-difficulty Q4 using the parent `player_mae` Q75 convention;
- per-player 2025 diagnostics for TEs with >=6 scored games;
- distribution-SD changes;
- maximum mean drift.

## Frozen qualification gates

Support:
- 2024 >=600 rows and >=60 TEs;
- 2025 >=600 rows and >=60 TEs;
- pooled >=1,300 rows and >=90 TEs.

Integrity:
- exact identity join;
- zero missing baseline arrays for eligible rows;
- max mean drift <= `1e-10`;
- sportsbook inputs = 0;
- OddsAPI calls = 0;
- 2026 outcomes read = 0;
- production changed = false.

Science:
1. named CRPS < canonical CRPS in 2024;
2. named CRPS < canonical CRPS in 2025;
3. named CRPS < pooled-overlay CRPS in 2024;
4. named CRPS < pooled-overlay CRPS in 2025;
5. pooled named-vs-canonical player-cluster 95% CI low > 0;
6. pooled named-vs-canonical game-cluster 95% CI low > 0;
7. 2025 named-vs-pooled player-cluster 95% CI low > 0;
8. 2025 named-vs-pooled game-cluster 95% CI low > 0;
9. pooled named 80% absolute coverage gap <= canonical gap;
10. pooled named 90% absolute coverage gap <= canonical gap;
11. at least one of the pooled 80/90 coverage gaps is strictly better;
12. high-difficulty Q4 named CRPS < canonical CRPS;
13. 2025 per-player median named-vs-canonical CRPS gain > 0;
14. 2025 fraction of qualifying players with positive named-vs-canonical gain > 0.50.

Disposition:
- all gates pass:
  `TE_PLAYER_ALLOCATION_MARGINAL_REC_YARDS_QUALIFIED`
- otherwise:
  `NO_ACTIONABLE_TE_PLAYER_ALLOCATION_MARGINAL_REC_YARDS`.

## If qualified

Qualification is research-only.

Immediately freeze a separate **2026 Week 6 prospective shadow lock** from the
already-existing pregame Week 6 football slate. Use a no-odds football artifact
or otherwise prove sportsbook data is downstream-only. Do not read Week 6
outcomes before the lock is uploaded.

Production promotion requires future prospective scoring and a separate
production-integration contract.

## If failed

Close this exact mapping. Do not rescue with:
- coefficient fitting;
- sigma caps/floors beyond the physical zero-yard floor already frozen;
- TE1/TE2 slices;
- different prior windows/minimum support;
- alternative YPT conversion;
- alternate noise family;
- alternate clipping/recentering rule;
- pooled-season-only qualification.

The closed TE joint Dirichlet mapping remains closed regardless of this result.
The older global TE-R5P Width V2 remains closed regardless of this result.

No paid OddsAPI.
No production mutation.
