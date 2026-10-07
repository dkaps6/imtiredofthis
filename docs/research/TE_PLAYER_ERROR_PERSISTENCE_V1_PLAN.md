# TE Player Error Persistence V1 — Frozen Diagnostic Plan

**STATUS: FROZEN BEFORE RESULT. DIAGNOSTIC ONLY. NO PRODUCTION CHANGE.**

## Purpose

Test whether the promoted TE-R5P receiving-yard architecture leaves stable, player-specific residual behavior after its individualized entitlement layer has already been applied.

This is the TE counterpart to prior WR-R3 / RB-PD2 player-reliability diagnostics, but it uses the **exact promoted TE-R5P OOS production authority** rather than a generic TE baseline.

Question:

> After TE-R5P has already individualized target entitlement, does a tight end's own strictly-prior projection-error history contain information about that same player's next TE-R5P receiving-yard error?

This audit does not fit a correction and does not alter TE-R5P.

## Exact parent authority

TE-R5P production-contract refit:
- run `34152797603` — SUCCESS
- job `101838375798`
- artifact `10029942404`
- digest `sha256:f9951441b748ef72514dbc81adf6fbe9cd023c9bf64ecb52a016840989ab4cdb`
- file: `data/backtests/te_r5p_production_contract_refit_v1/te_r5p_oos_player_casebook.csv`

Frozen authority invariants:
- 3,214 exact OOS TE player-games;
- 2023: 1,019;
- 2024: 1,082;
- 2025: 1,113;
- projection: `candidate_rec_yards_r5p`;
- actual: `rec_yards`;
- pooled authority MAE: `15.850607934418186`;
- pooled authority RMSE: `23.06142088072041`;
- no sportsbook inputs;
- same/future participation observations used = 0.

Do not substitute TE-R5, B0, a rebuilt generic ensemble, or the later failed TE Width V2 candidate.

## Protected / closed TE science

Preserve:
- TE-R5P entitlement;
- finite TE-room conservation;
- current production-order simulation;
- TE-R5P Receiving-Yards Width V2 = **FAILED CLOSED**;
- generic TE target-pool/pool-only corrections already closed;
- public TE-YPT additive integration candidate already closed.

This diagnostic does **not** authorize another generic width multiplier merely if player difficulty persists.

## Historical feature construction

Target seasons:
- 2024
- 2025

For each exact TE-R5P target player-game, collect the same player's TE-R5P OOS rows strictly before that target in chronological order.

History may cross seasons.

Use the latest up to **8** strictly-prior same-player TE-R5P games.

Minimum prior support:
- **4** prior TE-R5P player-games.

Pregame features:

1. `prior8_signed_error_mean`
   - mean of `candidate_rec_yards_r5p - rec_yards`

2. `prior8_abs_error_mean`
   - mean absolute TE-R5P receiving-yard error

3. `prior8_miss30_rate`
   - fraction of prior rows with absolute error >= 30 yards

No target-game row may enter its own feature.

No sportsbook line/price/edge/market residual is permitted.

## Target quantities

For each scoreable target row:

- `current_signed_error = candidate_rec_yards_r5p - rec_yards`
- `current_abs_error = abs(current_signed_error)`
- `current_miss30 = 1[current_abs_error >= 30]`

## Frozen diagnostics

### A — Directional bias persistence

Per 2024, 2025, and pooled:
- Spearman(`prior8_signed_error_mean`, `current_signed_error`);
- sign agreement rate between prior mean and current error, excluding exact zero ties.

A passes only if:
1. Spearman > 0 in 2024;
2. Spearman > 0 in 2025;
3. pooled player-cluster bootstrap P(Spearman > 0) >= 0.95;
4. pooled sign agreement > 0.50.

### B — Individual difficulty persistence

Per 2024, 2025, pooled:
- Spearman(`prior8_abs_error_mean`, `current_abs_error`).

B passes only if:
1. Spearman > 0 in 2024;
2. Spearman > 0 in 2025;
3. pooled player-cluster bootstrap P(Spearman > 0) >= 0.95.

### C — Extreme-miss persistence

Per 2024, 2025, pooled:
- Spearman(`prior8_miss30_rate`, `current_miss30`).

C passes only if:
1. Spearman > 0 in 2024;
2. Spearman > 0 in 2025;
3. pooled player-cluster bootstrap P(Spearman > 0) >= 0.95.

No quartile threshold, player carveout, or alternate miss threshold is searched.

## Support gate

Each target season must have:
- >= 400 scoreable player-games;
- >= 50 distinct TE player identities.

If support fails, disposition is source-limited rather than scientific null.

## Cluster bootstrap

- cluster = stable TE player identity (`player_clean_key`);
- resample players with replacement;
- each sampled player contributes all eligible target rows;
- 5,000 replicates;
- deterministic seed `20261007`;
- evaluate each of A/B/C Spearman statistics separately.

This preserves within-player dependence.

## Disposition

`TE_PLAYER_ERROR_PERSISTENCE_DETECTED`

requires:
- support gate passes;
- at least **2 of A/B/C** pass;
- at least one passing family is B or C.

`NO_ACTIONABLE_TE_PLAYER_ERROR_PERSISTENCE`

if support passes but the above signal gate fails.

`TE_PLAYER_ERROR_PERSISTENCE_SOURCE_LIMITED`

if support fails.

Why 2-of-3: signed bias, absolute difficulty, and extreme misses are different manifestations of player-specific reliability. One isolated positive family is insufficient to declare a general TE player-state signal.

## Interpretation boundaries

A positive result means only that TE-R5P leaves stable player-specific residual structure.

It does **not** tell us the correct intervention.

In particular:
- TE Width V2 remains closed;
- no mean correction is authorized;
- no width correction is authorized;
- no player-specific betting filter is authorized.

A detected signal would justify a separate mechanism audit asking **why** certain individual TEs remain systematically easier/harder to project after entitlement.

A null closes same-player TE-R5P error persistence as a useful player-state direction.

## Explicit anti-rescue

After results:
- no alternate 4/6/10-game windows;
- no min-3/min-5 history;
- no 20+/40+ miss threshold search;
- no TE1-only carveout;
- no high-target TE carveout;
- no 2023 exclusion;
- no signed-bias shrink;
- no direct width multiplier;
- no sportsbook conditioning.

Models fit: **0**  
Production mutations: **0**  
Sportsbook inputs: **0**  
2026 outcomes: **0**
