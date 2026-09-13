# RB-PD2 Multi-Season Current-Production-Route Replication V1 — Frozen Plan

**STATUS: PRE-EXECUTION AMENDMENT FROZEN BEFORE ANY 2021-2024 PERSISTENCE RESULT. RESEARCH ONLY. NO PRODUCTION CHANGE.**

## Amendment record

The initial preregistration at commit `241841ec685b7cbde16a34681e77ae40bca79932` was **not executed** and produced **no 2021-2024 persistence result**. Before implementation, two architecture facts forced a correction:

1. canonical STACK2's 50/50 allocation anchor is **M94C raw opportunity share**, not `stack_share`; the initial text misstated this;
2. more importantly, the production ledger `RB_P3_WEEK1_PROMOTION_2026_09_05.md` explicitly promotes P3 only for **Week 1**. Weeks 2-18 enriched-allocation/P3 is **not promoted** and production remains on the base calibrated ensemble.

Backporting the unpromoted Weeks2-18 STACK2/P3 route would therefore be less representative of the actual season-long production system. This amendment replaces that proposed backport before any scientific result is observed.

The literal PD5/PD6 memo requirement for a multi-season "P3-equivalent" panel remains a **separate unresolved PD6 blocker**. This experiment does not claim to satisfy it. Its narrower purpose is to determine whether PD2's untouched uncertainty-width authorization survives on the **current production-equivalent RB rushing mean route**.

## Purpose

RB-PD2 detected player-specific carry/yard error persistence on the already-observed 2025 STACK1 cohort and authorized three follow-ups, including an untouched difficulty -> MC-width/uncertainty lane.

This replication asks:

> Do the original PD2 persistence signals reproduce on non-2025 seasons when RB carries/rushing yards use the same temporal calibrated full-stack ensemble that underlies current production outside the Week-1-only P3 override?

A PASS can authorize only a separately frozen PD2 uncertainty-width experiment. It does not authorize PD6, a mean correction, or a production change.

## Frozen target seasons

Targets: **2021, 2022, 2023, 2024**.

Reason, fixed before results:

- M95Q run `33450395426` provides parity-checked M91 temporal component reconstructions for 2020-2024;
- every target season can therefore use an immediately prior full season to fit ensemble weights;
- 2025 is excluded because it is the already-observed PD2/PD3/PD4/PD5 cohort;
- 2020 is excluded because using it would require a 2019 fit season outside the M95Q comparability panel used for this replication.

No 2021-2024 persistence statistic has been inspected before this amended plan is frozen.

## Canonical historical evidence

Use only M95Q successful run `33450395426`:

- artifacts `m95q-m91-2020` through `m95q-m91-2024`;
- exact `component_predictions.csv` per season;
- final M95Q artifact `9779790912` with `M95Q_EXPANDED_PANEL_READY`;
- M91 2024 universe parity PASS.

No sportsbook data enters this experiment.

## Production-route-equivalent mean reconstruction

For each target season `S` in 2021-2024:

1. read exact M91 component rows for `S-1` and `S`;
2. fit canonical `fit_market_weights()` on **S-1 only**;
3. freeze those weights and call `apply_ensemble()` on S;
4. retain RB/HB/FB `rush_att` and `rush_yards` rows;
5. pivot one row per `(season, week, team, player)` with:
   - `pred_carry = ensemble_proj` for `rush_att`;
   - `pred_yard = ensemble_proj` for `rush_yards`;
   - corresponding target-season actuals from the same timestamp-safe M91 trace.

This is the exact temporal analogue of STACK1's `ensemble_2024_frozen` construction used by original PD2.

### Current-production relationship

- Week 1 P3's promoted `WEEK1_STACK_OVERRIDE` equals the calibrated STACK1 rushing-yard projection unchanged.
- Weeks 2-18 currently do **not** use the unresolved enriched P3 route; the production ledger keeps the base calibrated ensemble.

Therefore the frozen-ensemble historical reconstruction is the appropriate season-long parent for testing whether PD2's error-persistence mechanism generalizes to the currently authorized production mean architecture.

## Integrity gates

All must pass before scientific interpretation:

1. target seasons exactly `{2021, 2022, 2023, 2024}`;
2. 2025 rows = 0;
3. target-season ensemble weights use only `S-1` component rows;
4. required `mc_proj`, `ml_proj`, `state_proj`, `actual`, market, player/team/week identity present;
5. final RB identity unique by `(season, week, team, player_key)`;
6. exactly one `rush_att` and one `rush_yards` source row per final player-game;
7. M95Q source parity status remains PASS;
8. sportsbook inputs used = 0;
9. production changed = false.

Failure => `RB_PD2_MULTISEASON_INTEGRITY_FAILURE`; no scientific disposition.

## Frozen history construction

Reuse original PD2 mechanics:

- same player only;
- chronological completed prior games only;
- last **8** eligible games;
- minimum **4** prior games;
- history may cross season boundaries within the 2021-2024 reconstructed panel, but never includes same/future games;
- carry error = predicted carries - actual carries;
- yard error = predicted rushing yards - actual rushing yards;
- prior8 bias/MAE computed exactly as original PD2.

## Original PD2 gates retained

### A. Carry directional persistence

Pooled PASS iff all:
- scoreable rows >=700;
- Spearman(prior carry bias, target carry error) >= +0.08;
- high-low quartile target carry-error gap >= +1.0 carry;
- sign agreement >=55% where |prior carry bias|>=0.5;
- Weeks5-12 gap >0;
- Weeks13-18 gap >0.

### B. Carry difficulty persistence

Pooled PASS iff all:
- scoreable rows >=700;
- Spearman(prior carry MAE, target absolute carry error) >= +0.08;
- high-low absolute-error gap >= +0.75 carry;
- Weeks5-12 gap >0;
- Weeks13-18 gap >0.

### C. Yard directional persistence

Pooled PASS iff all:
- scoreable rows >=700;
- Spearman(prior yard bias, target yard error) >= +0.08;
- high-low target yard-error gap >= +6 yards;
- sign agreement >=55% where |prior yard bias|>=3 yards;
- Weeks5-12 gap >0;
- Weeks13-18 gap >0.

### D. Yard difficulty persistence

Pooled PASS iff all:
- scoreable rows >=700;
- Spearman(prior yard MAE, target absolute yard error) >= +0.08;
- high-low target absolute-error gap >= +5 yards;
- Weeks5-12 gap >0;
- Weeks13-18 gap >0.

## Multi-season confirmation guard

A diagnostic is `REPLICATED` only if:

- its original pooled gate passes; and
- both Spearman and quartile-gap direction are positive in at least **3 of 4** target seasons; and
- neither 2023 nor 2024 has both Spearman <=0 and quartile gap <=0.

This can only make the original gate stricter; it cannot rescue a pooled failure.

## Dispositions

- integrity failure: `RB_PD2_MULTISEASON_INTEGRITY_FAILURE`;
- no replicated diagnostics: `NO_MULTISEASON_RB_PLAYER_ERROR_PERSISTENCE`;
- >=1 replicated diagnostic: `MULTISEASON_RB_PLAYER_ERROR_PERSISTENCE_REPRODUCED`.

Width-lane authorization is narrower:

- carry uncertainty-width work unlocks only if **B** replicates;
- yard uncertainty-width work unlocks only if **D** replicates;
- A/C do not authorize width changes;
- any unlocked width experiment requires a new frozen plan before implementation.

## Explicit non-claims / forbidden actions

- this does **not** resolve the PD5/PD6 memo's literal multi-season P3-equivalent blocker;
- no unpromoted Weeks2-18 P3 backport;
- no M94C/STACK2 substitute invented after results;
- no sportsbook features;
- no 2025 confirmation rows;
- no gate/window/feature/model tuning after results;
- no production change.
