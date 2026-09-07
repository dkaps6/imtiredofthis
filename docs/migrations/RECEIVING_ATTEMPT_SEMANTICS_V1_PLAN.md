# Receiving Attempt Semantics V1 — Frozen Plan

## Purpose

Test one isolated football-semantic issue already exposed by the canonical historical component code:

- `rules_pass_rate` is documented as **dropbacks / plays**;
- `pass_attempts_per_dropback` is separately derived leakage-safely for official pass attempts;
- current `simulation_v2` uses its dropback-count draw directly as the target-allocation count for receivers.

A sack/scramble dropback is not a receiver target opportunity. V1 tests whether converting dropbacks to official pass attempts **before receiver target allocation** materially improves receiver projections.

This lane is independent of Joint Pass/Receiving Conservation V1. It is deliberately not folded into that already-frozen 2x2 experiment.

## Lineage

- Branch: `research-receiving-attempt-semantics-v1`
- Parent SHA: `dc120eb186bf0f63c465de15fea2f02e5cf1c14f`
- Exact receiving production parent for historical replay: M38 `b98518d97b3038f471aee9ae3201009b2c70bb29`
- Existing code evidence: `component_predictions._attach_historical_passing_volume` explicitly labels `rules_pass_rate` as dropbacks/plays and computes `mc_pass_attempts_per_dropback`.
- Sportsbook inputs: **0**
- Production change: **0**

## Frozen variants

Exactly two:

- `B0_CURRENT`: exact current M38 simulation.
- `C4_ATTEMPT_CONVERTED`: exact current receiving inputs/shares/efficiency, but receiver target opportunities are generated from official pass attempts rather than dropbacks.

No other target-volume variant is authorized in this migration.

## Frozen historical scope

- 2020-2025 regular season.
- 2020 Weeks 1-17; 2021-2025 Weeks 1-18.
- Walk-forward pregame context only.
- Positions: WR/LWR/RWR/SWR, TE, RB, FB.
- 2,000 MC iterations, week seed `42 + week` for B0.
- C4 uses deterministic independent seed `400000 + 42 + week`.

## Frozen C4 mechanics

For each team/iteration:

1. Generate plays and **dropbacks** from the exact current `_team_inputs` / game-pace process.
2. Read leakage-safe `pass_attempts_per_dropback` from the historical team form using the same semantics as `component_predictions._attach_historical_passing_volume`.
3. Valid attempt-conversion range is `[0.50, 1.00]`; invalid/missing values use the existing fallback `1.0` and are counted.
4. Generate official attempts:
   `official_attempts ~ Binomial(dropbacks, pass_attempts_per_dropback)`.
5. Allocate receiver targets from `official_attempts` using the exact current target probabilities, including M38 within-WR sharpening and the existing residual bucket.
6. Generate receptions and receiving yards using the exact current B0 catch-rate / YPT / receiving-yard formula. This migration does **not** introduce the C2/C3 completed-pass yard-conservation formula.
7. Rushing opportunity is not recalculated in C4. This is a receiving-only isolation test; RB P3/current rushing mechanics remain untouched.

## Frozen outputs

Pooled + season-level:
- conversion-source coverage and distribution of `pass_attempts_per_dropback`;
- B0 dropbacks vs C4 official-attempt opportunity means;
- actual team pass attempts and team receiver targets where available;
- modeled WR/TE/RB+FB target-count MAE/RMSE/bias;
- player WR/TE/RB target MAE/RMSE/bias/correlation;
- player WR/TE/RB reception MAE/RMSE/bias/correlation;
- player WR/TE/RB receiving-yard MAE/RMSE/bias/correlation;
- all-receiver receiving-yard MAE parity for B0.

## Frozen integrity gates

Scientific interpretation stops if any fail:

1. B0 2025 all-receiver rec-yard cohort must reproduce `n=4647` and MAE within **0.05 yards** of `17.099904733366`.
2. Valid/fallback attempt-rate accounting must cover every evaluated team-game.
3. At least **95%** of evaluated team-games must have a valid historical pregame attempt-rate source rather than fallback 1.0.
4. No target-week outcome may enter C4.
5. Sportsbook inputs = 0.
6. RB rushing outputs are not modified/evaluated as a C4 candidate effect.

## Frozen scientific gates

`ATTEMPT_SEMANTICS_CANDIDATE_PASS` requires all:

1. pooled team modeled-receiver target-count MAE improves by **>= 0.50 targets** vs B0;
2. pooled macro-average WR/TE/RB player target MAE improves by **>= 0.03 targets**;
3. no WR/TE/RB pooled player target MAE worsens by more than **0.03 targets**;
4. no WR/TE/RB pooled reception MAE worsens by more than **0.03 receptions**;
5. no WR/TE/RB pooled receiving-yard MAE worsens by more than **0.50 yards**;
6. pooled macro-average WR/TE/RB receiving-yard MAE is lower than B0;
7. macro receiving-yard MAE is lower than B0 in **>= 4 of 6 seasons**;
8. pooled 2024-2025 macro receiving-yard MAE is lower than B0;
9. no single season/position receiving-yard MAE regresses by more than **1.50 yards**.

Otherwise: `ATTEMPT_SEMANTICS_CANDIDATE_FAIL`.

Passing this gate authorizes a separate integration test with the joint receiving architecture. It does not directly change production.

## Stopping rule

No alternate attempt-rate window, range, fallback, deterministic conversion, target pool, or gate may be added after results are visible. Mechanical repairs may only restore the exact frozen semantics above.