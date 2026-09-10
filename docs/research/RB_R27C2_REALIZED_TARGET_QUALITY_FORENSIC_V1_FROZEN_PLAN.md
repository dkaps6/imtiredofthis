# RB R27C2 — Realized Target-Quality Forensic V1 Frozen Plan

Status: `FROZEN BEFORE TARGET-GAME PBP FORENSIC EXECUTION / DIAGNOSTIC ONLY / NO CANDIDATE`

## Parent authority

- R27C result commit: `b2907345c50dceffdfb5c074ce9d556ff6d764c6`
- R27C canonical run: `34430754033`
- R27C job: `102725570088`
- R27C artifact: `10134387002`
- R27C artifact digest: `sha256:0c915a1c413d66c550990a1cc8bd45e0dc20a18e110fe1b96414a79eb7b598e9`
- R27B V2 parent run: `34428917229`
- R27B V2 artifact: `10134023092`
- R27B V2 digest: `sha256:1df1a14c9900b890fe86e04269849dc90d76fc03a9215ad8612c1820dee5341f`
- Protected production-code authority: `bb76ba9eabb08e2f0875a9af49301c3877f4141f`

## Purpose

R27C established that the unresolved 2023 vacancy-RB1 failure is a realized yards-per-target problem rather than an obvious catch-rate problem, while the V2 30+ miss-rate regression is a separate small set of explosive RB1 outcomes. R27C did not identify the physical target-quality component responsible.

R27C2 will use target-game play-by-play only as retrospective labels to decompose realized RB receiving value into observable football components. It will not create or score any new pregame prediction candidate.

## Primary questions

1. Is the 2023 vacancy-RB1 YPT shortfall primarily associated with shallower actual target depth, lower YAC per reception, lower explosive-play rate, altered screen/behind-LOS usage, or a combination?
2. Does 2023 vacancy-RB1 actual target quality differ materially from non-2023 vacancy-RB1 even though the prior context features looked similar?
3. Do the few V2-created 30+ misses represent a qualitatively different target-quality state from ordinary vacancy RB1 games?
4. Did the target-game target shape move away from the strict-prior player/team target-shape features in a way that explains why V2 failed to foresee the realized efficiency?
5. Is there one physical mechanism strong enough to motivate a separately frozen future predictive study, or is the residual mostly irreducible/explosive variance?

## Frozen cohorts

Using exact R27B/V2 preserved prediction rows:
- `VACANCY_RB1_INCUMBENT`
- `VACANCY_RB2PLUS_INCUMBENT`
- `2023_VACANCY_RB1_INCUMBENT`
- `NON2023_VACANCY_RB1_INCUMBENT`
- `2023_VACANCY_ACTIVE`
- `NON2023_VACANCY_ACTIVE`

Tail cohorts are inherited exactly from R27C definitions:
- `V2_INTO_30PLUS_VACANCY_RB1`: B1 AE <30 and C1 AE >=30
- `V2_OUT_OF_30PLUS_VACANCY_RB1`: B1 AE >=30 and C1 AE <30
- `V2_BOTH_30PLUS_VACANCY_RB1`: B1 AE >=30 and C1 AE >=30

No player-specific anecdote may define a future mechanism.

## Realized target-game PBP fields

For each RB player-game, reconstruct from regular-season target rows:
- PBP target count
- PBP receptions / completed targets
- PBP receiving yards
- actual catch rate
- air yards per target
- YAC per reception, completed targets only
- screen/behind-LOS target rate, frozen definition `air_yards <= 0`
- explosive-20 target rate, frozen definition receiving `yards_gained >= 20`
- yards per reception when receptions >0
- yards per target when targets >0
- share of receiving yards coming from explosive-20 target plays
- maximum receiving gain

## Decomposition comparisons

For each frozen cohort report n, mean, median, p25, p75 for:
- actual targets
- actual receptions
- actual receiving yards
- actual catch rate
- actual air yards/target
- actual YAC/reception
- actual screen rate
- actual explosive20 target rate
- actual YPR
- actual YPT
- explosive20 share of receiving yards
- max receiving gain

For `2023_VACANCY_RB1_INCUMBENT` versus `NON2023_VACANCY_RB1_INCUMBENT`, report absolute differences for every metric.

## Prior-to-realized shape drift

For rows with available strict-prior V2 features, compare target-game realized values to the matching frozen prior features:
- actual air yards/target minus `player_air_yards_per_target_prior`
- actual YAC/reception minus `player_yac_per_reception_prior`
- actual screen rate minus `player_screen_target_rate_prior`
- actual explosive20 target rate minus `player_explosive20_target_rate_prior`

Report these drifts for 2023 RB1 and non-2023 RB1. This is retrospective diagnostic evidence only.

## Arithmetic decomposition

Use the identity:

`YPT = catch_rate × YPR`

and report cohort-level actual catch rate and actual YPR alongside production catch rate and production implied YPR (`production_ypt / production_catch_rate` where valid).

Also report:
- actual YPT minus production YPT
- actual catch rate minus production catch rate
- actual YPR minus production implied YPR

This is descriptive decomposition, not a fitted attribution model.

## Explosive-tail forensic

For inherited RB1 30+ crossing cohorts report the same target-quality metrics. In particular test descriptively whether `V2_INTO_30PLUS_VACANCY_RB1` has unusually high actual explosive20 rate, max gain, YAC/reception, or air yards/target relative to the full vacancy-RB1 cohort.

The sample is expected to be small. No numeric threshold, router or future candidate may be chosen from these tail rows.

## Source / identity rules

- Use NFL regular-season PBP for 2020–2025 only.
- Use the existing repository canonical player-name normalization and weekly-position resolution pattern already source-audited in R27B V2.
- Restrict receiver positions to RB/FB/HB/TB.
- Join player-game rows to exact preserved V2 evidence using season, week, team and canonical player key.
- Report join coverage and PBP-vs-preserved target/reception/yard gaps.
- Fail mechanically if identity coverage is insufficient to support cohort comparison.

## Integrity rules

- Exact R27B V2 artifact ID/digest must be verified.
- Exact R27C artifact ID/digest must be verified for inherited forensic evidence.
- No new model fit.
- No hyperparameter tuning.
- No new pregame candidate.
- No sportsbook input.
- Target-game PBP is allowed only as postgame diagnostic labels.
- No production change.
- No R26 change.
- No R22 change.

## Allowed terminal labels

- `R27C2_FORENSIC_COMPLETE_PHYSICAL_TARGET_QUALITY_HYPOTHESIS_IDENTIFIED`
- `R27C2_FORENSIC_COMPLETE_NO_SINGLE_PHYSICAL_MECHANISM_IDENTIFIED`
- `R27C2_MECHANICAL_OR_IDENTITY_FAILURE_NO_FORENSIC_CONCLUSION`

There is no production PASS state.

## Next-step boundary

Only after R27C2 is complete may a new predictive study be designed. Any future candidate must use strictly pregame reconstructable variables, be frozen before scoring, and must not use target-game PBP features directly.
