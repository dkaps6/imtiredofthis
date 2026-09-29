# RB-PD2 Yard-Difficulty Width — 2026 Week 3 Observation #1

Status: COMPLETE OBSERVATION — HOLD / INSUFFICIENT SUPPORT  
Date: 2026-09-29

Frozen prospective lock authority:
- original paid shadow run `36330757181`
- recovered immutable lock run `36331514633`
- recovered artifact `10935529140`
- recovered artifact digest `sha256:94f168125dc889b6a747da5f3b3e3829d2fd4bf7a9b6b3d9f4d2d79ac0b59b6e`

Postgame grade authority:
- run `36629683048` = SUCCESS
- artifact `11061697097`
- digest `sha256:1472ccd4290be2c3841265af68c68c5b5523d18eaef8ef11be8349233124ec1b`

Frozen plan:
`docs/research/RB_PD2_FORWARD_SHADOW_CONFIRMATION_V1_PLAN.md`

## Disposition

`NO_FORWARD_CONFIRMATION_INSUFFICIENT_SUPPORT`

This is **not** a scientific failure and **not** a confirmation.

Current support:
- prospectively locked weeks: 1 / 8 required
- unique eligible player-games: 46 / 400 required
- games: 15

No PASS/FAIL may be issued until both support floors are reached.

## Pooled Week-3 observation

Baseline:
- mean CRPS: 16.513152
- 80% coverage: 52.17%; absolute gap to nominal 27.83pp
- 90% coverage: 69.57%; absolute gap 20.43pp
- Brier >=50: 0.184883
- Brier >=75: 0.168414
- Brier >=100: 0.041522
- point MAE: 22.213108

Candidate:
- mean CRPS: 16.305948
- 80% coverage: 58.70%; absolute gap 21.30pp
- 90% coverage: 73.91%; absolute gap 16.09pp
- Brier >=50: 0.185754
- Brier >=75: 0.160870
- Brier >=100: 0.040266
- point MAE: 22.213108

Observed CRPS gain `baseline - candidate`: **+0.207204**.

Mean-neutrality remained exact:
- maximum rowwise mean gap: 1.42e-14
- pooled point-MAE difference: 0.0

## Dependence-aware observation

Game-cluster bootstrap, 10,000 reps, seed 42027:
- observed mean CRPS gain: +0.207204
- 95% CI: [+0.074031, +0.345209]

Crossed player × game bootstrap:
- P(candidate CRPS - baseline CRPS < 0): 0.9374

The eventual frozen crossed-robustness gate is >=0.95. This Week-3-only
observation is below that value, but because support is far below the frozen
minimum it must not be classified as a scientific non-confirmation.

## High-difficulty Q75 slice

Frozen Week-3 Q75 difficulty threshold: 0.771353  
Rows: 12

Baseline -> candidate:
- CRPS 22.672719 -> 21.979044
- 80% coverage gap 30.0pp -> 5.0pp
- 90% coverage gap 15.0pp -> 1.67pp
- Brier >=75 0.371252 -> 0.344992
- Brier >=100 0.078522 -> 0.074839

The high-difficulty observation is directionally favorable.

## Frozen guardrails — descriptive only at Week 3

Directionally satisfied this week:
- pooled 80% coverage-gap non-worse
- pooled 90% coverage-gap non-worse
- at least one pooled coverage gap strictly better
- high-Q75 CRPS strictly better
- high-Q75 80% and 90% gaps non-worse, with strict improvement
- Brier >=75 non-worse
- Brier >=100 strictly better

Not satisfied on this single observation week:
- Brier >=50 non-worse: candidate 0.185754 vs baseline 0.184883
- crossed player×game probability >=0.95: observed 0.9374

These are **observations, not final failed gates**, because the support floor has
not been reached.

## Boundary

- no target-game outcomes were present in the lock
- no sportsbook inputs were used in the candidate
- production remained unchanged
- no width cap/onset/window/reference/threshold was retuned
- no subgroup rescue is authorized

Continue the exact frozen forward shadow prospectively. Week 3 is Observation
Week #1 and must be accumulated unchanged with future qualifying locked weeks.
