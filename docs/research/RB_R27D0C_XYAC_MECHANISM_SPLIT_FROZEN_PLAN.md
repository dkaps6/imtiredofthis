# RB R27D0C — xYAC Mechanism Split Frozen Plan

Status: `FROZEN BEFORE TARGET-GAME XYAC DIAGNOSTIC / NO MODEL / NO CANDIDATE`

## Parent authority

- R27D0B source-extension result commit: `4e19b549dad2141ba3d55b77c563947342426461`
- R27D0 source-audit result commit: `8663b132c9910d07679d7b5e03ee8a8c9542f332`
- R27C2 result commit: `2cfca82d919290ac18ed01559994d2e0239b0798`
- Exact R27B V2 artifact: `10134023092`
- Exact R27B V2 artifact digest: `sha256:1df1a14c9900b890fe86e04269849dc90d76fc03a9215ad8612c1820dee5341f`
- Protected production-code authority: `bb76ba9eabb08e2f0875a9af49301c3877f4141f`

## Question

R27C2 showed that 2023 vacancy-RB1 receiving value compressed after the catch. Before fitting a predictive model, determine whether that physical YAC compression arose primarily because:

1. completed RB catches had lower **expected YAC** given their play/catch context; or
2. RB/offense/defense execution produced lower **YAC over expected (YACOE)** than the catch context implied; or
3. both mechanisms contributed.

This is retrospective decomposition only. Target-game PBP is outcome evidence and may never be used as a candidate input.

## Frozen calculations

For each completed RB catch with non-null `xyac_mean_yardage`:

- `expected_yac = xyac_mean_yardage`
- `yacoe = yards_after_catch - xyac_mean_yardage`

Aggregate to player-game using exact `(season, week, team, player_clean_key)` identity.

For each player-game report:
- PBP targets
- PBP receptions
- xYAC-observed receptions
- actual YAC/reception on xYAC-observed catches
- expected YAC/reception
- YACOE/reception
- actual receiving yards/reception and yards/target from preserved parent evidence

## Frozen cohorts

Primary:
- `2023_VACANCY_RB1_INCUMBENT`
- `NON2023_VACANCY_RB1_INCUMBENT`

Reference:
- `VACANCY_RB1_INCUMBENT`
- `VACANCY_RB2PLUS_INCUMBENT`
- `VACANCY_ACTIVE`

Tail reference from immutable V2 evidence:
- RB1 rows where B1 AE < 30 and C1 AE >= 30 (`RB1_INTO_30PLUS`)
- RB1 rows where B1 AE >= 30 and C1 AE >= 30 (`RB1_BOTH_30PLUS`)

No player-level anecdote may determine the conclusion.

## Frozen decomposition

For 2023 RB1 versus non-2023 RB1, report:

- difference in actual YAC/reception
- difference in expected YAC/reception
- difference in YACOE/reception
- fraction of the signed YAC/reception difference attributable to expected-YAC difference and to YACOE difference where algebraically defined

Also report mean/median/p25/p75 of player-game expected YAC/reception and YACOE/reception to determine whether the difference is broad or tail-driven.

For tail-reference cohorts report expected YAC/reception and YACOE/reception to determine whether extreme right-tail games were expected-context rich, execution-over-expected, or both. This may inform architecture but cannot alter R22.

## Integrity requirements

- exact R27B V2 artifact ID/digest must be verified
- regular-season target-game PBP only
- >=98% xYAC-observed reception coverage in each primary cohort
- no model fit
- no candidate projection
- no threshold search
- no sportsbook input
- no production change
- no R26 change
- no R22 change

## Allowed disposition

- `R27D0C_XYAC_MECHANISM_SPLIT_COMPLETE`
- `R27D0C_MECHANICAL_OR_COVERAGE_FAILURE_NO_CONCLUSION`

The result may identify which physical component should motivate the separately frozen R27D predictive feature family. It cannot authorize production or candidate integration.
