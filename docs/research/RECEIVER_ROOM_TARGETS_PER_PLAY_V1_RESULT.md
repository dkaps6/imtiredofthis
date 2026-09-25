# Receiver Room Targets-Per-Play V1 — 2022-2023 Temporal Screen Result

Date: 2026-09-25

Disposition: **RECEIVER_ROOM_TARGETS_PER_PLAY_V1_SUPPORTED**

This is a historical temporal qualification. It does not authorize production.

## Authority

- branch: `research-receiver-room-targets-per-play-v1`
- run: `36172644864`
- job: `108195660546`
- tested head: `40d1d9c716243eb1e78e3597823629128afdcf3b`
- artifact: `10880948137`
- digest: `sha256:961fcaa805e13818ade3473ee87799558c97c17317fab0913b2d893fd04835ff`
- discovery seasons: 2022-2023
- parameters fit: 0
- candidate variants: 1
- sportsbook inputs: 0
- target-game outcomes upstream: 0

## Exact frozen candidate

For room g in WR / TE / RB_FB:

`R_g_play = sum(strict-prior room targets) / sum(strict-prior offensive plays)`

`candidate room targets = projected offensive plays * R_g_play`

This does not change production pass rate, QB opportunity, rushing opportunity,
or the fixed 57% canonical pass/dropback state. It is a receiver-only
opportunity forecast.

## 2022

- WR MAE: `4.761250 -> 4.498155`
- WR p90: `9.519272 -> 9.023064`
- TE MAE: `2.828880 -> 2.594462`
- TE p90: `5.479486 -> 5.083496`
- RB_FB MAE: `2.730621 -> 2.535164`
- RB_FB p90: `5.589022 -> 5.155948`
- macro MAE: `3.440250 -> 3.209260`
- macro abs bias: `0.526882 -> 0.112578`

## 2023

- WR MAE: `4.538430 -> 4.467731`
- WR p90: `9.506628 -> 8.966222`
- TE MAE: `2.800682 -> 2.587555`
- TE p90: `5.553391 -> 5.280620`
- RB_FB MAE: `2.368523 -> 2.326150`
- RB_FB p90: `4.855590 -> 4.950083`
- macro MAE: `3.235879 -> 3.127145`
- macro abs bias: `0.495015 -> 0.392965`

## Pooled room result

- WR MAE: `4.649635 -> 4.482915`
- WR p90: `9.517370 -> 9.017551`
- TE MAE: `2.814755 -> 2.591002`
- TE p90: `5.493863 -> 5.173002`
- RB_FB MAE: `2.549238 -> 2.430465`
- RB_FB p90: `5.197636 -> 5.112040`
- macro MAE: `3.337876 -> 3.168127`
- macro p90: `6.736290 -> 6.434198`
- macro abs bias: `0.323767 -> 0.253029`
- all-room-row candidate closer rate: `54.0209%`

## Summed-room result

- MAE: `6.213005 -> 5.974676`
- absolute bias: `0.899338 -> 0.759088`
- p90: `12.609038 -> 12.322572`
- candidate closer rate: `53.4070%`

## Frozen gates

All 23 frozen gates passed, including:
- macro MAE improvement in both seasons;
- WR MAE improvement in both seasons;
- TE/RB_FB pooled nonworse;
- room p90 guards;
- pooled macro p90 improvement;
- pooled macro absolute-bias improvement;
- summed-room MAE/bias/p90 improvement;
- candidate closer >50%;
- zero fallback rows;
- finite bounded rates;
- zero sportsbook;
- zero target-game outcomes upstream;
- parameters fit = 0;
- candidate variants = 1.

## Parallel state diagnostics considered

Two independently run discovery diagnostics were reconciled before advancing:

### WR1 Current-State Anchor Diagnostic V1

- strong evidence that WR1 absolute/team target share is under-anchored;
- blend-4 state improved WR1 absolute share;
- but state-normalized WR1 within fixed WR-room composition worsened;
- WR1-only candidate was therefore not justified.

### Active-Roster Receiver Room State V1

Disposition:
`ACTIVE_ROSTER_RECEIVER_ROOM_STATE_V1_NOT_SUPPORTED`

It slightly improved macro room composition and TE/RB_FB but failed WR room
composition in both discovery seasons. Therefore active-roster room
redistribution is not an authorized competing solution.

These diagnostics strengthen the interpretation that the new gain comes from
the receiver-opportunity denominator itself, not from a post-hoc WR carveout or
room-composition rescue.

## Interpretation

The previous room-targetable/dropback formulation failed because a useful room
targets/dropback rate was multiplied by a negatively biased fixed-57%
dropback forecast.

Direct room targets per offensive play removes that two-stage denominator error
without changing the broader pass/run model.

This is the first room-opportunity formulation in this sequence to improve:
- WR;
- TE;
- RB_FB;
- macro MAE;
- macro tails;
- macro bias;
- summed-room MAE;
- summed-room bias;
- summed-room tails

under one frozen parameter-free formula.

## Next step

Freeze an unchanged 2024-2025 confirmation before inspecting those room outcomes.

Confirmation baseline must use authorized specialist order:
- 2024: TE-R5P + WR-R15;
- 2025: TE-R5P only;
- WR-R15 retrospective application in 2025 remains forbidden.

Candidate formula, history horizon, fallback and room definitions remain
unchanged.

No player-level integration or production change is authorized until that
confirmation passes.
