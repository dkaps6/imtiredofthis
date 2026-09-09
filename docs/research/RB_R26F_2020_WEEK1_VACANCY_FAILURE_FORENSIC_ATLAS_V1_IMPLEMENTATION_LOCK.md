# RB R26F — 2020 Week 1 Vacancy Failure Forensic Atlas V1 — Implementation Lock

Status: LOCKED BEFORE NEW SUBGROUP OUTCOME SLICING
Date: 2026-09-09

This lock resolves mechanical details left qualitative in the frozen plan. It does not change any forensic dimension and was written before executing the R26F subgroup outcome atlas.

## Source-only quantile construction

For Week-1 vacancy incumbents pooled across 2020-2025, calculate quantile bins from prediction/source state only:

- `abs_target_move = abs(candidate_targets - baseline_targets)`
- `abs_reception_move = abs(candidate_receptions - baseline_receptions)`
- `abs_r9_residual = abs(r9_calibrated_residual)`

Use `pandas.qcut(..., q=4, duplicates='drop')` on finite values. Quantile labels are Q1-low through Q4-high. These bins are created without using target-game outcomes.

## Direction groups

Using a numerical tolerance of `1e-12`:
- INCREASE if candidate - baseline > tolerance
- DECREASE if candidate - baseline < -tolerance
- UNCHANGED otherwise.

## Room-order flip

Within each `(season, week=1, team)` vacancy-active room:
- baseline leader = player with highest `baseline_targets`, deterministic tie break by `player_clean_key`;
- candidate leader = player with highest `candidate_targets`, same tie break;
- `leader_flip = 1` when identities differ.

This is prediction-state only and uses no target-game outcome.

## Forensic state table

Evaluate the following dimension/level states exactly:

- role: `RB1`, `RB2+`
- exits: `ONE_EXIT`, `TWO_PLUS_EXITS`
- entrants: `NO_ENTRANT`, `ONE_PLUS_ENTRANT`
- exit-vs-entrant balance: `EXITS_GT_ENTRANTS`, `EXITS_EQ_ENTRANTS`, `EXITS_LT_ENTRANTS`
- prior depth: `AVAILABLE`, `UNAVAILABLE`
- target movement quartile: source-only Q1-Q4
- reception movement quartile: source-only Q1-Q4
- target direction: `INCREASE`, `DECREASE`, `UNCHANGED`
- reception direction: `INCREASE`, `DECREASE`, `UNCHANGED`
- R9 residual sign: `POSITIVE`, `NEGATIVE`, `ZERO`
- R9 residual magnitude quartile: source-only Q1-Q4
- room target-leader flip: `FLIP`, `NO_FLIP`

No additional dimension may be added after outcomes are inspected in this R26F version.

## Mechanical harmful-state rule

For every dimension/level state, calculate reception `effect_delta_abs_error = candidate_abs_error - baseline_abs_error` for labeled Week-1 vacancy incumbents.

A state is `2020_MATERIALLY_HARMFUL` only if all hold:

1. 2020 support `n >= 10` labeled incumbent player-games;
2. 2020 mean effect delta > 0;
3. 2020 candidate receptions MAE worsens by more than 2% versus baseline;
4. the state's 2020 **net positive absolute-error delta sum** is at least 20% of the total 2020 vacancy-incumbent net positive absolute-error delta sum.

The contribution denominator is the total signed sum of candidate AE minus baseline AE across all labeled 2020 Week-1 vacancy incumbents; because 2020 is known from R26E to worsen overall, this denominator must be positive.

## Cross-season replication rule

A `2020_MATERIALLY_HARMFUL` state is `REPLICATED_HARMFUL_STATE` only if:

- the same dimension/level has mean reception effect delta > 0 in at least **2 of 5** seasons from 2021-2025;
- across those supporting harmful seasons, combined labeled support is at least **15** player-games.

No magnitude threshold is required outside 2020; direction is the predeclared replication requirement.

## Room-allocation diagnosis

For 2020 Week-1 vacancy team-weeks, compare:
- change in room-total reception absolute error;
- change in summed player reception absolute error.

Classify `WITHIN_ROOM_ALLOCATION_DOMINANT` if:
- summed player AE worsens overall;
- and room-total AE change is <= 25% of the summed-player AE worsening, or room-total AE improves.

Classify `ROOM_TOTAL_AND_ALLOCATION_BOTH_HARMFUL` if both worsen and room-total AE change exceeds 25% of player AE worsening.

Otherwise classify `ROOM_DIAGNOSIS_MIXED`.

## Final disposition

- `WEEK1_FAILURE_MECHANISM_IDENTIFIED_REPLICATED` if at least one state is `REPLICATED_HARMFUL_STATE`.
- `WEEK1_2020_FAILURE_LOCALIZED_NOT_REPLICATED` if at least one state is `2020_MATERIALLY_HARMFUL` but none replicate.
- `WEEK1_FAILURE_MECHANISM_UNRESOLVED` if no state is materially harmful under the frozen rule.
- `WEEK1_FORENSIC_INTEGRITY_FAILURE` if immutable evidence or required fields fail.

R26F remains diagnostic only. No disposition authorizes shadow or production.
