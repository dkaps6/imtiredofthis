# PRISM Opportunity Quality — Parent Denominator Persistence V1 Addendum

Frozen: 2026-10-16

Status: **MECHANICAL SOURCE-PERSISTENCE ADDENDUM / FROZEN BEFORE CODE CHANGE / NO PREDICTIVE SCORING**

Scientific parent:
`PRISM_OPPORTUNITY_QUALITY_SOURCE_READY_ALL_RECONCILED`

Canonical source result:
`docs/research/PRISM_OPPORTUNITY_QUALITY_V1_RECONCILED_SOURCE_RESULT.md`

Stage-2 source parent:
- run `38955368643` — SUCCESS;
- exact head `cb65a634a0e09296bfeabc94ffe1e77363a30f8a`;
- artifact `12070806828`;
- digest `sha256:e91a108495d7a2b6eb289e0618ab6af09e3d6cdf6895e534742a07340e4a2a83`;
- disposition `PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_SOURCE_READY_PARTIAL`.

## Why this addendum exists

The qualified Stage-2 source evaluator already reconstructs the numeric canonical
pregame opportunity denominator:

RB rushing:
`rush_denominator_mean = rules_plays_est * (1 - rules_pass_rate)`.

Receiving:
`target_denominator_mean = rules_plays_est * rules_pass_rate`.

The source audit proved these denominators finite under frozen gates, but the
published player-level join artifact retained only:

`denominator_finite`

instead of the numeric value itself.

PRISM opportunity quality should remain physically attached to the existing
canonical pregame opportunity quantity.

Therefore the numeric value must be persisted before a quality mechanism is
designed/scored.

## Frozen mechanical change

The Stage-2 source evaluator may add exactly one player-level field:

`authority_denominator_mean`

Semantics by cell:
- `RB_RUSHING` -> exact existing `rush_denominator_mean`;
- `RB_RECEIVING` -> exact existing `target_denominator_mean`;
- `TE_RECEIVING` -> exact existing source-qualified target denominator;
- `QB_PASSING` may persist its existing pass denominator for schema parity;
- `WR_RECEIVING` may persist only where its existing source authority is legal;
  2025 WR remains source-blocked and may not be rescued.

No denominator is recomputed from a target-game outcome.

No denominator formula changes.

No entitlement changes.

No source gate changes.

## Required parity

The refreshed source run must reproduce the prior canonical result exactly in
scientific disposition:

`PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_SOURCE_READY_PARTIAL`

with the same ready/blocked cells:
- QB passing ready;
- RB rushing ready;
- RB receiving ready;
- TE receiving ready;
- WR receiving blocked because 2025 independent OOS authority is unavailable.

For every authority-covered row:
- `authority_denominator_mean` finite iff prior `denominator_finite == true`;
- denominator > 0 where finite;
- no duplicate player-game identity introduced;
- entitlement `authority_q` unchanged versus the prior canonical artifact on
  the exact keyed intersection.

Required max absolute entitlement drift:
`<= 1e-12`.

## Output

The refreshed immutable source artifact must retain:
- all existing source outputs;
- `authority_denominator_mean` in
  `active_role_conditional_allocation_source_join_detail.csv`;
- a parity report against canonical artifact `12070806828`.

The refreshed run is an output-lineage authority only.

It does not supersede the scientific source result unless every parity check
passes.

## Integrity boundary

At this freeze:
- predictive candidates scored: 0;
- target-game outcomes used as predictors: 0;
- denominator formulas changed: 0;
- entitlement formulas changed: 0;
- source gates changed: 0;
- sportsbook inputs: 0;
- OddsAPI calls: 0;
- 2026 outcomes: 0;
- production changed: false.
