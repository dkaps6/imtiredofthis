# PRISM Player High-Value Conversion Identity V1 — Predictive Mechanism Freeze

Date: 2026-10-17

Status: **FROZEN BEFORE SCORING / RESEARCH ONLY / NO ATD CERTIFICATION**

Source authority:
- run `39093918886` — SUCCESS;
- exact head `5b517c1ac71a1f67d24cddf510c0f7ef5fe191b9`;
- artifact `12123727812`;
- source disposition `PRISM_PLAYER_HIGH_VALUE_CONVERSION_IDENTITY_SOURCE_READY_PARTIAL`.

Eligible cells only:
- `RB_RUSH_INSIDE10`;
- `TE_RECEIVING_REDZONE`.

`RB_RECEIVING_REDZONE` is source-blocked and may not be scored.

## Exact scientific question

> Conditional on the named player actually receiving the frozen scoring-area
> opportunity type, does his own strictly-prior TD-conversion history improve
> conversion probability over a strictly-prior channel-wide baseline?

This isolates **individual conversion efficiency** from:
- opportunity quantity;
- teammate entitlement;
- team scoring-area creation;
- opponent/environment;
- sportsbook pricing.

Target-game high-value opportunities are used only to condition/grade
conversion. They are not a pregame opportunity predictor in this experiment.

## Chronology

- 2022-2023: prior-history seed only;
- 2024: development;
- 2025: independent confirmation;
- no 2026 outcomes.

All prior statistics stop before the target week. The entire target game is
excluded from its own prior state.

## Frozen baseline

For channel `c` and target season/week `t`:

`p_base = (prior_channel_TD + 0.5) / (prior_channel_HV_opps + 1)`

where the prior channel totals include only completed regular-season weeks
strictly before target week, across the historical 2022+ source window.

This is a Jeffreys-smoothed strictly-prior channel baseline.

## Frozen named-player candidate

Support:
- same stable player identity across teams;
- >=3 strictly-prior high-value opportunities in the same frozen channel.

Fixed shrinkage strength:
**4 channel-baseline pseudo-opportunities**.

`p_player = (prior_player_TD + 4 * p_base) / (prior_player_HV_opps + 4)`

No fitting or search chooses the 4. It is fixed before outcomes are scored.

No recency window, same-team carveout, role subset or threshold search is
allowed after scoring.

## Frozen outcomes

### Primary — event conversion

Each frozen high-value opportunity is binary:
- 1 = native TD conversion;
- 0 = no TD conversion.

Primary metric:
**Brier improvement = baseline Brier - candidate Brier**.

Positive means the named-player conversion identity improves probability.

### Secondary — conditional player-game anytime conversion

For a target player-game with observed high-value opportunity count `H`:

`p_any = 1 - (1 - p)^H`.

Outcome:
- 1 if the player converted at least one of those frozen high-value
  opportunities into a TD;
- 0 otherwise.

This is **conditional conversion grading only**. It is not a pregame ATD model
because `H` is the realized target-game high-value opportunity count.

Secondary metric:
player-game anytime-conversion Brier improvement.

Log loss is reported as a protection metric at both levels.

## Frozen inference

2025 independent confirmation uses 5,000 deterministic bootstrap replicates
(seed 20261017):

Primary event-level Brier improvement:
- named-player cluster CI;
- target-game cluster CI;
- equal-player mean improvement + CI.

Secondary conditional player-game Brier improvement:
- named-player cluster CI;
- target-game cluster CI.

Report:
- per-player mean improvement;
- median supported-player improvement;
- share of supported players improved.

## Frozen qualification gates — per cell

A cell qualifies only if **all** are true:

2024 development:
1. event Brier improvement > 0;
2. conditional player-game Brier improvement > 0.

2025 confirmation:
3. event Brier improvement > 0;
4. event log-loss improvement >= 0;
5. conditional player-game Brier improvement > 0;
6. conditional player-game log-loss improvement >= 0;
7. event player-cluster CI lower bound > 0;
8. event game-cluster CI lower bound > 0;
9. event equal-player CI lower bound > 0;
10. conditional-game player-cluster CI lower bound > 0;
11. conditional-game game-cluster CI lower bound > 0.

No pooled-season result can rescue a failed 2025 gate.

## Anti-rescue

If a cell fails, do not rescue with:
- another pseudo-opportunity strength;
- alternate minimum history;
- inside-5 or inside-20 substitution;
- end-zone-target substitution;
- recent-window conversion;
- same-team only conversion;
- starter/TE1/lead-back subsets;
- teammate or opponent interactions;
- pooled RB/TE conversion;
- TD-count thresholds;
- sportsbook/market information;
- 2026 outcomes.

A failed cell closes this exact direct persistent-conversion identity mechanism.

## Promotion boundary

Qualification would establish only:

`NAMED_PLAYER_SCORING_AREA_CONVERSION_IDENTITY_SUPPORTED`.

It would still **not** certify:
- team high-value opportunity creation;
- player high-value entitlement;
- full pregame ATD probability;
- sportsbook value;
- production promotion.

Those require separate, already-governed upstream/downstream science.

## Integrity at freeze

- predictive rows scored: 0
- 2026 outcomes read: 0
- sportsbook inputs: 0
- OddsAPI calls: 0
- production mutation: false
