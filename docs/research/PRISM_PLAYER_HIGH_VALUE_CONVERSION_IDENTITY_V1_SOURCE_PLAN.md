# PRISM Player High-Value Conversion Identity V1 — Source / Support Audit Plan

Date: 2026-10-17

Status: **FROZEN BEFORE SOURCE AUDIT / NO PREDICTIVE SCORING / RESEARCH ONLY**

Version:
`PRISM_PLAYER_HIGH_VALUE_CONVERSION_IDENTITY_V1_SOURCE_AUDIT`

## Scientific object

PRISM projects the named player in his actual football situation.

This audit asks a downstream question that is deliberately separate from the
closed high-value-opportunity-share persistence family:

> Conditional on a named player actually receiving a scoring-area opportunity,
> is historical touchdown-conversion identity source-valid and sufficiently
> supported to justify a separately frozen predictive test?

This is **conversion efficiency conditional on opportunity**, not opportunity
entitlement.

## Anti-retest boundary

Protected closures remain protected.

Do not:
- reopen `PLAYER_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_CLOSED`;
- use recent scoring-area share as the candidate;
- change yardline bands after seeing support;
- use TD outcomes to rescue the closed entitlement formula;
- pool channels/positions to clear support;
- weaken support gates after inspection;
- use sportsbook, OddsAPI or 2026 outcomes;
- mutate production or generic ATD.

The prior high-value source authority explicitly stated that its first test was
**not touchdown probability** and that any later TD-conversion stage required a
separate freeze. This audit is that source/support boundary only.

## Frozen channels

Only channels with qualified PRISM Stage-2 allocation are eligible:

1. `RB_RUSH_INSIDE10`
   - rush attempt;
   - named rusher family RB/FB/HB;
   - `yardline_100 <= 10`;
   - conversion label = native `rush_touchdown`.

2. `RB_RECEIVING_REDZONE`
   - targeted pass to named receiver family RB/FB/HB;
   - `yardline_100 <= 20`;
   - conversion label = native `pass_touchdown`.

3. `TE_RECEIVING_REDZONE`
   - targeted pass to named receiver family TE;
   - `yardline_100 <= 20`;
   - conversion label = native `pass_touchdown`.

WR is not opened because WR Stage-2 remains source-blocked/not scored.

## Historical source

- canonical nflverse/nflfastR PBP via repo `get_pbp`;
- canonical weekly-roster GSIS bridge reused from the already-qualified
  high-value-opportunity source audit;
- seasons 2022-2025 regular season;
- no 2026 outcomes.

The audit may inspect historical conversion labels only to establish source
semantics and support. It may not compare a candidate probability with future
conversion outcomes.

## Strict-prior support construction

For every named player-game, first aggregate the frozen high-value opportunity
events and TD conversions for that game.

Then compute same-player history from **completed earlier games only**:
- all eligible seasons in chronological order;
- target game excluded in full;
- events earlier within the same game may not enter the prior state.

Frozen minimum history for a later predictive candidate:
**>=3 prior high-value opportunities**.

This source audit does not choose a coefficient, probability formula or
shrinkage strength.

## Frozen source gates

For every season/channel:
- nonzero events;
- native TD label finite coverage >= 99.9%;
- stable player identity nonempty >= 99.0%;
- weekly-roster family bridge >= 98.0%;
- duplicate player-play contradictions = 0;
- receiving TD implies native complete pass (contradictions = 0);
- touchdown labels binary after numeric coercion.

For each 2024 and 2025 channel:
- >=350 target player-games with >=3 strictly-prior same-player high-value
  opportunities;
- >=40 distinct supported named players.

All gates are AND gates.

If any channel fails, that channel is source-blocked. No post-result pooling or
gate reduction is allowed.

## What source readiness would authorize

Only a separately frozen predictive mechanism test comparing:
- a strict-prior channel/family conversion baseline;
- versus a shrunk strictly-prior same-player conversion identity.

The later test, if authorized, must keep target-game opportunity quantity
conditional/descriptive so conversion is not confused with entitlement.

It would **not** certify ATD, production promotion, or betting.

## Integrity

Required:
- predictive candidates scored = 0;
- 2026 outcomes read = 0;
- sportsbook inputs = 0;
- OddsAPI calls = 0;
- production mutation = false.
