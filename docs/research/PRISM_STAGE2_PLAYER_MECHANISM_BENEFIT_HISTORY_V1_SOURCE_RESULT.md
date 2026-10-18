# PRISM Stage-2 Player Mechanism Benefit History V1 — Source Result

Date: 2026-10-18

Status: **SOURCE READY ALL / HIERARCHICAL MECHANISM-ELIGIBILITY SIGNAL MAY BE FROZEN / NO PERSONALIZATION RULE SCORED**

Model: **PRISM — Player Role & Individual Situation Model**

Frozen source plan:
`docs/research/PRISM_STAGE2_PLAYER_MECHANISM_BENEFIT_HISTORY_V1_SOURCE_PLAN.md`

Pre-implementation freeze:
`2a7d0f4a33b4669207f34bd98ee44128de7522c2`

Canonical execution:
- run `39098772787` — **SUCCESS**;
- exact head `744b2e920592803937e8a8c94e7a0c89d6ebd4eb`;
- job `117590554580` — **SUCCESS**;
- artifact `12126215015`;
- digest `sha256:5249f41f72a248443c8a596de2a3cf4211928369b8867b3f3a4d16e70601c78d`.

Parent Stage-2 authority:
- run `38957575645`;
- exact head `eba5787c0b06aa5f975008965df7d023b3525ba7`;
- artifact `12070774378`;
- digest `sha256:e07edbad1bf97a4917ba95c58156c8941448e7b2668bd0eae2e9bcad5d4945b0`.

Overall disposition:

`PRISM_STAGE2_PLAYER_MECHANISM_BENEFIT_HISTORY_SOURCE_READY_ALL`

## 1. Exact source object

The immutable parent Stage-2 artifact already preserves row-level OOS evidence for:

- stable player identity;
- season/week/game/team;
- exact qualified cell;
- canonical pregame entitlement `authority_q`;
- qualified Stage-1 `p_active`;
- baseline finite-count CRPS;
- Stage-2 candidate finite-count CRPS;
- exact row-level `delta_crps = baseline_crps - candidate_crps`;
- expected-count absolute-error deltas.

The source audit reconstructs, for each target row, only **strictly-prior**
same-player + same-cell mechanism-benefit history.

The current target row is appended to history only **after** its pregame state is
captured.

Chronology violations: **0**.

Duplicate player/cell/week identities: **0**.

Stable player identity: **100%**.

Canonical entitlement-rank state derivable: **100%**.

## 2. RB rushing — SOURCE READY

Disposition:

`RB_RUSHING_STAGE2_PLAYER_BENEFIT_HISTORY_SOURCE_READY`

2025 target:
- 1,419 player-games;
- 154 distinct RBs;
- 1,278 rows with >=3 strictly-prior same-player mechanism observations;
- 142 players with >=3;
- 128 players with >=6;
- 117 players with >=10;
- 107 players span both 2024 and 2025;
- 32 teams.

Prior-history count:
- median 14;
- p25 8;
- p75 18.

2025 support tiers:
- Tier 0: 47 rows;
- Tier 1 (1–2): 94;
- Tier 2 (3–5): 126;
- Tier 3 (6–9): 158;
- Tier 4 (10+): 994.

Every frozen source gate passed.

## 3. RB receiving — SOURCE READY

Disposition:

`RB_RECEIVING_STAGE2_PLAYER_BENEFIT_HISTORY_SOURCE_READY`

2025 target:
- 1,419 player-games;
- 154 distinct RBs;
- 1,278 rows with >=3 strictly-prior same-player mechanism observations;
- 142 players with >=3;
- 128 players with >=6;
- 117 players with >=10;
- 107 players span both 2024 and 2025;
- 32 teams.

Prior-history count:
- median 14;
- p25 8;
- p75 18.

2025 support tiers:
- Tier 0: 47 rows;
- Tier 1 (1–2): 94;
- Tier 2 (3–5): 126;
- Tier 3 (6–9): 158;
- Tier 4 (10+): 994.

Every frozen source gate passed.

This cell is especially important because the qualified parent Stage-2 mechanism
improved only **47.11%** of evaluable named players in 2025 despite passing its
equal-player inference gate. There is therefore real population value plus real
named-player heterogeneity to explain.

## 4. TE receiving — SOURCE READY

Disposition:

`TE_RECEIVING_STAGE2_PLAYER_BENEFIT_HISTORY_SOURCE_READY`

2025 target:
- 1,305 player-games;
- 141 distinct TEs;
- 1,215 rows with >=3 strictly-prior same-player mechanism observations;
- 127 players with >=3;
- 120 players with >=6;
- 110 players with >=10;
- 106 players span both 2024 and 2025;
- 32 teams.

Prior-history count:
- median 15;
- p25 10;
- p75 19.

2025 support tiers:
- Tier 0: 35 rows;
- Tier 1 (1–2): 55;
- Tier 2 (3–5): 76;
- Tier 3 (6–9): 122;
- Tier 4 (10+): 1,017.

Every frozen source gate passed.

The qualified parent Stage-2 mechanism improved **57.89%** of evaluable named
TEs in 2025, again leaving meaningful individual heterogeneity.

## 5. What this result proves

PRISM has enough legal historical OOS evidence to ask a new player-specific
question without inventing a tiny bespoke model for each player:

> Does this named player's own **prior out-of-sample benefit from an already-qualified mechanism** contain stable information about whether that same mechanism should receive authority for him in a future game?

This is source readiness, not predictive proof.

It supports a hierarchy of:

`market prior -> current role/archetype prior -> same-player prior mechanism evidence -> current player state`.

It does **not** authorize:
- a free coefficient per player;
- a player whitelist;
- an activation threshold selected after outcomes;
- a Stage-2 formula change;
- projection mutation;
- production mutation.

## 6. Role/archetype source

A pregame role object is available without outcomes:

`canonical within-room entitlement rank/state`.

The source audit can derive:
- ZERO;
- RANK1;
- RANK2;
- RANK3_PLUS

from the already-canonical pregame entitlement.

That provides a legal role/archetype prior for a later hierarchical signal
without reopening blocked depth-chart or room-competition sources.

## 7. Anti-retest boundary

This result does not reopen:
- QB Stage-2;
- WR Stage-2;
- short-yardage carry-type identity;
- high-value conversion identity;
- raw recent-share replacement;
- room competition;
- depth-role transition;
- role elasticity;
- generic matchup;
- NGS RB efficiency;
- closed receiving-efficiency or trust rescues.

The new object is **prior utility of an already-qualified mechanism**, not a new
version of the underlying football feature.

## 8. Next authorized action

Freeze one hierarchical **mechanism-benefit persistence / eligibility signal**
before looking at whether prior benefit predicts future benefit.

The first signal should:
- keep Stage-2 itself unchanged;
- use current entitlement-rank state as the role/archetype prior;
- use same-player prior OOS `delta_crps` as individual evidence;
- shrink player evidence toward the role prior;
- require >=3 prior same-player observations before individual deviation;
- use 2024 as development and 2025 as independent confirmation;
- score player-aware incremental prediction of future Stage-2 utility;
- not change any actual player projection yet.

Only if that signal survives may a separate conservation-safe Stage-2 authority
bridge be frozen.

## 9. Integrity

Canonical source run:
- personalization rules scored: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 outcomes used: **0**;
- Week-6 outcomes used: **0**;
- sportsbook inputs: **0**;
- OddsAPI calls: **0**;
- production changed: **false**.

Final disposition:

`PRISM_STAGE2_PLAYER_MECHANISM_BENEFIT_HISTORY_SOURCE_READY_ALL`.
