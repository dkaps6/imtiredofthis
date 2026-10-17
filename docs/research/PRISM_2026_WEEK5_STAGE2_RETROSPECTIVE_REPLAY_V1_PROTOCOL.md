# PRISM 2026 Week-5 Stage-2 Retrospective Replay V1 — Protocol

Date: 2026-10-17

Status: **FROZEN RETROSPECTIVE QA CONTRACT / EXACT QUALIFIED MECHANISM / NO RETUNING**

Model:
**PRISM — Player Role & Individual Situation Model**

## 1. Purpose

The Week-5 player-first forensic replay showed that a large share of the largest
named-player misses involved opportunity/allocation, while the same player can
change failure mechanism from one week to the next.

The next question is therefore not to fit another share model from Week-5
outcomes.

It is:

> If the already-qualified PRISM Stage-2 active-role -> conditional-allocation
> mechanism had been available for Week 5, would its exact frozen law have
> improved the named-player opportunity allocation that actually failed?

This is a retrospective current-season QA replay of an already-frozen
historical mechanism.

## 2. Frozen scientific parent

Stage-2 parent:

`PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED_PARTIAL`

Canonical mechanism freeze:
- commit `87400a9c9c1db465c9c4bd70b0cb241dba27c6c9`;
- contract
  `docs/research/PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_V1_MECHANISM_FREEZE.md`.

Canonical historical scored authority:
- run `38957575645` — SUCCESS;
- job `117164201650`;
- exact head `eba5787c0b06aa5f975008965df7d023b3525ba7`;
- artifact `12070774378`;
- digest
  `sha256:e07edbad1bf97a4917ba95c58156c8941448e7b2668bd0eae2e9bcad5d4945b0`.

Historically qualified cells:
- RB rushing;
- RB receiving;
- TE receiving.

Closed / unavailable:
- QB passing — CLOSED;
- WR receiving — 2025 OOS parent source-blocked / not scored.

This replay may not reopen QB or manufacture WR Stage-2.

## 3. Exact mechanism remains unchanged

For every positive canonical entitlement `q_i`:

`A_i ~ Bernoulli(p_active_i)`.

The candidate exactly enumerates the active-set states and conditions the
existing canonical entitlement on the active set while preserving the anonymous
complement/residual.

No share coefficient is fit.
No activity threshold is fit.
No current-season outcome is allowed to alter `q_i` or `p_active_i`.

The Stage-1 probability remains the exact qualified V1 law:
- same named player;
- same team;
- last up to four strictly-prior eligible games;
- minimum three named-player games;
- otherwise position-family baseline;
- prior-strength semantics unchanged.

For target Week 5, Stage-1 history is **Weeks 1-4 only**.

## 4. Week-5 frozen pregame state

Canonical player-state authority:
- run `37717016519` — SUCCESS;
- exact head `35820953a255ce0f8413c2c468501df7f267fa96`;
- artifact `11524296325`;
- digest
  `sha256:e043133d4b9a48ce472dde8810d4d77d636f4ce2fbcee14708b23fe0ea35f6a0`.

Required target file:
`player_landscape_week5_player_state.csv`.

This file contains the Week-5 football-only player state used for:
- canonical receiving entitlement `entitlement_tgt_share`;
- canonical rule rushing state `rules_rush_share`;
- team play / pass-rate state;
- player/team/opponent identity.

It contains no sportsbook input for this replay.

RB rushing canonical `q` must use the exact same conserved top-five rushing
entitlement construction used by the Week-6 Stage-2 shadow.

RB/TE receiving canonical `q` must use the frozen Week-5
`entitlement_tgt_share`.

## 5. Week-5 target availability and identity

Target player IDs and definitive inactive state must be reconstructed only from
pregame football authorities:
- nflverse weekly roster identity for 2026 Week 5;
- official NFL Week-5 inactive reports.

No target-game stat row may determine whether a player was eligible.

Fail closed if:
- target stable-ID coverage <99%;
- the official inactive identity bridge <95%;
- room entitlement conservation fails;
- any source chronology violation occurs.

## 6. Strict-prior Stage-1 source

Only Weeks 1-4 may enter Week-5 `p_active`.

Use the same source family as the canonical Week-6 shadow:
- nflverse weekly rosters;
- official NFL inactive reports;
- PFR/nflverse offensive snap source;
- exact stable player identity.

No Week-5 snap/outcome may enter `p_active`.

## 7. Outcome scoring

Week-5 outcomes are read **only after** the pregame candidate rows are fixed.

Outcome source:
public weekly player statistics with stable IDs.

For each cell:
- RB rushing -> player carries / team rushing attempts;
- RB receiving -> player targets / team targets;
- TE receiving -> player targets / team targets.

Primary retrospective allocation score:
the same named-player finite-count CRPS comparison used by the frozen Stage-2
mechanism, conditional on realized team opportunity count.

Also report:
- expected-count absolute error;
- room allocation error;
- player-level direction of change;
- exact named-player rows from the already-frozen Week-5 failure casebook.

The realized team count is used for **scoring the allocation**, not as a pregame
predictor.

Separately retain the frozen pregame team opportunity mean to show how the
allocation shift would have affected full pregame expected opportunity.

## 8. Player-first diagnostics

Priority names from the already-frozen forensic evidence include, where the cell
is legally covered:
- Jahmyr Gibbs — RB rushing;
- Aaron Jones — RB rushing;
- Ollie Gordon II — RB rushing;
- Braelon Allen — RB rushing if identity/source gates clear;
- Brock Bowers — TE receiving.

Malik Nabers remains outside this replay because WR Stage-2 is source-blocked.

Do not select or tune the mechanism from these names. They are explanatory
case rows under the exact same all-player replay.

## 9. Interpretation boundary

This replay was designed after Week-5 outcomes existed.

Therefore:
- it is valid retrospective QA of a mechanism already frozen independently;
- it is not a new prospective qualification;
- it may show whether later-qualified PRISM science would have repaired earlier
  player-level failures;
- it may expose source/transmission seams;
- it may not fit a Week-5-specific correction.

A favorable Week-5 replay does not by itself promote Stage-2 to production.
The immutable Week-6 prospective shadow remains the prospective authority.

## 10. Anti-retest

Do not:
- alter Stage-1 window/minimum;
- alter Stage-2 active-set law;
- change canonical entitlement from Week-5 outcomes;
- use raw recent share as replacement;
- create WR Stage-2;
- rescue QB Stage-2;
- use sportsbook data;
- make an OddsAPI call;
- read Week-6 outcomes;
- change production.

## 11. Required outputs

1. all-player Week-5 Stage-2 replay rows;
2. room-level allocation rows;
3. source / chronology certificate;
4. cell-level retrospective metrics;
5. frozen-casebook named-player comparison rows;
6. explicit integrity certificate.

Disposition at protocol freeze:

`PRISM_2026_WEEK5_STAGE2_RETROSPECTIVE_REPLAY_V1_PROTOCOL_FROZEN`.
