# NFL HANDOFF — 2026-10-16 — PRISM DENOMINATOR PERSISTED / OPPORTUNITY-QUALITY MECHANISM NEXT

**Status:** CURRENT / ELITE CONTINUITY HANDOFF  
**Repository:** `dkaps6/imtiredofthis`  
**Model:** **PRISM — Player Role & Individual Situation Model**  
**Production branch:** `main`  
**Active research branch:** `research-individual-opportunity-roadmap-2026-10-08`  
**Research PR:** #672 — OPEN / DRAFT  
**Canonical research status:** `PLAYER_INDIVIDUALIZATION_PARTIAL`

This handoff supersedes the earlier 2026-10-16 active-role / Stage-2 / Week-6-shadow / opportunity-quality handoffs for startup purposes. Do not recursively read older handoffs unless this file explicitly points to one.

## 1. User objective — non-negotiable

PRISM is an **individual-player projection model**.

The user has been explicit that the scientific object is the named player, not a position-average error statistic.

Canonical order:

`PLAYER IDENTITY -> CURRENT ROLE/STATE -> TEAMMATE COMPETITION -> OPPORTUNITY ALLOCATION -> OPPORTUNITY QUALITY -> OPPONENT/ENVIRONMENT -> EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`

Position MAE/RMSE is downstream QA. If individual player projections improve, aggregate position error should improve as a consequence.

The model must reason about:
- who this player is;
- his own strictly-prior history;
- his current role;
- teammate competition;
- finite opportunity allocation;
- quality/type/location of opportunity;
- this week's team/opponent environment;
- player-specific efficiency and uncertainty;
- the exact market being projected.

Do not drift back into "reduce average QB/RB/WR/TE MAE" as the primary modeling objective.

## 2. Paid-data guardrail

Exactly **one** paid Week-6 OddsAPI acquisition has already been used.

Preserved origin:
- run `38791377571`
- artifact `11999923880`
- digest `sha256:0916cd1cbec93f9a2529190b3df7c39172eee011a31eb9df429a35cdb9b41f61`

Do **not** call OddsAPI again without explicit user authorization.

Sportsbook data remains downstream comparison only and may not enter upstream PRISM football probability construction.

## 3. Stage 1 — named-player active-role state QUALIFIED

Disposition:

`PLAYER_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED_ALL`

Canonical scored authority:
- run `38907459996` SUCCESS
- job `116994537005`
- exact head `1c2c92ca498e8dce1eee19d39b2f276dfd37d8f7`
- artifact `12049120604`
- digest `sha256:41b1acd3f1b8d77abb53980db160ece32f8fc40351939d4827248b9caad0c64b`

Exact mechanism:
- pregame universe = weekly roster ACT + stable identity - exact official inactives;
- target = any offensive snap;
- position-family Jeffreys strictly-prior baseline;
- named-player same-team history = most recent up to 4 eligible games;
- minimum 3;
- prior strength 2.

2025 Brier improvement:
- QB +0.092720
- RB +0.038602
- WR +0.015566
- TE +0.006296

This establishes a real player-specific zero/active-role layer. It does not by itself authorize multiplying every projection by p_active.

## 4. Stage 2 — active-role -> conditional allocation HISTORICALLY QUALIFIED PARTIAL

Historical bridge disposition:

`PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED_PARTIAL`

Frozen-before-scoring commit:
`87400a9c9c1db465c9c4bd70b0cb241dba27c6c9`

Canonical scored authority:
- run `38957575645` SUCCESS
- job `117164201650`
- exact head `eba5787c0b06aa5f975008965df7d023b3525ba7`
- artifact `12070774378`
- digest `sha256:e07edbad1bf97a4917ba95c58156c8941448e7b2668bd0eae2e9bcad5d4945b0`

Qualified:
- `RB_RUSHING_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED`
- `RB_RECEIVING_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED`
- `TE_RECEIVING_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED`

Closed:
- `QB_PASSING_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_CLOSED`

Not scored:
- WR receiving because legitimate 2025 WR-R15 independent OOS authority is unavailable/forbidden.

Important QB lesson:
the exact bridge improved CRPS but worsened expected pass attempts and room allocation and failed 2025 dependence-aware confirmation. Do not rescue it with thresholds, starter carveouts, correlations, backup reassignment or pooled evidence.

## 5. Week-6 prospective Stage-2 shadow — LOCKED

Disposition:

`PRISM_WEEK6_SUNMON_STAGE2_SHADOW_LOCKED`

Canonical lock:
- run `38991032116` SUCCESS
- exact head `3c0190cc8f1cd6271bfd6912cf84cff82008ff80`
- artifact `12084848700`
- digest `sha256:ec1376dfa90fff3b5fc31e199588f7fe5f570017946adfd0b85be847af901e6e`

Population:
- 13 Sunday/Monday Week-6 games;
- SEA-DEN excluded because it had already played before Stage-2 was frozen;
- 248 player-cell rows;
- RB rushing / RB receiving / TE receiving only.

All lock-time integrity gates passed:
- prior roster team-game coverage 100%;
- official inactive team-game coverage 100%;
- prior snap team-game coverage 100%;
- stable identity 100%;
- inactive -> roster qualified-skill bridge 97.5535%;
- positive-snap -> roster qualified-skill bridge 99.6386%;
- Stage-1 p_active attach 100%;
- canonical q finite 100%;
- max candidate mass gap 2.22e-16;
- Week-6 target outcomes read 0;
- sportsbook upstream 0;
- OddsAPI calls 0;
- production mutation false.

Earlier dry-run identity issues are **closed, not current blockers**.

Final identity-only reconciliations include:
- Audric Estim -> Audric Estimé -> GSIS `00-0039373`;
- Drew Ogletree -> Andrew Ogletree -> GSIS `00-0037292`.

Do not reopen the interim Cleveland-ID failure.

Do not grade the shadow until all 13 included Sunday/Monday games are final.

## 6. Opportunity quality/topology — source authority READY

This is the current research layer after opportunity quantity.

Do not confuse it with the closed experiment:
`PLAYER_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_CLOSED`.

That closed V1 asked whether a named player's recent scoring-area share directly persists better than ordinary usage. It did not.

The current question is different:

> How much high-value opportunity does the TEAM create, conditional on its finite opportunity quantity, and how should that environment flow into the current named player's already-qualified entitlement?

### Source authority A — player/Stage-2 compatibility

Run:
- `38991429245` SUCCESS
- head `d14ba07f7f0e79f2b3f16a0b018a50c30744be91`
- artifact `12085262613`
- digest `sha256:77739799f34237538d7ffcba9715939c59d6a5b50f811f06f2f5836ff9deb31d`

Disposition:
`PRISM_OPPORTUNITY_QUALITY_SOURCE_READY_ALL`

Source-ready cells:
- RB rush inside-10;
- RB receiving red-zone targets;
- TE receiving red-zone targets.

### Source authority B — full finite team pools

Canonical hardened run:
- `38991605252` SUCCESS
- exact head `e590ce4754da2c7f7aa01beab2ee148aa5b613a0`
- artifact `12085367606`
- digest `sha256:68b680c745ae8b8479b3dbaf8f41dcc4f6cb728a0f39b0ac10b77dc3ac8e4b90`

Disposition:
`PRISM_OPPORTUNITY_QUALITY_TEAM_VOLUME_SOURCE_READY`

2022-2025 all passed:
- scheduled team-game coverage 100%;
- rusher identity 100%;
- receiver identity 100%;
- strictly-prior offense state attach 100%;
- strictly-prior opponent state attach 100%;
- drive topology source-ready;
- optional end-zone source-ready.

Optional source readiness is not permission to use every field predictively.

### Exact source reconciliation

Freeze:
`docs/research/PRISM_OPPORTUNITY_QUALITY_SOURCE_RECONCILIATION_V1_FREEZE.md`

Canonical run:
- `38991737241` SUCCESS
- job `117275405644`
- exact head `c1abb0bacd0c78dd0b38861214bd28c04cc9f376`
- artifact `12085278127`
- digest `sha256:0be3d571129876ce0a8df755762215e147faec2ff1ac1822e05cbd84508a12bb`

Disposition:
`PRISM_OPPORTUNITY_QUALITY_SOURCE_RECONCILED_READY_ALL`

Exact 2024/2025 Weeks 5-18 parity:
- 832 / 832 team-games;
- zero left/right-only keys;
- zero mismatch rows;
- team rushes drift 0;
- team inside-10 rushes drift 0;
- team targets drift 0;
- team red-zone targets drift 0.

No predictive opportunity-quality candidate was scored.

## 7. Numeric parent denominator lineage — NOW PASSED

The prior checkpoint said this run was active. It has completed successfully.

Frozen addendum:
`docs/research/PRISM_OPPORTUNITY_QUALITY_PARENT_DENOMINATOR_PERSISTENCE_V1_ADDENDUM.md`

Freeze commit:
`3a980c9ba821c5e0a3b41231314db5c21bb0adef`

Canonical execution:
- run `38991915503` — **SUCCESS**
- job `117275946409`
- exact head `02e96ad9faabee62c1ecb31303b48746e4945a8c`
- artifact `12085194208`
- digest `sha256:c2be00311f9cf2149ebef36d419851214add08cae83bbc3b947990ce60a5e569`

Result:
`docs/research/PRISM_OPPORTUNITY_QUALITY_PARENT_DENOMINATOR_PERSISTENCE_V1_RESULT.md`

Disposition:
`PRISM_PARENT_DENOMINATOR_PERSISTENCE_PARITY_PASS`

Exact parity:
- rows 14,284;
- key sets exact true;
- denominator finite flags exact true;
- ready cells exact true;
- scientific disposition exact true;
- max absolute authority-q drift `2.0816681711721685e-16`.

The existing canonical numeric pregame denominator `authority_denominator_mean` is now immutable source lineage.

No denominator formula changed.

## 8. Exact next scientific action

The next chat should **not** redo source work.

The source seam is complete.

Now freeze **one** opportunity-quality mechanism BEFORE scoring.

The first mechanism must isolate opportunity quality before opponent/environment transmission.

Conceptually:
1. preserve the existing canonical pregame opportunity denominator;
2. estimate the TEAM'S high-value-opportunity creation state using only strictly-prior football information;
3. keep the existing PRISM player entitlement / Stage-2 allocation as the named-player side;
4. test whether the quality/topology layer improves the individual player's relevant opportunity distribution.

Source-ready quality pools:
- RB rushing: inside-10 team rushing opportunities;
- RB receiving: team red-zone targets;
- TE receiving: team red-zone targets.

Development:
- 2024.

Independent confirmation:
- 2025.

The mechanism/gates must be frozen before target scoring.

### Anti-retest

Do **not**:
- reopen recent player high-value-share persistence V1;
- replace canonical player entitlement with recent red-zone/inside-10 player share;
- change inside-10/red-zone bands after seeing outcomes;
- use TD outcomes to rescue;
- add opponent/environment transmission in this first mechanism;
- use sportsbook/market inputs;
- use 2026 outcomes;
- lower support/inference gates after scoring;
- production-promote anything automatically.

A failed mechanism closes that exact formulation.

## 9. Other protected terminal state

Preserve:
- player role-elasticity V1 source block;
- QB exact Stage-2 V1 closure;
- WR 2025 OOS source block;
- generic recent-share allocators closed;
- existing RB carry/snap and RB receiving-room science;
- WR/TE trajectory science and contracts;
- WR/TE player-difficulty trust qualifications and Week-6 locks;
- generic matchup / Vegas / game-script closed lanes;
- current production model unless separately authorized.

Generic ATD remains uncertified.

## 10. Live repository state at handoff preparation

Immediately before this handoff write:
- `main` newest canonical docs checkpoint was around `7ad49a1d3f1cdf771ad85f3fab7e72d5ca686da7`;
- research branch / PR #672 head was `b4603132e927aa3d9ecf8a312a8bb445ff4b39c1`;
- PR #672 OPEN / DRAFT.

This handoff write itself advances documentation heads. The next chat must reconcile live:
- main;
- research branch;
- PR #672;
- Issue #535 newest comments;
- exact-head Actions.

GitHub is canonical.

## 11. Minimal startup read order for next chat

Read only:

1. `AGENTS.md`
2. ONLY the newest TOP checkpoint in `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. this handoff
4. `docs/research/PRISM_OPPORTUNITY_QUALITY_PARENT_DENOMINATOR_PERSISTENCE_V1_RESULT.md`
5. latest relevant Issue #535 comments after the opportunity-quality reconciliation
6. live PR #672 / main / research head / exact-head Actions

Do not recursively ingest old handoffs.

Then immediately continue the opportunity-quality mechanism freeze.

The user should not have to re-explain PRISM, the individual-player philosophy, Stage 1/2, the Week-6 shadow, or why position-average MAE is not the objective.
