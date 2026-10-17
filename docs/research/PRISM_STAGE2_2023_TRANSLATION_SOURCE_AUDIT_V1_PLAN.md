# PRISM Stage-2 2023 Independent Market-Translation Source Audit V1 — Frozen Plan

Date: 2026-10-17

Status: **FROZEN SOURCE / LINEAGE AUDIT BEFORE ANY 2023 MARKET-OUTCOME SCORING**

Model:
**PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_STAGE2_2023_TRANSLATION_SOURCE_AUDIT_V1`

## 1. Governing question

The Week-4/5 forensic replay and finite-allocation transmission audit identified a named-player finite-pool coherence problem.

The qualified Stage-2 architecture already solves a related historical allocation problem for:
- RB rushing;
- RB receiving;
- TE receiving.

Before translating that architecture into market-specific PRISM outcomes, ask one source question on an untouched/pre-authority season:

> Can the exact frozen Stage-1 active-role probability, the legitimate canonical pregame entitlement `q_i`, and the exact Stage-2 joint active-set semantics be reconstructed for 2023 with source parity strong enough to support an independent market-translation test?

This audit scores **no player outcome and no market projection**.

## 2. Scientific parents — immutable

Stage 1:
`PLAYER_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED_ALL`.

Stage 2:
`PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED_PARTIAL`.

Frozen Stage-2 mechanism:
- commit `87400a9c9c1db465c9c4bd70b0cb241dba27c6c9`;
- contract `docs/research/PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_V1_MECHANISM_FREEZE.md`.

Qualified cells only:
1. `RB_RUSHING`;
2. `RB_RECEIVING`;
3. `TE_RECEIVING`.

Not authorized:
- QB Stage 2 — CLOSED;
- WR Stage 2 — source-blocked/not scored.

No frozen Stage-1 probability law, Stage-2 active-set law, entitlement construction, window, fallback, threshold or conservation rule may be changed by this audit.

## 3. Target chronology

Target season:
- **2023** regular season.

Translation-audit target weeks:
- Weeks **5-18**.

Strict-prior state may use:
- 2022;
- earlier completed 2023 games;

but never the target game itself.

The source audit may inspect target-game identity/schedule membership only where it is a true pregame authority.

Target-game carries, targets, receptions, yards, snaps or other outcomes are forbidden as predictors or as a way to repair identity/availability.

## 4. Required Stage-1 pregame-universe parity

The 2023 target universe must reproduce the frozen Stage-1 semantics rather than invent a looser historical proxy.

Required:
1. weekly roster authority for the target team-week;
2. roster `ACT` semantics where the source supplies status;
3. stable player identity;
4. exact pre-kickoff official game-day inactive identity removed;
5. valid team-week offensive-snap source for later participation-history construction;
6. zero-snap target players remain eligible if they were legitimately in the pregame universe;
7. no target-game stats/snaps used to decide membership.

Frozen source gates:
- scheduled team-game roster coverage >= 99%;
- stable target identity >= 99%;
- exact official-inactive team-game coverage >= 99%;
- official-inactive -> roster identity bridge >= 95%;
- team-week snap-source coverage >= 99%;
- positive-snap identity -> roster bridge >= 98%;
- chronology violations = 0.

A roster status field is **not** automatically equivalent to official game-day inactive authority.
A target-game snap zero is **not** an inactive source.

If exact 2023 official-inactive parity cannot be established, Stage 1 is source-blocked and the overall translation bridge must fail closed.

## 5. Required canonical entitlement parity

### RB rushing

Reconstruct the exact canonical historical RB rushing entitlement:
- existing pregame rules state;
- existing top-five rushing selection;
- existing 0.95 named-mass cap/residual semantics;
- finite team rushing denominator;
- no new share model.

Required:
- target team-game coverage >= 99%;
- canonical positive/zero entitlement preserved;
- finite denominator coverage >= 99%;
- room conservation >= 99%;
- ambiguous identity <= 0.5%;
- chronology violations = 0.

### RB receiving

Reconstruct the exact canonical target entitlement already present in the historical PRISM state:
- existing target entitlement;
- existing residual/complement;
- finite target denominator;
- no new receiving-share proxy.

Use the same coverage, identity, denominator, conservation and chronology gates as RB rushing.

### TE receiving

Use only the exact TE-R5P OOS fold authority for **test season 2023**.

Required:
- `test_season = 2023` fold exists;
- its training lineage precedes the target season under the frozen OOS contract;
- exact recovered coefficients/scaler lineage;
- strictly-prior participation semantics;
- TE pool conservation;
- target team-game coverage >= 99%;
- entitlement coverage >= 95%;
- finite denominator coverage >= 99%;
- chronology violations = 0.

A later TE final fit may not be relabeled as 2023 OOS authority.

## 6. Frozen Stage-2 semantics

If and only if Stage-1 plus the cell's canonical `q_i` source are ready, the later translation test may use the already-frozen Stage-2 law:

`A_i ~ Bernoulli(p_active_i)`.

For each active-set state:
- inactive positive-entitlement named player receives zero;
- active named players preserve relative canonical entitlement;
- anonymous complement remains explicit;
- finite opportunity is conserved exactly;
- all-off state goes to complement.

No correlation, threshold, depth-chart override, new share coefficient, vacancy transfer or recent-share replacement may be added.

This audit does **not** run that outcome score.

## 7. Source dispositions

Per cell:
- `RB_RUSHING_STAGE2_2023_TRANSLATION_SOURCE_READY`
- `RB_RECEIVING_STAGE2_2023_TRANSLATION_SOURCE_READY`
- `TE_RECEIVING_STAGE2_2023_TRANSLATION_SOURCE_READY`

or matching `..._SOURCE_BLOCKED`.

Overall:
- all ready:
  `PRISM_STAGE2_2023_TRANSLATION_SOURCE_READY_ALL`;
- some ready:
  `PRISM_STAGE2_2023_TRANSLATION_SOURCE_READY_PARTIAL`;
- none / common mandatory Stage-1 blocker:
  `PRISM_STAGE2_2023_TRANSLATION_SOURCE_BLOCKED`;
- integrity violation:
  `PRISM_STAGE2_2023_TRANSLATION_SOURCE_INTEGRITY_FAILURE`.

A common Stage-1 source blocker prevents scoring every cell even if a cell-specific entitlement source exists.

## 8. If source-ready

Do not score immediately from this document.

Freeze a separate market-translation test before opening 2023 target outcomes.

Order:
1. coherent named-player opportunity count/distribution;
2. then a separately frozen bridge to the exact market:
   - RB rushing -> carries before rushing yards;
   - RB receiving -> targets before receptions/receiving yards;
   - TE receiving -> targets before receptions/receiving yards;
3. existing efficiency authority remains unchanged.

No threshold/window/weight search.

## 9. If source-blocked

Fail closed.

Do not:
- drop official inactives from the Stage-1 contract;
- use target-game snaps as availability;
- use sportsbook listing;
- use a later-season depth chart;
- substitute raw current/recent share;
- approximate TE-R5P with a later fit;
- score a partially reconstructed 2023 proxy.

Move to prospective/future-week mechanics that can persist exact Stage-1, canonical `q_i`, active-set allocation and market-translation trace **before kickoff**, while protecting immutable Week-6 evidence.

## 10. Integrity

This source audit authorizes:
- 2023 target outcomes scored: **0**;
- 2026 outcomes used: **0**;
- sportsbook inputs: **0**;
- OddsAPI calls: **0**;
- fitted models: **0**;
- threshold searches: **0**;
- projection changes: **0**;
- production changes: **0**.

Disposition at freeze:

`PRISM_STAGE2_2023_TRANSLATION_SOURCE_AUDIT_V1_FROZEN`.
