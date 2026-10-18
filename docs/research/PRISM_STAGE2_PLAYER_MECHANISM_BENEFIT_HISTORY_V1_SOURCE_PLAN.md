# PRISM Stage-2 Player Mechanism Benefit History V1 — Frozen Source Audit Plan

Date: 2026-10-18

Status: **FROZEN SOURCE / LINEAGE AUDIT ONLY / NO PERSONALIZATION RULE SCORED**

Model: **PRISM — Player Role & Individual Situation Model**

Parent architecture:
`docs/research/PRISM_PLAYER_SPECIFIC_RULE_LEARNING_V1_ANTI_RETEST_SOURCE_SUPPORT_MAP.md`

Version:
`PRISM_STAGE2_PLAYER_MECHANISM_BENEFIT_HISTORY_V1_SOURCE_AUDIT`

## 1. Frozen question

For each already-qualified Stage-2 cell, can PRISM reconstruct enough
**strictly-prior same-player OOS mechanism-benefit history** to support a later,
separately frozen hierarchical player-specific mechanism-eligibility experiment?

Qualified parent cells only:

- RB rushing;
- RB receiving;
- TE receiving.

This source audit does **not** test whether player-specific gating improves a
projection.

It only tests whether the required historical OOS evidence exists with clean
chronology and stable player identity.

## 2. Scientific distinction from closed work

This is **not** a retest of the Stage-2 mechanism.

Stage-2's frozen formula remains unchanged:

`qualified Stage-1 active-role probability -> active-set state -> existing canonical entitlement -> finite opportunity allocation`.

This audit does not alter:
- active-role probability;
- entitlement;
- active-set enumeration;
- conservation;
- anonymous complement;
- opportunity denominator;
- any Stage-2 coefficient, because Stage-2 has none to fit.

It asks a new meta-question:

> Has this named player historically benefited from this already-qualified
> mechanism often and stably enough, using only prior OOS rows, to justify later
> deviation from a broader role/archetype prior?

It is also distinct from:
- mechanism-instability trust V1, which asked whether switching between
  opportunity/efficiency failure modes predicts projection fragility;
- player high-value conversion identity, which tested scoring conversion;
- short-yardage carry-type identity, which tested a direct same-player situational
  carry scalar;
- raw recent-share replacement;
- role elasticity / vacancy transfer.

## 3. Canonical parent authority

Historical Stage-2 result:

`PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED_PARTIAL`

Canonical scored run:
- `38957575645` — SUCCESS;
- exact scored head `eba5787c0b06aa5f975008965df7d023b3525ba7`;
- artifact `12070774378`;
- digest `sha256:e07edbad1bf97a4917ba95c58156c8941448e7b2668bd0eae2e9bcad5d4945b0`.

Parent source run:
- `38955368643` — SUCCESS;
- exact head `cb65a634a0e09296bfeabc94ffe1e77363a30f8a`;
- artifact `12070806828`;
- digest `sha256:e91a108495d7a2b6eb289e0618ab6af09e3d6cdf6895e534742a07340e4a2a83`.

Use the exact parent row authority if recoverable.
Do not rebuild a different Stage-2 formula.

## 4. Frozen source fields required

For every historical Stage-2 scored row used in this audit, require:

- season;
- week;
- game identity;
- team;
- opponent;
- stable player ID / canonical player key;
- position/family;
- qualified Stage-2 cell;
- canonical pregame entitlement;
- Stage-1 active-role probability;
- baseline row score contribution;
- Stage-2 candidate row score contribution;
- enough information to reproduce the already-completed row-level OOS mechanism
  benefit:
  `benefit = baseline_loss - candidate_loss`.

Preferred primary loss:
- named-player finite-count CRPS, matching Stage-2's primary score.

Secondary disclosure only:
- expected-count absolute-error improvement;
- room-allocation error improvement where row semantics permit.

The target row's realized outcome may be used only to reproduce its already-OOS
historical score contribution.
That row's benefit may not enter its own pregame player history.

## 5. Frozen chronology

For any target historical row `t`, prior player mechanism-benefit history must
use only rows satisfying:

- same stable player identity;
- same qualified Stage-2 cell;
- prior season, or same season with week < target week;
- no same/future target row;
- no target-game outcome used in pregame state.

For a 2025 target row, 2024 same-player OOS Stage-2 rows are eligible prior
history if exact player identity and cell semantics match.

The source audit must report chronology violations and fail if any are nonzero.

## 6. Frozen support-reporting bins

Report, but do not turn into a predictive rule:

- Tier 0: 0 prior mechanism rows;
- Tier 1: 1–2;
- Tier 2: 3–5;
- Tier 3: 6–9;
- Tier 4: 10+.

For each cell and target season, report:

- total target player-games;
- distinct players;
- rows / players by support tier;
- median / p25 / p75 prior-history count;
- number of players with >=3, >=6 and >=10 prior observations;
- number of players spanning both 2024 and 2025;
- team coverage;
- identity failures;
- duplicate identities;
- chronology violations.

Do not inspect whether any support tier “works better” in this audit.

## 7. Frozen source-readiness gates

The later hierarchical eligibility experiment is authorized only for cells that
pass **all** source gates.

### RB rushing

For 2025 targets:
- >=900 target player-games with >=3 prior same-player Stage-2 rows;
- >=90 distinct players with >=3 prior rows;
- >=60 players with >=6 prior rows;
- >=28 teams;
- stable player identity >=99%;
- duplicate player/cell/week rows = 0;
- chronology violations = 0.

### RB receiving

For 2025 targets:
- >=900 target player-games with >=3 prior same-player Stage-2 rows;
- >=90 distinct players with >=3 prior rows;
- >=60 players with >=6 prior rows;
- >=28 teams;
- stable player identity >=99%;
- duplicates = 0;
- chronology violations = 0.

### TE receiving

For 2025 targets:
- >=800 target player-games with >=3 prior same-player Stage-2 rows;
- >=75 distinct players with >=3 prior rows;
- >=50 players with >=6 prior rows;
- >=28 teams;
- stable player identity >=99%;
- duplicates = 0;
- chronology violations = 0.

These gates may not be lowered after the audit runs.

If a cell fails, it is source-blocked for V1 personalization even though the
parent Stage-2 mechanism itself remains qualified.

## 8. Role/archetype-prior source inventory

The source audit must also report whether the historical Stage-2 rows contain a
clean pregame role/archetype object usable **without outcomes**.

Preferred candidates, in order:

1. canonical within-room entitlement rank;
2. canonical entitlement share / state already used by the parent;
3. exact current pregame role label only if historical/live semantic parity is
   already proven.

This audit may report support for these fields.

It may not select the best archetype by looking at mechanism benefit.

No clustering on outcome or benefit is allowed.

If only canonical entitlement rank/share is clean, that is sufficient for a
later role-prior design.

## 9. Explicit prohibited actions

This audit must not:

- fit a player-specific model;
- fit a role model;
- choose a shrinkage coefficient;
- choose an activation threshold;
- create an “always on for these players” list;
- use 2026 outcomes;
- read/grade Week-6 outcomes;
- use sportsbook lines/odds;
- call OddsAPI;
- change production;
- change the parent Stage-2 formula;
- rescue QB Stage-2;
- score WR Stage-2;
- reopen any closed mechanism.

## 10. Possible source dispositions

Per cell:

- `RB_RUSHING_STAGE2_PLAYER_BENEFIT_HISTORY_SOURCE_READY`
- `RB_RECEIVING_STAGE2_PLAYER_BENEFIT_HISTORY_SOURCE_READY`
- `TE_RECEIVING_STAGE2_PLAYER_BENEFIT_HISTORY_SOURCE_READY`

or corresponding `..._SOURCE_BLOCKED`.

Overall:

- all three ready:
  `PRISM_STAGE2_PLAYER_MECHANISM_BENEFIT_HISTORY_SOURCE_READY_ALL`;
- one/two ready:
  `PRISM_STAGE2_PLAYER_MECHANISM_BENEFIT_HISTORY_SOURCE_READY_PARTIAL`;
- none ready:
  `PRISM_STAGE2_PLAYER_MECHANISM_BENEFIT_HISTORY_SOURCE_BLOCKED`.

## 11. If source-ready

Do **not** immediately optimize player weights.

Freeze a separate hierarchical mechanism-eligibility experiment first.

The first scored candidate should prefer a bounded/shrunk authority state over
free player coefficients.

Its evaluation must include:
- equal-player;
- player-cluster;
- game/team cluster;
- named-player improvement share;
- median player change;
- tail-harm protection;
- support-tier stability;
- 2025 independent confirmation;
- finite-opportunity conservation.

## 12. Integrity at freeze

- predictive candidates scored: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 outcomes used: **0**;
- Week-6 outcomes used: **0**;
- sportsbook inputs: **0**;
- OddsAPI calls: **0**;
- production changed: **false**.

Disposition at freeze:

`PRISM_STAGE2_PLAYER_MECHANISM_BENEFIT_HISTORY_V1_SOURCE_AUDIT_FROZEN`.
