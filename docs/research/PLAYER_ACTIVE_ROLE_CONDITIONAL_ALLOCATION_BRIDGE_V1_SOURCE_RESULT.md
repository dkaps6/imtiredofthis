# Player Active-Role -> Conditional Allocation Bridge V1 — Source Result

Date: 2026-10-16

Project name: **PRISM — Player Role & Individual Situation Model**

Status: **SOURCE READY PARTIAL / FOUR CELLS ADVANCE / WR RECEIVING BLOCKED**

Version:
`PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_V1_SOURCE_AUDIT`

Frozen source contract:
`docs/research/PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_V1_SOURCE_PLAN.md`

Scientific parent:
`PLAYER_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED_ALL`

## Canonical authority

- workflow run: `38955368643` — **SUCCESS**
- job: `117156780470`
- exact source-audit head: `cb65a634a0e09296bfeabc94ffe1e77363a30f8a`
- artifact: `12070806828`
- artifact digest: `sha256:e91a108495d7a2b6eb289e0618ab6af09e3d6cdf6895e534742a07340e4a2a83`
- strict repository audit: **SUCCESS**

This supersedes the earlier blocked source-audit attempts. Those earlier results
were implementation/lineage-reconciliation states, not scientific failures and
not authorization to weaken any frozen source gate.

## Overall disposition

`PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_SOURCE_READY_PARTIAL`

Source-ready cells:
- `QB_PASSING_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_SOURCE_READY`
- `RB_RUSHING_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_SOURCE_READY`
- `RB_RECEIVING_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_SOURCE_READY`
- `TE_RECEIVING_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_SOURCE_READY`

Blocked:
- `WR_RECEIVING_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_SOURCE_BLOCKED`

The WR block is exact and narrow:
- 2024 WR receiving source is READY;
- 2025 WR receiving is BLOCKED because exact WR-R15 OOS authority does not
  exist for 2025 under the frozen chronology contract;
- the WR-R15 final fit overlaps the 2025 training period and may not be
  relabeled OOS;
- do not rescue this cell with a new share mean, final-fit replay, weaker gate,
  alternate subset, or sportsbook information.

## Important Stage-1 fallback reconciliation

The qualified Stage-1 mechanism freezes two legal participation states:

1. named-player `p_candidate` when the player has the required same-team
   strictly-prior support;
2. the frozen strictly-prior Jeffreys position baseline as the explicit
   unsupported-player fallback.

The source audit therefore uses:

`p_active_role = p_candidate when supported, otherwise p_baseline`.

This is not a new model or post-result rescue. It is the deployment fallback
explicitly required in
`PLAYER_ACTIVE_ROLE_PARTICIPATION_V1_MECHANISM_FREEZE.md`.

The final exact-head rerun reconstructs that fallback from the **full earlier
eligible chronology**, so Weeks 1-4 correctly contribute prior evidence for
Week 5. Stage-1 probability attach is therefore 100% for every source-ready
cell/season.

## Specialist overlay semantics

WR-R15 and TE-R5P are specialist **overlays**, not player-universe filters.

Where an exact legal OOS specialist row exists, it replaces the canonical base
entitlement for that row. Players not touched by the specialist retain their
canonical base entitlement. The complete finite room remains represented and
the residual bucket remains explicit.

No specialist absence is reinterpreted as zero entitlement.

## QB authority reconciliation

QB is source-ready; it is not source-blocked.

The repository already contains a leakage-safe per-QB passing-opportunity
authority:
- historical pregame primary-QB selection;
- strictly-prior `qb_pass_att_share`;
- finite team passing/dropback opportunity denominator;
- explicit backup zero-share semantics;
- no target-week participation used to select the passer.

No new QB share model was invented in this audit.

## Exact 2024 source metrics

### QB passing
- target player-games: **838**
- authority player-games: **837**
- authority players: **85**
- scheduled team-game coverage: **100%**
- stable identity: **100%**
- Stage-1 probability attach: **100%**
- canonical entitlement coverage: **99.8807%**
- finite denominator coverage: **100%**
- conservation pass rate: **100%**
- ambiguous identity: **0%**
- chronology violations: **0**
- disposition: READY

### RB rushing
- target player-games: **1,400**
- authority player-games: **1,386**
- authority players: **156**
- scheduled team-game coverage: **100%**
- stable identity: **100%**
- Stage-1 probability attach: **100%**
- canonical entitlement coverage: **99.0000%**
- finite denominator coverage: **100%**
- conservation pass rate: **100%**
- ambiguous identity: **0%**
- chronology violations: **0**
- disposition: READY

### RB receiving
- target player-games: **1,400**
- authority player-games: **1,386**
- authority players: **156**
- scheduled team-game coverage: **100%**
- stable identity: **100%**
- Stage-1 probability attach: **100%**
- canonical entitlement coverage: **99.0000%**
- finite denominator coverage: **100%**
- conservation pass rate: **100%**
- ambiguous identity: **0%**
- chronology violations: **0**
- disposition: READY

### WR receiving
- target player-games: **2,151**
- authority player-games: **2,101**
- authority players: **240**
- scheduled team-game coverage: **100%**
- stable identity: **100%**
- Stage-1 probability attach: **100%**
- canonical entitlement coverage: **97.6755%**
- finite denominator coverage: **100%**
- conservation pass rate: **100%**
- ambiguous identity: **0%**
- chronology violations: **0**
- disposition: READY for 2024 only

### TE receiving
- target player-games: **1,334**
- authority player-games: **1,334**
- authority players: **138**
- scheduled team-game coverage: **100%**
- stable identity: **100%**
- Stage-1 probability attach: **100%**
- canonical entitlement coverage: **100%**
- finite denominator coverage: **100%**
- conservation pass rate: **100%**
- ambiguous identity: **0%**
- chronology violations: **0**
- disposition: READY

## Exact 2025 source metrics

### QB passing
- target player-games: **848**
- authority player-games: **835**
- authority players: **94**
- scheduled team-game coverage: **100%**
- stable identity: **100%**
- Stage-1 probability attach: **100%**
- canonical entitlement coverage: **98.4670%**
- finite denominator coverage: **100%**
- conservation pass rate: **100%**
- ambiguous identity: **0%**
- chronology violations: **0**
- disposition: READY

### RB rushing
- target player-games: **1,431**
- authority player-games: **1,419**
- authority players: **154**
- scheduled team-game coverage: **100%**
- stable identity: **100%**
- Stage-1 probability attach: **100%**
- canonical entitlement coverage: **99.1614%**
- finite denominator coverage: **100%**
- conservation pass rate: **100%**
- ambiguous identity: **0%**
- chronology violations: **0**
- disposition: READY

### RB receiving
- target player-games: **1,431**
- authority player-games: **1,419**
- authority players: **154**
- scheduled team-game coverage: **100%**
- stable identity: **100%**
- Stage-1 probability attach: **100%**
- canonical entitlement coverage: **99.1614%**
- finite denominator coverage: **100%**
- conservation pass rate: **100%**
- ambiguous identity: **0%**
- chronology violations: **0**
- disposition: READY

### WR receiving
- target player-games: **2,139**
- target players: **247**
- exact legal 2025 WR-R15 OOS authority: **NONE**
- disposition:
  `WR_RECEIVING_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_SOURCE_BLOCKED`
- frozen blocker:
  `WR_R15_2025_OOS_AUTHORITY_FORBIDDEN_BY_FROZEN_CONTRACT__FINAL_FIT_TRAINING_OVERLAP_CANNOT_BE_RELABELED_OOS`

### TE receiving
- target player-games: **1,312**
- authority player-games: **1,305**
- authority players: **141**
- scheduled team-game coverage: **100%**
- stable identity: **100%**
- Stage-1 probability attach: **100%**
- canonical entitlement coverage: **99.4665%**
- finite denominator coverage: **100%**
- conservation pass rate: **100%**
- ambiguous identity: **0%**
- chronology violations: **0**
- disposition: READY

## Integrity

Canonical final run:
- historical specialist reconstruction:
  `EXACT_OOS_FOLD_COEFFICIENTS_STRICT_PRIOR`
- target-game outcomes used as predictors: **0**
- sportsbook inputs: **0**
- OddsAPI calls: **0**
- 2026 outcomes: **0**
- new share models fit: **0**
- production changed: **false**
- Stage-1 chronology violations: **0**
- ambiguous identity in ready cells: **0**
- WR 2025 OOS authority used: **false**
- invented QB share model: **false**

## Scientific meaning

PRISM now has legal historical source lineage for the first four Stage-2
active-role bridges:
- QB passing opportunity;
- RB rushing opportunity;
- RB receiving opportunity;
- TE receiving opportunity.

This result does **not** show that conditioning allocation on Stage-1 active-role
probability improves projection quality. Source readiness is not mechanism
qualification.

It only establishes that those four cells can now be tested without replacing
the existing allocation science or contaminating the chronology.

WR receiving remains outside the scored bridge until a genuinely legal 2025 OOS
WR entitlement authority exists.

## Next authorized step

For the four READY cells only, freeze a separate Stage-2 bridge mechanism
**before scoring any bridge outcome**.

That freeze must define:
- exact `p_active_role` construction, including the frozen baseline fallback;
- active-set sampling law;
- random-number isolation / reproducibility;
- conditioning and renormalization of existing entitlement only;
- residual/other-player handling;
- all-off / degenerate-room fallback;
- exact finite-opportunity conservation;
- evaluation metrics;
- player-, team/game- and equal-player dependence-aware inference;
- support and qualification gates;
- anti-rescue rules.

The bridge may not:
- fit a new opportunity-share mean;
- change team opportunity volume;
- retune QB/RB/TE entitlement formulas;
- use realized target-game participation as a pregame input;
- use sportsbook information;
- call OddsAPI;
- read 2026 outcomes;
- mutate production.

WR receiving stays fail-closed and may not enter the bridge scoring run.

Canonical project status remains:
`PLAYER_INDIVIDUALIZATION_PARTIAL`.
