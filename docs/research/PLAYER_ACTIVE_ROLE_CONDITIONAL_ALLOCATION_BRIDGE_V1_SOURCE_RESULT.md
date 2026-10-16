# Player Active-Role -> Conditional Allocation Bridge V1 — Source Result

Date: 2026-10-16

Status: **SOURCE READY PARTIAL — QB PASSING, RB RUSHING, RB RECEIVING, TE RECEIVING READY; WR RECEIVING BLOCKED. NO BRIDGE OUTCOME HAS BEEN SCORED.**

Version:
`PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_V1_SOURCE_AUDIT`

Frozen parent:
`docs/research/PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_V1_SOURCE_PLAN.md`

PRISM architecture:
`PLAYER IDENTITY -> CURRENT ROLE/STATE -> TEAMMATE COMPETITION -> OPPORTUNITY ALLOCATION -> OPPORTUNITY QUALITY -> OPPONENT/ENVIRONMENT -> EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`

This is a source/lineage result only. It does not qualify a scored Stage-2 bridge and does not mutate production.

## Canonical authority

Corrected chronology-preserving source audit:
- workflow run: `38955368643` — SUCCESS
- exact head: `cb65a634a0e09296bfeabc94ffe1e77363a30f8a`
- artifact: `12070806828`
- artifact digest: `sha256:e91a108495d7a2b6eb289e0618ab6af09e3d6cdf6895e534742a07340e4a2a83`

The corrected head reconstructs the Stage-1 fallback from the same full-season eligible chronology as the qualified Stage-1 mechanism. Weeks 1-4 remain strictly-prior evidence for Week 5+ bridge rows.

Overall disposition:

`PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_SOURCE_READY_PARTIAL`

Ready cells:
- `QB_PASSING_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_SOURCE_READY`
- `RB_RUSHING_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_SOURCE_READY`
- `RB_RECEIVING_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_SOURCE_READY`
- `TE_RECEIVING_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_SOURCE_READY`

Blocked cell:
- `WR_RECEIVING_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_SOURCE_BLOCKED`

## Source gate result

All source-ready cells pass the frozen gates in both 2024 and 2025:
- scheduled team-game coverage >= 99%;
- stable identity coverage >= 99%;
- Stage-1 history attach >= 95%;
- canonical pregame entitlement coverage >= 95%;
- finite denominator coverage >= 99%;
- conservation >= 99%;
- ambiguous identity <= 0.5%;
- chronology violations = 0;
- frozen support floors.

### QB passing

2024:
- authority player-games: **837**
- players: **85**
- scheduled team-game coverage: **100%**
- Stage-1 history attach: **100%**
- canonical entitlement coverage: **99.8807%**
- finite denominator coverage: **100%**
- conservation: **100%**

2025:
- authority player-games: **835**
- players: **94**
- scheduled team-game coverage: **100%**
- Stage-1 history attach: **100%**
- canonical entitlement coverage: **98.4670%**
- finite denominator coverage: **100%**
- conservation: **100%**

The audit recovered the existing pregame QB passing-opportunity authority; it did not fit a new QB share model.

### RB rushing

2024:
- authority player-games: **1,386**
- players: **156**
- scheduled team-game coverage: **100%**
- Stage-1 history attach: **100%**
- canonical entitlement coverage: **99.0000%**
- finite denominator coverage: **100%**
- conservation: **100%**

2025:
- authority player-games: **1,419**
- players: **154**
- scheduled team-game coverage: **100%**
- Stage-1 history attach: **100%**
- canonical entitlement coverage: **99.1614%**
- finite denominator coverage: **100%**
- conservation: **100%**

The canonical top-five rushing allocation and residual semantics remain unchanged.

### RB receiving

2024:
- authority player-games: **1,386**
- players: **156**
- scheduled team-game coverage: **100%**
- Stage-1 history attach: **100%**
- canonical entitlement coverage: **99.0000%**
- finite denominator coverage: **100%**
- conservation: **100%**

2025:
- authority player-games: **1,419**
- players: **154**
- scheduled team-game coverage: **100%**
- Stage-1 history attach: **100%**
- canonical entitlement coverage: **99.1614%**
- finite denominator coverage: **100%**
- conservation: **100%**

No replacement RB receiving proxy was fitted. The source audit uses the canonical historical target entitlement already present in the reconstructed pregame model state.

### TE receiving

2024:
- authority player-games: **1,334**
- players: **138**
- scheduled team-game coverage: **100%**
- Stage-1 history attach: **100%**
- canonical entitlement coverage: **100%**
- finite denominator coverage: **100%**
- conservation: **100%**

2025:
- authority player-games: **1,305**
- players: **141**
- scheduled team-game coverage: **100%**
- Stage-1 history attach: **100%**
- canonical entitlement coverage: **99.4665%**
- finite denominator coverage: **100%**
- conservation: **100%**

The TE authority is reconstructed from the exact recovered OOS TE-R5P fold coefficients using strictly-prior participation semantics.

## WR receiving — blocked, not scientifically failed

2024 itself is source-ready:
- authority player-games: **2,101**
- players: **240**
- scheduled team-game coverage: **100%**
- Stage-1 history attach: **100%**
- canonical entitlement coverage: **97.6755%**
- finite denominator coverage: **100%**
- conservation: **100%**

2025 is intentionally blocked:

`WR_R15_2025_OOS_AUTHORITY_FORBIDDEN_BY_FROZEN_CONTRACT__FINAL_FIT_TRAINING_OVERLAP_CANNOT_BE_RELABELED_OOS`

Therefore WR does not advance to the scored V1 bridge. A good 2024 result may not rescue the absent 2025 OOS authority.

## Integrity

Canonical audit reports:
- Stage-1 chronology violations: **0**
- target-game outcomes used as predictors: **0**
- new share models fit: **0**
- sportsbook inputs used: **0**
- OddsAPI calls: **0**
- 2026 outcomes used: **0**
- production changed: **false**
- WR 2025 OOS authority used: **false**
- QB share model invented: **false**
- historical specialist reconstruction: `EXACT_OOS_FOLD_COEFFICIENTS_STRICT_PRIOR`
- participation-snap duplicate rate: **0.0**

## Interpretation

The missing source seam is now resolved for four PRISM opportunity cells.

This does **not** mean those Stage-2 bridges work. It means they can now be tested honestly.

The next legal action is to freeze, before scoring:
1. the active-set probability law from qualified Stage-1 `p_active`;
2. exact conditioning of the existing canonical entitlement on that active set;
3. complement/residual and degenerate all-off behavior;
4. finite-opportunity conservation;
5. individual-player and room-level proper scoring;
6. support and clustered inference;
7. anti-rescue rules.

Only the four source-ready cells may enter that scored bridge.

WR receiving remains outside V1 until a legitimate 2025 OOS canonical authority exists.

No sportsbook.
No OddsAPI.
No 2026 outcomes.
No production mutation.
