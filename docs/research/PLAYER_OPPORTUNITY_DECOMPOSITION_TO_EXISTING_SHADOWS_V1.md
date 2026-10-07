# PLAYER OPPORTUNITY DECOMPOSITION TO EXISTING SHADOWS V1 — CROSSWALK

Date: 2026-10-07
Branch: `research-player-opportunity-decomposition-to-existing-shadows-v1`

## Purpose

Reconcile the successful 2026 Weeks 1-4 player opportunity decomposition against
already-frozen player-centric mechanisms before opening any new science.

Parent result:
`PLAYER_OPPORTUNITY_VOLUME_VS_SHARE_DECOMPOSITION_V1`

Disposition:
`QB_TEAM_VOLUME_DOMINANT__RB_WR_TE_PLAYER_SHARE_DOMINANT`

This document is a research-governance crosswalk only. It does not fit or
promote anything.

## 1. WR target share

Parent decomposition:
- player-share oracle removes **68.6%** of WR target MAE;
- team-volume oracle removes only **3.7%**.

Existing frozen mechanism:
- `PLAYER_TARGET_SHARE_TRAJECTORY_V1` historically CONFIRMED;
- exact rule requires four prior same-season team games;
- therefore 2026 Weeks 1-4 are legally ineligible;
- Week-5 prospective shadow already frozen.

Crosswalk:
`DIRECT_MECHANISM_MATCH__ALREADY_COVERED_PROSPECTIVELY`

The decomposition identifies the same layer that the target-share trajectory
shadow is designed to attack: individual receiving entitlement/share, not team
dropback volume.

Decision:
- do not create a second WR share model;
- do not weaken the four-prior-game rule;
- preserve Week-5+ prospective trajectory evaluation as the authorized WR share
  lane.

## 2. TE target share

Parent decomposition:
- player-share oracle removes **77.1%** of TE target MAE;
- team-volume oracle removes only **0.5%**.

Existing frozen mechanism:
- same `PLAYER_TARGET_SHARE_TRAJECTORY_V1` authority as WR;
- Week-5 prospective shadow already frozen.

Crosswalk:
`DIRECT_MECHANISM_MATCH__ALREADY_COVERED_PROSPECTIVELY`

Decision:
- no second TE share model;
- no retrospective W1-4 substitute;
- evaluate the already-frozen Week-5+ trajectory shadow.

## 3. RB carry share

Parent decomposition:
- player-share oracle removes **64.5%** of RB carry MAE;
- team-volume oracle removes only **4.7%**.

Existing frozen mechanism:
- `RB_PLAYER_STATE_ALLOCATION_SHADOW_V1`;
- Week-5 pregame shadow already frozen;
- strict-prior individual carry share / snap participation drive room allocation;
- prospective sample requirement remains binding.

Crosswalk:
`DIRECT_MECHANISM_MATCH__ALREADY_COVERED_PROSPECTIVELY`

Decision:
- do not back-apply the Week-5 shadow to Weeks 1-4;
- do not reopen M96 retrospective RB router/threshold/feature research;
- carry-share evidence strengthens the rationale for the existing prospective RB
  shadow but does not authorize a new carry allocator.

## 4. RB target share

Parent decomposition:
- player-share oracle removes **76.5%** of RB target MAE;
- team-volume oracle removes only **2.8%**.

Current known player-state inventory:
- live RB player state already carries strict-prior individual receiving history,
  including recent target-history fields;
- current production/shadow lineage was frozen primarily around RB room
  carry/snap allocation;
- no separately validated RB receiving target-share allocator has been frozen.

Crosswalk:
`PARTIALLY_COVERED_STATE__EXPLICIT_RB_TARGET_SHARE_MECHANISM_UNRESOLVED`

This is the only RB/WR/TE share family from the decomposition that is not already
cleanly mapped to a frozen prospective mechanism.

Decision:
authorize a bounded **coverage/consumption audit**, not a fitted model:

`RB_RECEIVING_SHARE_STATE_COVERAGE_V1`

Required questions:
1. What strict-prior RB target-share / target-count state already exists in
   PlayerForm and the live player-state artifacts?
2. Which of those fields are actually consumed by the canonical target
   entitlement allocator for RBs?
3. Are RB target shares currently inherited from generic M38/receiving-share
   logic, historical priors, or a dedicated RB rule?
4. Is recent RB receiving role present but unused, or already consumed and
   overly shrunk?
5. Can the uncovered mechanism be isolated without reopening R23-R27D receiving
   mean science or M96?

No candidate transform may be fit until this audit is frozen and reviewed.

## 5. QB pass-attempt volume

Parent decomposition:
- actual team pass volume removes **60.3%** of QB attempt MAE;
- actual QB share removes **24.7%**;
- in 41+ attempt games, actual team volume reduces bias from about **-17.6** to
  **-0.6** attempts.

Existing QB anti-reinvention ledger:
- M40-M42 / M16-M21 team pass-rate and game-script repackaging:
  `FULL_STACK_TESTED_CLOSED`;
- M63 directional high/low attempt surprise:
  `SIGNAL_SCREEN_FAILED`;
- M64-M65 possession/dropback generative state:
  `FULL_STACK_TESTED_CLOSED`;
- M73 attempt-opportunity oracle recoverability:
  `FULL_STACK_TESTED_CLOSED` as a deployable mechanism, while confirming
  structural headroom;
- M67-M69 opening/playcaller/tendency family:
  failed to become actionable;
- same-data model-zoo rescue is prohibited.

Later shared-opportunity research localized the residual to week-specific
within-state / first-down play choice and likely pregame game-plan intent.
The only legitimate reopen condition is materially new pregame intent/opportunity
information, not another pass-rate retune.

Crosswalk:
`KNOWN_STRUCTURAL_HEADROOM__SAME_DATA_QB_LANE_REMAINS_CLOSED`

Decision:
- do not reopen generic QB pass-volume science from the W1-4 decomposition;
- do not retune the 57/43 architecture;
- do not reopen M63/M64/M65;
- only a genuinely new pregame intent observable may reopen QB volume work;
- manual public-intent archaeology remains paused; automation-first only under
  the prior hard time cap if revisited.

## Global decision

The decomposition does **not** justify four new models.

It maps as follows:

| Family | Decomposition result | Existing coverage | Decision |
|---|---|---|---|
| WR target share | player-share dominant | Week-5 target-share trajectory shadow | evaluate existing shadow |
| TE target share | player-share dominant | Week-5 target-share trajectory shadow | evaluate existing shadow |
| RB carry share | player-share dominant | Week-5 RB player-state allocation shadow | evaluate existing shadow |
| RB target share | player-share dominant | partial state only; no explicit validated target-share allocator | audit current coverage/consumption |
| QB pass attempts | team-volume dominant | same-data volume families already exhausted/closed | no reopen absent new information |

## Immediate next action

Proceed with:

`RB_RECEIVING_SHARE_STATE_COVERAGE_V1`

Audit-only first.

No paid OddsAPI.
No sportsbook inputs.
No production promotion.
No retrospective Week-5 shadow application.
No threshold fitting.
No reopening closed RB receiving-yard mean science.
