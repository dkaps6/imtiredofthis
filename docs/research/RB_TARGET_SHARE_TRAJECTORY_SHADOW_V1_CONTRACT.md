# RB TARGET SHARE TRAJECTORY SHADOW V1 — WEEK-5 PROSPECTIVE CONTRACT

**STATUS: FROZEN BEFORE WEEK-5 OUTCOMES**

Date: 2026-10-07  
Branch: `research-rb-target-share-trajectory-shadow-v1`

## Purpose

Test a strictly prospective, player-specific RB receiving-opportunity state
overlay without reopening closed RB receiving efficiency/mean science.

The newly completed player-share decomposition and stage audit identify general
RB target-share allocation as an uncovered skill-position mechanism. They do
**not** authorize a retrospective fitted repair.

This shadow is justified independently by prior untouched evidence:

- `CURRENT_SEASON_STATE_PERSISTENCE_V1` (evaluation 2022-2025) found RB
  target-share current state replicates in 2025:
  - prior-only MAE: `0.04611`
  - current-only MAE: `0.04418`
  - Blend-4 MAE: `0.04268`
  - relative blend gain vs prior: **7.45%**
  - state-delta Spearman: **+0.3610**
- exact two-completed-game 2025 bucket:
  - prior MAE `0.04462`
  - current-only `0.04106`
  - Blend-4 `0.04170`

The formula below is inherited unchanged from the already-frozen
WR/TE Target Share Trajectory Shadow V1. No RB-specific coefficient, threshold,
window, or outcome-selected rule is fit.

## Scientific boundary

This is:
- a Week-5 pregame shadow only;
- receiving **opportunity allocation** only;
- no production change;
- no sportsbook input;
- no Week-5 outcome input;
- no retrospective W1-4 grading or fitting.

This is **not**:
- an RB receiving-yard mean change;
- a YPT/YPR/catch-rate change;
- a receiving-width change;
- an R23-R27D rescue;
- an M96 rushing reopen;
- a replacement for R26;
- a production promotion.

## Parent entitlement

Use the current protected player target entitlement after:

`M38 -> TE-R5P -> WR-R15`

For Week 5:
- R22 is not applicable;
- R26 is Week-1-only and not applicable;
- the new shadow operates only on RB/FB target entitlement.

Non-RB/FB entitlement must remain exact.

## Locked player cohort

Week-5 pregame football universe:
- positions RB / HB / FB normalized to RB-family;
- same public pregame input authority used by the existing WR/TE Week-5
  trajectory shadow;
- all cohort rows retained even when trajectory feature is unavailable.

Unavailable trajectory feature => exact no-change weight
(`trajectory_delta = 0`).

## Frozen trajectory feature

For each RB/FB player on the same current team:

Eligibility:
- at least four completed same-season team games before Week 5;
- stable player identity;
- strictly prior 2026 PBP only;
- no Week-5-or-later PBP or weekly stats.

State:
- `RECENT2` = team Weeks 3-4 (or latest two completed prior team games);
- `EARLIER` = all earlier same-season same-team games, requiring >=2.

For each window:

`player_target_share = player_targets / team_targets`

Then:

`trajectory_delta = recent2_share - earlier_share`

No clipping. No rising-only/falling-only route.

## Frozen no-fit transform

For each RB/FB player:

`weight_i = baseline_entitlement_i * exp(trajectory_delta_i)`

Then renormalize **inside the RB/FB room only**:

`shadow_i = RB_room_pool * weight_i / sum(room_weights)`

where:

`RB_room_pool = sum(baseline_entitlement_i for RB/FB on team)`

The exact RB/FB room target pool is preserved.

If the room has no positive weight or zero pool, retain a zero allocation under
the canonical fail-safe.

## Required conservation

Fail closed unless:

- RB/FB room pool gap <= `1e-12` for every room;
- team modeled target-mass gap <= `1e-12`;
- non-RB/FB entitlement max absolute delta <= `1e-12`;
- same/future feature violations = 0;
- Week-5 outcomes read = 0;
- sportsbook inputs used = 0;
- parameters fit = 0;
- production changed = false.

## Required lock outputs

1. `rb_target_share_trajectory_shadow_week5_lock.csv`
2. `rb_target_share_trajectory_shadow_week5_rooms.csv`
3. `rb_target_share_trajectory_shadow_week5_lock.json`

Row output must include:
- season / week / event / team / opponent
- player / player key / position
- stable receiver identity status
- recent2 share
- earlier share
- trajectory delta
- feature max week
- baseline RB entitlement
- trajectory weight
- shadow RB entitlement
- entitlement delta
- route / availability flag

## Prospective grading accumulation

Week 5 is lock #1.

No final PASS/FAIL before at least:
- 4 prospectively locked weeks;
- 200 scoreable RB/FB player-games;
- 70 distinct RB/FB identities;
- 80 scoreable team-RB/FB room-games.

Primary future grading:
- paired baseline vs shadow player target-share MAE;
- paired active-player target-share MAE;
- room-level allocation error;
- directional error reduction;
- downstream receptions / rec-yards diagnostic only, with receiving efficiency
  held fixed.

The formula must remain frozen through accumulation.

## No rescue

After Week-5 outcomes, do not try:
- different exponent coefficient;
- recent1 / recent3 / recent4;
- clipping;
- high-target-only application;
- rising-only application;
- RB1-only application;
- snap-conditioned multiplier;
- sportsbook conditioning;
- outcome-selected role thresholds.

Any materially different candidate requires a new independent scientific basis.

## Relationship to other shadows

- RB carry allocation remains governed by the separately frozen
  `RB_PLAYER_STATE_ALLOCATION_SHADOW_V1`; this receiving shadow does not alter
  carries.
- WR/TE Target Share Trajectory shadow remains unchanged.
- No shadow is production-active by virtue of this lock.
