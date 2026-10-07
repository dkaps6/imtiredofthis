# RB TARGET SHARE TRAJECTORY SHADOW V1 — WEEK-5 PREGAME LOCK

**STATUS: IMMUTABLY FROZEN BEFORE WEEK-5 OUTCOMES**

Frozen contract:
`docs/research/RB_TARGET_SHARE_TRAJECTORY_SHADOW_V1_CONTRACT.md`

## Certified authority

- branch: `research-rb-target-share-trajectory-shadow-v1`
- lock run: `37696979325` — **SUCCESS**
- job: `113051008665`
- exact run head: `afc6549b773c5210873ad5ae44179c38ce6619d9`
- artifact: `11515727766`
- artifact digest:
  `sha256:b40d2f2faeae4501dd5910fc3631389995baefc9d3f274aa0f6fac7450c45c3f`
- canonical row CSV digest:
  `sha256:23ed4317a9e5acb12207a1f6cb67947f71a6fed50d915c711b08b08a1276d94e`

## Lock status

`WEEK5_PREGAME_RB_TARGET_SHARE_LOCK_FROZEN`

Boundary:
- season: 2026
- week: 5
- Week-5 outcomes read: **0**
- sportsbook inputs used: **0**
- fitted parameters: **0**
- production changed: **false**
- same/future feature violations: **0**
- R22 applicable Week 5: **false**
- R26 applicable Week 5: **false**

## Locked RB/FB universe

- locked players: **108**
- locked teams / RB rooms: **30**
- stable-ID coverage: **84.2593%**
- trajectory available: **91 / 108 = 84.2593%**
- unavailable trajectory rows remain in the lock under the frozen no-fit
  `trajectory_delta=0` fail-safe.

## Frozen transformation

Strict-prior state:

`trajectory_delta = RECENT2 target share - EARLIER target share`

with:
- at least four completed same-season team games;
- latest two completed games in RECENT2;
- at least two earlier completed team games;
- player target share measured from strict-prior PBP only.

No fitted coefficient:

`weight_i = baseline_RB_entitlement_i * exp(trajectory_delta_i)`

Then re-normalize inside the exact RB/FB receiving room only:

`shadow_i = RB_room_pool * weight_i / sum(room_weights)`

Parent entitlement:

`M38 -> TE-R5P -> WR-R15`

The shadow changes neither team target mass nor receiving efficiency.

## Pregame materiality

- changed players: **108**
- changed RB/FB rooms: **30**
- median absolute player entitlement change:
  **0.0009433874**
  - approximately **0.094 target-share percentage points**
- maximum absolute player entitlement change:
  **0.0111904106**
  - approximately **1.119 target-share percentage points**

This is intentionally conservative. It is a room-preserving player-state
redistribution, not a new team-volume or efficiency model.

## Conservation / integrity

- max RB/FB room pool gap:
  `2.7755575615628914e-17`
- max team modeled target-mass gap:
  `1.1102230246251565e-16`
- max non-RB entitlement delta:
  `0.0`
- TE parent model:
  `TE_R5P_PRODUCTION_MODEL_V1`
- WR parent model:
  `WR_R15_PRODUCTION_MODEL_V1`

All frozen safeguards passed.

Therefore the lock does **not**:
- change RB carries;
- change team pass/dropback volume;
- change total modeled team target mass;
- change WR/TE/QB entitlement;
- change YPT/YPR/catch rate;
- change R22/R26;
- change production.

## Scientific authority

This prospective test was not created from a W1-4 high-target cutoff.

Independent pre-2026 authority:
`CURRENT_SEASON_STATE_PERSISTENCE_V1`

2025 RB target-share replication:
- prior-only MAE: `0.04611`
- current-only MAE: `0.04418`
- Blend-4 MAE: `0.04268`
- relative blend gain vs prior: **7.45%**
- state-delta Spearman: **+0.3610**

The new shadow inherits the already-frozen WR/TE trajectory mathematics rather
than fitting an RB-specific exponent or window.

## Prospective accumulation

Week 5 is prospective lock **#1**.

No final promotion PASS/FAIL is permitted until at least:
- 4 prospectively locked weeks;
- 200 scoreable RB/FB player-games;
- 70 distinct RB/FB identities;
- 80 scoreable team-RB/FB room-games.

Grade:
- paired baseline vs shadow player target-share MAE;
- paired active-player target-share MAE;
- room-level allocation error;
- directional error reduction;
- receptions / rec-yards downstream only as diagnostics with efficiency held
  fixed.

## Binding no-rescue rule

After Week-5 outcomes do not try:
- exponent tuning;
- recent1/recent3/recent4;
- clipping;
- high-target-only routing;
- rising-only routing;
- RB1-only routing;
- snap-conditioned multiplier;
- sportsbook conditioning;
- outcome-selected thresholds.

The formula remains fixed through prospective accumulation.

## Relationship to the player-centric stack

The current individual-player prospective opportunity stack is now:

- **QB:** protected production M89/M90/C2; no new generic attempt-volume retune.
- **RB rushing allocation:** existing
  `RB_PLAYER_STATE_ALLOCATION_SHADOW_V1` Week-5 prospective lock.
- **RB receiving target share:** this newly frozen Week-5 shadow.
- **WR target share:** existing Target Share Trajectory Week-5 shadow.
- **TE target share:** existing Target Share Trajectory Week-5 shadow.

The failed pooled W1-4 symmetric target-depth distribution transform remains
unpromoted and is not justified as part of a combined player-centric candidate.

No production change is authorized by this lock.
