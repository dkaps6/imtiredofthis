# PLAYER-CENTRIC WEEK-5 SHADOW MANIFEST V1 — FROZEN CONTRACT

**STATUS: FROZEN BEFORE WEEK-5 OUTCOMES**

Date: 2026-10-07  
Branch: `research-player-centric-week5-shadow-manifest-v1`

## Purpose

Create one immutable, outcome-blind Week-5 manifest of the currently authorized
**individual-player prospective state** across QB/RB/WR/TE.

This is a composition/audit artifact, not a new model and not a production
promotion.

The manifest prevents three independently frozen player-state shadows from
drifting, overlapping incorrectly, or being interpreted as position-level
adjustments.

## Immutable component authorities

### RB rushing allocation
- family: `RB_PLAYER_STATE_ALLOCATION_SHADOW_V1`
- run: `37560824479`
- artifact: `11456916566`
- artifact digest:
  `sha256:edf51bbd93920ef0af580a0396422af3288cf511785be59deeea95c117062af4`
- canonical row digest:
  `sha256:66a428f0c19ee1dc356db117fa8e39092204896356461358667224826826e90a`

### WR/TE target-share trajectory
- family: `PLAYER_TARGET_SHARE_TRAJECTORY_SHADOW_V1`
- run: `37654382316`
- artifact: `11497153776`
- artifact digest:
  `sha256:c12878ed97a4543a907ac54f4cbadadda7178db068491373091d2c048f176e88`
- canonical row digest:
  `sha256:afbfd7f360c50fcd4850c0967be40f9a333da1bd5f835cdc676c2e88d777c1f3`

### RB receiving target-share trajectory
- family: `RB_TARGET_SHARE_TRAJECTORY_SHADOW_V1`
- run: `37696979325`
- artifact: `11515727766`
- artifact digest:
  `sha256:b40d2f2faeae4501dd5910fc3631389995baefc9d3f274aa0f6fac7450c45c3f`
- canonical row digest:
  `sha256:23ed4317a9e5acb12207a1f6cb67947f71a6fed50d915c711b08b08a1276d94e`

These exact artifacts are the only allowed shadow inputs.

## Week-5 football universe

Build the same public leakage-safe Week-5 pregame universe used by the
WR/TE and RB receiving trajectory locks.

Supported player positions:
- QB
- RB / HB / FB / TB
- WR / LWR / RWR / SWR
- TE

This manifest is a research universe, not a claim that it reproduces the
richer live production availability stack. Do not use it to reopen availability
science.

## Player-level composition

### QB

No new player shadow.

Every QB row is labeled:

`PROTECTED_PRODUCTION_QB_NO_NEW_PLAYER_SHADOW`

The M89/M90/C2 production authority remains untouched. The replay-confirmed team
pass-volume issue is a closed generic lane absent genuinely new pregame intent
information.

### RB rushing

Where an exact Week-5 RB rushing lock identity exists, carry state is attached:

- control recent carry share
- frozen player-state rushing share
- carry share delta

The manifest **does not** convert this room share into a production rushing-yard
projection. That integration remains outside the parent shadow's authority.

Players not in the frozen rushing lock remain explicit with:
`rb_rush_shadow_available = false`.

### RB receiving

Attach the exact locked:
- baseline target entitlement
- shadow target entitlement
- entitlement delta
- trajectory availability/state

No recalculation or refit.

### WR/TE receiving

Attach the exact locked:
- protected parent baseline target entitlement
- shadow target entitlement
- entitlement delta
- trajectory availability/state

No recalculation or refit.

## Identity rules

Preferred joins:
- target-share artifacts: exact
  `season/week/event_id/team/player_clean_key`
- RB rushing artifact:
  - exact GSIS ID where the manifest can resolve one from strict-prior identity;
  - otherwise exact team + normalized canonical player identity;
  - ambiguous matches fail closed.

No fuzzy matching.

## Required outputs

1. `player_centric_week5_shadow_manifest.csv`
2. `player_centric_week5_shadow_manifest_summary.json`
3. `player_centric_week5_shadow_conflict_audit.csv`

Each row must identify:
- season / week / event / team / opponent
- player / player_clean_key / normalized position family
- player-state route(s)
- target baseline / target shadow / target delta where applicable
- RB carry control / carry shadow / carry delta where applicable
- component source run / row digest
- whether each shadow is available
- production-changed = false

## Conflict/integrity requirements

Fail closed unless:

- every WR/TE locked row maps to exactly one same-universe player;
- every RB target-share locked row maps to exactly one same-universe player;
- every RB rushing locked row maps to at most one same-universe RB/FB player;
- no QB receives a shadow target/carry modification;
- no WR/TE receives an RB shadow;
- no RB receives a WR/TE shadow;
- target locks do not overlap position families;
- all target baseline/shadow values are finite and nonnegative;
- all RB carry control/shadow shares are finite and in [0,1];
- per-team RB carry control and shadow sums equal 1 within `1e-12` over the
  locked rushing cohort;
- no Week-5 outcome field is read;
- no sportsbook field is read;
- parameters fit = 0;
- production changed = false.

## Explicit exclusions

The failed W1-4 pooled symmetric target-depth distribution transform is **not**
part of this combined player-centric manifest.

No new QB volume candidate is part of this manifest.

No raw-history/Bayes replacement is authorized.

## Scientific purpose

The manifest is the immutable Week-5 player-centric scoreboard key.

Future grading can evaluate each previously frozen mechanism separately and,
only where scientifically authorized, evaluate whether their combined
player-state direction is coherent.

The manifest itself can never constitute evidence for production promotion.
