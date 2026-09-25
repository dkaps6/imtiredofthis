# Active-Roster Receiver Room State V1 — Frozen Diagnostic Plan

Date: 2026-09-25

Status: **FROZEN BEFORE DIAGNOSTIC COMPUTE**

This is diagnostic-only. It does not qualify a production change.

## 1. Why this exists

The current receiver research now has a coherent chain:

1. strict-prior targetable-dropback rate improves team receiver-target volume in
   2022, 2023, 2024 and 2025;
2. uniform translation of that better team pool through the current player
   entitlement stack fails full-stack protection;
3. the current-stack compensation audit shows:
   - WR room mass is underallocated;
   - TE and RB/FB room mass are overallocated;
   - WR2+ target MAE improves under the smaller targetable pool;
   - WR1 is already underallocated and gets worse;
4. WR1 Current-State Anchor Diagnostic V1 shows:
   - the existing validated blend-4 current-season state materially improves
     WR1 absolute team target-share prediction;
   - but normalizing that state inside a fixed WR room is worse.

The missing opportunity therefore appears to cross the **position-room
boundary**.

Closed C1 does not settle this question because C1 used recent **team-level
position-group history** plus empirical-Bayes shrinkage. It did not compose room
state from the identities and target-share states of the players actually in
the active pregame roster.

## 2. New question

Does an active-roster composition built from already-validated strict-prior
player target-share state improve WR / TE / RB_FB room composition relative to
the current explicit entitlement room masses?

This is a personnel/state question, not a new group-history window.

## 3. Anti-retest boundaries

Do not:
- reopen C1's last-8 position-group target-share calibration;
- change its prior target count or search nearby windows;
- retune M38 hierarchy multipliers;
- retune WR-R15 or TE-R5P;
- use targetable-dropback results to choose a threshold;
- use sportsbook inputs;
- inspect 2024-2025 in this diagnostic.

## 4. Discovery data only

Discovery seasons:
- 2022
- 2023

2024 and 2025 remain untouched and are reserved for a separately frozen
confirmation only if this diagnostic passes its predeclared rule.

For target week W >= 2:
- prior player state = season S-1 completed games only;
- current player state = season S games with week < W only;
- target-week/future information is prohibited upstream.

## 5. Baseline room authority

Reconstruct current explicit target entitlement through:
- historical pregame context;
- current rules/Bayesian path;
- M38;
- explicit finite target entitlement.

WR-R15 and TE-R5P are not needed to define room mass because both production
specialists conserve their existing WR/TE room mass exactly.

Rooms:
- WR
- TE
- RB_FB

Baseline room mass is the sum of explicit player entitlement in each room.

Baseline composition is:

`baseline_room_mass / sum(WR + TE + RB_FB room mass)`

## 6. Active-roster state authority

Use the already-validated Current-Season State Persistence V1 target-share
construction for each modeled active receiver:

- prior target share = prior-season cumulative player targets / cumulative team
  targets;
- current target share = completed-current-season cumulative player targets /
  cumulative team targets;
- `w_current = current_games / (current_games + 4)`;
- blend-4 target share =
  `(1-w_current)*prior_share + w_current*current_share`.

Identity must be recovered from strictly prior football history only.

### Missing-state fallback

For an active modeled receiver without a valid prior+current blend-4 state:
- retain that player's current explicit baseline entitlement.

This is a fail-safe fallback, not a fitted imputation.

For each player:
- `state_player_share = blend4` when eligible;
- otherwise `state_player_share = baseline explicit entitlement`.

No coefficient is fit.

## 7. State room construction

For each team-game:

`raw_state_room_mass = sum(state_player_share in room)`

Then normalize only for the composition diagnostic:

`state_room_composition =
 raw_state_room_mass / sum(raw_state_room_mass across WR,TE,RB_FB)`

This does not change any production mass.

## 8. Actual room quantities

Attach outcomes only after every pregame quantity is frozen.

For the modeled WR / TE / RB / FB receiver universe:
- actual room targets;
- actual modeled receiver targets;
- actual complete team targets.

Derive:

`actual_room_composition =
 actual_room_targets / actual_modeled_receiver_targets`

and:

`actual_room_team_share =
 actual_room_targets / actual_complete_team_targets`

## 9. Primary diagnostics

For each room, season and pooled:

### A. Composition share
Compare:
- baseline room composition;
- active-roster state room composition;
against actual room composition.

Report:
- MAE
- RMSE
- bias
- correlation
- p90 AE

### B. Oracle-volume room targets
To isolate composition from total volume, multiply each predicted composition by
the **actual modeled receiver target total** for that team-game.

Compare baseline vs state oracle room targets against actual room targets.

Report:
- MAE
- RMSE
- bias
- p90 AE

This is diagnostic attribution only; actual volume is never a candidate input.

### C. Absolute team target share
Compare:
- baseline room mass;
- raw state room mass;
against actual room targets / actual complete team targets.

This tells us whether player-state composition also carries useful absolute room
mass information.

## 10. State coverage

Report by room and pooled:
- player rows;
- rows using blend-4 state;
- row coverage;
- entitlement-weighted coverage;
- number of team-games with at least one fallback;
- number of team-games with full state coverage.

No coverage threshold is tuned.

## 11. WR1 interaction

Descriptive only:

For each team-game, retain the M38 WR1 identity and report:
- WR1 blend-4 minus M38 absolute share gap;
- state-room-mass minus baseline WR-room-mass gap;
- actual WR-room share error.

Report whether upward WR1 state gaps are associated with upward needed WR-room
mass corrections.

This does not authorize a WR1 carveout.

## 12. Frozen diagnostic rule

`ACTIVE_ROSTER_RECEIVER_ROOM_STATE_V1_SUPPORTED` requires all:

1. pooled macro composition-share MAE improves;
2. pooled WR composition-share MAE strictly improves;
3. pooled TE composition-share MAE is nonworse;
4. pooled RB_FB composition-share MAE is nonworse;
5. WR composition-share MAE improves separately in both 2022 and 2023;
6. pooled macro oracle-volume room-target MAE improves;
7. pooled WR oracle-volume room-target MAE improves;
8. no target-week/future state leakage;
9. sportsbook inputs = 0;
10. parameters fit = 0;
11. candidate variants scored = 0.

If any fail:
`ACTIVE_ROSTER_RECEIVER_ROOM_STATE_V1_NOT_SUPPORTED`

No rescue/search is authorized.

## 13. If supported

Freeze exactly one later candidate:

- preserve total modeled player target mass and residual bucket;
- use active-roster state only to set WR/TE/RB_FB **room composition**;
- preserve current relative player entitlement inside each room;
- preserve M38 / WR-R15 / TE-R5P relative ordering/redistribution semantics;
- no targetable-volume combination in the first candidate;
- score 2024-2025 once under a separate frozen confirmation;
- only after independent qualification may a combined targetable-volume +
  room-state architecture be considered.

Any historical pass still requires prospective 2026 confirmation before
production.

No production change is authorized by this plan.
