# INDIVIDUAL OPPORTUNITY — WEEK-6 PROSPECTIVE LOCK CONTINUATION V1

Frozen: 2026-10-11  
Branch: `research-individual-opportunity-roadmap-2026-10-08`

Status: **READINESS / CONTINUATION CONTRACT ONLY. DO NOT GRADE WEEK 5. DO NOT CREATE A WEEK-6 LOCK FROM INCOMPLETE PRIOR-WEEK STATE.**

Parent roadmap:
- `docs/research/INDIVIDUAL_OPPORTUNITY_ALLOCATION_ROADMAP_V1.md`
- scientific headline: `OPPORTUNITY_DOMINANT_ACROSS_ALL_PRIMARY_MARKETS__INDIVIDUAL_ROLE_SHARE_ALLOCATION_NEXT`

## Purpose

Preserve the exact already-frozen prospective opportunity mechanisms across
Week 6 without allowing Week-5 results, market information, or post-hoc tuning
to alter their formulas, support rules, room definitions, grading rules, or
acceptance gates.

This contract does **not** score Week 5 and does **not** build a Week-6 lock
tonight.

A valid Week-6 lock must use:
- a complete strictly-prior football state through Week 5;
- a legitimate current Week-6 pregame player/role universe;
- only source information available before the target Week-6 game;
- the exact mechanisms frozen before Week-5 outcomes.

## Scientific boundary

Week 5 remains prospective lock #1 for the existing share/allocation shadows.

Week 6 will be prospective lock #2 only after its own immutable pregame lock is
successfully created.

Nothing in this continuation contract authorizes:
- grading Week 5;
- changing a formula because of Week-5 performance;
- dropping poor Week-5 rows;
- adding Week-5-derived thresholds;
- sportsbook conditioning;
- production promotion.

## 1. RB PLAYER STATE ALLOCATION — CARRY / SNAP

Week-5 frozen authority:
- branch: `research-rb-player-state-allocation-shadow-v1`
- run: `37560824479`
- source SHA: `48f718b075b9f417145042c5cbb48ea6ba8beb2e`
- artifact: `11456916566`
- artifact digest:
  `sha256:edf51bbd93920ef0af580a0396422af3288cf511785be59deeea95c117062af4`
- canonical row digest:
  `sha256:66a428f0c19ee1dc356db117fa8e39092204896356461358667224826826e90a`

Frozen Week-6 formula:

For each legitimate current Week-6 RB room:

1. compute each player's strictly-prior recent RB-room carry share;
2. renormalize current room carry state to 1.0;
3. compute each player's strictly-prior recent RB-room offensive-snap fraction;
4. renormalize current room snap state to 1.0;
5. preserve the exact frozen equal-weight rule:

`shadow_share = 0.50 * carry_state + 0.50 * snap_state`

6. defensively renormalize the resulting current room to 1.0.

No coefficient fit.
No subgroup.
No threshold.
No realized Week-5 carry-based selector.

Protected:
- projected team rush volume;
- rushing efficiency/YPC;
- QB/WR rushing;
- receiving science;
- betting/pricing.

Existing prospective final support floor remains unchanged:
- >=4 distinct future locked weeks;
- >=80 scoreable team-games;
- >=200 scoreable RB player-games.

## 2. RB RECEIVING ROOM SHARE

Week-5 frozen authority:
- branch: `research-rb-receiving-room-share-week5-lock-v1`
- run: `37705464974`
- artifact: `11518604971`
- artifact digest:
  `sha256:2174af9aeb75100f29f9965f808be03d9d1d3d38d91366b539d7d25909cfe029`

Retrospective mechanism-impact authority:
- run: `37703522415`

Preserved retrospective W1-W4 effects:
- target MAE improvement: 7.60%;
- receptions MAE improvement: 3.83%;
- receiving-yards MAE improvement: 2.89%;
- rush+receiving-yards MAE improvement: 1.66%;
- rushing projections unchanged.

Frozen Week-6 rule:

For every legitimate current Week-6 RB room:
- attach each current frozen identity's exact-GSIS strictly-prior
  `prior_rb_room_share`;
- fail closed for any required frozen identity without genuine strictly-prior
  receiving-room history;
- normalize `prior_rb_room_share` within the exact current RB room;
- resulting normalized share is the candidate receiving-room allocation.

No fitted coefficient.
No fitted fallback.
No name-based fabrication.
No substitution from Week-5 realized receiving usage.

Protected:
- total RB+FB receiving target mass;
- R26 vacancy ordering;
- R22 tail logic;
- team pass volume;
- per-target receiving efficiency.

## 3. WR / TE TARGET SHARE TRAJECTORY

Week-5 frozen authority:
- branch: `research-player-target-share-trajectory-shadow-v1`
- corrected run: `37654382316`
- source SHA: `e2c57e17c1df95130b70d3a3926b090ccd26e847`
- artifact: `11497153776`
- artifact digest:
  `sha256:c12878ed97a4543a907ac54f4cbadadda7178db068491373091d2c048f176e88`
- canonical row digest:
  `sha256:afbfd7f360c50fcd4850c0967be40f9a333da1bd5f835cdc676c2e88d777c1f3`

Frozen trajectory eligibility:
- same season;
- same current target team;
- completed team games strictly before target week;
- >=4 prior same-team target-team games;
- RECENT2 = two most recent completed target-team games;
- EARLIER = all prior target-team games before RECENT2;
- EARLIER must contain >=2 games;
- unsupported player receives trajectory delta = 0 and remains in the room.

Frozen feature:

`trajectory_delta = RECENT2 target share - EARLIER target share`

Frozen transform:

`weight_i = baseline_entitlement_i * exp(trajectory_delta_i)`

Then normalize within the protected room.

TE:
- reweight full current TE room;
- preserve exact baseline TE-room target mass.

WR:
- M38 WR1 remains exactly frozen at baseline;
- reweight WR2+ only;
- preserve exact WR2+ pool;
- preserve exact full WR-room target mass.

Protected:
- team pass volume;
- M38 WR1;
- TE-R5P;
- WR-R15;
- receiving efficiency;
- QB science;
- distribution/simulation formulas.

Existing final prospective gate remains unchanged:
- >=4 future locked weeks;
- >=400 scoreable WR/TE player-games;
- >=120 distinct WR/TE identities;
- >=80 scoreable team-position rooms;
- pooled target-share MAE improves;
- pooled RMSE non-worse;
- WR pooled MAE improves;
- TE pooled MAE improves;
- >=3 of first 4 qualified weeks non-worse;
- room-rank correlation non-worse;
- player-cluster bootstrap P(MAE improvement > 0) >=0.80;
- zero chronology/conservation/sportsbook violations.

## Week-6 lock prerequisites

A Week-6 lock for any lane may be created only after the following are true.

### A. Complete strictly-prior football state

The historical/statistical source used for target-week features must contain the
complete Week-5 prior-game state for every Week-6 target team whose Week-5 game
has been completed.

Do not create a mixed Week-6 lock where some teams include a completed Week-5
game and another target team's still-pending Week-5 game is silently treated as
missing history.

If a prior game is not complete / source data are not yet final, that team's
Week-6 lock must wait or the entire intended all-team lock must fail closed
according to the lane's frozen population contract.

### B. Legitimate current Week-6 pregame identity

Week-5 identity artifacts are not Week-6 identity authority.

Week 6 requires a newly certified current pregame player/role universe with:
- target Week-6 team;
- current position/room membership;
- current starter/active-role authority where applicable;
- exact stable player identity;
- chronology-safe source dates.

No Week-5 frozen player set may simply be copied forward.

### C. Strict chronology

Every feature source max week / timestamp must be strictly before the target
Week-6 game.

No Week-6 box score, PBP, target, carry, snap, reception, or yardage may be read
when creating the Week-6 lock.

### D. No outcome-based selection

Week-5 results are not required to decide whether the formulas continue.

The formulas already continue unchanged by this contract.

Week-5 grading is a separate later scoring operation and may not:
- change Week-6 membership rules;
- change trajectory support;
- change 50/50 carry/snap weights;
- change RB receiving normalization;
- change room definitions;
- change final prospective gates.

### E. Sportsbook separation

Sportsbook/game-market inputs authorized for the lock: **0**.

OddsAPI calls authorized for the lock: **0**.

## Hard-coded Week-5 implementation warning

The existing lock implementations are not generic future-week executors:

- `scripts/research/lock_rb_player_state_allocation_shadow_v1.py`
  hard-codes season 2026 / Week 5 and the Week-5 live-state parent authority;

- `scripts/research/lock_rb_receiving_room_share_week5_v1.py`
  hard-codes Week 5 plus the Week-5 RB player-state and live-state artifacts;

- `scripts/research/lock_player_target_share_trajectory_shadow_v1.py`
  hard-codes Week 5 and Week-5 output filenames.

Therefore:

**Do not obtain Week-6 lock #2 by changing only the constant `WEEK=5` on the old
scripts.**

A future-week executor must parameterize the target week and receive a newly
certified target-week pregame identity parent while preserving every scientific
rule above.

That implementation work is mechanical only. It may not alter science.

## Week-6 readiness disposition

Current state on 2026-10-11:

`WEEK6_CONTINUATION_RULES_FROZEN__LOCK_EXECUTION_WAITING_FOR_COMPLETE_PRIOR_STATE_AND_CURRENT_WEEK6_IDENTITY`

This is not a scientific failure.

It is the required prospective boundary.

## Relationship to QB V1L

QB prospective intent V1L is separate from the three allocation shadows above.

V1L Week-6 opening collection has already been certified:
- run `38187787580` SUCCESS;
- artifact `11719983641`;
- digest
  `sha256:f5c65d6ae651141fb8120cb3d726687cac6e35174d1175d460e25e5d37315aa2`;
- 28/28 scheduled team-weeks attempted;
- zero retained source rows in the opening Sunday-night snapshot;
- source coverage is descriptive only.

Do not use V1L source availability to change the RB/WR/TE lock formulas.

## Integrity

This continuation contract:
- reads no football outcomes;
- reads no model residuals;
- uses no sportsbook/game-market inputs;
- makes no paid OddsAPI calls;
- fits no football predictor;
- changes no production model;
- grades no Week-5 prospective lock.

Production mutations authorized: **0**
