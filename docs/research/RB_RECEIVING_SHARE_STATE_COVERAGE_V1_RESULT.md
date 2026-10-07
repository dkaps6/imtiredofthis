# RB RECEIVING SHARE STATE COVERAGE V1 — RESULT

Date: 2026-10-07  
Branch: `research-rb-receiving-share-state-coverage-v1`  
Successful head: `71e47bd5c5e6a75943610c2e9bc2ca10a226b629`  
GitHub Actions run: `37702111845` — **SUCCESS**  
Artifact: `11518640307`  
Artifact digest: `sha256:955946b42be34f5ca03323b3716e8b4522c586d6865db770abdc9184c14817e7`

## Disposition

`EXISTING_RB_RECEIVING_STATE_PRESENT_BUT_NOT_GENERICALLY_CONSUMED`

The residual RB receiving allocation problem identified by the player-opportunity
decomposition has a concrete strict-prior player-state signal already present in
the repository.

The generic RB target allocator does not consume the richer RB receiving-room
identity fields directly.

A raw, no-fit within-RB-room carry-forward of prior receiving-room share improves
allocation error versus the current model.

No production promotion is authorized by this result.

## Mechanical validity

Run `37702111845` passed:

- frozen RB receiving-share tests;
- frozen parent decomposition artifact recovery;
- strict-prior receiving identity reconstruction;
- room-share scoring;
- source-consumption audit;
- certification boundary;
- strict repository audit.

Boundary:

- rows: **436**
- scoreable rows: **432**
- team rooms: **128**
- scoreable team rooms: **127**
- parameters fit: **0**
- automatic promotion: **false**
- sportsbook inputs: **false**
- paid OddsAPI: **false**

The first attempted run `37701998598` failed only in a static test that
expected a literal feature-name occurrence inside R26. R26 consumes the imported
`FEATURES` bundle indirectly. No scientific scoring occurred in that failed
run. The test/plumbing repair was committed before the successful run.

## Current model within-RB-room target allocation

Current model RB-room share MAE:

**0.2233629239**

Bias:

approximately **0.0** by construction after room normalization.

Current model RB target-room leader match rate:

**75.59%**

This means the current stack often knows which back is the receiving-room leader,
but its within-room share magnitudes remain too compressed.

## Strict-prior state coverage

Coverage across 436 RB/FB target player-games:

- any prior player history: **92.20%**
- same-team prior history: **83.72%**
- previous-season history: **81.65%**

Key strict-prior room-share state:

| Field | Coverage | Spearman vs realized RB-room share | Spearman vs current model room residual |
|---|---:|---:|---:|
| prior_rb_room_share | 92.20% | **0.582** | **-0.333** |
| last8_rb_room_share | 92.20% | 0.559 | -0.277 |
| prev_season_rb_room_share | 81.65% | 0.550 | -0.250 |
| same_team_prior_rb_room_share | 83.72% | 0.572 | -0.296 |

The negative residual correlation is directionally important: backs with larger
strict-prior receiving-room roles tend to be **under-allocated** by the current
model, while smaller-role backs tend to be relatively over-allocated.

That is directly consistent with the middle-compression pattern found in the
all-player replay.

## No-fit proxy scoreboard

Each proxy below simply normalizes the already-existing strict-prior RB room
share inside the active RB room. No blend coefficient, threshold, or model fit is
used.

| Raw strict-prior proxy | Current model MAE on same rows | Proxy MAE | Absolute improvement | Relative improvement |
|---|---:|---:|---:|---:|
| prior_rb_room_share | 0.22336 | **0.18807** | **0.03530** | **15.8%** |
| last8_rb_room_share | 0.22336 | **0.18809** | **0.03527** | **15.8%** |
| prev_season_rb_room_share | 0.22336 | 0.19392 | 0.02944 | 13.2% |
| same_team_prior_rb_room_share | 0.22336 | 0.19586 | 0.02751 | 12.3% |

All four predeclared room-share state variants outperform the current allocator
on the scoreable population.

This is not a one-window accident where only one selected lookback works:
long-run prior, last-8, prior-season, and same-team history all point in the same
direction.

## Leader identity

Leader-match rates:

- current model: **75.59%**
- prior_rb_room_share: **75.59%**
- last8_rb_room_share: **74.80%**
- prev_season_rb_room_share: **71.65%**
- same_team_prior_rb_room_share: **71.65%**

The main gain is therefore **not** better identification of the room leader.

The gain is better **share magnitude / concentration** once the room hierarchy is
known.

That matters because it localizes the failure more precisely:

> The current RB target allocator is too compressed inside the room, not simply
> choosing the wrong receiving back.

## Existing production consumption

Generic canonical path:

- empirical-Bayes target share: player-specific but position-shrunk
- RB default target-share prior: **0.08**
- target-share group prior strength: **3.0 equivalent games**
- player prior cap: **6.0 games**
- rules layer: coarse RB receiving matchup multiplier
- explicit entitlement: consumes the resulting generic target share
- TE-R5P / WR-R15: preserve non-TE / non-WR room mass

The richer fields such as:

- prior_rb_room_share
- last8_rb_room_share
- same_team_prior_rb_room_share
- previous-season RB room share

are **not directly consumed by the generic Bayes/rules/entitlement path**.

They do exist in RB specialist code:

- R26 Week-1 receptions adapter consumes the strict-prior receiving-identity
  feature bundle, but R26 is Week-1-specific and not a general Weeks 2+ target
  allocator.
- R22 also consumes RB receiving identity state for tail/distribution risk, but
  preserves reception/target means and does not solve the target-share center.

Therefore the signal exists; it is simply not wired as the general RB receiving
share authority.

## Scientific interpretation

This is a legitimate uncovered player-level mechanism.

The earlier decomposition showed that supplying realized RB player share removes
**76.5%** of RB target MAE while supplying realized team dropback volume removes
only **2.8%**.

This audit now shows that a fully pregame, strict-prior, already-available RB
room-share state improves within-room allocation by about **12-16%** with no
fitting.

The mechanism is coherent with the broader replay diagnosis:

- current allocator knows much of the hierarchy;
- position/player shrinkage pulls allocations toward the middle;
- focal receiving backs are under-allocated;
- low-role backs receive too much relative room mass;
- raw strict-prior RB receiving-room state restores some concentration.

## What this does NOT authorize

Do not:

- retrofit a selected coefficient onto Weeks 1-4;
- choose a post-hoc threshold;
- reopen R23-R27D receiving-yard mean science;
- reopen M96 retrospective RB router/threshold research;
- change R22 tail science;
- back-apply the Week-5 carry/snap allocation shadow;
- alter WR/TE trajectory rules;
- promote the raw proxy directly to production.

## Next authorized step

Freeze a separate **prospective RB receiving-share shadow** for Week 5+.

Recommended mechanism boundary:

`RB_RECEIVING_ROOM_SHARE_SHADOW_V1`

Principles:

1. preserve the model's existing total RB+FB target-entitlement pool;
2. redistribute only **within that RB/FB pool**;
3. use strict-prior RB receiving-room share state only;
4. do not create or remove team target opportunity;
5. do not change WR/TE entitlement;
6. do not change RB receiving YPT/yardage efficiency;
7. do not change R22 distribution tails;
8. unavailable-history players fall back to current canonical allocation;
9. no fitted coefficient from Weeks 1-4;
10. freeze Week-5 pregame values before outcomes.

Because all four predeclared historical room-share variants improved in the same
direction, the prospective shadow should prefer a simple, prespecified
strict-prior room-state rule rather than fit a blend to Weeks 1-4.

No paid OddsAPI.
No sportsbook inputs.
No automatic production promotion.
