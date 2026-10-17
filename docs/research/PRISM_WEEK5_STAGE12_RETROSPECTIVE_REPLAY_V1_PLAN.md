# PRISM Week-5 Stage-1/Stage-2 Retrospective Replay V1 — Frozen Plan

Date: 2026-10-17

Status: **FROZEN BEFORE REPLAY IMPLEMENTATION / INDEPENDENT-MECHANISM CURRENT-SEASON DIAGNOSTIC**

Model:
**PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_WEEK5_STAGE12_RETROSPECTIVE_REPLAY_V1`

## 1. Scientific question

PRISM Stage 1 and Stage 2 were frozen, developed and qualified from 2024/2025
evidence with **zero 2026 outcomes**.

Week 5 of 2026 has already been played, but it has a preserved leakage-safe
pregame player-state authority from before those PRISM mechanisms were created.

The frozen question is:

> If the already-qualified PRISM Stage-1 active-role and Stage-2 conditional
> allocation mechanisms had been applied unchanged to the preserved Week-5
> pregame football state, would they have improved named-player opportunity
> projection on Week 5?

This is a retrospective current-season diagnostic of independently frozen
science.

It is **not** a prospective Week-5 claim and cannot be retuned from Week-5
outcomes.

## 2. Immutable scientific parents

Stage 1:
`PLAYER_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED_ALL`.

Canonical authority:
- run `38907459996`;
- exact head `1c2c92ca498e8dce1eee19d39b2f276dfd37d8f7`;
- artifact `12049120604`.

Stage 2:
`PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED_PARTIAL`.

Canonical authority:
- freeze commit `87400a9c9c1db465c9c4bd70b0cb241dba27c6c9`;
- scored run `38957575645`;
- exact scored head `eba5787c0b06aa5f975008965df7d023b3525ba7`;
- artifact `12070774378`;
- digest
  `sha256:e07edbad1bf97a4917ba95c58156c8941448e7b2668bd0eae2e9bcad5d4945b0`.

No Stage-1 or Stage-2 coefficient, threshold, window, support rule or allocation
law may change for this replay.

## 3. Preserved Week-5 pregame authority

Use only the preserved player-state artifact from the leakage-safe Player
Landscape Transmission audit:

- run `37717016519` — SUCCESS;
- exact head `35820953a255ce0f8413c2c468501df7f267fa96`;
- artifact `11524296325`;
- digest
  `sha256:e043133d4b9a48ce472dde8810d4d77d636f4ce2fbcee14708b23fe0ea35f6a0`;
- file:
  `player_landscape_week5_player_state.csv`.

That artifact was built by:
- `build_all_historical_inputs(... weeks=[5])`;
- a Week-5 pregame universe;
- 2026 player logs with max completed week <5;
- zero Week-5 outcomes;
- zero sportsbook input.

It contains the exact pregame fields required for this replay, including:
- player/team/event identity;
- `rules_plays_est`;
- `rules_pass_rate`;
- `rules_rush_share`;
- `entitlement_tgt_share`;
- old Week-5 `expected_carries`;
- old Week-5 `expected_targets`.

## 4. Target cells

Score only the cells already qualified by frozen Stage 2:

1. **RB rushing opportunity**
   - observed count: player rush attempts;
2. **RB receiving opportunity**
   - observed count: player targets;
3. **TE receiving opportunity**
   - observed count: player targets.

Do not score:
- QB Stage 2 — CLOSED;
- WR Stage 2 — source blocked/not scored.

## 5. Strict-prior Stage-1 history

For target Week 5, Stage-1 history is **2026 Weeks 1-4 only**.

Preserve exact Stage-1 semantics:
- same named player;
- same team;
- last up to four eligible prior player-games;
- minimum three for named-player Stage 1;
- otherwise exact position-family fallback;
- official inactive identity removed from prior eligible universes;
- target-game performance never used as a predictor.

The Week-5 target active universe may use only pregame football availability
facts, including exact official inactive identity.

No target-game participation may be used to decide target eligibility.

## 6. Frozen Week-5 target identity / availability rules

Start from the preserved Week-5 player-state population.

Attach target stable player identity from the same weekly-roster semantics used
by the qualified Stage-1 source.

Remove exact official Week-5 inactive players from the target eligible
population using fixed historical official-inactive facts only.

Fail closed if:
- stable target identity <99%;
- a target row cannot be reconciled without an explicitly frozen identity rule;
- exact target official-inactive source cannot be reconstructed with the frozen
  Stage-1 semantics.

Do not infer target availability from:
- Week-5 snaps;
- Week-5 carries/targets;
- Week-5 box score;
- sportsbook listing;
- whether the player eventually played.

## 7. Frozen canonical entitlement

### RB rushing

Construct the exact Stage-2 canonical RB rushing entitlement from the preserved
Week-5 `rules_rush_share` using the same frozen `_top_n_shares(..., 5)`,
0.95 cap and anonymous complement semantics used by the Week-6 Stage-2 shadow.

Team opportunity mean:
`rules_plays_est * (1 - rules_pass_rate)`.

### RB / TE receiving

Canonical target entitlement:
`entitlement_tgt_share`.

Team opportunity mean:
`rules_plays_est * rules_pass_rate`.

Do not fit or replace either entitlement.

## 8. Frozen candidate

For canonical-positive named players:

`A_i ~ Bernoulli(p_active_i)`.

Use the exact frozen Stage-2 active-set enumeration.

For every active-set state:
- preserve the canonical entitlement;
- renormalize only among active canonical-positive named players plus the
  anonymous complement;
- canonical-zero player remains zero;
- all-off state goes to anonymous complement.

Candidate expected opportunity is the active-state weighted expected share
times the frozen team opportunity mean.

No fitted weights.
No threshold search.
No role subgroup rescue.

## 9. Three replay comparisons

Where available, report all three named-player expected-opportunity errors.

### A. Original Week-5 production

From the preserved Week-5 player-state artifact:
- RB rushing: `expected_carries`;
- RB/TE receiving: `expected_targets`.

This answers:
> how wrong was the model actually used in Week 5?

### B. Frozen Stage-2 canonical parent

`canonical_q * frozen team opportunity mean`.

This isolates the already-qualified Stage-2 parent entitlement under the exact
replay law.

### C. Frozen Stage-1/Stage-2 candidate

`candidate_expected_q * frozen team opportunity mean`.

This answers:
> would the newer player-first active-role conditioning have improved the named
> player's opportunity estimate?

## 10. Outcome boundary

Candidate rows must be fully built and hashed **before** any Week-5 outcome file
is loaded in the scoring step.

After candidate certification, Week-5 outcomes may be loaded only to score:
- rush attempts;
- targets;
- finite team opportunity totals.

The candidate builder must accept no target-game outcome file.

The workflow must persist:
- candidate row digest before scoring;
- exact source/run/head provenance;
- confirmation that Week-5 outcomes read by builder = 0.

## 11. Primary diagnostics

This is a one-week current-season replay, not a new qualification trial.

Report by cell:
- rows;
- unique players;
- games/rooms;
- named-player Stage-1 support rate;
- original Week-5 production mean absolute opportunity error;
- canonical Stage-2 parent mean absolute opportunity error;
- Stage-1/Stage-2 candidate mean absolute opportunity error;
- candidate minus parent AE improvement;
- candidate minus original-production AE improvement;
- share of named player rows improved vs parent;
- share improved vs original production.

Also report:
- deterministic top Week-5 opportunity misses;
- candidate movement for those exact named players;
- whether the candidate moved in the correct direction;
- room-level conservation.

No position-average conclusion may replace the named-player diagnostics.

## 12. Forensic casebook linkage

Join the already-frozen Week-5 player-failure casebook only **after** scoring.

Authority:
- run `38397361891`;
- artifact `11812403368`;
- digest
  `sha256:4cdef29912f35368473d57e15b38efdc3dd0f75e897bbbdc3b97fe1ff8f67670`.

For opportunity-dominant major misses, report:
- old predicted opportunity;
- actual opportunity;
- frozen Stage-2 parent expected opportunity;
- frozen Stage-1/Stage-2 candidate expected opportunity;
- whether the new PRISM mechanism moved toward or away from reality.

Do not select or tune cases based on candidate success.

## 13. Interpretation rules

Possible diagnostic conclusions:

- `WEEK5_STAGE12_CURRENT_SEASON_REPLAY_DIRECTIONALLY_SUPPORTIVE`
- `WEEK5_STAGE12_CURRENT_SEASON_REPLAY_MIXED`
- `WEEK5_STAGE12_CURRENT_SEASON_REPLAY_UNSUPPORTIVE`
- `WEEK5_STAGE12_REPLAY_SOURCE_BLOCKED`
- `WEEK5_STAGE12_REPLAY_INTEGRITY_FAILURE`.

These are **diagnostic**, not promotion dispositions.

No Week-5 result may:
- refit Stage 1;
- refit Stage 2;
- change a threshold;
- promote production;
- rescue a closed cell.

## 14. Protected state

Unchanged:
- Week-6 Sun/Mon Stage-2 shadow remains immutable and ungraded;
- Week-6 WR/TE trust locks remain immutable and ungraded;
- Weeks 7-10 WR dual prospective protocol remains intact;
- Room Competition State V1 remains SOURCE BLOCKED;
- raw recent-share replacement remains CLOSED;
- generic opponent/matchup families remain closed;
- QB Stage 2 remains CLOSED;
- WR Stage 2 remains source blocked.

## 15. Integrity

Required:
- paid OddsAPI calls: 0;
- sportsbook predictors: 0;
- target outcomes used in candidate construction: 0;
- new fitted models: 0;
- threshold searches: 0;
- production changes: 0.

This plan is frozen before replay implementation/scoring.
