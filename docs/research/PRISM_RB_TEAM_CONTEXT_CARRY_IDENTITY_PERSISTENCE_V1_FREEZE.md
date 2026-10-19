# PRISM RB Team-Context Carry Identity Persistence V1 — Frozen Predictive Contract

Date: 2026-10-18

Status: **FROZEN BEFORE SCORING / NAMED-RB IN-SEASON TEAM-CHANGE CARRY OPPORTUNITY ONLY**

Model:
**PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_RB_TEAM_CONTEXT_CARRY_IDENTITY_PERSISTENCE_V1`

## 1. Player-first question

Canonical Team-Context Transition V1 established that PRISM can source-certify
when this exact named player moves from one NFL team context to another before
kickoff.

This V1 asks one market-specific predictive question:

> For this exact RB in his first source-certified **in-season** new-team
> context, is his own strictly-prior old-team carry-share identity more
> predictive of his first new-team carry share than the already-canonical
> current-team RB rushing entitlement?

This is current-role arbitration.

It is not a universal new-team penalty or boost.

It does not project rushing yards, efficiency or touchdowns.

## 2. Why V1 is in-season only

The canonical historical RB rushing entitlement authority is protected and
source-ready for Weeks 5-18 in 2024 and 2025.

V1 therefore scores only source-certified:

`INSEASON_TEAM_CHANGE`

events whose first new-team roster observation falls in Weeks 5-18.

Offseason Week-1 transitions are out of scope.

Do not rebuild or extrapolate an unqualified Week-1 Stage-2 authority merely to
increase support.

This restriction is frozen before outcomes are scored.

## 3. Immutable source parents

### A. Team-context transition parent

Canonical authority:
- freeze `bd254ec9435cd0a17e2c7d82d78532e67c2cd23e`;
- run `39182076127` — SUCCESS;
- exact source head `6c047e4df67702affde52493f7d1b90296751968`;
- job `117836769489` — SUCCESS;
- artifact `12156204035`;
- artifact name
  `prism-player-team-context-transition-v1-source-39182076127`;
- digest
  `sha256:abc2a5b9d3ea2a4d683e56626ea2d33495fa1e7cad2c36c56c01071ccc0339c8`;
- disposition
  `PRISM_PLAYER_TEAM_CONTEXT_TRANSITION_V1_SOURCE_READY`.

Use only transition rows whose:
- family is RB;
- transition kind is in-season;
- stable ID is nonblank;
- previous team is nonblank and differs from current team;
- prior-team roster support is >=4.

The looser `fc63601c...` lineage is noncanonical and forbidden.

### B. Canonical RB rushing entitlement parent

Canonical source authority:
- run `38955368643` — SUCCESS;
- exact head `cb65a634a0e09296bfeabc94ffe1e77363a30f8a`;
- artifact `12070806828`;
- digest
  `sha256:e91a108495d7a2b6eb289e0618ab6af09e3d6cdf6895e534742a07340e4a2a83`;
- disposition includes
  `RB_RUSHING_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_SOURCE_READY`.

Use its exact RB rushing pregame:
- `authority_q`;
- `authority_denominator_mean`;
- target player/team/week identity;
- finite-room conservation semantics.

No new baseline rushing entitlement may be fit.

### C. Actual / strictly-prior carry-count authority

Use the existing canonical historical count construction:

`scripts/research/audit_player_opportunity_allocation_uncertainty_betabinomial_v1.py::build_target_counts`

Required exact fields:
- season;
- week;
- team;
- stable player identity / canonical player identity;
- rushes;
- team_rushes.

The scorer must source-certify exact stable-ID linkage from the team-context
parent to the count authority.

Name-only matching is forbidden.

## 4. Anti-retest declaration

This V1 is not a rescue or re-test of:

- global raw recent-share replacement — CLOSED;
- Opportunity Authority Priority V1 — CLOSED;
- RB carry/snap 50/50 state — protected;
- active-role participation — QUALIFIED;
- active-set conditional allocation — QUALIFIED for RB rushing;
- teammate-vacancy transfer — CLOSED / blocked family;
- teammate-availability Role Elasticity V1 — SOURCE BLOCKED;
- RB depth-role transition — SOURCE BLOCKED;
- playcaller transition — SOURCE BLOCKED.

The genuinely new information is the source-certified **named-player team
boundary**.

The old-team carry share is used only as a first-new-team-game identity
candidate conditional on that boundary.

It may never become a global recent-share replacement rule.

## 5. Eligible transition event

A target event is eligible only when all are true:

1. season is 2024 or 2025;
2. target week is 5-18;
3. family is RB;
4. state is source-certified team-context transition;
5. transition kind is exactly `INSEASON_TEAM_CHANGE`;
6. stable player ID is exact and nonblank;
7. previous team != current team;
8. exact canonical current-team RB rushing `authority_q` is present;
9. `0 < authority_q < 1`;
10. actual target-game team rushing denominator is finite and >0;
11. exact player target-game rush count is finite and >=0;
12. at least four valid strictly-prior carry-share observations exist for the
    same named player on the immediately previous team.

No Week-1 offseason transition can enter V1.

## 6. Frozen old-team carry identity

For each eligible transition RB, look only at exact same-player observations
satisfying:

- team == source-certified previous team;
- observation strictly earlier than target game;
- team_rushes finite and >0;
- player rushes finite and >=0;
- player rushes <= team_rushes.

Per-observation:

`old_team_carry_share = rushes / team_rushes`

Take the latest **up to eight** valid previous-team observations.

Require at least **four**.

Frozen anchor:

`old_team_identity_q = arithmetic mean(old_team_carry_share)`

No recency weighting.

No median alternative.

No last-3 alternative.

No coefficient.

No position prior.

No current-team target-game usage.

The 4-to-8 support window is frozen before scoring and may not be searched.

## 7. Baseline

Baseline for the target event:

`q_base_i = canonical current-team RB rushing authority_q`

from the immutable Stage-2 source parent.

This is the already-canonical pregame current-team entitlement.

V1 does not replace the baseline outside source-certified transition events.

## 8. Frozen finite-room candidate

For every target team-week, construct the baseline RB room:

- every covered canonical RB rushing `q_base_j`;
- anonymous residual:
  `q_base_residual = 1 - sum(q_base_j)`.

Require:
- every q finite and >=0;
- baseline residual >=0 within floating tolerance;
- total baseline room mass = 1 within `1e-10`.

For all eligible transition RBs in that room:

`q_raw_i = old_team_identity_q_i`

If the sum of transition raw anchors exceeds 0.95:

`q_candidate_transition_i =
 q_raw_i * (0.95 / sum(q_raw_transition))`

Otherwise:

`q_candidate_transition_i = q_raw_i`

Let:

`T_base = sum(q_base_transition)`

`T_candidate = sum(q_candidate_transition)`

Baseline remaining mass:

`R_base = 1 - T_base`

Candidate remaining mass:

`R_candidate = 1 - T_candidate`

Require `R_base > 0`.

Every non-transition canonical RB and the anonymous residual retain their
relative baseline competition weights:

`q_candidate_j = q_base_j * (R_candidate / R_base)`

for each non-transition RB and residual.

Thus:
- only transition-RB identity mass is replaced;
- current-team teammate/residual competition proportions are preserved;
- candidate room mass remains exactly 1.

No opportunity is created or destroyed.

## 9. What is scored

### Primary named-player endpoint

For each eligible transition RB:

`actual_share_i = target_game_rushes_i / target_game_team_rushes`

Baseline absolute share error:

`AE_base_i = abs(q_base_i - actual_share_i)`

Candidate absolute share error:

`AE_candidate_i = abs(q_candidate_i - actual_share_i)`

Improvement:

`share_AE_improvement_i = AE_base_i - AE_candidate_i`

Positive means old-team player identity improved the first new-team carry-share
projection.

The named player is the primary unit.

### Secondary expected carry-count endpoint

Using the **actual team-rush denominator only for historical scoring**:

`count_AE_base = abs(N * q_base_i - rushes_i)`

`count_AE_candidate = abs(N * q_candidate_i - rushes_i)`

This is evaluation only.

No target-game denominator is a predictor.

### Room protection endpoint

For the full RB + residual room, use actual team rushing counts to score
baseline vs candidate multinomial negative log likelihood.

The candidate must not improve the transition player by causing supported harm
to the finite current-team room.

## 10. Named-player aggregation

Within each season:

- average episode improvement within named player first;
- then average players equally.

Report:
- eligible events;
- distinct named RBs;
- equal-player mean share-AE improvement;
- median player improvement;
- improved / tied / worsened player shares;
- equal-player expected-count AE improvement;
- room log-score change;
- event support by previous/current team;
- support by in-season target week.

Do not lead with position MAE.

## 11. 2024 development gates

All must pass:

1. eligible transition events >= 20;
2. distinct named RBs >= 15;
3. exact stable-ID count linkage >= 99%;
4. canonical entitlement coverage >= 99%;
5. old-team >=4 valid carry-share support = 100% of scored rows;
6. chronology violations = 0;
7. finite-room mass violations = 0;
8. equal-player mean share-AE improvement > 0;
9. median player share-AE improvement >= 0;
10. improved-player share > worsened-player share;
11. equal-player expected-count AE improvement >= 0;
12. mean room log-score change is not harmful.

If any development gate fails, V1 closes and 2025 may be reported as
confirmation diagnostics only; it cannot rescue development.

## 12. 2025 independent confirmation gates

If development passes, all must pass:

13. eligible transition events >= 20;
14. distinct named RBs >= 15;
15. exact stable-ID count linkage >= 99%;
16. canonical entitlement coverage >= 99%;
17. old-team support / chronology / conservation gates all pass;
18. equal-player mean share-AE improvement > 0;
19. median player share-AE improvement >= 0;
20. improved-player share > worsened-player share;
21. equal-player expected-count AE improvement >= 0;
22. mean room log-score change is not harmful;
23. named-player cluster bootstrap 95% lower bound for share-AE improvement > 0.

Bootstrap:
- 5,000 replicates;
- seed `20261018`;
- sample named players with replacement;
- preserve all eligible events belonging to the sampled player.

Do not change these gates after scoring.

## 13. Dispositions

Qualification:

`PRISM_RB_TEAM_CONTEXT_CARRY_IDENTITY_PERSISTENCE_V1_QUALIFIED`

Failure:

`PRISM_RB_TEAM_CONTEXT_CARRY_IDENTITY_PERSISTENCE_V1_CLOSED`

Source failure before outcome scoring:

`PRISM_RB_TEAM_CONTEXT_CARRY_IDENTITY_PERSISTENCE_V1_SOURCE_BLOCKED`

No subgroup rescue.

## 14. What qualification would mean

A pass would mean:

> for source-certified in-season RB team changes, the named player's own
> previous-team carry-share identity contains replicated information beyond the
> current canonical new-team rushing entitlement for his first new-team game.

A pass would authorize a separately frozen integration/authority test.

It would **not** authorize immediate production use.

It would not authorize:
- rushing-yard changes;
- efficiency changes;
- touchdown changes;
- applying old-team share after the first new-team game;
- applying the rule to offseason transitions;
- applying it to WR/TE targets.

## 15. What failure would mean

Failure closes this exact:

`in-season team change + old-team carry identity -> first new-team carry share`

family.

Do not rescue by:
- changing 4-8 history support;
- using last-3;
- using median instead of mean;
- fitting a blend coefficient;
- applying only to RB1;
- applying only to trades vs waivers;
- splitting by old/new team;
- splitting by competitor count;
- adding snaps;
- adding depth chart;
- adding playcaller;
- moving the failed signal downstream to rushing yards.

The team-context source state remains valid even if this predictive mechanism
fails.

## 16. Integrity at freeze

Before this freeze:
- eligible historical transition outcomes scored: **0**;
- old-team anchor vs target outcome comparisons inspected: **0**;
- candidate coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 outcomes used: **0**;
- sportsbook/OddsAPI used: **0**;
- paid OddsAPI calls: **0**;
- production changed: **false**.

Disposition at freeze:

`PRISM_RB_TEAM_CONTEXT_CARRY_IDENTITY_PERSISTENCE_V1_FROZEN`.
