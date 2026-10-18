# PRISM Injury-Partial Receiving Exposure Decomposition V1 — Frozen Predictive Contract

Date: 2026-10-18

Status: **FROZEN BEFORE PREDICTIVE SCORING / NAMED-PLAYER RECEIVING ROLE ONLY**

Model: **PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_INJURY_PARTIAL_RECEIVING_EXPOSURE_DECOMPOSITION_V1`

## 1. Core player-first question

PRISM's objective is the named player's next-game projection.

The current PlayerForm history can see a partial game such as:
- reduced offensive participation;
- reduced full-game target total;

and allow that full-game target total to pull down the player's ordinary role
identity.

The prior complete-deletion candidate is CLOSED:
`PRISM_COMPARABLE_OBSERVATION_RECEIVING_OPPORTUNITY_IDENTITY_V1_CLOSED`.

The partial game still contains football information.

This V1 asks:

> When a named player's recent history contains an injury-corroborated
> participation collapse, is his next comparable receiving role better
> represented by separately estimating:
>
> 1. his expected ordinary participation exposure; and
> 2. his target earning conditional on the offensive snaps he actually played?

This is not a position correction.

## 2. Immutable source authority

Use only:

- run `39161711741` — SUCCESS;
- artifact `12149181200`;
- artifact name
  `prism-receiving-opportunity-earning-participation-exposure-v1-source-39161711741`;
- digest
  `sha256:6cae427b667db4a120a1308747d00b82b6770fe0adbc46d3efc68fce3e1db240`;
- disposition
  `PRISM_RECEIVING_OPPORTUNITY_EARNING_PARTICIPATION_EXPOSURE_V1_SOURCE_READY`;
- file
  `receiving_opportunity_earning_participation_exposure_rows.csv`.

No provider rebuild is permitted in the scorer.

## 3. Scope

Families:
- RB;
- WR;
- TE.

Development:
- 2024.

Independent confirmation:
- 2025.

No 2026 outcomes.

Every rule is identical across families.

Family is reporting/guardrail context only.

## 4. Target-game eligibility

A target player-game is eligible only if:

1. stable player identity is present;
2. target family is RB/WR/TE;
3. target row has positive offensive participation;
4. target row state is exactly
   `PARTICIPATION_NOT_COLLAPSED`;
5. target team has positive team target volume;
6. same player + same team + same season has at least four strictly-prior
   positive-participation rows.

The target game's own targets/target share are outcome only.

## 5. Frozen four-observation player window

For each eligible target player-game:

- same named player;
- same team;
- same season;
- strictly earlier weeks only;
- take exactly the latest four positive-participation rows.

The V1 action cohort requires all of the following inside those four rows:

1. at least one
   `PARTICIPATION_COLLAPSE_INJURY_CORROBORATED`;
2. zero
   `PARTICIPATION_COLLAPSE_UNRESOLVED`;
3. at least three rows that are not major participation collapses.

Because the window contains four rows, this ordinarily means one resolved
injury-partial observation plus three comparable observations.

If the window contains an unresolved participation collapse:
- fail closed for that target;
- do not score the candidate.

## 6. Source-only power check before scoring

Using only identity, chronology and already-certified participation states:

2024 eligible action cohort:
- target player-games: **211**;
- named players: **66**;
- RB: 56 games / 20 players;
- WR: 111 games / 32 players;
- TE: 44 games / 14 players.

2025 eligible action cohort:
- target player-games: **193**;
- named players: **64**;
- RB: 50 games / 16 players;
- WR: 85 games / 30 players;
- TE: 58 games / 18 players.

No target-share outcome relationship was inspected to obtain these counts.

## 7. Frozen baseline

Use the exact same four prior player rows as the candidate.

Raw recent receiving-role baseline:

`baseline_target_share =
sum(prior4 targets) / sum(prior4 team_targets)`

This intentionally treats every prior game's team target volume as full-game
exposure.

It is a within-player baseline for isolating the participation/exposure
mechanism.

It is not a position prior.

## 8. Frozen participation component

The injury-corroborated partial row remains in the earning history, but its
reduced snap percentage may not define ordinary expected participation.

For the same four prior rows:

`comparable_rows = rows whose state is not
PARTICIPATION_COLLAPSE_INJURY_CORROBORATED`

There must be at least three comparable rows.

Expected ordinary offensive participation:

`expected_offense_pct =
median(comparable_rows offense_pct)`

No target information enters this participation estimate.

No position average enters it.

## 9. Frozen target-earning component

Use **all four resolved prior rows**, including the injury-corroborated partial
row.

Participation-conditioned target earning:

`earning_intensity =
sum(prior4 targets)
/
sum(prior4 team_targets * prior4 offense_pct)`

Interpretation:
- numerator = targets the named player actually earned;
- denominator = team target volume scaled by the player's actual offensive snap
  exposure.

This is a snap-conditioned target-earning measure.

It is **not**:
- targets per route;
- route participation;
- first-read rate.

No route claim is authorized.

## 10. Frozen candidate

`candidate_target_share_raw =
expected_offense_pct * earning_intensity`

`candidate_target_share =
clip(candidate_target_share_raw, 0, 1)`

No coefficient is fit.

No blend weight is fit.

No position-specific multiplier exists.

No alternate formula is scored in V1.

## 11. Target outcome

`actual_target_share = target-game targets / target-game team_targets`

The target game is guaranteed by the eligibility gate to be a
`PARTICIPATION_NOT_COLLAPSED` observation.

Therefore an injury/partial target game is not treated as an ordinary role
label.

## 12. Named-player scoring

For each target player-game:

`player_game_closeness_gain =
abs(baseline_target_share - actual_target_share)
-
abs(candidate_target_share - actual_target_share)`

Positive:
candidate is closer for that named player-game.

Negative:
candidate is worse.

Then aggregate **within named player first**.

For each named player:
- average his target-game closeness gain.

Primary reporting:
- target games;
- named players;
- equal-player average closeness gain;
- median named-player gain;
- improved / tied / worsened player share;
- p10 / p25 / p75 / p90 named-player gain;
- named-player bootstrap interval;
- per-player target-game count.

A pooled position average is not the objective.

## 13. Frozen qualification gates

### Integrity
1. exact source run/artifact/name/digest match;
2. chronology violations = 0;
3. target rows with major participation-collapse state = 0;
4. unresolved prior-window rows = 0;
5. every scored prior window has exactly four rows;
6. every scored prior window has >=3 comparable rows;
7. expected_offense_pct finite and >0;
8. earning_intensity finite and >=0;
9. candidate share finite and in [0,1];
10. coefficients fit = 0;
11. thresholds searched = 0;
12. 2026 outcomes = 0;
13. sportsbook/OddsAPI = 0.

### 2024 development — named players
14. target games >=100;
15. named players >=40;
16. equal-player average closeness gain > 0;
17. median named-player gain >= 0;
18. improved-player share > worsened-player share.

### 2025 independent confirmation — named players
19. target games >=100;
20. named players >=40;
21. equal-player average closeness gain > 0;
22. median named-player gain >= 0;
23. improved-player share > worsened-player share;
24. named-player bootstrap 95% lower bound > 0.

Bootstrap:
- sample named players with replacement;
- preserve all target games for sampled players;
- 5,000 replicates;
- seed `20261018`.

## 14. Family diagnostics

Report RB / WR / TE separately:
- target games;
- named players;
- equal-player average gain;
- median player gain;
- improved/worsened shares.

No family gets a different formula.

No family result can be used to rescue a failed overall frozen contract.

## 15. Anti-rescue

If V1 fails, do not:
- change latest-4;
- change the minimum comparable-row count;
- change the injury-collapse thresholds;
- use unresolved collapses as though injury-caused;
- substitute mean for median snap exposure;
- tune a snap multiplier;
- use target-game injury state as a predictor;
- add recency weights;
- add a position coefficient;
- subset stars / WR1 / TE1 / lead backs;
- call snaps routes;
- use sportsbook information;
- use 2026 outcomes.

A failure closes this exact decomposition.

## 16. What qualification means

Qualification means:

> For named pass catchers with a resolved injury-partial observation in recent
> history, separating ordinary expected participation from target earning during
> actual participation produces a more faithful next comparable receiving-role
> estimate.

It still does not authorize direct production mutation.

Next would be a separately frozen integration into the real finite-opportunity
stack so teammate competition and team target mass remain conserved.

## 17. What failure means

Failure means this exact snap-conditioned decomposition is not a reliable
general correction.

It does not permit returning to:
- "partial game = regression";
- DNP = zero performance;
- unexplained anomaly = trend.

The permanent player-state anomaly gate remains binding.

## 18. Integrity at freeze

- candidate outcome scores inspected: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 outcomes used: **0**;
- sportsbook/OddsAPI used: **0**;
- production changed: **false**.

Disposition at freeze:

`PRISM_INJURY_PARTIAL_RECEIVING_EXPOSURE_DECOMPOSITION_V1_FROZEN`.
