# OL Roster Pairwise Cohesion Qualification V1

**Status:** FROZEN PRE-QUALIFICATION — OUTCOME-FREE  
**Parent:** `PERSONNEL_CONTINUITY_CONTRACT_V1_FROZEN`  
**Prior qualified input:** `ol_roster_continuity_share_prev_game`  
**Predictive outcomes authorized:** false  
**Production changes authorized:** false

## Scientific question

Can accumulated shared OL roster history identify a pregame line-cohesion state that is materially different from simple last-game personnel turnover and not reconstructible from existing strict-prior team state plus the already-qualified immediate continuity feature?

This is explicitly designed to satisfy the post-M77 requirement for a mechanism **beyond discontinuity counts**.

## Distinction from closed M77 / OL continuity V1

This V1 does **not** ask who changed since the previous game.

It asks how much shared historical team-roster experience the **current OL group** has accumulated together before the target game.

Two target games may have identical last-game continuity while having very different accumulated pairwise shared history. That is the intended incremental mechanism.

No QB/RB outcome is read during qualification.

## Frozen primary candidate

`ol_roster_pairwise_cohesion_prior_share`

For target scheduled team-game G:

1. form the current-week OL roster set C from stable GSIS IDs using the same frozen OL eligibility as OL Roster Continuity V1;
2. gather up to the team's **20 most recent strictly prior scheduled regular-season games** across season boundaries;
3. for every unordered pair `{i,j}` in C, compute:
   `prior_pair_share(i,j) = prior games in which both i and j were on that team's OL weekly roster / available prior games`;
4. set:
   `ol_roster_pairwise_cohesion_prior_share = mean(prior_pair_share(i,j))`
   across all unordered current-OL pairs.

If fewer than two current OL stable IDs exist, the state is unknown.

If no strictly prior scheduled team game is available in the source horizon, the state is
`UNKNOWN_NO_PRIOR_HISTORY`, not zero.

The 20-game lookback is frozen before result inspection.

## Supporting diagnostics only

- current OL roster count
- current unordered OL pair count
- prior scheduled team-games available, capped at 20
- mean prior co-rostered games across current pairs
- median prior co-rostered games across current pairs
- share of current pairs with zero prior co-rostered games
- immediate prior-game continuity from the already-qualified V1 candidate

No alternative cohesion formula is inspected in V1.

## Frozen OL eligibility

Exactly reuse the V1 weekly-roster eligibility:

- `position` in `C,G,T,OT,OG,OL`; or
- `depth_chart_position` in `LT,LG,C,RG,RT,OT,OG,OL,G,T`.

Backups remain in the broad roster state.

Do not substitute target-game starters, snaps or participation.

## Historical scope

Seasons: **2019–2025**.

Target grain:

`season, week, team` for scheduled regular-season team-games.

Strict-prior pair history may cross season boundaries but may never use a roster snapshot from the target or future game as historical support.

2019 cold-start rows remain in the broad denominator.

## Identity / integrity

Frozen gates:

- stable GSIS identity coverage >= **0.99**
- ambiguous same-week GSIS/team conflicts = **0**
- duplicate published team-week keys = **0**
- schedule join fanout = **0**
- current OL stable-ID count >= 2 for a known state
- unknown is never coerced to zero
- names diagnostic only; no fuzzy matching

## Pregame coverage

Broad denominator: all scheduled 2019–2025 regular-season team-games.

Frozen gates:

- eligible team-game rows >= **500**
- known pregame cohesion coverage >= **0.80**

No late-season-only or starter-only rescue.

## Temporal legality

For target kickoff/game week T:

- current weekly roster is the pregame current-group source;
- pairwise history uses only scheduled team-games strictly before T;
- target-game snaps, target-game participation, target-game PBP and postgame lineup labels are forbidden;
- future-week rosters are forbidden.

Temporal violations must be **0**.

## Stability

Unlike immediate continuity, pairwise cohesion is intended to be an accumulated team/group state, so adjacent-game repeatability is applicable.

Report:

- adjacent-game pair count;
- adjacent-game Spearman;
- median absolute adjacent change.

Frozen stability gate:

- adjacent-game pairs >= **500**
- adjacent-game Spearman >= **0.50**

Failure of this gate yields `DESCRIPTIVE_ONLY` unless a harder integrity/source gate produces a different fail-closed disposition.

Do not lower the threshold after result inspection.

## Outcome-free redundancy

Test whether pairwise cohesion is reconstructible from already-known pregame context.

Frozen reconstruction inputs:

- `ol_roster_continuity_share_prev_game`
- prior completed team-game `pressure_rate_allowed`
- prior completed team-game `success_rate_off`
- prior completed team-game `dropback_rate`
- prior completed team-game `plays_est`
- prior completed team-game `proe`
- team one-hot
- target week numeric

Train seasons: **2019–2023**  
Holdout seasons: **2024–2025**

Linear least-squares with intercept.

Numeric missingness is imputed with train medians.
Team categories are learned from train only.

Minimum rows:

- train >= **1,000**
- holdout >= **500**

Frozen redundancy interpretation:

- R2 >= **0.90**: `HIGHLY_RECONSTRUCTIBLE_REDUNDANT`
- 0.75 <= R2 < 0.90: `REDUNDANCY_REVIEW`
- R2 < 0.75: `INCREMENTAL_INFORMATION_SURVIVES_REDUNDANCY_GATE`
- insufficient rows: `REDUNDANCY_UNRESOLVED_SOURCE_THIN`

A candidate at or above 0.90 is `DESCRIPTIVE_ONLY`.
A candidate in the 0.75–0.90 review band is not automatically experiment-ready; V1 closes as `DESCRIPTIVE_ONLY` pending a separately frozen rationale.

## Qualification disposition

`READY_FOR_FROZEN_EXPERIMENT` requires:

- identity/integrity clean;
- broad coverage >= 0.80;
- eligible rows >= 500;
- stability gate passes;
- clear accumulated-cohesion mechanism;
- redundancy R2 < 0.75;
- no leakage violation.

Other dispositions:

- `ENGINEERING_READY_SOURCE_THIN`
- `DESCRIPTIVE_ONLY`
- `SOURCE_BLOCKED`
- `REJECTED_INTEGRITY`

Qualification alone does not authorize outcome testing.

## No-retest / no-rescue rules

Do not:

- turn this into another previous-game turnover ratio;
- use target-game starters/snaps/participation;
- restrict to five starters after seeing the result;
- shorten/lengthen the 20-game lookback after seeing the result;
- add pair weights based on target-game role;
- lower coverage, stability or redundancy gates;
- inspect QB/RB/WR/TE target outcomes during qualification;
- use sportsbook information;
- touch Issue #535.

If pairwise cohesion fails, preserve the failure and move to another distinct mechanism.

## Pre-result disposition

`OL_ROSTER_PAIRWISE_COHESION_QUALIFICATION_V1_FROZEN_PRE_RESULT`
