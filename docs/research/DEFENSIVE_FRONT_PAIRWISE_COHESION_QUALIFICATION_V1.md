# Defensive Front Pairwise Cohesion Qualification V1

**Status:** FROZEN PRE-QUALIFICATION — OUTCOME-FREE  
**Predictive outcomes authorized:** false  
**Production changes authorized:** false

## Scientific question

Can accumulated shared roster history among the current defensive front-seven group be
materialized pregame with broad coverage, stable identity, repeatability and incremental
information beyond immediate defensive-front roster turnover and prior aggregate defense
state?

This is a different unit and football mechanism from the failed OL-cohesion pressure
experiment. Qualification is outcome-free and does not assume predictive value.

## Frozen primary candidate

`def_front_pairwise_cohesion_prior_share`

For target scheduled team-game G:

1. form the current-week defensive-front roster set C from stable GSIS IDs;
2. gather up to the team's **20 most recent strictly prior scheduled regular-season
   games**, crossing season boundaries when available;
3. for every unordered pair `{i,j}` in C, compute:
   `prior_pair_share(i,j) = prior games in which both i and j were on that team's
   defensive-front weekly roster / available prior roster games`;
4. publish the mean of those pair shares.

If fewer than two current stable front IDs exist, state is unknown.
If no strict-prior scheduled history exists, state is `UNKNOWN_NO_PRIOR_HISTORY`.

The 20-game lookback is frozen before result inspection.

## Frozen defensive-front eligibility

A weekly-roster row is included when either normalized `position` or
`depth_chart_position` is one of:

- `DE`
- `DT`
- `NT`
- `DL`
- `EDGE`
- `OLB`
- `ILB`
- `LB`

This intentionally represents a broad front-seven personnel room rather than claiming
exact pass-rush assignment.

Backups are retained.

No target-game snaps, participation or postgame starter labels are allowed.

## Supporting diagnostics only

- current front roster count
- current unordered pair count
- prior scheduled games in 20-game window
- prior roster games available
- mean/median prior co-rostered games
- share of current pairs with zero prior co-rostered games
- immediate prior-game defensive-front roster continuity share

No alternate formula is inspected in V1.

## Historical scope

Seasons: **2019–2025**.

Target grain:

`season, week, team` for scheduled regular-season team-games.

2019 source-horizon cold starts stay in the denominator.

## Identity / integrity

Frozen gates:

- stable GSIS coverage >= **0.99**
- ambiguous same-week GSIS/team conflicts = **0**
- duplicate published team-week keys = **0**
- join fanout = **0**
- current stable front-ID count >=2 for a known state
- names diagnostic only; no fuzzy matching
- unknown never becomes zero

## Coverage

Broad denominator: all scheduled 2019–2025 regular-season team-games.

Frozen gates:

- eligible team-game rows >= **500**
- known pregame coverage >= **0.80**

No late-season or starter-only rescue.

## Temporal legality

Current weekly roster is the pregame current-group source.

Historical pair support uses only scheduled team-games strictly before the target.

Forbidden:

- target-game snaps/participation;
- target-game PBP;
- future-week rosters;
- postgame starter/assignment labels.

Temporal violations must be zero.

## Stability

This is an accumulated group state, so repeatability is a hard gate.

Report:

- adjacent-game pairs;
- adjacent-game Spearman;
- median absolute adjacent change.

Frozen gate:

- adjacent-game pairs >= **500**
- adjacent-game Spearman >= **0.50**

## Outcome-free redundancy

Reconstruct the candidate from:

- `def_front_roster_continuity_share_prev_game`
- prior completed team-game `pressure_rate_generated`
- prior completed team-game `success_rate_def`
- prior completed team-game `def_pass_epa`
- prior completed team-game `explosive_play_rate_allowed`
- team one-hot
- target week numeric

Train: **2019–2023**  
Holdout: **2024–2025**

Linear least-squares with intercept.
Numeric missingness uses train medians.
Team categories learned from train only.

Minimum rows:

- train >= **1,000**
- holdout >= **500**

Interpretation:

- R2 >=0.90: `HIGHLY_RECONSTRUCTIBLE_REDUNDANT`
- 0.75 <= R2 <0.90: `REDUNDANCY_REVIEW`
- R2 <0.75: `INCREMENTAL_INFORMATION_SURVIVES_REDUNDANCY_GATE`
- insufficient support: `REDUNDANCY_UNRESOLVED_SOURCE_THIN`

R2 >=0.75 is not experiment-ready in V1.

## Qualification disposition

`READY_FOR_FROZEN_EXPERIMENT` requires all frozen integrity, coverage, stability and
redundancy gates.

Otherwise use the existing fail-closed dispositions:

- `ENGINEERING_READY_SOURCE_THIN`
- `DESCRIPTIVE_ONLY`
- `SOURCE_BLOCKED`
- `REJECTED_INTEGRITY`

Qualification alone does not authorize a predictive experiment.

## Anti-retest boundary

M77 already tested defensive pass-rush **discontinuity counts/role deltas** in a QB
point-mean correction and failed.

This candidate is allowed only because it measures accumulated pairwise shared history,
not last-game turnover. It may not later be converted into another M77-style QB mean
correction without a separate anti-retest review.

Do not claim exact blocker-rusher assignments; M85 remains source-blocked for that
information.

## No-rescue rules

Do not:

- change the 20-game lookback;
- drop backups;
- redefine the front after seeing results;
- restrict to obvious edge rushers after seeing results;
- add snap weights;
- use target-game outcomes during qualification;
- lower coverage/stability/redundancy gates;
- use sportsbook data;
- touch Issue #535.

## Pre-result disposition

`DEFENSIVE_FRONT_PAIRWISE_COHESION_QUALIFICATION_V1_FROZEN_PRE_RESULT`
