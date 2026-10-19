# PRISM Return-State Receiving Opportunity Source V1 — Frozen Source / Lineage Plan

Date: 2026-10-18

Status: **FROZEN SOURCE AUDIT ONLY / NO PREDICTIVE OPPORTUNITY SCORING**

Model: **PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_RETURN_STATE_RECEIVING_OPPORTUNITY_V1_SOURCE_AUDIT`

## 1. Player-first question

Return-State Next-Role Persistence V1 qualified one upstream state fact:

> after a named player's participation is source-certified restored, his next
> clean participation observation behaves much more like his own pre-injury
> participation role than like the interruption / continued-decline
> participation state.

That result is **participation state only**.

It does not authorize assuming that restored snaps restore targets.

The next legal question is therefore source/lineage only:

> For this same named player, can PRISM attach exact historical receiving
> opportunity evidence to a source-certified restored-return episode at stable
> player + team + week grain, with exact finite team target volume and strictly
> prior same-player opportunity history, before any target-restoration
> relationship is scored?

This audit must answer source support only.

It does not test whether targets restore.

## 2. Scientific separation

Keep these objects separate:

1. participation restoration;
2. target earning / receiving opportunity;
3. efficiency;
4. receiving yards / receptions / touchdown outcomes.

A source-ready result in this audit authorizes only a later separately frozen
predictive opportunity-state test.

It does not authorize:
- target-share restoration;
- target-count restoration;
- route restoration;
- efficiency restoration;
- yards restoration;
- touchdown restoration.

Snaps are snaps, never routes.

## 3. Anti-retest declaration

This audit may not rescue or repackage:

- Comparable-Observation Receiving Opportunity Identity V1 — CLOSED;
- Injury-Partial Receiving Exposure Decomposition V1 — CLOSED;
- complete deletion of injury observations;
- partial weighting / reweighting of injury observations;
- the closed
  `expected participation x snap-conditioned target earning` formula;
- raw recent-share authority replacement;
- WR/TE target-share trajectory retuning;
- family-specific opportunity multipliers;
- position coefficients.

This is a new source membership / finite-opportunity bridge only.

## 4. Immutable source parents

### A. Return-from-Injury Role Restoration V1

Use only:

- run `39162437467`;
- artifact `12149417234`;
- artifact name
  `prism-return-from-injury-role-restoration-v1-source-39162437467`;
- digest
  `sha256:7d69111925b7ad48f16b07150f5e5fbfac5e592bd01430d03efd907ca778c51e`;
- file
  `return_from_injury_role_restoration_episodes.csv`;
- disposition
  `PRISM_RETURN_FROM_INJURY_ROLE_RESTORATION_V1_SOURCE_READY`.

The source-certified state used here is exactly:

`RETURN_PARTICIPATION_RESTORED`.

### B. Receiving Opportunity Earning / Participation Exposure V1 source

Use only:

- run `39161711741`;
- artifact `12149181200`;
- artifact name
  `prism-receiving-opportunity-earning-participation-exposure-v1-source-39161711741`;
- digest
  `sha256:6cae427b667db4a120a1308747d00b82b6770fe0adbc46d3efc68fce3e1db240`;
- file
  `receiving_opportunity_earning_participation_exposure_rows.csv`;
- disposition
  `PRISM_RECEIVING_OPPORTUNITY_EARNING_PARTICIPATION_EXPOSURE_V1_SOURCE_READY`.

This parent provides exact stable player + team + week identity, raw weekly
targets, exact team target volume, target share, offensive participation and
source-certified observation state.

The scorer from the CLOSED injury-partial decomposition is **not** a parent.

No provider rebuild is permitted.

## 5. Scope

Seasons:
- 2024;
- 2025.

Families:
- RB;
- WR;
- TE.

QB is excluded because this audit is receiving-opportunity state for PRISM pass
catchers / receiving backs, not QB passing opportunity.

The named player is the primary unit.

Family summaries are source diagnostics only.

No 2026 rows.

## 6. Restored-episode membership

Start only with source episodes whose exact return state is:

`RETURN_PARTICIPATION_RESTORED`.

Require:
- stable player ID nonblank;
- same team;
- same season;
- family RB/WR/TE;
- finite episode-start week;
- finite return week;
- return week > episode-start week;
- finite positive pre-episode snap baseline.

No still-reduced or no-return episode enters the restored cohort.

## 7. Exact receiving-opportunity membership

All joins are exact:

`season + team + stable_id + week`

No fuzzy names.

No position-only join.

No outcome-based identity repair.

For every restored episode, source-map these three objects.

### A. Strictly-prior same-player pre-injury opportunity history

From the immutable receiving source:
- same named player;
- same team;
- same season;
- positive offensive participation;
- strictly before episode-start week;
- latest up to four rows;
- minimum three rows required.

Persist raw source fields only:
- week;
- targets;
- team_targets;
- target_share_game;
- offense_snaps;
- offense_pct;
- observation_state;
- `playerform_weekly_row_present`.

Do not create an opportunity anchor in this audit.

Do not average, weight, regress, or compare target share to future outcomes.

### B. Restored return row

Require exact source membership at return week.

Persist:
- return week;
- targets;
- team_targets;
- target_share_game;
- offense_snaps;
- offense_pct;
- observation_state;
- `playerform_weekly_row_present`.

The return row's target result is source evidence only.

It may not define a predictive target-restoration label in this audit.

### C. Immediate next same-player observation

Identify the immediately next scheduled same-team observation after return week
from the receiving source panel.

Persist:
- next week when present;
- observation_state;
- positive-participation flag;
- targets;
- team_targets;
- target_share_game;
- `playerform_weekly_row_present`.

Do not skip a reported/unresolved immediate observation to find a later clean
one.

The source audit must report the exact next-state distribution.

It must not compare target values across states.

## 8. PlayerForm row-presence rule

A numeric zero target is source-valid opportunity evidence only when the exact
weekly PlayerForm row is present.

Therefore every positive-participation row used as:
- a strictly-prior opportunity-history row;
- a restored return opportunity row;
- or a future clean opportunity row in a later scorer

must have:

`playerform_weekly_row_present == true`.

A missing PlayerForm row may not silently become a zero-target observation.

This is a hard source-integrity rule.

## 9. Finite team opportunity / teammate competition

Receiving opportunity is finite.

For every team-week represented by a mapped player row:

- `team_targets` must be finite and > 0;
- all player rows for that team-week must resolve to one exact
  `team_targets` value;
- summed RB + WR + TE targets must not exceed exact team_targets beyond floating
  tolerance;
- residual team targets:
  `team_targets - sum(RB + WR + TE targets)`
  must be >= 0 within tolerance.

The residual is preserved.

No player target share may be altered in this source audit.

No teammate allocation is changed.

## 10. Source-only support reporting

Report separately for 2024 and 2025:

- source-certified restored RB/WR/TE episodes;
- distinct restored named players;
- episodes with >=3 exact strictly-prior receiving rows;
- episodes with exact restored-return receiving row;
- episodes with exact immediate-next observation;
- exact immediate-next state counts;
- episodes whose immediate-next observation is
  `NO_REPORT_PARTICIPATED`;
- positive-participation PlayerForm-row presence coverage;
- positive team-target coverage;
- finite-opportunity team-week count;
- finite-opportunity residual violations;
- duplicate identity count;
- chronology violations.

By family report only counts / coverage.

Do **not** report:
- whether target share increased or decreased after return;
- correlations with return state;
- target prediction error;
- target restoration rate;
- yards, receptions, touchdowns, fantasy points or sportsbook outcomes.

## 11. Frozen source gates

Adjudicate 2024 and 2025 separately.

### Identity / chronology

1. restored RB/WR/TE episodes >= 100;
2. restored named players >= 90;
3. stable-ID coverage = 100%;
4. duplicate episode identities = 0;
5. duplicate receiving-source
   `season + week + team + stable_id` identities = 0;
6. return week > episode-start week for every restored episode;
7. every pre-injury history row week < episode-start week;
8. every immediate-next observation week > return week;
9. no skipped immediate-next observations.

### Strictly-prior same-player opportunity support

10. episodes with >=3 exact strictly-prior positive-participation opportunity
    rows >= 95% of restored episodes;
11. every admitted prior row has
    `playerform_weekly_row_present == true`;
12. every admitted prior row has finite targets >= 0;
13. every admitted prior row has finite team_targets > 0;
14. every admitted prior row has finite target_share_game.

### Restored-return row support

15. exact restored-return receiving-row coverage >= 99%;
16. every admitted restored-return row has positive participation;
17. every admitted restored-return row has
    `playerform_weekly_row_present == true`;
18. every admitted restored-return row has finite targets >= 0;
19. every admitted restored-return row has finite team_targets > 0;
20. every admitted restored-return row has finite target_share_game.

### Immediate-next source support

21. exact immediate-next observation identified for >= 75% of restored episodes;
22. immediate-next state vocabulary contains only source-certified states;
23. immediate-next clean `NO_REPORT_PARTICIPATED` episodes > 0;
24. any positive-participation immediate-next row has
    `playerform_weekly_row_present == true` before it can be used by a later
    predictive scorer.

### Finite opportunity

25. exact team-target value is unique within every represented team-week;
26. represented team-weeks with finite positive team_targets = 100%;
27. RB + WR + TE target sum exceeds team_targets in zero represented team-weeks
    beyond tolerance;
28. residual team targets are never materially negative.

### Integrity

29. predictive target relationships scored = 0;
30. target-restoration labels created = 0;
31. coefficients fit = 0;
32. thresholds searched = 0;
33. routes inferred from snaps = false;
34. 2026 outcomes used = 0;
35. sportsbook/OddsAPI inputs used = 0;
36. paid OddsAPI calls = 0;
37. production changed = false.

Do not lower gates after execution.

## 12. Required outputs

Persist:
- one restored-episode source-membership row per episode;
- one normalized strictly-prior history row file;
- immediate-next source-state counts;
- finite team-opportunity audit;
- family/season coverage summary;
- source result JSON.

The episode source-membership artifact may retain raw target and team-target
fields for a later frozen scorer.

No predictive relationship may be computed in this source audit.

## 13. If source-ready

A pass authorizes only a separate predictive freeze.

The next legal question would be:

> Once this named player's participation is source-certified restored, does his
> receiving opportunity state return toward his own strictly-prior pre-injury
> opportunity identity, after preserving finite team targets and teammate
> competition?

That future contract must be frozen before target outcomes are compared.

It must choose exactly one opportunity-state formula and exactly one comparator
before scoring.

It may not assume snap restoration equals target restoration.

## 14. If source-blocked

Fail closed.

Document the exact missing identity, chronology, PlayerForm membership or finite
team-opportunity support.

Do not:
- rebuild providers to rescue the join;
- convert missing PlayerForm rows to zero targets;
- fuzzy-match players;
- lower source gates;
- skip bad immediate-next states;
- reuse the closed exposure x earning formula;
- use 2026 outcomes;
- use sportsbook information.

## 15. Integrity at freeze

Before this freeze:
- target-restoration relationships inspected: **0**;
- predictive target outcomes scored: **0**;
- target-restoration labels created: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- routes inferred from snaps: **false**;
- 2026 outcomes used: **0**;
- sportsbook/OddsAPI used: **0**;
- paid OddsAPI calls: **0**;
- production changed: **false**.

Disposition at freeze:

`PRISM_RETURN_STATE_RECEIVING_OPPORTUNITY_V1_SOURCE_FROZEN`.
