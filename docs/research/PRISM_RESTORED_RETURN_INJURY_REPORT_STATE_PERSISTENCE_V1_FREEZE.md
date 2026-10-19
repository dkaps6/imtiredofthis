# PRISM Restored-Return Injury-Report State Persistence V1 — Frozen Predictive Contract

Date: 2026-10-18

Status: **FROZEN BEFORE PREDICTIVE SCORING / NAMED-PLAYER HEALTH-STATE RELIABILITY ONLY**

Model: **PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_RESTORED_RETURN_INJURY_REPORT_STATE_PERSISTENCE_V1`

## 1. Player-first question

PRISM now has qualified evidence that a source-certified restored participation
state persists toward the named player's own pre-injury participation role.

But two named players can both have
`RETURN_PARTICIPATION_RESTORED` while differing in an important current-state
fact:

- one remains on the weekly injury report at the restored return;
- the other has no matching weekly injury-report row at that return.

This V1 asks:

> Once this named player's participation is source-certified restored, does the
> exact injury-report state at the return observation distinguish whether his
> immediately next scheduled observation is a clean no-report participation
> state versus another reported / unresolved participation state?

This is a current health-state / reliability mechanism.

It is not a target, carry, route, efficiency, yard or touchdown mechanism.

## 2. Important semantic boundary

`return_injury_reported == false` means only:

> no matching weekly injury-report row was present in the source for that
> player-week.

It does **not** mean:
- medically healthy;
- fully recovered;
- unrestricted;
- identical role;
- guaranteed future availability.

`return_injury_reported == true` means only:
- a matching weekly injury-report row remained present.

No injury severity score is inferred.

## 3. Distinctness / anti-retest

This mechanism is distinct from:

- Return-State Next-Role Persistence V1 — QUALIFIED STATE ONLY;
- the source-blocked receiving-opportunity bridge;
- Comparable-Observation Receiving Opportunity Identity V1 — CLOSED;
- Injury-Partial Receiving Exposure Decomposition V1 — CLOSED;
- raw recent-share authority;
- active-role allocation;
- teammate-vacancy elasticity.

It does not change:
- the frozen return-restoration threshold;
- the pre-injury snap anchor;
- any opportunity formula;
- any position coefficient.

No alternate injury-report subgroup rule may be searched after outcomes.

## 4. Immutable source authority

### Return-state source

Use only:
- run `39162437467`;
- artifact `12149417234`;
- digest
  `sha256:7d69111925b7ad48f16b07150f5e5fbfac5e592bd01430d03efd907ca778c51e`;
- file
  `return_from_injury_role_restoration_episodes.csv`.

Required source state:

`RETURN_PARTICIPATION_RESTORED`.

Required current-state field:

`return_injury_reported`.

### Observation-health source

Use only:
- run `39154145666`;
- artifact `12146625382`;
- digest
  `sha256:d74f2d3c37a88aa639527aedf719e7e08001d464c722613b46291cda0f74cbeb`;
- file
  `player_observation_health_state_rows.csv`.

No provider rebuild.

## 5. Scope

Development:
- 2024.

Independent confirmation:
- 2025.

Families:
- QB;
- RB;
- WR;
- TE.

Named player is primary.

Family/position summaries are diagnostics only.

No 2026 rows.

## 6. Eligible restored return

Episode must have:
1. exact stable player ID;
2. exact same team + season;
3. return state exactly
   `RETURN_PARTICIPATION_RESTORED`;
4. finite return week;
5. return week > episode-start week;
6. exact unique return health row;
7. return health row has positive offensive participation;
8. return-source `return_injury_reported` exactly matches health-source
   `injury_reported`.

Any mismatch fails closed.

## 7. Frozen current health-state label

Exactly two current states:

### `RESTORED_RETURN_REPORT_PRESENT`

`return_injury_reported == true`

### `RESTORED_RETURN_NO_REPORT`

`return_injury_reported == false`

No third state.

No injury/body-part subset.

No questionable/doubtful/out subset.

No practice-status subset.

No severity inference.

## 8. Frozen target observation

Use the **immediately next scheduled same-team player observation after return**
from the immutable health source.

Do not skip forward.

Chronology:

`episode_start_week < return_week < next_week`

If no immediate next observation exists:
- exclude from predictive scoring;
- count as `NO_NEXT_OBSERVATION`.

For observed next rows, retain exact source observation state.

Allowed:
- `NO_REPORT_PARTICIPATED`;
- `REPORTED_INJURY_PARTICIPATED`;
- `REPORTED_INJURY_NO_OFFENSIVE_SNAPS`;
- `NO_REPORT_NO_OFFENSIVE_SNAPS_UNRESOLVED`.

No invented healthy/inactive label.

## 9. Frozen binary state outcome

For the immediate next observation:

`next_clean_no_report_participation = 1`

only if exact state is:

`NO_REPORT_PARTICIPATED`.

Otherwise:

`next_clean_no_report_participation = 0`.

This outcome means only clean source state under the V1 vocabulary.

It does not mean medically healthy.

## 10. Named-player aggregation

Within each season and current return-report state:

1. for each named player, average his binary
   `next_clean_no_report_participation` across eligible episodes;
2. average those player means equally.

Define:

`no_report_equal_player_clean_rate`

and

`report_present_equal_player_clean_rate`.

Frozen contrast:

`report_state_clean_rate_gain =
no_report_equal_player_clean_rate
- report_present_equal_player_clean_rate`

Positive means absence of a weekly injury-report row at the restored return is
associated with a higher next-observation clean-state rate than continued
report presence.

No coefficient is fit.

## 11. Player-cluster bootstrap

2025 confirmation only:

- sample named players with replacement from the union of eligible players;
- preserve all episodes and current-state labels belonging to each sampled
  player;
- recompute equal-player state rates and their difference;
- 5,000 replicates;
- seed `20261018`.

If a bootstrap replicate lacks either current state, discard that replicate.

Require at least 4,500 valid replicates.

## 12. Frozen support gates

Per season:

1. scored episodes >= 120;
2. scored named players >= 100;
3. `RESTORED_RETURN_REPORT_PRESENT` episodes >= 30;
4. report-present named players >= 25;
5. `RESTORED_RETURN_NO_REPORT` episodes >= 30;
6. no-report named players >= 25;
7. immediate-next observation coverage >= 75% of restored episodes;
8. stable-ID coverage = 100%;
9. duplicate episode identities = 0;
10. return-source / health-source injury-report mismatches = 0;
11. chronology violations = 0;
12. skipped immediate-next observations = 0;
13. next-state values are within the frozen source vocabulary.

Do not lower support gates.

## 13. Frozen qualification gates

### 2024 development

14. `report_state_clean_rate_gain > 0`;
15. no-report equal-player clean rate >
    report-present equal-player clean rate.

### 2025 independent confirmation

16. `report_state_clean_rate_gain > 0`;
17. no-report equal-player clean rate >
    report-present equal-player clean rate;
18. valid player-cluster bootstrap replicates >= 4,500;
19. bootstrap 95% lower bound for
    `report_state_clean_rate_gain` > 0.

All gates must pass.

Qualification:

`PRISM_RESTORED_RETURN_INJURY_REPORT_STATE_PERSISTENCE_V1_QUALIFIED_RELIABILITY_ONLY`

Failure:

`PRISM_RESTORED_RETURN_INJURY_REPORT_STATE_PERSISTENCE_V1_CLOSED`

No family rescue.

## 14. What qualification would mean

Qualification would mean:

> for a named player whose participation has already restored, whether a weekly
> injury-report row remains present carries additional source-valid information
> about persistence into the next clean no-report participation state.

This may later support a **reliability / unresolved-health-state flag**.

It would not authorize:
- changing projected snaps automatically;
- changing target/carry share automatically;
- a yards multiplier;
- an injury severity coefficient;
- a position-wide penalty.

## 15. What failure would mean

Failure closes this exact weekly-report-presence persistence mechanism.

Do not rescue with:
- injury body part;
- practice status;
- questionable/doubtful/out subgroups;
- position-specific cuts;
- alternate lookahead;
- changed return thresholds;
- 2026 outcomes;
- sportsbook availability.

Failure does not weaken the qualified participation-restoration state.

## 16. Integrity at freeze

Before this freeze:
- target outcome rows scored under this mechanism: **0**;
- report-state predictive contrasts inspected: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- targets/carries/receptions/yards/TDs used: **0**;
- routes inferred from snaps: **false**;
- 2026 outcomes used: **0**;
- sportsbook/OddsAPI used: **0**;
- paid OddsAPI calls: **0**;
- production changed: **false**.

Disposition at freeze:

`PRISM_RESTORED_RETURN_INJURY_REPORT_STATE_PERSISTENCE_V1_FROZEN`.
