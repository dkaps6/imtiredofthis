# QB First-Down Choice Economics Source Audit V1 — Frozen Plan

## Purpose

Audit whether a genuinely new, leakage-safe pregame information family can be constructed before any predictive test of the newly localized shared QB/receiver first-down opportunity mechanism.

Parent work established the chain:

`TEAM PASS OPPORTUNITY -> PASS-OPPORTUNITY RATE -> WITHIN-STATE PASS PROPENSITY -> FIRST-DOWN PLAY SELECTION`

The new source hypothesis is **choice economics**, not another pass-tendency statistic:

> Before kickoff, does the data support estimating how much better an offense has been at pass-origin plays versus designed runs on first down, and how much more vulnerable the upcoming opponent has been to pass-origin plays versus designed runs on first down?

This migration audits source construction only. It is prohibited from reading the parent first-down residual, QB/WR residuals, passing-yard outcomes, sportsbook inputs, or any target-game outcome as a predictor.

## Parent lineage

- Parent branch: `research-qb-pass-rate-state-shared-attribution-v1`
- Parent result commit: `13d9ad428deaada4d92fe303a55948375dc1a491`
- Parent run: `34544457497`
- Parent job: `103093958683`
- Parent artifact: `10178499888`
- Parent digest: `sha256:db401ff605820441605c985bf2262cb6114e1b06c7468605bc6b0a0708c84f4d`
- Parent disposition: `FIRST_DOWN_SHARED_PRIMARY_DIAGNOSTIC`

## Explicit novelty boundary

This is not:

- M42 generic team pass-rate history;
- M56 all-down defense-only `pass_funnel_epa = def_pass_epa - def_rush_epa`;
- M64/M65 pace, possession, score-state, or occupancy modeling;
- M67 early-down/neutral/third-long DBR tendency;
- M68 opening-script/playcaller/leverage history;
- M77-M79 generic personnel/inactive correction;
- M81 motion/screen/RPO or other FTN tactical-history family;
- M83 comparable-opponent defensive adaptive gameplan;
- M87/M88 pass-funnel + short/intermediate threshold regime;
- Phase-J QB distribution-state selection.

The exact new information object is the **first-down relative value of choosing pass-origin versus designed-run football**, separately for the target offense and opponent defense. The source audit may not combine these values into a fitted model or select a weighting after results.

## Source

Primary source: nflverse regular-season play-by-play loaded through the repository's maintained nflreadpy/nflverse path.

Historical seasons required:

- 2022: prior-history support for 2023 targets;
- 2023;
- 2024;
- 2025.

Target source-audit seasons:

- 2023;
- 2024;
- 2025.

Target team-week/opponent identities come from nflverse regular-season schedules, not target-game PBP outcomes.

No 2026 outcomes are read.

## Frozen first-down opportunity semantics

A source play is eligible when all are true:

- regular season;
- valid possession team and defense team;
- `down == 1`;
- `qb_dropback == 1` OR `rush_attempt == 1`;
- `two_point_attempt != 1` when field exists;
- `no_play != 1` when field exists.

### Pass-origin choice

`PASS_ORIGIN = qb_dropback == 1`

This intentionally includes sacks and QB scrambles as pass-origin decisions, consistent with corrected M89 team pass-opportunity semantics.

### Designed-run choice

`DESIGNED_RUN = rush_attempt == 1 AND qb_dropback != 1`

If `qb_kneel` is available, kneels are excluded from designed-run economics. A play cannot be both `PASS_ORIGIN` and `DESIGNED_RUN`.

No target-game play is ever allowed into its own pregame history.

## Frozen source quantities

For every completed team-game, construct offense and defense-observed first-down quantities:

### EPA

- offense first-down pass-origin EPA;
- offense first-down designed-run EPA;
- defense first-down pass-origin EPA allowed;
- defense first-down designed-run EPA allowed.

### Success rate

- offense first-down pass-origin success rate;
- offense first-down designed-run success rate;
- defense first-down pass-origin success rate allowed;
- defense first-down designed-run success rate allowed.

Also retain pass-origin and designed-run play counts for source-density auditing.

## Strict-prior pregame construction

For each target `(season, week, team, opponent)` in 2023-2025:

- offense history = target offense's last 8 completed regular-season games strictly before the target week, including the previous season where applicable;
- opponent-defense history = upcoming opponent's last 8 completed regular-season defensive games strictly before the target week, including the previous season where applicable;
- league prior = all eligible games strictly before the target week.

Every source rate/mean is shrunk by **4 league-equivalent games** toward the corresponding strictly-prior league first-down quantity.

For each side/channel, the shrinkage weight is based on eligible first-down **play count**, not merely number of games:

`shrunk_value = (history_play_count * history_value + league_equivalent_play_count * league_value) / (history_play_count + league_equivalent_play_count)`

where:

`league_equivalent_play_count = 4 * strictly_prior_league_mean_play_count_per_team_game_for_that_channel`

This preserves the frozen four-game shrinkage concept while respecting unequal pass/run sample counts.

No target-game PBP enters these quantities.

## Frozen primitive outputs

For every target team-week, output these eight strictly-prior primitives:

1. `off_fd_pass_epa`
2. `off_fd_run_epa`
3. `def_fd_pass_epa_allowed`
4. `def_fd_run_epa_allowed`
5. `off_fd_pass_success`
6. `off_fd_run_success`
7. `def_fd_pass_success_allowed`
8. `def_fd_run_success_allowed`

Also output four predeclared, deterministic **difference descriptors** for downstream use if the source qualifies:

- `off_fd_epa_pass_minus_run = off_fd_pass_epa - off_fd_run_epa`
- `def_fd_epa_pass_minus_run = def_fd_pass_epa_allowed - def_fd_run_epa_allowed`
- `off_fd_success_pass_minus_run = off_fd_pass_success - off_fd_run_success`
- `def_fd_success_pass_minus_run = def_fd_pass_success_allowed - def_fd_run_success_allowed`

These descriptors are not scored against outcomes in this migration.

No combined offense/opponent weighting is authorized yet.

## Frozen source audit outputs

For each target season and pooled 2023-2025 report:

- target team-weeks;
- source-row coverage for all eight primitives;
- finite coverage for all four difference descriptors;
- median / p10 prior offense games;
- median / p10 prior opponent-defense games;
- median / p10 prior first-down pass-origin plays by offense;
- median / p10 prior first-down designed-run plays by offense;
- median / p10 prior first-down pass-origin plays faced by opponent defense;
- median / p10 prior first-down designed-run plays faced by opponent defense;
- EPA non-null rate among eligible source plays;
- success non-null rate among eligible source plays;
- source exclusivity/reconciliation checks for pass-origin vs designed-run plays;
- maximum prior `(season, week)` ordinal used relative to each target ordinal.

Also report 2026 deployment feasibility from source schema only:

- schedule source available;
- PBP source contract supports in-season completed-game updates;
- all required fields are part of the same historical/live nflverse PBP contract.

No 2026 target outcomes are required or permitted.

## Frozen source-eligibility gates

The family qualifies for exactly one later predictive screen only if **all** are true:

1. exact expected target team-weeks from regular-season schedules for each of 2023, 2024, 2025;
2. no duplicate `(season, week, team)` target keys;
3. all eight primitive source quantities have >= `99%` finite target-row coverage pooled and >= `98%` in every target season;
4. all four difference descriptors have >= `99%` finite target-row coverage pooled and >= `98%` in every target season;
5. >= `95%` of target rows have at least 4 prior offense games;
6. >= `95%` of target rows have at least 4 prior opponent-defense games;
7. >= `95%` of target rows have at least 40 prior first-down pass-origin plays for the offense;
8. >= `95%` have at least 30 prior first-down designed-run plays for the offense;
9. >= `95%` have at least 40 prior first-down pass-origin plays faced by the opponent defense;
10. >= `95%` have at least 30 prior first-down designed-run plays faced by the opponent defense;
11. EPA non-null coverage among eligible historical source plays >= `99%`;
12. success non-null coverage among eligible historical source plays >= `99%`;
13. pass-origin and designed-run source choices are mutually exclusive;
14. every target source value uses only games with ordinal strictly less than the target ordinal;
15. zero sportsbook inputs;
16. zero parent first-down residual / QB residual / WR residual reads;
17. zero model fitting;
18. zero production changes.

The numeric density gates may not be relaxed after results are visible.

## Frozen disposition

Exactly one of:

- `FIRST_DOWN_CHOICE_ECONOMICS_SOURCE_QUALIFIED`
- `FIRST_DOWN_CHOICE_ECONOMICS_SOURCE_BLOCKED`
- `MECHANICAL_OR_INTEGRITY_FAIL_NO_SCIENCE`

A scientifically clean source failure is `SOURCE_BLOCKED`; it is not a reason to loosen thresholds or inspect predictive outcomes.

## If source qualifies

A separate migration must be frozen before any first-down residual is opened.

Because 2024-2025 were used to discover/localize the mechanism, the first predictive development screen should preferentially use **2023 first-down choice residuals with 2022 strict-prior history** to select/falsify a fixed candidate construction. Only if that separate 2023 screen passes may an unchanged construction be evaluated on the 2024-2025 M89/shared receiver cohorts, with explicit acknowledgement that those seasons are mechanism-discovery data rather than pristine untouched holdouts.

No current production change is authorized by this source audit.
