# WR Player Opportunity-Difficulty Trust Selectivity V1 — Frozen Plan

Date: 2026-10-15

Status: **FROZEN BEFORE SCORING / RESEARCH ONLY**

Version:
`WR_PLAYER_OPPORTUNITY_DIFFICULTY_TRUST_SELECTIVITY_V1`

## Parent evidence

Named-WR opportunity difficulty is already confirmed:

- parent mechanism:
  `WR_PLAYER_PERSISTENCE_MIXED_MECHANISM`
- run `37634044900`
- artifact `11487359185`
- digest `sha256:13bb2df4d239f9f2832370520c3d31a08ed4f39c782766fbfb96295edc11c4e3`

Player-specific opportunity uncertainty:
- `WR_PLAYER_OPPORTUNITY_UNCERTAINTY_CONFIRMED`
- run `38595088817`
- artifact `11908832554`
- pooled CRPS gain about 1.47%
- player-cluster CI strictly positive.

The exact WR allocation-only reinterpretation is CLOSED and is not reopened.

## Question

Without changing WR-R15 targets, receiving-yard mean, distribution, team target
mass, or any production value:

> Does the named WR's own strictly-prior opportunity-projection difficulty
> identify when his **final WR-R15 receiving-yard point projection** is
> comparatively reliable versus fragile?

This is a trust/selectivity question, not another width law.

## Frozen sources

### Same-player difficulty source

Use only:
`wr_player_mechanism_persistence_rows.csv`

from run `37634044900` / artifact `11487359185`.

Required:
- season
- week
- team
- player_clean_key
- wr_rank
- prior_games
- prior8_abs_opportunity_error_mean
- prior8_abs_efficiency_error_mean
- current_opportunity_error
- current_efficiency_error
- current_total_error
- current_abs_total_error

Target seasons:
- 2023 development
- 2024 independent confirmation.

### Exact WR-R15 mean authority

Use only recovered exact:
`wr_r15_confirmation_predictions.csv`

from recovery run `37782189538` / artifact `11552447166`.

Filter exactly:
`variant == WR_R15_WR1_ANCHORED_PARTICIPATION`.

Required:
- event_id
- season
- week
- team
- player_clean_key
- player
- wr_rank
- mc_rec_yards
- actual_rec_yards.

## Frozen parent parity gate

Join one-to-one on:
`season, week, team, player_clean_key`.

Require:
- exactly 3,224 matched scoreable parent rows;
- exactly 211 distinct WRs;
- no duplicate identities;
- exact WR-R15 absolute error
  `abs(mc_rec_yards - actual_rec_yards)`
  matches parent `current_abs_total_error` within `1e-9`;
- signed WR-R15 error
  `mc_rec_yards - actual_rec_yards`
  matches parent `current_total_error` within `1e-9`;
- decomposition identity
  `current_opportunity_error + current_efficiency_error == current_total_error`
  within `1e-9`.

If parity fails, fail closed before scientific scoring.

## Frozen trust score

For every exact `(season, week)`:
1. `player_mae = prior8_abs_opportunity_error_mean`;
2. rank `player_mae` across eligible WRs using
   `rank(method="average", pct=True)`;
3. lower = more trustworthy.

Define:
- `HIGH_TRUST <= 0.25`
- `LOW_TRUST >= 0.75`
- otherwise `MIDDLE`.

No alternate threshold.

For the reliability target:
- `abs_total_error = abs(mc_rec_yards - actual_rec_yards)`;
- within the same `(season, week)`, calculate
  `error_badness_rank` with the same average-percentile convention.

Primary association:
Pearson correlation between the two within-week percentile ranks.

This is a rank-on-rank association; the week-wide difficulty level is removed
by construction.

## Frozen support

2023:
- >=1,200 rows
- >=140 WRs
- HIGH_TRUST >=250 rows
- LOW_TRUST >=250 rows.

2024:
- >=1,600 rows
- >=160 WRs
- HIGH_TRUST >=350 rows
- LOW_TRUST >=350 rows.

Pooled:
- >=3,000 rows
- >=190 WRs.

## Frozen gates

Qualification requires ALL:

1. source/parity integrity passes;
2. support passes;
3. 2023 primary rank association > 0;
4. 2024 primary rank association > 0;
5. pooled player-cluster 95% CI lower bound > 0;
6. pooled game-cluster (`event_id`) 95% CI lower bound > 0;
7. 2024 player-cluster 95% CI lower bound > 0;
8. 2024 game-cluster 95% CI lower bound > 0;
9. 2023 HIGH_TRUST raw absolute error < overall raw absolute error;
10. 2024 HIGH_TRUST raw absolute error < overall;
11. 2023 LOW_TRUST raw absolute error > overall;
12. 2024 LOW_TRUST raw absolute error > overall;
13. 2023 LOW_TRUST raw absolute error > HIGH_TRUST;
14. 2024 LOW_TRUST raw absolute error > HIGH_TRUST;
15. sportsbook inputs = 0;
16. OddsAPI calls = 0;
17. 2026 outcomes = 0;
18. models fit = 0;
19. projection/distribution values changed = 0;
20. production unchanged.

Bootstrap:
- 5,000 reps;
- paired rows;
- player cluster and event/game cluster separately;
- seeds fixed in evaluator before output.

## Diagnostics only

Report but do not allow to rescue:
- WR1 vs WR2+;
- prior efficiency-difficulty rank association;
- raw player-MAE / week-pool-MAE ratio;
- high/low trust raw projection bias.

## Frozen dispositions

All gates:
`WR_PLAYER_OPPORTUNITY_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED`

Otherwise:
`NO_ACTIONABLE_WR_PLAYER_OPPORTUNITY_DIFFICULTY_TRUST_SELECTIVITY`

Integrity failure:
`WR_PLAYER_OPPORTUNITY_DIFFICULTY_TRUST_INTEGRITY_FAILURE`

## Boundaries

Do not:
- modify WR-R15;
- reinterpret LOW_TRUST as automatic under/fade;
- reinterpret HIGH_TRUST as automatic over/bet;
- fit thresholds or weights;
- rescue by WR1/WR2+;
- alter the last-8/min-4 parent;
- use 2025/2026 outcomes;
- reopen WR allocation-only width;
- reopen target-depth width;
- use sportsbook lines.

A positive result authorizes only a separately frozen future-week WR trust
shadow around the unchanged WR projection.
