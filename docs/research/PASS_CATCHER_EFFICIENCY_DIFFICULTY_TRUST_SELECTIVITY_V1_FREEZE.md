# Pass-Catcher Efficiency-Difficulty Trust Selectivity V1 — Frozen Plan

Date: 2026-10-15

Status: **FROZEN BEFORE SCORING / RESEARCH ONLY / NO PRODUCTION CHANGE**

Version:
`PASS_CATCHER_EFFICIENCY_DIFFICULTY_TRUST_SELECTIVITY_V1`

## Why this is a distinct lane

Closed family:
`PLAYER_EFFICIENCY_TAIL_STABILITY_NULL` tested whether a player's own recent realized YPC/YPT **volatility** predicted next-game efficiency tail risk. It failed the frozen three-season replication gate and may not be rescued by alternate windows, subgroups, pooling, thresholds or matchup interactions.

Also closed:
- receiver drop/catchability conversion V1 as a mean mechanism;
- WR route-matchup direct full-stack mean integration;
- generic WR/TE depth-YPT direct transmission;
- WR allocation-only second width bridge;
- TE marginal-width and joint-Dirichlet mappings.

This V1 does **not** retest those families.

The surviving parent evidence is different:
- WR parent `WR_PLAYER_PERSISTENCE_MIXED_MECHANISM`:
  - run `37634044900`, artifact `11487359185`;
  - 3,224 rows / 211 WRs;
  - efficiency-difficulty persistence E2 PASS:
    2023 rho +0.3254, 2024 +0.2501, pooled +0.2833;
  - signed efficiency persistence E1 failed.
- TE parent `TE_PLAYER_PERSISTENCE_MIXED_MECHANISM`:
  - run `37631011196`, artifact `11486082927`;
  - 2,005 rows / 121 TEs;
  - efficiency-difficulty persistence E2 PASS:
    2024 +0.1806, 2025 +0.2592, pooled +0.2240;
  - signed efficiency persistence E1 failed.

The question is therefore reliability/selectivity, not direction.

## Frozen question

Without changing any WR-R15 / TE-R5P mean or distribution:

> Does the named WR/TE's own strictly-prior **efficiency projection difficulty**
> identify when his existing final receiving-yard projection is comparatively
> reliable versus fragile, **after controlling for his strictly-prior
> opportunity projection difficulty**?

This is intended to isolate a named-player efficiency-reliability state from the already-qualified opportunity-reliability state.

It does **not** authorize:
- a receiving-yard mean correction;
- a YPT mean correction;
- a distribution-width law;
- an over/under direction;
- an automatic bet/fade rule.

## Frozen sources

### WR

Use only:
`wr_player_mechanism_persistence_rows.csv`

from run `37634044900` / artifact `11487359185`.

Target seasons:
- 2023 development;
- 2024 independent confirmation.

Exact expected parent:
- 3,224 rows;
- 211 WRs.

Required columns:
- season
- week
- team
- player_clean_key
- wr_rank
- prior_games
- prior8_abs_opportunity_error_mean
- prior8_abs_efficiency_error_mean
- current_opportunity_error
- current_abs_opportunity_error
- current_efficiency_error
- current_abs_efficiency_error
- current_total_error
- current_abs_total_error

### TE

Use only:
`te_player_mechanism_persistence_rows.csv`

from run `37631011196` / artifact `11486082927`.

Target seasons:
- 2024 development;
- 2025 independent confirmation.

Exact expected parent:
- 2,005 rows;
- 121 TEs.

Required columns:
- season
- week
- team
- player_clean_key
- prior_games
- prior8_abs_opportunity_error_mean
- prior8_abs_efficiency_error_mean
- current_opportunity_error
- current_abs_opportunity_error
- current_efficiency_error
- current_abs_efficiency_error
- current_total_error
- current_abs_total_error

## Frozen integrity

For each position:
- exact expected row count and player count;
- unique `season, week, team, player_clean_key`;
- `prior_games >= 4`;
- all difficulty/error fields finite;
- nonnegative absolute-error fields;
- decomposition identity:
  `current_opportunity_error + current_efficiency_error == current_total_error`
  within `1e-9`;
- absolute identity:
  `abs(current_total_error) == current_abs_total_error`
  within `1e-9`;
- no 2026 outcomes;
- no sportsbook input;
- no OddsAPI call;
- no production mutation.

Any integrity failure is terminal for this V1.

## Frozen pregame state

Within each exact `season, week`:

1. `efficiency_fragility_rank` =
   average percentile rank of
   `prior8_abs_efficiency_error_mean`.

2. `opportunity_fragility_rank` =
   average percentile rank of
   `prior8_abs_opportunity_error_mean`.

Lower rank = historically easier/more stable.
Higher rank = historically harder/more fragile.

No alternate history window, threshold, subgroup or transform is allowed.

## Frozen realized reliability outcomes

Within the same `season, week`:

Primary final-projection outcome:
`total_error_badness_rank` =
average percentile rank of `current_abs_total_error`.

Mechanism check:
`efficiency_error_badness_rank` =
average percentile rank of `current_abs_efficiency_error`.

The mechanism check must show that the prior efficiency-difficulty state still tracks next-game efficiency difficulty. The primary question is whether it also adds information about the **final** receiving-yard projection after opportunity difficulty is controlled.

## Frozen incremental statistic

Primary statistic is the partial rank correlation:

`corr(efficiency_fragility_rank, total_error_badness_rank | opportunity_fragility_rank)`

computed by the standard three-correlation partial-correlation identity.

Because every input is first ranked within `season, week`, week-wide conditions are removed before the partial association is evaluated.

Secondary mechanism statistic:

`corr(efficiency_fragility_rank, efficiency_error_badness_rank | opportunity_fragility_rank)`.

No predictive coefficient is fit.

## Frozen dependence-aware inference

For the **primary** partial rank association:
- 5,000 player-cluster bootstrap replicates;
- 5,000 team-week/game-proxy cluster bootstrap replicates using
  `season-week-team` as the conservative within-team-game dependence key available in both exact parents;
- recompute the partial correlation inside every bootstrap replicate;
- fixed seeds in implementation.

Report:
- pooled;
- independent confirmation season only.

## Frozen support

WR:
- 2023 >=1,200 rows / >=140 WRs;
- 2024 >=1,600 rows / >=160 WRs;
- pooled >=3,000 rows / >=190 WRs.

TE:
- 2024 >=800 rows / >=80 TEs;
- 2025 >=800 rows / >=80 TEs;
- pooled >=1,800 rows / >=110 TEs.

## Frozen qualification gates

A position qualifies only if ALL of the following pass:

1. source/integrity passes;
2. support passes;
3. development primary partial association > 0;
4. confirmation primary partial association > 0;
5. pooled player-cluster 95% CI lower bound > 0;
6. pooled team-week cluster 95% CI lower bound > 0;
7. confirmation player-cluster 95% CI lower bound > 0;
8. confirmation team-week cluster 95% CI lower bound > 0;
9. development secondary mechanism partial association > 0;
10. confirmation secondary mechanism partial association > 0;
11. lowest-quartile efficiency-fragility rows have lower raw final absolute error than highest-quartile rows in development;
12. same low-vs-high ordering holds in confirmation;
13. sportsbook inputs = 0;
14. OddsAPI calls = 0;
15. 2026 outcomes = 0;
16. predictive models fit = 0;
17. projection/distribution values changed = 0;
18. production unchanged.

Quartiles use the frozen within-week `efficiency_fragility_rank`:
- HIGH_EFFICIENCY_TRUST <= 0.25
- LOW_EFFICIENCY_TRUST >= 0.75.

These bucket contrasts are guardrails; they cannot rescue a failed partial-association gate.

## Frozen dispositions

WR:
- pass:
  `WR_PLAYER_EFFICIENCY_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED`
- otherwise:
  `NO_ACTIONABLE_WR_PLAYER_EFFICIENCY_DIFFICULTY_TRUST_SELECTIVITY`

TE:
- pass:
  `TE_PLAYER_EFFICIENCY_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED`
- otherwise:
  `NO_ACTIONABLE_TE_PLAYER_EFFICIENCY_DIFFICULTY_TRUST_SELECTIVITY`

Overall:
- both qualify:
  `PASS_CATCHER_EFFICIENCY_DIFFICULTY_TRUST_SELECTIVITY_BOTH_QUALIFIED`
- one qualifies:
  `PASS_CATCHER_EFFICIENCY_DIFFICULTY_TRUST_SELECTIVITY_PARTIAL`
- neither qualifies:
  `PASS_CATCHER_EFFICIENCY_DIFFICULTY_TRUST_SELECTIVITY_CLOSED`
- integrity failure:
  `PASS_CATCHER_EFFICIENCY_DIFFICULTY_TRUST_SELECTIVITY_INTEGRITY_FAILURE`.

## If qualified

Qualification authorizes only a separately frozen **future-week efficiency trust shadow** around the unchanged WR-R15 / TE-R5P projection.

Opportunity trust and efficiency trust remain separate axes until a later frozen combination study proves that combining them adds value.

Do not automatically interpret:
- HIGH trust = OVER;
- LOW trust = UNDER;
- LOW trust = FADE;
- HIGH trust = BET.

## Anti-rescue

After scoring, do not:
- change last-8 / min-4 history;
- substitute raw YPT volatility;
- change the control variable;
- remove opportunity difficulty control;
- pool WR and TE;
- lower support floors;
- rescue WR1/WR2+/TE1 subsets;
- change quartiles;
- use alternate partial-correlation transforms;
- fit a mean correction;
- fit a width multiplier;
- use sportsbook lines;
- use 2026 outcomes.

No paid OddsAPI.
No production mutation.
