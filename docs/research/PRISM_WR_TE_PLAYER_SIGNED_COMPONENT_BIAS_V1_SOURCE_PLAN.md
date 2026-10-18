# PRISM WR/TE Player Signed Component Bias V1 — Frozen Source / Anti-Retest Plan

Date: 2026-10-18

Status: **FROZEN SOURCE-ONLY AUDIT / NO SIGNED-BIAS SIGNAL SCORED**

Model: **PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_WR_TE_PLAYER_SIGNED_COMPONENT_BIAS_V1_SOURCE_AUDIT`

## 1. Scientific question

Before scoring any player-specific correction, determine whether canonical
historical component-decomposition rows can legally support this question:

> Does PRISM have a stable, strictly-prior signed bias for this named player's
> opportunity component or efficiency component?

The source audit does **not** test whether the bias persists.

It only proves whether the required player/component history exists without
leakage and with enough named-player support.

## 2. Anti-retest boundary

This is materially distinct from the CLOSED:

`PRISM_PLAYER_MECHANISM_INSTABILITY_TRUST_V1_CLOSED`.

That closed experiment used:
- prior mechanism switch rate;
- prior dominant-state share;
- absolute component difficulty;
- total-error fragility / trust as the target.

This V1 source object explicitly excludes:
- mechanism switch rate as a predictor;
- dominant-state labels as a predictor;
- absolute total error as the target;
- projection width / trust / selectivity.

The new object is **signed component mean error**:

- opportunity error: systematically too high vs too low;
- efficiency error: systematically too high vs too low.

If later predictive evidence exists, a separate bridge would ask whether a
component mean should be corrected.

That later bridge is not authorized here.

## 3. Canonical source authority

Reuse the immutable component-decomposition source from:

- run `39061606501` — SUCCESS;
- artifact `12111278777`;
- artifact name
  `prism-player-mechanism-instability-source-v1-39061606501`;
- digest
  `sha256:f64c6e7efde9deaf91b7bc19076693a5106cb632338c461acc4d2cfe2ae37ea9`.

Files:
- `wr_mechanism_instability_source_rows.csv`;
- `te_mechanism_instability_source_rows.csv`.

The source artifact is reused only as immutable historical component lineage.
Its old instability predictor is not reused.

## 4. V1 positions / components

WR:
- opportunity signed error;
- efficiency signed error.

TE:
- opportunity signed error;
- efficiency signed error.

TE catch-rate error is source-visible but excluded from V1 to avoid opening a
third mechanism after seeing support.

QB excluded by prior anti-retest/source boundaries.

RB excluded because this source authority was not proven for RB.

## 5. Exact source fields

Require:

Common:
- season;
- week;
- team;
- stable player key;
- prior component-row count;
- prior component min/max week;
- source eligibility flag;
- decomposition identity gap.

WR:
- `prior8_opportunity_error_mean`;
- `prior8_efficiency_error_mean`;
- `current_opportunity_error`;
- `current_efficiency_error`.

TE:
- same four signed fields.

Absolute-error fields may be retained only for integrity diagnostics.
They are not V1 predictors or targets.

## 6. Strict chronology

For every source-eligible target row require:

- prior component rows >=4;
- prior component rows <=8;
- prior component max week < target week;
- prior component min week < target week;
- all history is same-season and strictly pre-target under the parent source
  contract;
- chronology violations = 0.

No same-week result may enter the target's prior signed bias.

## 7. Frozen source windows

Preserve the parent source seasons exactly.

WR:
- development = 2023;
- independent confirmation = 2024.

TE:
- development = 2024;
- independent confirmation = 2025.

No 2026 rows.
No Week-6 2026 outcomes.

## 8. Frozen support gates

Source eligibility uses the existing parent flag.

### WR 2023
- >=700 eligible target rows;
- >=120 distinct WRs;
- >=28 teams;
- prior signed opportunity field complete = 100%;
- prior signed efficiency field complete = 100%;
- current signed opportunity field complete = 100%;
- current signed efficiency field complete = 100%;
- decomposition identity max gap <=1e-10;
- chronology violations = 0.

### WR 2024
- >=1,100 eligible rows;
- >=135 WRs;
- same integrity gates.

### TE 2024
- >=575 eligible rows;
- >=70 TEs;
- same integrity gates.

### TE 2025
- >=600 eligible rows;
- >=75 TEs;
- same integrity gates.

These gates were frozen from source inventory only.
They may not be lowered after any predictive relationship is inspected.

## 9. Support reporting

For each position/season report:
- eligible rows;
- distinct players;
- teams;
- prior component-row count median / p25 / p75;
- rows with 4 / 5 / 6 / 7 / 8 prior component rows;
- signed-field missingness;
- chronology violations;
- max decomposition identity gap.

Do not report:
- signed correlation;
- sign accuracy;
- MAE improvement;
- player improvement rate;
- any predictive outcome relationship.

## 10. Later hierarchy if source-ready

A separately frozen signal may use:

`position prior signed bias -> same-player prior signed bias -> current player component error`.

The broader position prior must:
- be strictly prior;
- exclude the target player's own history;
- exclude same-week outcomes.

Player evidence:
- exact parent `prior8_*_error_mean`;
- support weight = exact `prior_component_rows_available`;
- minimum support already fixed at 4 by the parent source.

If later frozen, the first shrinkage candidate should use exactly four
position-prior pseudo-observations, matching the minimum legal player support.

That is not scored here.

## 11. Anti-rescue / excluded families

Do not use this lane to reopen:
- mechanism-instability trust;
- generic efficiency volatility;
- WR opportunity-difficulty trust as a mean correction;
- WR efficiency-difficulty trust as a mean correction;
- TE allocation-difficulty trust as a mean correction;
- raw recent-share replacement;
- same-QB allocation;
- WR target-quality chain;
- TE target-quality efficiency;
- generic DvP.

A future signal must predict **signed component error itself**.

## 12. Integrity at freeze

- predictive component-bias signals scored: **0**;
- component means changed: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 outcomes used: **0**;
- Week-6 outcomes used: **0**;
- sportsbook/OddsAPI: **0**;
- production changed: **false**.

Disposition:

`PRISM_WR_TE_PLAYER_SIGNED_COMPONENT_BIAS_V1_SOURCE_AUDIT_FROZEN`.
