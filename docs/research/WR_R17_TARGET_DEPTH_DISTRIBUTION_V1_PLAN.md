# WR-R17 Target-Depth Distribution V1 — Frozen Plan

**STATUS: FROZEN BEFORE ANY WR-R17 RESULT. RESEARCH ONLY. NO PRODUCTION CHANGE. NO PAID FULL SLATE.**

## 1. Decision and lineage

This plan is the single next WR receiving-yard experiment after the authority-exact anti-retest + feature-availability audit and Claude's independent Issue #535 review.

Audit lineage:
- canonical main at audit start: `de6aed84d474867d81427f4e8277219868ac9d50`
- audit branch: `research-wr-yard-efficiency-feature-audit-v1`
- audit commit: `0e013445d815fc869630b6bc4380d909914739fa`
- audit doc: `docs/research/WR_RECEIVING_YARD_EFFICIENCY_ANTIRETEST_FEATURE_AUDIT_V1.md`

WR-R16 is **subsumed/deferred** and must not be run in parallel. Claude independently confirmed that R16's `wr_target_depth_sd8` already overlaps this mechanism and that R16's receiver history is keyed by canonicalized name instead of GSIS ID. The R16 `team_*` delivery family also materially overlaps the M70/M71 CPOE/completed-air/deep-efficiency construction family. This plan therefore does **not** inherit R16's `team_*` features or its four team/WR interactions.

User instruction keeps R3 closed regardless of the separate documentation caveat raised in Issue #535. No R3 rescue is authorized here.

## 2. Football hypothesis

Current WR opportunity is comparatively healthy under M38 + `WR_R15_PRODUCTION_MODEL_V1`, while receiving-yard translation remains weak. Prior research has already rejected simple persistent mean air-yards/target, YAC, explosive rates, opponent explosive allowances, tracking marginals/interactions, generic NGS target lifts, and QB-tail selectors.

The new question is:

> Does the **shape of a WR's strictly-prior targeted-pass depth distribution** contain reproducible information about next-game receiving-yard residual after the frozen M38/R15 opportunity-based projection?

This is specifically a distribution-shape question, not another mean-aDOT test. The intended mechanism is that two receivers can have similar mean target depth and similar projected target opportunity while differing materially in how target depth is distributed (stable intermediate usage versus high dispersion / meaningful deep-target mass), which may alter receiving-yard translation and right-tail exposure.

## 3. Exact authority cohort

Use only the exact WR-R15 OOS authority artifact:
- run: `34238301577`
- artifact: `10061328722` (`wr-r15-wr1-anchor-participation-v1`)
- digest: `sha256:8df31b5e136621d959272daf0422dfc665593cd0da2eb0892b4aa69c1417f3ce`
- variant exactly: `WR_R15_WR1_ANCHORED_PARTICIPATION`
- expected rows exactly: `4,193`
- 2023: `2,076` development rows
- 2024: `2,117` untouched holdout rows

Authority fields are immutable, including `mc_rec_yards`, `mc_receptions`, `pred_targets`, `entitlement_tgt_share`, and `wr_rank`.

Primary outcome:
`receiving_yard_residual = actual_rec_yards - mc_rec_yards`.

No new cohort selection is allowed after results. Rows may be feature-ineligible only under the fixed source/support rules below; ineligible rows remain part of coverage accounting.

## 4. Source and temporal contract

Historical target events come from nflverse/nflreadpy regular-season PBP for history seasons needed to construct strictly-prior features for 2023-2024 scoring.

For each authority row `(season S, week W, team, player)`:
- only PBP events from games strictly before `(S, W)` may feed features;
- target-game PBP is forbidden as a predictor source;
- sportsbook lines, prices, odds, and market-derived features are forbidden;
- postseason data are forbidden unless already part of the exact authority artifact (the PBP history feature source itself is regular season only);
- receiver history lookup must be **GSIS/player-ID first**;
- canonicalized name may be used only as an audited fallback when a stable ID is absent;
- no fuzzy post-result identity rescue is allowed;
- unmatched/ambiguous rows must be counted and reported.

## 5. History window and support

For a target row, define the receiver's last **8 eligible prior target-bearing games** before `(S,W)`.

A feature row is valid only when all are true:
- at least `4` prior target-bearing games exist;
- at least `12` prior target events with non-null `air_yards` exist across the last-8 history;
- receiver identity is unambiguous under the ID-first contract.

These thresholds are frozen before result exposure.

## 6. Frozen candidate signals

Exactly three distribution-shape signals are permitted. No others may be added after results.

### Signal 1 — `DEPTH_SD8` (priority 1)
Population standard deviation (`ddof=0`) of `air_yards` across all valid target events in the receiver's last-8 eligible prior target-bearing games.

This is the primary signal and the cleanest direct dispersion measure. It is the distribution-shape feature that appeared in the unrun R16 proposal but has not been evaluated as a standalone WR receiving-yard mechanism.

### Signal 2 — `DEPTH_IQR8` (priority 2)
`Q75(air_yards) - Q25(air_yards)` across the same last-8 target-event history, using pandas/numpy linear quantile interpolation.

This is a robust dispersion measure intended to determine whether any result is specific to standard-deviation sensitivity to outliers.

### Signal 3 — `DEEP15_TARGET_SHARE8` (priority 3)
Number of target events with `air_yards >= 15` divided by all valid target events in the last-8 history.

This represents deep-target probability mass, not realized explosive-yard outcomes. The 15-yard threshold is frozen because it is already a conventional deep-attempt boundary in the repo's prior passing research; no alternate 10/20/25-yard threshold search is allowed.

### Explicitly excluded
Do not include:
- mean air yards / target;
- historical YPT/YPR/YAC as candidate signals;
- EXP20/EXP40 receiving outcomes;
- opponent explosive/YAC/air allowances;
- team/QB CPOE, completed-air, or deep-completion features;
- WR-CB assignment proxies;
- NGS separation/cushion/YACOE features;
- sportsbook data.

Mean air yards / target may be computed **only as a robustness control** described below, never as a candidate signal.

## 7. Development / holdout separation

- 2023 = development / signal-existence season only.
- 2024 = untouched confirmation holdout.
- Do not pool 2023+2024 to choose a signal, sign, threshold, support rule, or coefficient.
- 2024 receiving-yard outcomes must not be read into the Stage-A selection logic.

At most one signal may advance. Frozen priority order:
1. `DEPTH_SD8`
2. `DEPTH_IQR8`
3. `DEEP15_TARGET_SHARE8`

Choose the **first** signal in that order that clears every Stage-A gate. Do not choose the numerically strongest passing signal.

## 8. Stage A frozen gates — 2023 only

For each valid signal report:
- valid N and coverage over all otherwise authority-eligible 2023 rows;
- Spearman correlation with receiving-yard residual;
- 2023-only Q4 minus Q1 mean receiving-yard residual gap;
- Q4/Q1 actual-100+ receiving-yard game rate ratio in the signal-consistent direction;
- Q4/Q1 absolute receiving-yard miss >=30 yards rate ratio in the signal-consistent direction;
- WR1 and WR2+ residual-gap slices where each has at least 150 valid rows.

A signal is `DEVELOPMENT_SUPPORTED` only if **all** are true:
1. coverage >= `60%`;
2. abs(Spearman) >= `0.08`;
3. abs(Q4-Q1 residual gap) >= `5.0` yards with the same sign as Spearman;
4. either the 100+ rate ratio >= `1.20` in the mechanism-consistent direction **or** the 30+ miss-rate ratio >= `1.20` in the same direction;
5. WR1 and WR2+ residual-gap directions are both coherent with the pooled direction when each slice has >=150 rows;
6. identity/source leakage audits pass.

If none pass, disposition is `NO_ACTIONABLE_WR_TARGET_DEPTH_DISTRIBUTION_SIGNAL`; do not inspect 2024 to rescue the family.

## 9. Role/endogeneity robustness — required before Stage B interpretation

Claude correctly identified selection-on-targets endogeneity as a major risk: thrown target depth may simply re-express receiver role/archetype already captured by M38/R15.

For the single Stage-A advancing signal only, perform this predeclared robustness analysis using 2023 development rows:

Fit an OLS control model on 2023 only:
`signal ~ entitlement_tgt_share + C(wr_rank_bucket) + mean_air_yards_per_target8`
where:
- `wr_rank_bucket = WR1` when `wr_rank == 1`, else `WR2PLUS`;
- `mean_air_yards_per_target8` is computed from the same strict-prior target events and is a control only.

Define `role_orthogonal_signal` as the OLS residual.

Report, but do not use to pick a different signal:
- Spearman(`role_orthogonal_signal`, receiving-yard residual);
- Q4-Q1 residual gap using 2023-only quartiles of `role_orthogonal_signal`;
- whether both preserve the raw signal's direction.

Interpretation rule:
- if both orthogonalized metrics reverse sign, label the Stage-A result `ROLE_MEDIATED_WARNING`;
- a `ROLE_MEDIATED_WARNING` does not invalidate Stage A mechanically, but a later holdout PASS may authorize only more research, not integration, until role mediation is resolved.

Freeze the 2023 OLS coefficients for any optional 2024 robustness report. They may not be refit on 2024.

## 10. Stage B holdout contract — 2024 untouched

If exactly one signal advances from Stage A, freeze before reading 2024 results:
- the advancing signal identity;
- its expected sign;
- 2023 Q25/Q75 thresholds;
- any 2023 OLS robustness coefficients.

Apply those unchanged to 2024.

Holdout confirmation requires all:
1. coverage >= `60%` of otherwise authority-eligible 2024 rows;
2. Spearman has the same sign and abs >= `0.06`;
3. frozen-threshold high-vs-low residual gap has the same sign and abs >= `4.0` yards;
4. 100+ receiving-yard rate ratio >= `1.15` **or** 30+ miss-rate ratio >= `1.15` in the same direction;
5. same-sign residual-gap direction in WR1 and WR2+ where each has >=150 valid rows;
6. zero target-game feature leakage, zero sportsbook inputs, and identity audit passes.

PASS disposition:
`WR_TARGET_DEPTH_DISTRIBUTION_REPLICATED_SIGNAL`.

FAIL disposition:
`NO_ACTIONABLE_WR_TARGET_DEPTH_DISTRIBUTION_SIGNAL`.

DATA disposition:
`WR_TARGET_DEPTH_DISTRIBUTION_DATA_BLOCKED`.

A PASS is **not** a production change. It authorizes only a separately preregistered integration experiment that must protect M38/R15 target/reception accuracy and improve receiving-yard football metrics.

## 11. Protection / football metrics for any later integration

No integration is authorized in V1, but any future integration candidate must prospectively score at minimum:
- receiving-yard MAE;
- RMSE;
- bias;
- correlation;
- median/p75/p90 absolute error;
- underprojection >=25 and >=50 yards;
- overprojection >=25 and >=50 yards;
- actual 100+ receiving-yard games;
- WR1 and WR2+ separately;
- target/reception parity or protection versus the frozen authority.

Historical market comparison remains downstream only.

## 12. Stop rules

After any result is visible, do **not**:
- change last-8 history;
- change minimum prior games or 12-target support;
- replace SD with a different variance estimator;
- change IQR interpolation;
- try deep thresholds other than 15 yards;
- swap quartiles for tertiles/deciles;
- change Stage-A or Stage-B numerical gates;
- choose a later-priority signal over an earlier passing one;
- add mean aDOT as a candidate;
- add QB/team delivery features;
- add opponent matchup interactions;
- add WR-CB assignments without an honest historical source;
- turn a failed mean signal into an uncertainty/width/tail-only rescue;
- reopen C2->WR1, C1, C3, R3, R7, R9-R11, M72/M75, ND3, or R16 under a new label;
- use sportsbook lines/odds upstream;
- alter M38/R15 opportunity science.

If Stage A fails, stop this specification. If Stage B fails, preserve the failure.

## 13. Collaboration disposition

Claude's Issue #535 review independently confirmed that:
- prior R7/ND6/M72/M75/R11 work used scalar means rather than within-player target-depth distribution shape;
- a target-depth dispersion/deep-mass mechanism conditional on M38/R15 opportunity is genuinely open;
- R16's name-first receiver history is a hard identity blocker;
- R16's `wr_target_depth_sd8` overlaps this lane and therefore R16 should not run independently;
- selection-on-targets / role endogeneity must be explicitly audited;
- the exact WR-R15 4,193-row cohort and 2023->2024 split should be preserved;
- the existing two-stage gate magnitudes are appropriate to reuse rather than retune.

This plan accepts those amendments before any WR-R17 result is exposed.
