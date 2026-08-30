# Migration 82 — Current Full-Stack Clean Walk-Forward Reconciliation

## Status

`PREREGISTERED / DIAGNOSTIC ONLY`

M82 re-anchors QB research to the actual production-style predictive stack after a long sequence of focused QB research harnesses. It does not introduce a new football feature and does not promote production logic.

## Why M82 exists

Focused residual and mechanism screens are useful for cheaply rejecting weak information, but they are not identical to testing the complete predictive system. M82 therefore answers two separate questions before any new QB frontier is opened:

1. What does the **current full production-style stack** actually score today on a clean, common 2024/2025 QB cohort?
2. Which M40-M81 research families are truly closed versus merely never receiving a legitimate full-stack integration test?

## Frozen football/data boundary

- No sportsbook player props, prop lines, odds, spreads, totals, implied points, or other market variables may enter any football projection, ensemble feature, cohort definition, or promotion decision.
- Historical inputs remain leakage-safe at each target week.
- Current production football rules are used unchanged.
- M82 does not rerun obsolete rule-calibration sweeps. It evaluates the rules that are currently in the production path.
- M82 changes no production coefficients or model defaults.

## Target seasons and run standard

- 2024 regular season Weeks 1-18, using 2023 as prior history.
- 2025 regular season Weeks 1-18, using 2024 as prior history.
- Monte Carlo: 2,000 iterations per target week/player.
- Current production-style component generator for every week.

## Full model components

M82 must generate the same three independent components used by the production architecture:

1. `mc_proj`
   - empirical Bayesian baseline
   - currently promoted empirical football rules
   - current joint Monte Carlo simulation
2. `ml_proj`
   - current leakage-safe supervised ML v2
3. `state_proj`
   - current leakage-safe state-transition v2

Bayesian is not counted as a fourth independent projection because it is already embedded in `mc_proj`.

## OOS ensemble contract

M82 must not fit and score ensemble weights on the same target rows.

### 2024
For each target week, ensemble weights may use only completed **earlier 2024 OOS component predictions**. Until at least 40 complete pass-yard component rows exist, the ensemble must explicitly fall back to MC-only.

### 2025
One frozen set of pass-yard ensemble weights is fit using complete **2024 OOS component predictions only** and applied unchanged to all 2025 rows.

No 2025 outcome may alter 2025 ensemble weights.

## Canonical-v3 common QB cohort

The authoritative QB comparison population is:

`qb_frontier_canonical_v3_football_only`

Expected SHA256:

`c4a481a760657bb7516d52b9aa9ba84af096063a4e78660b12a541f59fd7b742`

Expected rows:

- 2024: 444
- 2025: 440
- combined: 884

M82 must join each current component to these exact canonical QB-game identities and must fail closed if the common-cohort identity contract drifts.

This eliminates the old problem of comparing a 643-game market-narrowed population with an 884-game football-only population.

## Required QB scoreboard

On the identical canonical-v3 QB games, report for:

- canonical-v3 point projection
- current MC/Bayesian/rules projection
- current ML projection
- current State projection
- OOS ensemble projection

Metrics:

- coverage
- MAE
- RMSE
- bias
- correlation
- 100+ yard misses
- metrics by 2024, 2025, and combined

Also report:

- pairwise signed residual-error correlations among available component models
- prediction correlations
- best-of-current-library hindsight oracle as diagnostic headroom only
- model-disagreement buckets versus absolute canonical/current error
- row-level common-cohort reconciliation trace

The hindsight oracle is non-deployable and cannot promote anything.

## Full-market diagnostic output

The normal walk-forward component evaluator must also score all supported prop markets for 2024 and 2025 so M82 remains an integrated system audit, not only a QB passing test.

## M40-M81 integration-eligibility ledger

M82 must classify major research families into exactly one disposition:

- `PROMOTED_FOUNDATION`
- `FULL_STACK_TESTED_CLOSED`
- `SIGNAL_SCREEN_FAILED`
- `PARTIAL_SIGNAL_INTEGRATION_CANDIDATE`
- `SOURCE_BLOCKED_NEW_INFORMATION_REQUIRED`

A failed idea is not eligible for another model merely because a different algorithm exists. Integration eligibility requires evidence that the prior experiment left a genuine architectural question open.

## Two user-generated frontier hypotheses preserved for the ledger

### A. `TOP_WEAPON_ESCAPE_HATCH`

Hypothesis: a QB can exceed a bad macro matchup because one WR/TE/RB has an unusually exploitable individual matchup and the offense funnels high-value targets through that player.

Important distinction: realized WR yards and realized QB yards are mechanically related and are **not** predictive evidence. A valid future test must ask whether a **pregame** top-weapon matchup score predicts positive QB residual conditional on the QB's macro matchup.

Prior overlap:

- M72 aggregate explosive-weapon x defense matchup was negative.
- M75 NGS receiver tracking / PFR secondary / interactions were negative on canonical-v3.

Therefore M82 may not relabel those same proxies as new. The exact escape-hatch hypothesis remains open only if materially new pregame observables are obtained, such as trustworthy route/responsibility-level receiver-defender exposure, role-specific defensive injury replacement, or another distinct player-level matchup mechanism.

Frozen disposition going into the ledger: `SOURCE_BLOCKED_NEW_INFORMATION_REQUIRED`, not a same-data retest.

### B. `DEFENSIVE_ADAPTIVE_GAMEPLAN`

Hypothesis: static defensive quality may be insufficient; defenses may systematically change blitz, shell, man/zone, box, bracket/double, or pressure structure based on the offense/QB archetype they face, and that adaptation may create week-specific QB over/underperformance.

Prior overlap:

- M56 tested richer static/lagged defensive opportunity, coverage and QB x defense interactions.
- M67-M69 tested offensive intent/opening/game-script mechanisms and their relation to defensive context.
- M80 held route x coverage-shell as a genuinely new interaction but deployment/source constrained.

What has **not** been cleanly proven is a leakage-safe model of the defense's *conditional tactical response to opponent offensive traits*. A future test would need to predict the target defense's tactical change from earlier games against comparable offenses; target-game realized scheme is forbidden.

Frozen disposition going into the ledger: `SOURCE_BLOCKED_NEW_INFORMATION_REQUIRED` / future frontier candidate, not a static-defense retest.

## M82 interpretation rule

M82 is a reconciliation checkpoint, not a tuning migration.

After the run:

1. declare the authoritative current clean QB scoreboard;
2. identify whether the production-style stack materially outperforms canonical-v3 on the same games;
3. identify whether component diversity is real or redundant;
4. finalize the M40-M81 integration ledger;
5. only then choose a genuinely eligible M83 information/integration test.

`production_actionable = false`
