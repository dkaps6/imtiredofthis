# TE Target-Quality Efficiency V1 — Source / Novelty Audit Plan

Date: 2026-09-24

Status: FROZEN SOURCE-AUDIT PLAN — NO PREDICTIVE FIT / NO PRODUCTION CHANGE

## Why this lane exists

The active 2026 evidence says TE entitlement is improving from current snap/role state, but receiving-yard translation remains the larger live mean problem.

Established authorities that must not be rerun:

- TE-R1: roughly 45.2% of TE receiving-yard error mass is TARGETS, while YPR + CATCH_RATE together are roughly 54.8%.
- TE Live Entitlement vs Efficiency V1: Week-2 current-snap entitlement improves target-share MAE, but the receiving-yard counterfactual does not improve; perfect realized efficiency recovers more error than perfect entitlement on the selected TE receiving-yard cohort.
- raw two-game TE YPT / catch-rate recency is too noisy to justify aggressive current-only efficiency learning.
- team-level man/zone coverage has already been ablation-tested and was near-neutral; player-level WR-CB matchup history is not reconstructable historically. Do not reopen that lane as if untested.

The new question is narrower:

> Can strict-prior **target-quality / receiving-execution information** explain and predict TE receiving efficiency better than raw recent YPT/catch-rate averages?

This is intended to improve the football mean, not merely the probability width.

## Novel data families to audit

### A. NFL Next Gen Stats receiving

Use the maintained nflreadpy receiving feed, resolved to TE identity/position.

Audit weekly TE support for:

- targets / receptions
- avg_separation
- avg_cushion
- avg_intended_air_yards
- avg_expected_yac
- avg_yac_above_expectation
- percent_share_of_intended_air_yards when present

Important: schema existence is not support. Player-week TE rows and non-null coverage must be measured season by season.

### B. nflverse play-by-play target quality

Resolve target receivers to TE using receiver GSIS/name plus weekly-player identity.

Audit TE target/catch support for:

- air_yards
- complete_pass
- yards_after_catch
- xyac_mean_yardage / compatible expected-YAC field
- xyac_success / xyac_fd when present
- pass_length / pass_location
- down / yards-to-go
- shotgun / no_huddle
- score differential

Derived source-audit quantities may include expected YAC and YAC above expectation, but **no predictive model or outcome-error scoring** is allowed in this phase.

## Leakage boundary

This audit may inspect source coverage and source semantics only.

It may NOT:

- fit a TE projection or residual model;
- score model prediction error;
- choose a feature because it correlates with target-game outcome;
- use sportsbook lines/odds;
- use target-week observations as pregame features;
- use 2026 game outcomes to select a feature/gate;
- change TE-R5P, PR #627, Width V2, or production.

Any later predictive study must construct every feature from games strictly before the target week and freeze its exact feature set/model/gates before outcome scoring.

## Coverage questions

For each season 2020-2026 where the provider supports the data, report:

- total TE target rows / completed catches (PBP);
- identity/position resolution rate;
- non-null rate for each target-quality field;
- TE NGS player-week rows and unique players;
- non-null rate for each NGS field;
- specifically for 2026, whether Weeks 1-2 provide enough strict-prior evidence to construct Week-3 inputs without using Week-3 outcomes.

The source audit is successful if at least one genuinely new target-quality family has adequate historical and 2026 strict-prior support to justify a separately frozen predictive study.

No fixed predictive MAE threshold is declared here because this phase is source/novelty only. Predictive gates will be frozen in a new plan **before** any relationship to projection error is scored.

## Fast decision rule

1. If TE NGS player-week support is dense and current through 2026 W1-W2, prefer a player-level tracking-quality predictive study first.
2. If NGS TE support is sparse/late, but PBP target-quality support is dense, use strict-prior PBP target-depth + expected-YAC/YACOE state instead.
3. If neither source is adequate, fail closed and do not create another generic raw-YPT model.

## Relationship to other active lanes

- RB Vacancy Opportunity V1 remains frozen/prospective on its own branch; this TE audit does not alter it.
- TE Width V2 remains a distribution/probability lane. Do not conflate a width improvement with a mean improvement.
- QB passing yards remains frozen/prospective.
- No paid OddsAPI pull is authorized.

## Immediate output

Produce a compact source-audit artifact and result document that states:
- what TE target-quality data actually exists;
- how complete it is;
- whether 2026 W1-W2 can legally feed the upcoming slate;
- which single predictive study, if any, is justified next.

No production mutation is authorized by the source audit alone.
