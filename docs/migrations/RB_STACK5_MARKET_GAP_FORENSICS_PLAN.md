# RB-STACK5 — Remaining Market-Gap Forensics

Status: NO-FIT DIAGNOSTIC PLAN
Parent: retained STACK3 central point = full-stack Week-1 override + STACK2 enriched-opportunity/full-stack-efficiency architecture otherwise.

## Purpose

The current football-only point has 2025 listed-market MAE 24.3158 on the exact 899 rows versus Vegas consensus 23.7019. This diagnostic asks what football mechanism explains the remaining gap.

No model fitting, threshold tuning, or sportsbook input to football projections. Vegas is used only to define disagreement strata after the football projection is frozen.

## Required decomposition

For every one of the 899 rows calculate:

- football projection, Vegas line, actual yards;
- signed and absolute football-minus-Vegas disagreement;
- whether football or Vegas was closer;
- football projected carries and implied YPC;
- actual carries and actual YPC;
- carry error and YPC error;
- opportunity-oracle yards = actual carries * frozen predicted YPC;
- efficiency-oracle yards = frozen predicted carries * actual YPC;
- recoverable absolute-error improvement from perfect opportunity vs perfect efficiency;
- pregame role/depth/snap/committee/injury/rookie/workload-risk states already present in STACK2/3.

## Fixed strata

- football 10+ above Vegas;
- football 10+ below Vegas;
- 5-10 above/below;
- within 5;
- Week 1 / Weeks 2-18;
- depth rank 1 / rank 2 / rank 3+;
- committee / concentrated;
- M95F workload-risk / non-risk;
- injury-report / no-report;
- rookie / veteran.

Postgame actual-carry and actual-YPC bands are diagnostic only.

## Outputs

- aggregate mechanism table by disagreement direction/magnitude;
- opportunity-vs-efficiency attribution by stratum;
- top football wins vs Vegas;
- top Vegas wins vs football;
- false-high and false-low casebooks;
- candidate missing-information ledger based only on pregame-observable explanations.

The next modeling/data migration must be justified by this forensic evidence rather than by another arbitrary feature search.
