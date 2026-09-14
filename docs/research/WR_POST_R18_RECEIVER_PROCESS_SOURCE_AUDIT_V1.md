# WR Post-R18 Receiver Process / Route-Matchup Source Audit V1

**STATUS: AUDIT ONLY. NO R19 AUTHORIZED. NO FOOTBALL OUTCOME RUN. 2024 WR-R15 HOLDOUT REMAINS SEALED.**

## Purpose

After WR-R18 receiver-attributed CPOE failed its formal 2023 Stage-A gate stack but retained `PARTIAL_DIRECTIONAL_EVIDENCE`, do not rescue R18 through WR1-only or tail-only slicing. The next WR receiving-yard lane must use materially new pregame football information rather than another transformation of CPOE, YPT, YAC, air-yards, explosiveness, or team/QB efficiency history.

This audit asks whether the repo now has a scientifically defensible and deployable receiver-process / route-matchup information source that satisfies both:
1. historical strictly-prior backtestability; and
2. live/in-season availability for eventual production use.

No candidate is frozen here.

## Canonical closures entering this audit

Closed / do not retest:
- R7 prior player YAC / air-per-target / explosive persistence;
- ND6 player explosive ceiling;
- R17 target-depth distribution shape;
- R18 receiver-target CPOE level and post-hoc threshold/WR1 rescue;
- M70-M71 QB efficiency / volatility history;
- M72 aggregate explosive weapon x defense;
- M75 NGS receiver tracking / secondary-quality family;
- R9-R11 vendor NGS target model;
- team-level man/zone coverage as a generic efficiency signal (existing ablation near-neutral);
- fake historical WR-CB assignments;
- R3 remains closed by explicit user instruction.

## Source finding 1 — nflverse participation has route + coverage-shell fields, but live deployability is the blocker

Current nflverse participation data dictionary includes:
- `route` for the primary receiver on the play;
- `defense_man_zone_type`;
- `defense_coverage_type` (Cover 0/1/2/3/4/6/9, 2-Man, combo, blown);
- on-field player lists, time to throw, pressure, personnel and formation.

This means historical route/coverage mechanics are not conceptually absent.

However, nflverse's current availability contract says participation data from 2023 onward is supplied by FTN only after the entire postseason is complete and **does not update during the season**. That makes the free participation feed unsuitable by itself as a live pregame production source for the current season.

Therefore a route/coverage-shell predictive lane remains source-blocked unless a live source with compatible semantics is established prospectively.

## Source finding 2 — nflverse FTN charting is historical and live/in-season, but is not route-level

The nflverse FTN charting feed is available from 2022 onward and updates in-season after games are charted. It contains process observables including:
- `read_thrown` (primary/secondary/later/checkdown/designed/scramble-drill progression state);
- `is_catchable_ball`;
- `is_contested_ball`;
- `is_created_reception`;
- `is_drop`;
- tactical / pressure fields.

This source is operationally attractive because it is both historical and in-season.

But M80-M81 already tested four FTN information families for **QB passing residual prediction**, including `THROW_DECISION_QUALITY` (`is_interception_worthy`, `read_thrown`, `is_catchable_ball`) and `RECEIVER_ERROR_ATTRIBUTION` (`is_drop`). M81 ended `NO_FTN_DEVELOPMENT_SURVIVOR`; no same-data QB model-zoo rescue is allowed.

That does not automatically prove a receiver-specific target-process signal is closed, because M81's aggregation levels were QB history, team history, opponent history and same-family QB x opponent interactions—not receiver-specific history. But aggregation level alone is not sufficient novelty.

## Narrow potentially-open question — NOT AUTHORIZED YET

The only FTN process observable that currently appears plausibly distinct enough to audit further is **receiver-specific catchable-target quality relative to the QB/team environment**.

Prospective mechanism:

> Some WRs may systematically receive a different quality of delivered targets than the same offense/QB generally produces, because route type, timing, alignment, defensive attention and QB-WR chemistry change whether throws to that receiver are charted as catchable. A receiver-specific, team-orthogonal catchable-target state is a direct process observable rather than another realized YPT/YAC/CPOE transform.

Why it may be new:
- `is_catchable_ball` is human-charted process quality, not completion outcome or expected-completion residual;
- M81 tested catchable-ball information at QB/team/opponent level for QB outcomes, not receiver-specific residualized history for WR receiving-yard residuals;
- R18 tested receiver-attributed CPOE, which remains outcome/model-derived and does not directly identify whether a throw was charted as catchable;
- the source is available historically from 2022 onward and in-season, so an eventual deployment contract is feasible.

Why it may still be too close / must be challenged:
- it is still a QB-WR delivery-quality family adjacent to R18;
- receiver-specific aggregation could simply repackage QB/team throw quality;
- catchability may be strongly mediated by target depth, route role or coverage difficulty;
- R5/R7 already show that historical receiver efficiency traits generally do not solve the YPR-dominant residual;
- M81's `THROW_DECISION_QUALITY` family failed QB development despite including catchable-ball rate.

A legitimate receiver-specific candidate would therefore need, before any outcomes are scored:
- GSIS-first receiver identity via PBP join from FTN game/play IDs;
- strictly-prior receiver target history;
- direct receiver catchable-target rate as the only primary candidate;
- team/QB catchable-target rate as a control, not a competing signal;
- mean target depth and frozen M38/R15 role/opportunity controls;
- no `is_contested_ball`, `is_created_reception`, `is_drop`, or `read_thrown` add-ons unless separately justified before results;
- no post-result threshold/WR1/tail rescue;
- exact WR-R15 authority cohort if source coverage is adequate;
- 2023 development only initially; 2024 remains sealed unless preregistered gates pass;
- adversarial Claude review before freezing any R19.

## Route / coverage source frontier

A stronger route/coverage candidate would be preferred scientifically if a compatible live source can be established.

Current free-source problem:
- historical participation provides route and coverage shell;
- 2023+ participation is not available during the season;
- existing repo WR-CB exposure has no honest historical responsibility assignment.

Potential external live-source direction to investigate before paying or integrating:
- a provider with historical + current route-level data using stable receiver identity and explicit route/alignment/coverage-shell semantics;
- export/API access sufficient for reproducible snapshots and strict-prior audits;
- a historical span covering at least the development/holdout seasons used by the WR authority.

Do not start a predictive route-matchup test until the historical/live semantic bridge is documented and reproducible.

## Preliminary disposition

`POST_R18_PROCESS_SOURCE_AUDIT__CATCHABLE_RATE_NOVELTY_REVIEW_REQUIRED`

No R19 is authorized yet.

Next collaboration task:
1. Claude independently challenge whether receiver-specific catchable-target rate is materially new versus M81 + R18 or merely aggregation-level repackaging.
2. Independently verify FTN/nflverse source coverage and joinability for 2022-2024 without using target-game information.
3. Check whether any prior WR branch already used receiver-specific `is_catchable_ball` history.
4. Propose a better genuinely-open receiver process / route-matchup observable if one exists.
5. Do not inspect WR-R15 2024 outcomes, do not reopen R18, do not reopen R3.

No production change, no paid Full Slate, no RB work.
