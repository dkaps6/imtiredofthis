# RB Vacancy Opportunity V1 — Frozen Research Plan

Date: 2026-09-24
Branch: `research-rb-vacancy-opportunity-v1`
Parent main at branch creation: `68a2aab2b7868c29e8c6cd7f57e64071bf368260`
Status: FROZEN BEFORE OUTCOME TESTING

## Why this lane exists

This is the sanctioned continuation after the TE Width V2 replay/parity blocker was solved but candidate validation remained mechanically unresolved. It does not reopen M96 retrospective router tuning.

Established authority that must not be re-derived:
- M96A showed RB rushing error is opportunity-dominant overall: perfect carries recover 7.6777 yards of MAE vs 6.7256 for perfect efficiency, with 59.73% of games opportunity-dominant.
- Current-season state persistence found RB rush share is the strongest live opportunity-state signal.
- Existing production injury logic can haircut an injured RB's own `rules_rush_share`, but there is no explicit RB-to-RB transfer of vacated rushing opportunity.

## Hypothesis

Pregame unavailability of a backfield teammate creates a measurable vacancy in team rushing opportunity that should be transferred to available successor RB/FBs according to strictly-prior role evidence rather than merely disappearing from the backfield allocation.

This is an opportunity/carries hypothesis only. It does not change YPC efficiency.

## Phase 0 — production correctness audit (must happen first)

Before any candidate science, trace the actual production path for a confirmed unavailable RB (`OUT`, `DOUBTFUL`, `IR`, `PUP`) from canonical context through rules, simulation, ensemble/synthesis, and pricing.

Questions:
1. Can a confirmed unavailable RB retain nonzero rushing opportunity in the final football projection/pricing universe?
2. Does any downstream roster/slate layer remove or zero that player after `simulation_rules.py` applies its 0.50 self-haircut?
3. If removed, at what exact seam and with what provenance?
4. If not removed, treat that as a production-correctness defect separately from this research hypothesis.

Do not infer final behavior from the source-line self-haircut alone.

## Candidate information contract

Allowed pregame inputs:
- teammate injury/availability state available before kickoff (`report_status`, `practice_status`, roster designations and their existing canonical provenance);
- strictly-prior current-season snap share / offense snap participation;
- strictly-prior rush share / carries state already available through canonical player evidence;
- roster/depth role available pregame;
- schedule/team/opponent identity.

Forbidden upstream inputs:
- sportsbook player lines or odds;
- target-game carries, yards, snaps, or efficiency;
- postgame target-week participation;
- retrospective M96 router thresholds/features fitted against exposed 2025 outcomes.

## Vacancy construction

For each team and target game, define the unavailable backfield set using only pregame status. Estimate each unavailable RB/FB's vacated opportunity from strict-prior role evidence. Keep snap-share and rush-share versions separately auditable; do not search arbitrary mixtures after outcomes are visible.

The first candidate must be deterministic and conservation-aware:
- unavailable player's candidate rushing share becomes zero only if the production eligibility audit establishes that confirmed unavailability means no game participation;
- vacated rushing share is redistributed only among eligible RB/FB teammates;
- successor weights come from strict-prior role evidence;
- team RB/FB rushing-share mass is conserved unless the canonical production model explicitly assigns some vacancy to QB/WR/non-RB rushing; any such exception must be predeclared from football mechanics, not fit from outcomes.

No YPC/efficiency change in V1.

## Evaluation discipline

This lane is authorized as genuinely new pregame information. Do not use it as a pretext to reopen exposed 2025 M96 threshold search.

Before evaluating outcomes, freeze:
- exact availability mapping;
- exact vacancy formula;
- exact successor weighting formula;
- exact cohort and identity rules;
- exact gates.

Prefer prospective/untouched 2026 evaluation where available. If historical data are used only to establish source coverage/mechanics, do not tune coefficients against exposed outcomes.

Primary point-mean questions once a legitimate untouched evaluation cohort exists:
- RB rushing-attempt MAE;
- RB rushing-yard MAE with efficiency held frozen;
- bias;
- team/backfield carry conservation;
- high-volume and vacancy-event subsets declared before grading.

A failure does not authorize coefficient/threshold search.

## Anti-retest / stopping rules

Do not:
- rerun M96A/M96E;
- search injury-status weights against exposed 2025 outcomes;
- search snap/rush-share blend weights after seeing target outcomes;
- alter YPC in this V1;
- use sportsbook information upstream;
- promote from two live weeks alone;
- mutate production from a research-only result without a separate promotion review.

If the required pregame availability source cannot be reconstructed with timestamp-safe provenance, fail closed and document the source gap.

## Immediate implementation sequence

1. Complete Phase 0 final-path inactive-RB audit.
2. Inventory exact current injury fields/provenance and strict-prior snap/rush-share fields.
3. Build a research-only vacancy-state table with no target outcomes attached.
4. Freeze deterministic V1 redistribution mechanics and invariants.
5. Only then attach a legitimate evaluation cohort and grade once.

No paid OddsAPI pull is required or authorized for this work.