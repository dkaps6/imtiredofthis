# Prospective Public-Intent Capture Contract V1

Status: **FROZEN TEMPLATE — OBSERVATIONAL ONLY**

Purpose: standardize future pregame public-intent capture for qualifying RB/FB vacancy events so the Week-3 DEN/PIT exercise can become a prospective evidence set instead of a retrospective narrative.

## Eligibility

A team/event may enter only when:
- an RB/FB is already a qualifying 'definitive_unavailable == 1' event under the frozen RB Vacancy Opportunity contract;
- the capture occurs before kickoff and before target-game outcomes are read;
- strict-prior football evidence is available;
- sportsbook/fantasy projections do not define the label.

## Source tiers

- **Tier A:** official team injury report, transaction, roster move, coach/team transcript.
- **Tier B:** attributable established local beat reporting, preferably direct coach/player quotation plus clearly labeled reporter interpretation.
- **Tier C:** fantasy/secondary inference. Archiveable, but cannot define the frozen label.

## Required frozen fields

Identity:
- season
- week
- team
- opponent
- qualifying unavailable player
- capture timestamp
- latest evidence timestamp

Vacancy state:
- unavailable strict-prior opportunity share
- remaining active RB/FB identities
- frozen mechanical Vacancy V1 redistribution, if already materialized

Public-intent state:
- 'intent_label'
- 'lead_identity'
- 'concentration_direction'
- 'confidence'
- source tier(s)
- exact attributable evidence summary
- contradiction flag if Tier A/B sources disagree
- later roster/inactive updates as versioned addenda, never rewrites

## Allowed label family

The label should describe only observable workload structure, not predict an exact carry total.

Allowed structural classes:
- 'CLEAR_SINGLE_SUCCESSOR_CONCENTRATION'
- 'LEAD_BACK_LEAN_WITH_DEPTH_SUPPORT'
- 'ROTATION_PRESERVED_NO_CLEAR_SUCCESSOR_CONCENTRATION'
- 'COMMITTEE_REALLOCATION_WITH_NAMED_PAIR'
- 'INSUFFICIENT_ATTRIBUTABLE_EVIDENCE'
- 'CONFLICTING_ATTRIBUTABLE_EVIDENCE'

'lead_identity' may be a player, a named pair, or 'NONE_CLEAR'.

Confidence:
- 'HIGH'
- 'MEDIUM_HIGH'
- 'MEDIUM'
- 'LOW'
- 'UNRESOLVED'

Confidence measures strength/clarity of the attributable pregame evidence, **not** probability that the label will prove correct.

## Versioning rule

Once a pregame label is frozen:
- later pregame news creates an addendum;
- the original label is never rewritten;
- addendum may say 'STRENGTHENED', 'WEAKENED', 'UNCHANGED', or 'CONTRADICTED';
- if contradicted, preserve both original and updated pregame states with timestamps.

## Postgame grading order

1. Grade frozen RB Vacancy Opportunity V1 independently.
2. Attach actual team/RB/FB carries, snaps, routes/targets when available.
3. Measure realized vacancy absorption without changing the pregame label.
4. Compare public-intent structure to the mechanical Vacancy V1 structure.
5. Do not fit a coefficient until a separately preregistered minimum prospective sample exists.

## Suggested descriptive outputs

For each event:
- unavailable player's strict-prior share;
- Vacancy V1 share assigned by successor;
- actual successor carry/share absorption;
- Herfindahl concentration of actual RB/FB carries;
- top-successor share of remaining RB/FB carries;
- whether frozen lead identity led the room;
- whether realized concentration direction matched the frozen structural label.

These are descriptive diagnostics only.

## No-go rules

Do not:
- use exact postgame outcomes to redefine a label;
- translate fantasy rankings into Tier A/B intent;
- fit workload percentages from coach adjectives;
- use sportsbook props/odds to infer intent;
- promote from DEN/PIT alone;
- tune label definitions after seeing which labels worked;
- alter the already-frozen Vacancy V1 candidate based on public-intent evidence.

Week-3 DEN/PIT is the first frozen observational cohort under this logic, not a training set.
