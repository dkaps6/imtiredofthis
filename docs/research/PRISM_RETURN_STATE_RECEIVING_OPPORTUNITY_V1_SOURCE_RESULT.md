# PRISM Return-State Receiving Opportunity V1 — Source Result

Date: 2026-10-18

Status: **SOURCE BLOCKED / EXACT PLAYER TARGET MEMBERSHIP INSUFFICIENT / NO PREDICTIVE SCORING**

Model: **PRISM — Player Role & Individual Situation Model**

Frozen source plan:
`docs/research/PRISM_RETURN_STATE_RECEIVING_OPPORTUNITY_V1_SOURCE_PLAN.md`

Freeze:
`3a2a51e5412ad83265700dbf088e2425bae359c3`

Canonical execution:
- run `39180266614` — **SUCCESS**;
- exact head `bd76fc9d123ed11d9049ce9b93660a25d60f0de6`;
- job `117831326382` — **SUCCESS**;
- artifact `12156245630`;
- artifact name
  `prism-return-state-receiving-opportunity-v1-source-39180266614`;
- digest
  `sha256:bd6a65e9cd7639aa8fb52a2f6d8ac60146d10aab20f64cb4d966a3bdc7b693c3`;
- strict repository audit: **PASS**.

Disposition:

`PRISM_RETURN_STATE_RECEIVING_OPPORTUNITY_V1_SOURCE_BLOCKED`

## Player-first question

Can PRISM attach exact same-player receiving opportunity evidence to a
source-certified restored-return episode, at stable player + team + week grain,
while preserving exact finite team target volume, before any target-restoration
relationship is scored?

Under the frozen source contract, the answer is **not cleanly enough**.

The team opportunity ledger is source-valid.

The blocking gap is exact **named-player weekly target membership**.

## What passed

### Stable player / episode identity

2024:
- restored RB/WR/TE episodes: **147**;
- named players: **133**;
- exact return-row identity coverage: **100%**;
- immediate next observation identified: **127 / 147 = 86.39%**.

2025:
- restored episodes: **173**;
- named players: **142**;
- exact return-row identity coverage: **100%**;
- immediate next observation identified: **148 / 173 = 85.55%**.

Chronology and exact identity integrity passed.

### Finite team target opportunity

Across **862** represented team-weeks:

- exact team-target uniqueness violations: **0**;
- nonfinite/nonpositive team-target weeks: **0**;
- RB + WR + TE target sum exceeding team targets: **0**;
- negative residual team-target weeks: **0**.

Therefore finite team opportunity / teammate competition can be preserved at
the team-target denominator.

That is not the blocker.

## Exact blocker

The immutable Receiving Opportunity Earning / Participation Exposure V1 parent
contains a field:

`playerform_weekly_row_present`

The parent construction also used:

`targets = 0.0 if the exact PlayerForm player-week row is absent`.

That behavior was acceptable for its earlier source-purpose audit, but it is
not sufficient for this stricter return-state target-membership contract.

A numeric zero can mean either:

1. the named player has an exact weekly PlayerForm row and truly recorded zero
   targets; or
2. the exact PlayerForm player-week row is absent and the parent filled zero.

For an individual-player opportunity model, those states cannot be conflated.

The frozen V1 therefore required every positive-participation opportunity row
to have exact PlayerForm row membership.

That gate failed.

### 2024

- restored episodes: **147**;
- episodes with >=3 exact valid strictly-prior receiving rows:
  **101 / 147 = 68.71%**.

Family diagnostics:
- RB: 44 restored / 39 players; prior support 36; return-valid 41;
- TE: 36 / 34; prior support 18; return-valid 24;
- WR: 67 / 60; prior support 47; return-valid 58.

### 2025

- restored episodes: **173**;
- episodes with >=3 exact valid strictly-prior receiving rows:
  **121 / 173 = 69.94%**.

Family diagnostics:
- RB: 39 restored / 33 players; prior support 31; return-valid 31;
- TE: 43 / 37; prior support 26; return-valid 31;
- WR: 91 / 72; prior support 64; return-valid 78.

The frozen >=95% prior-support gate failed materially in both seasons.

The exact positive-participation PlayerForm-membership gates also failed for
prior rows, restored-return rows and some immediate-next rows.

## Why this is a real fail-close

PRISM may not say:

> no PlayerForm row = zero target earning.

That would repeat the same class of causal mistake as treating:
- DNP as zero performance;
- injury-shortened production as ordinary decline.

Missing source membership is a source question, not a player-performance value.

Therefore the bridge is blocked rather than rescued.

## Immediate-next source states

These were recorded only as source state counts.

2024:
- `NO_REPORT_PARTICIPATED`: **80**;
- `REPORTED_INJURY_PARTICIPATED`: **47**.

2025:
- `NO_REPORT_PARTICIPATED`: **102**;
- `REPORTED_INJURY_PARTICIPATED`: **46**.

No target-value relationship across these states was scored.

## What was NOT done

- no target-restoration relationship;
- no target-share comparison;
- no target prediction;
- no carry/reception/yard/TD relationship;
- no alternate target provider rebuild;
- no missing-row-to-zero rescue;
- no lowered source gate;
- no fitted coefficient;
- no searched threshold;
- no route inference from snaps;
- no 2026 outcome;
- no sportsbook/OddsAPI;
- no production mutation.

## Anti-rescue

Do not rescue this exact V1 by:
- accepting the parent's filled zero as exact target evidence;
- lowering the 95% prior-support gate;
- reducing minimum prior history;
- fuzzy matching players;
- dropping TE or another family because its coverage is weaker;
- using only the subset with convenient target rows as a production rule;
- reusing the CLOSED exposure x earning formula;
- switching to position-level averages.

Disposition remains source blocked.

## Scientific consequence

Qualified upstream state remains intact:

`PRISM_RETURN_STATE_NEXT_ROLE_PERSISTENCE_V1_QUALIFIED_STATE_ONLY`.

What failed is only the exact downstream historical receiving-opportunity
membership bridge.

PRISM knows more about the named player's participation state than it can
currently certify about his historical target-earning state from this exact
parent.

That distinction must be preserved.

## Exact next boundary

Per queue governance, do not manufacture a receiving-opportunity result from a
source-blocked bridge.

Move only to a genuinely distinct named-player **CURRENT ROLE / STATE**
mechanism.

A clean next state question is:

> When the named player's participation has returned to
> `RETURN_PARTICIPATION_RESTORED`, does the weekly injury report still show an
> unresolved injury state, or has the player also cleared the report?

This separates:
- participation restoration;
- injury-report clearance;
- later opportunity earning.

It uses source-certified player health state and does not require missing target
membership.

Any predictive health-clearance mechanism must be frozen separately before
future state outcomes are scored.

## Integrity

- workflow: **SUCCESS**;
- immutable parent verification: **PASS**;
- focused tests: **PASS**;
- strict repo audit: **PASS**;
- finite team opportunity audit: **PASS**;
- predictive target relationships scored: **0**;
- target-restoration labels created: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- routes inferred from snaps: **false**;
- 2026 outcomes used: **0**;
- sportsbook/OddsAPI used: **0**;
- paid OddsAPI calls: **0**;
- production changed: **false**.
