# PRISM Return-State Receiving Opportunity V1 — Corrected Source Result

Date: 2026-10-18

Status: **SOURCE BLOCKED / EXACT PLAYER TARGET MEMBERSHIP INSUFFICIENT / CORRECTED CHRONOLOGY / NO PREDICTIVE SCORING**

Model: **PRISM — Player Role & Individual Situation Model**

Frozen source plan:
`docs/research/PRISM_RETURN_STATE_RECEIVING_OPPORTUNITY_V1_SOURCE_PLAN.md`

Initial document-only freeze:
`3a2a51e5412ad83265700dbf088e2425bae359c3`

Authoritative pre-implementation chronology amendment / implementation freeze:
`930907dd91c1b4139f2aa6be52db23a730c7a96b`

Canonical corrected execution:
- run `39180575868` — **SUCCESS**;
- exact head `af3ca4469ae3fad59b219a36a79adbbb1b3ed640`;
- job `117832254171` — **SUCCESS**;
- artifact `12156510164`;
- artifact name
  `prism-return-state-receiving-opportunity-v1-source-39180575868`;
- digest
  `sha256:ff57c3b2f3b700444ffb5fcc89f9ffa98242861711c7f143ec948d6ad3a03bf6`.

Disposition:

`PRISM_RETURN_STATE_RECEIVING_OPPORTUNITY_V1_SOURCE_BLOCKED`

## Lineage correction

Earlier run `39180266614` used the stale pre-amendment freeze
`3a2a51e...` and discovered the immediate next observation from a
positive-participation-only receiving panel.

That run is **noncanonical science** because it could skip a reported no-snap /
DNP / unresolved observation.

The corrected run uses immutable Observation Health-State V1 as the chronology
authority for the exact immediately next scheduled same-team player observation,
then left-joins receiving evidence onto that exact week.

No target-restoration relationship was scored in either run.

## Player-first question

Can PRISM attach exact same-player receiving opportunity evidence to a
source-certified restored-return episode, at stable player + team + week grain,
while preserving exact finite team target volume, before any target-restoration
relationship is scored?

The corrected answer remains **not cleanly enough**.

The team opportunity ledger is source-valid.

The blocker is exact named-player weekly target membership.

## 2024 corrected source support

- restored RB/WR/TE episodes: **147**;
- named players: **133**;
- exact return-row identity coverage: **100%**;
- exact >=3-row strictly-prior target support:
  **101 / 147 = 68.71%**;
- immediate next scheduled observation:
  **133 / 147 = 90.48%**;
- clean `NO_REPORT_PARTICIPATED`: **77**.

Immediate-next source states:
- `NO_REPORT_NO_OFFENSIVE_SNAPS_UNRESOLVED`: **4**;
- `NO_REPORT_PARTICIPATED`: **77**;
- `REPORTED_INJURY_NO_OFFENSIVE_SNAPS`: **9**;
- `REPORTED_INJURY_PARTICIPATED`: **43**.

Family diagnostics:
- RB: 44 restored / 39 players; prior support 36; return-valid 41;
- TE: 36 / 34; prior support 18; return-valid 24;
- WR: 67 / 60; prior support 47; return-valid 58.

## 2025 corrected source support

- restored episodes: **173**;
- named players: **142**;
- exact return-row identity coverage: **100%**;
- exact >=3-row strictly-prior target support:
  **121 / 173 = 69.94%**;
- immediate next scheduled observation:
  **156 / 173 = 90.17%**;
- clean `NO_REPORT_PARTICIPATED`: **97**.

Immediate-next source states:
- `NO_REPORT_NO_OFFENSIVE_SNAPS_UNRESOLVED`: **8**;
- `NO_REPORT_PARTICIPATED`: **97**;
- `REPORTED_INJURY_NO_OFFENSIVE_SNAPS`: **8**;
- `REPORTED_INJURY_PARTICIPATED`: **43**.

Family diagnostics:
- RB: 39 restored / 33 players; prior support 31; return-valid 31;
- TE: 43 / 37; prior support 26; return-valid 31;
- WR: 91 / 72; prior support 64; return-valid 78.

## Finite team opportunity passed

Across **858** represented team-weeks:
- team-target uniqueness violations: **0**;
- nonfinite/nonpositive team-target weeks: **0**;
- RB + WR + TE target sums above team targets: **0**;
- negative residual team-target weeks: **0**.

Finite team opportunity / teammate competition is not the blocker.

## Exact blocker

The immutable receiving parent can contain:

`playerform_weekly_row_present = false`

while carrying a filled numeric:

`targets = 0`.

For an individual-player opportunity model, those states cannot be conflated.

A missing exact weekly PlayerForm target row is not evidence that the named
player earned zero targets.

The frozen >=95% exact strictly-prior target-support gate therefore fails
materially in both seasons, and exact PlayerForm membership also fails for some
restored-return / positive immediate-next rows.

Do not lower the gate.

## Scientific meaning

This source block says only:

> PRISM can certify the named player's participation restoration and finite team
> target environment, but this exact immutable parent cannot certify enough of
> the player's own historical target-earning rows to test target restoration
> cleanly.

It does **not** say injury context is unimportant.

It does **not** say targets do or do not restore.

It does **not** permit missing target rows to become zeroes.

## Anti-rescue

Do not rescue with:
- missing-row zeroes;
- fuzzy identity;
- a lower history-support gate;
- fewer required prior rows;
- family cherry-picks;
- deletion/reweighting;
- the closed exposure x earning formula;
- route claims from snaps;
- 2026 outcomes;
- sportsbook/OddsAPI.

## Integrity

- corrected chronology authority: **PASS**;
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

## Next boundary

The source-block fallback permitted one genuinely distinct current health-state
mechanism. That follow-up has now been scored separately and is CLOSED:

`PRISM_RESTORED_RETURN_INJURY_REPORT_STATE_PERSISTENCE_V1_CLOSED`.

Do not continue searching injury-report variants.

Return to the broader player-first Current Role-State Arbitration frontier and
source-map a genuinely new pregame regime-transition object under its anti-retest
contract.
