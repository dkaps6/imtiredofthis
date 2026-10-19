# PRISM Restored-Return Injury-Report State Persistence V1 — Result

Date: 2026-10-18

Status: **CLOSED / HEALTH-STATE RELIABILITY SIGNAL DID NOT CONFIRM / NO PRODUCTION CHANGE**

Model: **PRISM — Player Role & Individual Situation Model**

Frozen contract:
`docs/research/PRISM_RESTORED_RETURN_INJURY_REPORT_STATE_PERSISTENCE_V1_FREEZE.md`

Freeze:
`513a5e5ee29b909aeaff387396352e1a10d83633`

Canonical execution:
- run `39180544730` — **SUCCESS**;
- exact head `5b2af3bda19fe16a150a6484e96050bd365f12db`;
- job `117832159685` — **SUCCESS**;
- artifact `12155682815`;
- artifact name
  `prism-restored-return-injury-report-state-persistence-v1-39180544730`;
- digest
  `sha256:0e868c6424f76abd423e4aff6cdebdeec6d5c4690f212759dfd4c67cc292fec8`.

Disposition:

`PRISM_RESTORED_RETURN_INJURY_REPORT_STATE_PERSISTENCE_V1_CLOSED`

## Player-first question

Once this named player's participation is source-certified restored, does
whether a weekly injury-report row remains present at the return observation
help distinguish whether the player's immediately next scheduled observation is
a clean no-report participation state?

Under the frozen V1 contract, the answer did **not** confirm strongly enough.

This is a health-state / reliability question only.

## 2024 development

- restored source episodes: **162**;
- scored episodes: **147**;
- named players: **135**;
- immediate-next coverage: **90.74%**.

Return state:
- no weekly report at restored return:
  **18 episodes / 18 players**;
- report still present:
  **129 episodes / 123 players**.

Equal-player clean-next rate:
- no-report return: **72.22%**;
- report-present return: **54.07%**;
- frozen gain: **+0.181572**.

Immediate-next source states:
- `NO_REPORT_NO_OFFENSIVE_SNAPS_UNRESOLVED`: **5**;
- `NO_REPORT_PARTICIPATED`: **84**;
- `REPORTED_INJURY_NO_OFFENSIVE_SNAPS`: **10**;
- `REPORTED_INJURY_PARTICIPATED`: **48**.

Frozen support failed because the no-report return group had fewer than the
required 30 episodes / 25 named players.

## 2025 independent confirmation

- restored source episodes: **187**;
- scored episodes: **168**;
- named players: **142**;
- immediate-next coverage: **89.84%**.

Return state:
- no weekly report at restored return:
  **30 episodes / 28 players**;
- report still present:
  **138 episodes / 119 players**.

Equal-player clean-next rate:
- no-report return: **66.07%**;
- report-present return: **63.03%**;
- frozen gain: **+0.030462**.

Immediate-next source states:
- `NO_REPORT_NO_OFFENSIVE_SNAPS_UNRESOLVED`: **8**;
- `NO_REPORT_PARTICIPATED`: **105**;
- `REPORTED_INJURY_NO_OFFENSIVE_SNAPS`: **10**;
- `REPORTED_INJURY_PARTICIPATED`: **45**.

Player-cluster bootstrap:
- valid replicates: **5000 / 5000**;
- 95% interval:
  **[-0.169569, +0.221526]**.

The confirmation lower bound did not stay above zero.

## Interpretation

The exact binary weekly-report-presence mechanism is CLOSED.

A named player can have restored participation while still appearing on the
weekly injury report, but the mere presence/absence of that report row is not a
reliable enough incremental predictor of the next clean source state under this
frozen contract.

Do not interpret `no report` as medically healthy.

Do not infer injury severity from this result.

Do not change projected snaps, target share, carry share, yards or touchdowns
from this mechanism.

## Anti-rescue

Do not rescue with:
- injury body part;
- questionable / doubtful / out subsets;
- practice-status subgroups;
- family / position cuts;
- alternate lookahead;
- changed return thresholds;
- 2026 outcomes;
- sportsbook availability.

This exact family is closed.

## What remains qualified

Return-State Next-Role Persistence V1 remains:

`PRISM_RETURN_STATE_NEXT_ROLE_PERSISTENCE_V1_QUALIFIED_STATE_ONLY`.

That means restored participation itself carries valid named-player state
information.

This closure says only that weekly report presence does not add a confirmed
reliability layer under this V1.

## Exact next boundary

Stop extending the injury-specific lane.

Return to:

`PRISM_CURRENT_ROLE_STATE_ARBITRATION_FRONTIER_OPEN__SOURCE_AUDIT_ONLY`.

The next legal work is a source/lineage audit for one genuinely new pregame
named-player regime-transition object that is:
- distinct from raw recent share;
- distinct from WR/TE target-share trajectory;
- distinct from RB carry/snap state;
- distinct from active-role / active-set allocation;
- distinct from blocked teammate-availability role elasticity;
- distinct from blocked depth-role transition;
- market-specific rather than a universal multiplier.

No outcome scoring until that source object passes parity and chronology gates.

## Integrity

- targets/carries/receptions/yards/TDs used: **0**;
- routes inferred from snaps: **false**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 outcomes used: **0**;
- sportsbook/OddsAPI used: **0**;
- paid OddsAPI calls: **0**;
- production changed: **false**.
