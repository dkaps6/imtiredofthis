# PRISM TE Signed-Opportunity Conserved Bridge — Independent Path Decision

Date: 2026-10-18

Status: **BLOCKED UNDER EXISTING FROZEN GATES / NO TE BRIDGE SCORE**

Model: **PRISM — Player Role & Individual Situation Model**

## Question

After the WR-R15 exact-authority recovery failed closed, can the already-frozen TE signed-opportunity conservation bridge execute through an independently valid source and execution path without depending on the failed WR recovery?

## Live reconciliation before this decision

Before any write:
- `main`: `d7c8d17a31b7d275ab82872e526c11d13342a030`;
- research branch: `research-individual-opportunity-roadmap-2026-10-08`;
- research branch head: `e1ce9880ba8fd2ccd71cd26fe609a1a96cf66c90`;
- PR #672: OPEN / DRAFT;
- exact-head generic Actions at `e1ce9880...`: two runs, both completed/failure;
- exact-head queued/in_progress runs observed: **0**;
- exact-head PR-triggered workflow runs returned: **0**;
- exact-head combined statuses returned: **0**.

No workflow was represented as running.

## TE source authority is independently preserved

The TE data source itself does **not** require the WR-R15 reconstruction.

The frozen source-recovery amendment authorizes TE-R5P authority from:
- run `36332720457`;
- artifact `10935999653`;
- name `opportunity-authority-priority-v1`;
- digest `sha256:4fde0a72c3aef7ae2412580333a7fefcd3078e53a31f0e9ebb945fd6a6ed1bb5`.

Live artifact reconciliation:
- expired = false;
- expires = 2026-10-27T16:39:32Z.

Frozen TE parent fields:
- target opportunity from `opportunity_authority_priority_direct_detail.csv`, family `TE_TARGET_OPPORTUNITY`, field `baseline_pred`;
- receiving yards from `opportunity_authority_priority_market_detail.csv`, position `TE`, market `rec_yards`, field `baseline_proj`;
- seasons 2024 and 2025.

The qualified signed-opportunity signal also remains live:
- run `39109423266`;
- artifact `12128983909`;
- digest `sha256:bb14d6474778d0afd53ca74bb60237b381aa4e973195bab3d09d54ed87ef2490`;
- expired = false.

Therefore the TE **source bytes/evidence are independently available**.

## Binding execution blocker

The already-frozen action contract
`docs/research/PRISM_WR_TE_PLAYER_OPPORTUNITY_BIAS_CONSERVATION_BRIDGE_V1_FREEZE.md`
contains one shared precondition for **both positions**:

> The source-recovery audit must be
> `PRISM_WR_TE_OPPORTUNITY_BIAS_CONSERVATION_BRIDGE_SOURCE_RECOVERED_READY_ALL`.

The frozen recovery workflow is combined WR/TE:
`.github/workflows/research-prism-wr-te-opportunity-bias-bridge-v1-source-recovery.yml`.

It first requires the rebuilt WR-R15 prediction file to match the immutable expected SHA exactly.

Run `39110742338` / job `117626156590` failed closed because:
- expected WR-R15 prediction SHA:
  `91d108ee7981935d78179257b204a0204e9a4f8061e6eba4d6810845ed057d05`;
- rebuilt SHA:
  `7dce6b0e194d37ec3b814caf79942eea3b83fcda520b481788ec6b2b3f0778c3`.

The byte-identity assertion fails before the combined recovered-source audit can certify `READY_ALL`.

Repository-tree audit at the reconciled research head found **no separately frozen TE-only signed-opportunity bridge contract or TE-only execution workflow**.

## Decision

The correct answer is:

> **TE source authority is independently valid, but the already-frozen TE bridge does not have an independently authorized execution path under the existing pre-result gates.**

Changing the shared `READY_ALL` precondition after the WR failure, creating a TE-only exception to that frozen contract, or weakening the WR byte gate would be a post-result rescue.

None is authorized.

Disposition:

`PRISM_TE_SIGNED_OPPORTUNITY_CONSERVED_BRIDGE_EXECUTION_BLOCKED_BY_FROZEN_COMBINED_SOURCE_GATE`

## Protected science

This does **not** reverse:
- `TE_OPPORTUNITY_PLAYER_SIGNED_COMPONENT_BIAS_SIGNAL_QUALIFIED`;
- `WR_OPPORTUNITY_PLAYER_SIGNED_COMPONENT_BIAS_SIGNAL_QUALIFIED`.

It also does not reopen:
- WR or TE efficiency signed-bias cells;
- WR-R15 recovery;
- any closed Stage-1/Stage-2 bridge.

No TE bridge outcome was scored.

## Next genuinely distinct player-specific mechanism

Advance to the already-confirmed **WR/TE player-specific opportunity-uncertainty integration/shadow** frontier.

Existing evidence:
- WR: `WR_PLAYER_OPPORTUNITY_UNCERTAINTY_CONFIRMED`;
- TE: `TE_PLAYER_OPPORTUNITY_UNCERTAINTY_CONFIRMED`.

This is scientifically distinct from the blocked signed-opportunity mean bridge:
- it changes uncertainty/distribution only;
- it preserves the promoted named-player mean;
- it preserves finite opportunity allocation;
- it uses the named player's own strictly-prior opportunity-difficulty history;
- position pooling remains only the comparator/fallback.

A new integration/shadow contract must be frozen before any score and must not use 2026 outcomes, sportsbook inputs, or production mutation.

## Integrity

- TE bridge outcomes scored: **0**;
- frozen source gate weakened: **false**;
- expected WR digest changed: **false**;
- reconstructed WR authority used for scoring: **false**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 outcomes used: **0**;
- Week-6 outcomes used: **0**;
- sportsbook/OddsAPI used: **0**;
- production changed: **false**.
