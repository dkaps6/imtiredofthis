# PLAYER SHARE INPUT COVERAGE AND RESIDUAL AUDIT V1 — PRE-RUN SEMANTIC AMENDMENT

**STATUS: FROZEN BEFORE ANY AUDIT RESULT OR ACTION RUN**

Date: 2026-10-07

Parent contract:
`docs/research/PLAYER_SHARE_INPUT_COVERAGE_AND_RESIDUAL_AUDIT_V1_CONTRACT.md`

## Why this amendment is required

The first frozen contract referenced the parallel
`PLAYER_OPPORTUNITY_VOLUME_VS_SHARE_DECOMPOSITION_V1` lineage that expressed
receiver actual share against official pass attempts.

Before this audit was ever executed, the simulator seam was re-audited and the
correct target-allocation denominator was confirmed:

- canonical receiver targets are allocated from the simulator's **dropback-side
  opportunity volume**;
- QB passing attempts separately convert dropbacks to official attempts;
- therefore comparing a target allocator probability to
  `actual targets / official pass attempts` is not like-for-like.

No audit run has occurred on this branch, so correcting the parent semantic now
does not inspect or respond to an audit result.

## Corrected immutable parent authority

Use the corrected decomposition:

- branch: `research-player-opportunity-volume-vs-share-decomposition-v1`
- run: `37694474836` — SUCCESS
- head: `3c557e846e731acb403e2df81255ec8cff403304`
- artifact: `11515041385`
- digest:
  `sha256:ea989763090af01230c94c76b63e7648f5afaa40e3a42f00b929ff56d360e16c`
- result commit:
  `e8d7a63c5d0c999787ae67ea49059887b87a3b32`

Corrected actual-share semantics:

- RB/WR/TE target share =
  `actual player targets / actual team dropback-side opportunity volume`;
- RB carry share =
  `actual player carries / canonical actual team non-dropback opportunity volume`;
- QB remains out of scope for this share-stage audit.

## Binding audit consequence

For RB/WR/TE target stages, every predicted allocator share must be compared to
the corrected parent `actual_player_share` above.

The audit may not:
- substitute official pass attempts as the receiver-share denominator;
- recompute realized share from a different denominator;
- use the earlier parallel decomposition artifact;
- change any football prediction or specialist.

All other frozen V1 contract rules remain unchanged.

## Production consequence

None.

This is a pre-result semantic correction only. Parameters fit = 0. Sportsbook
inputs = 0. Production changes = 0.
