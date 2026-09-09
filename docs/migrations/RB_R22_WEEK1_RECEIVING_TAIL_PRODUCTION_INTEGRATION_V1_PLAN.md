# RB R22 Week-1 Receiving Tail Production Integration V1 — Frozen Plan

Date frozen: 2026-09-08/09 UTC
Branch: `research-cross-position-catastrophic-casebook-v1`
Parent scientific/deployment lineage: R16 -> R17 -> R18 -> R19 -> R20
Prospective control/shadow lock: R21 run `34294227588`, artifact `10082525892`
Status: PLAN FROZEN BEFORE PRODUCTION INTEGRATION EXECUTION AND BEFORE 2026 WEEK-1 OUTCOMES

## User/product objective

The production model must be ready to generate the best validated pregame Week-1 player distributions and player-prop probabilities before kickoff. Prospective grading is a parallel evidence ledger, not a reason to intentionally leave an already-supported distribution improvement out of the live Week-1 model.

## Research decision

R16-R20 together are sufficient to authorize a governed Week-1 **RB receiving-yard distribution** integration candidate now:

- R16: strict-prior upside-tail state is predictable OOS.
- R17: mean-preserving tail mixture passed every frozen distribution gate.
- R18: canonical Monte Carlo adapter parity passed with exact mean preservation/non-RB protection.
- R19: strict-prior 2026 deployable scorer serialized and passed all parity/leakage/integrity gates.
- R20: exact scorer/adapter passed on the real governed 2026 Week-1 Full Slate with 94 RBs and 10,000 draws, zero sportsbook inputs, zero outcome inputs, zero production parameter changes and effectively exact mean parity.
- R21: exact Week-1 CONTROL and SHADOW draws were sealed prospectively before kickoff, preserving a clean old-vs-new prospective benchmark even if the candidate is promoted for live Week 1.

This plan does **not** authorize a new target/reception mean model and does not promote R12.

## Frozen production scope

Candidate name: `RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_INTEGRATION_V1`

Qualified scope:

- season: 2026
- week: 1
- player family adapted: RB only (FB remains canonical)
- distribution markets allowed to change: `rec_yards`, plus exact corresponding delta in `rush_rec_yards`
- receiving-yard mean: must remain canonical
- receptions: unchanged
- target entitlement: unchanged
- rush attempts/yards: unchanged
- QB/WR/TE distributions: unchanged relative to the already-certified V3 stack
- sportsbook inputs upstream: zero

The production integration is a **post-simulation distribution adapter**, not a rewrite of `scripts/simulation_v2.py`.

## Frozen model/pool lineage

R19 source:

- run: `34288244770`
- artifact: `10080377483`
- artifact digest: `sha256:11432b9d7b7f2367935a862b63c30df9f40955e479806ab67d90921b63a907c7`
- head SHA: `6ac1342f737f142acac6a3e4b459f442faf1442a`
- model file SHA256: `9ed6a98b0022e86992fb468df40a9fd79a54bc87885777ac5955a898b5c292ba`
- residual-pool NPZ SHA256: `c69a268a5a1683e846bcb5f59fe55bcae20d70c679792449ea77e6548b37a362`

Residual semantic hashes:

- non-tail: count 3,890; `f677e91cd25cdbd6db044e9decccec6312943b99f8ffc25a827527e54d4d7b1d`
- tail30-49: count 174; `3da7bf656fcab3c111c8d5fb60e38fb7f735d7fcce639dabad899431f613c225`
- tail50+: count 79; `ee203d3ffa01b8687e7774d817d0dce6cdc8b56ca62321a5583d5287dfd5ee43`

Production must fail closed if the exact pinned R19 model/pools are unavailable or drift.

## Frozen architecture

The V3 production stack is preserved intact:

`M38 -> explicit target entitlement -> TE-R5P -> WR-R15 -> canonical joint MC -> mean-neutral QB C2 selector`

R22 adds one seam after that:

`... -> QB C2 selected result -> RB R19/R17 receiving-tail adapter -> downstream M89/M90 QB mean / RB P3 mean authorities / sportsbook pricing`

Because QB C2 changes only qualified QB `pass_yards` distribution shape and R22 changes only RB `rec_yards`/matching `rush_rec_yards`, the two adapters must be compositionally isolated.

## Frozen production scorer semantics

Use exactly the R19 forward chain already reproduced by R20:

`R8 strict-prior identity -> R9 reliability/target-delta feature -> R11 high-state probability -> R16 p30/p50 -> R17 residual pools -> R18 rank-preserving mean-neutral adapter`.

R9 target delta remains a **scorer feature only**. It may not alter live target entitlement, receptions, or the receiving-yard mean.

R11 state probability remains TOP20 identity only; REST80 exactly zero.

History source must be strict-prior through 2025 for 2026 Week 1.

Adapter seed remains `918`, matching R18/R20.

## Frozen integration gates

R22 integration passes only if **all** gates pass on the governed Week-1 replay:

1. exact R19 artifact/model/pool lineage and hashes;
2. V3 certified stack passes before R22;
3. 16 games / 32 teams and the governed player universe are preserved;
4. exactly 94 Week-1 RBs are adapted;
5. no FB is adapted;
6. max RB CONTROL-vs-candidate `rec_yards` mean delta <= `1e-8` yards;
7. minimum RB CONTROL-vs-candidate receiving-yard Spearman >= `0.9999`;
8. all adapted receiving-yard draws finite and nonnegative;
9. deterministic replay under frozen seed;
10. every non-RB simulation array is exactly unchanged;
11. for RBs, every market except `rec_yards` and `rush_rec_yards` is exactly unchanged;
12. each adapted `rush_rec_yards` draw equals adapted `rush_yards + rec_yards` within `1e-10`;
13. target entitlement trace is byte/semantic unchanged by R22;
14. TE-R5P audit remains valid and unchanged in scientific meaning;
15. WR-R15 audit remains valid and unchanged in scientific meaning;
16. QB C2 audit remains valid and its selected QB arrays are unchanged by R22;
17. RB P3 Week-1 rushing pricing authority remains intact;
18. `receptions` priced rows/distributions are unchanged;
19. priced-row universe is unchanged;
20. only RB `rec_yards`/`rush_rec_yards` distribution-derived probabilities/odds may differ; their football mean must remain equal within `1e-8`;
21. no sportsbook variable enters R19 scoring or the adapter;
22. no 2026 outcome enters scoring;
23. zero unrelated production parameters changed;
24. all existing Full Slate mechanical/model-quality validators pass after R22.

Do not weaken these gates after seeing the integration result.

## Output provenance requirements

Final pricing output must expose auditable RB receiving-tail lineage for qualified rows, at minimum:

- `rb_receiving_tail_applied`
- `rb_receiving_tail_version`
- `rb_receiving_tail_model_run`
- `rb_receiving_tail_mean_preserved`

A separate audit must record per adapted RB:

- canonical/adapted mean
- mean delta
- rank correlation
- canonical/adapted q50/q75/q90/q95
- R19 state probability, p30 and p50
- exact model/pool lineage.

## Promotion decision

A governed R22 replay PASS authorizes the R22 adapter for **2026 Week 1 production Full Slate only**.

It does not:

- promote an RB reception model;
- change target entitlement;
- authorize Weeks 2-18 without a separately qualified runtime/history contract;
- remove the R21 prospective ledger;
- auto-authorize any future retrained version.

R21 remains the clean pre-outcome prospective comparator for evaluating whether this Week-1 production decision helped distribution quality.
