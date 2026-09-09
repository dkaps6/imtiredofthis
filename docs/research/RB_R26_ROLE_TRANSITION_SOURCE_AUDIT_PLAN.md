# RB R26 — Role-Transition Entitlement Source Audit Plan

Status: **SOURCE AUDIT ONLY — NO SCIENTIFIC CANDIDATE SCORED**

Branch: `research-rb-r26-role-transition-entitlement-v1`
Base lineage: repaired R25 head `a74f8091edf12d3652345a945003509c58c9561c`
Protected production authority: `main@f8417f55b04ce0e19baf260e9d532765034c47f1`

## Why R26 exists

R23/R24 found a repeatable target/reception allocation signal on 2023-25, but universal redistribution did not survive receiving-yard / RB1 / tail gates. R25 then tested the fixed R23 opportunity mechanism on 2020-22 with receiving-yard means frozen. R25 failed overall; its apparent LOW_HISTORY reception-MAE gain was caused primarily by target suppression and came with worse RMSE, materially worse p90 error, and a large negative reception bias. That pattern must not be promoted as a subgroup rescue.

The next materially different football hypothesis is therefore **role-transition reliability**, not stronger historical target reallocation:

> When an RB has sparse or unstable prior receiving history, current pregame roster/depth/competition state may determine how much historical receiving identity should be trusted.

This source audit asks only whether the repo has timing-safe historical inputs capable of testing that hypothesis. It does **not** score the hypothesis.

## Hard governance

1. No target-game player outcomes may be loaded by this audit.
2. No receptions, targets, receiving yards, carries, fantasy points, model residuals, sportsbook lines, odds, or target-game participation may be used.
3. No model is fit.
4. No scientific gate is evaluated.
5. No production runtime, model asset, Full Slate workflow, R22 adapter, P3, M89/M90, WR-R15, TE-R5P, or QB C2 file may change.
6. Target-game participation is postgame information and remains forbidden.
7. Date-bearing depth-chart data is not usable until an as-of-before-kickoff contract is proven. The current historical universe builder correctly refuses to merge date-based depth rows without that proof.
8. Week-tagged depth data is only a candidate pregame source; this audit reports its coverage and schema but does not assume depth order is football truth. Prior direct depth-order remap work failed.
9. Current-week weekly roster snapshots may define the pregame universe, consistent with the existing historical input contract.
10. If timing-safe role state is absent or too sparse, R26 stops at source audit. We do not weaken the timing contract.

## Temporal exposure ledger

There is no genuinely untouched historical RB-receiving scoring block remaining in the current research archive:

- 2015-17: R9 scoring / confirmation lineage.
- 2017-19: R8 scoring / confirmation lineage.
- 2020-22: R25 scoring lineage (and 2021-22 also appear in R10 modern stability).
- 2021-25: R10/R12 modern-state lineage.
- 2023-25: R23/R24 candidate scoring lineage.

Accordingly, any later R26 historical outcome test must be labeled **predeclared retrospective mechanism confirmation**, not untouched OOS evidence. The 2026 Week-1 lock is the genuinely prospective evidence stream.

## Source-audit seasons

Audit seasons `2014-2025` so we can see the full historical transition of nflverse roster/depth semantics.

For each season, report:

- weekly-roster load success / row count / relevant schema;
- depth-chart load success / row count / relevant schema;
- regular-season week count;
- RB/HB/FB rows in weekly rosters;
- RB/HB/FB rows in depth data when position semantics are identifiable;
- depth timing class: `WEEK_TAGGED`, `DATE_BEARING`, `NO_BOUNDARY`, or `UNAVAILABLE`;
- whether the existing leakage-safe pregame-universe builder actually merges depth;
- RB pregame-universe rows and nonblank role coverage overall and Week 1;
- roster fields capable of representing current player state (status / roster status / position);
- depth fields capable of representing current role/competition state (`depth_position`, `depth_team`, equivalents);
- explicit warnings when a source cannot satisfy a pregame as-of contract.

## Exposure labels

The audit also emits a deterministic season exposure ledger with previously used scoring studies. This is governance metadata only and is not an outcome feature.

## What would authorize a later R26 model plan

A later scientific plan may be written only if this source audit identifies at least one timing-safe role-state construction with enough coverage to be meaningful. The later plan must be frozen **before** any R26 outcome scoring and must remain materially different from R12 and from direct depth-order remapping.

The intended architecture, if sources support it, is:

`production baseline RB receiving opportunity`

+ `persistent receiving identity (strict prior)`

+ `current pregame role-transition reliability state`

→ **reliability-weighted identity adjustment inside an exactly conserved RB target room**

with:

- established/stable backs anchored strongly to production;
- transition backs allowed to move only when timing-safe current role evidence supports the move;
- production catch conversion unchanged unless separately supported;
- receiving-yard mean unchanged;
- R22 receiving-yard tail unchanged;
- no direct assumption that depth rank itself equals target entitlement.

## Stop condition

If the historical source audit cannot establish timing-safe role-state coverage, the historical R26 mechanism is not run. The next valid path would be a prospective 2026 role-state lock/shadow using current pregame roster/depth evidence, not a relaxed historical proxy.
