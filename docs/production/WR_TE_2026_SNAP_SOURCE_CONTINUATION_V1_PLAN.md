# WR/TE 2026 Snap Source Continuation — Production Integration V1

**Status:** FROZEN BEFORE CODE CHANGE  
**Branch:** `production-wr-te-2026-snap-continuation-v1`  
**Parent research authority:** `WR_TE_2026_SNAP_SOURCE_CONTINUATION_READY`  
**Research run:** `35742765095`  
**Research artifact:** `10699873027`  
**Research digest:** `sha256:dbd15f7f9d2eb34d2c28aec07fb71dc0dccb80e8f6d40f0165b92c91eee62ddd`

## Purpose

Prospectively continue the already-authorized strict-prior snap source used by
TE-R5P and WR-R15 into 2026, beginning with target Week 3.

No model coefficients, learned features, pool conservation rules, entitlement
logic, or sportsbook inputs change.

## Activation boundary

The continuation is **prospective only**:

- all target seasons <= 2025: legacy source `2020..2025`;
- 2026 Week 1: legacy source `2020..2025`;
- 2026 Week 2: legacy source `2020..2025`;
- 2026 Week 3+: source `2020..2026`.

This prevents rewriting the historical Week-1/Week-2 production record and
preserves the paid Week-2 replay exactly.

## Implementation contract

1. Add a pure source-season resolver in the shared TE-R5P adapter.
2. Make the shared snap loader accept target season/week.
3. TE-R5P must derive one unique target season/week from its entitlement frame
   and request the corresponding source contract.
4. WR-R15 must do the same through the shared loader.
5. Existing strict-prior feature constructors remain unchanged.
6. Frozen model JSONs remain unchanged.
7. Audit output must disclose the actual snap source seasons used.

## Required gates

- pure activation-boundary unit test;
- 2025 target resolves legacy source;
- 2026 W1/W2 resolve legacy source;
- 2026 W3+ resolves legacy + 2026;
- mixed target season/week frame fails closed;
- existing TE-R5P / WR-R15 tests remain green;
- Repo CI green;
- preserved paid Week-2 Full Slate replay green and identical in the protected
  replay contract;
- no paid/live OddsAPI pull.

## Production decision

Merge only if all required gates pass and automated review finds no unresolved
correctness/integrity defect.

## Disposition

`WR_TE_2026_SNAP_SOURCE_CONTINUATION_PRODUCTION_PLAN_FROZEN`
