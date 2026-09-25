# Receiver Room Targetable-Rate V1 — Historical Baseline Authority Amendment

Date: 2026-09-25

Status: **FROZEN BEFORE 2022-2023 ROOM OUTCOME SCORING**

The parent plan correctly requires a leakage-safe baseline, but its wording
"current historical receiver stack" needs a provenance clarification.

## Specialist authority boundary

The preserved fold-safe specialist replay authority is:

- TE-R5P: authorized historical production-order replay for 2024 and 2025;
- WR-R15: authorized historical production-order replay for 2024 only;
- WR-R15 2025 retrospective application is explicitly forbidden by its contract.

There is no authorized fold-safe TE-R5P / WR-R15 backcast for 2022-2023.

Therefore those later specialists must **not** be applied retrospectively to the
2022-2023 first temporal screen.

## Frozen 2022-2023 baseline

For 2022-2023 only, baseline room target forecasts use the leakage-safe
historical entitlement stack available before those specialists:

1. historical football context;
2. Bayesian/rules target-share authority;
3. promoted M38 WR hierarchy semantics already present in the historical
   simulation stack;
4. explicit target entitlement materialization;
5. no TE-R5P or WR-R15 backcast.

Baseline room targets are:

baseline projected dropbacks * summed baseline entitlement for that room

This creates a like-for-like historical room-opportunity comparison without
injecting future-trained specialist coefficients.

## Candidate unchanged

The candidate formula remains exactly frozen:

R_g = sum(strict-prior room targets) / sum(strict-prior team dropbacks)

candidate room targets = projected dropbacks * R_g

No parameter, window, gate, room definition, or fallback changes.

## If 2022-2023 passes

A separately frozen unchanged 2024-2025 confirmation must use the authorized
current specialist order for those seasons:
- 2024: TE-R5P + WR-R15
- 2025: TE-R5P only, with WR-R15 retrospective application forbidden

The 2022-2023 screen and 2024-2025 confirmation must be reported separately.

No result has been inspected before this amendment.
