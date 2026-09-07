# WR-R8 Target-Dominant Role Signals — Frozen Plan

## New question
WR-R5 showed that WR receiving-yard error is player-specific: among 133 qualifying WRs, 72 were TARGETS-dominant, 44 MIXED, 16 YPR-dominant, and 1 CATCH-dominant. WR-ND5 previously tested participation/depth signals across all WRs and did not find an actionable global signal.

WR-R8 does **not** retry ND5 globally or relax its gates. It asks a materially new question enabled by the later WR-R5 decomposition:

**Do the exact pregame snap/depth signals already recovered in ND5 become predictive when evaluated only for WRs whose historical error mechanism is TARGETS-dominant?**

This is a 2025 discovery diagnostic only. It cannot change production. Any discovery pass would still require a separately frozen multi-season/source-coverage replication and then full-stack integration.

## Frozen evidence
- WR-R5 run: `34066489537`, artifact `wr-r5-target-catch-ypr-individual-decomposition`.
- WR-ND5 run: `34055534002`, artifact `wr-nd5-snap-depth-entitlement-datetime-fix`.
- Expected ND5 casebook rows: **2130**.
- Expected WR-R5 qualifying profiles: **133**.
- Expected TARGETS-dominant qualifying players: **72**.
- No sportsbook data.

## Cohort
Join WR-R5 `wr_r5_individual_mechanisms.csv` to ND5 `wr_nd5_casebook.csv` using normalized player key. Retain only players with frozen WR-R5 `dominant_mechanism == TARGETS`.

The target remains ND5's leakage-safe target-allocation residual:

`allocation_residual = actual_wr_share - pred_wr_share`

Also retain `raw_target_error = actual_targets - pred_targets` and ND5's predeclared entitlement tail.

## Exact signals
No new thresholds, windows, combinations, or features are permitted. Test only the exact ND5 signals already available pregame:
1. `SNAP_LEVEL_PRIOR1` — prior-game offensive snap percentage; quartiles.
2. `SNAP_ACCEL_1V4` — prior-game snap percentage minus prior-4 mean; quartiles.
3. `DEPTH_TOP2_STATE` — binary current pregame top-2 depth state.
4. `DEPTH_RANK_PROMOTION` — positive promotion vs non-positive.

No interactions or combinations.

## Metrics per signal
Within TARGETS-dominant WRs report:
- conditioned N and valid N;
- coverage;
- Spearman with `allocation_residual`;
- high-minus-low `allocation_residual` gap;
- high-minus-low `raw_target_error` gap;
- high-state entitlement-tail enrichment vs valid-cohort tail rate;
- W2-18 allocation gap;
- W13-18 allocation gap;
- number of players with >=8 valid games;
- fraction of those players whose within-player Spearman/sign association is positive.

For binary/positive-vs-nonpositive signals, use the frozen ND5 state definition rather than inventing quartiles.

## Frozen discovery gate
A signal passes only if **all** are true:
1. conditioned cohort N >= **600**;
2. coverage >= **0.80**;
3. Spearman >= **0.10**;
4. high-minus-low allocation-residual gap >= **0.030**;
5. high-minus-low raw-target-error gap >= **1.00 target**;
6. entitlement-tail enrichment >= **1.20x**;
7. W2-18 allocation gap > 0;
8. W13-18 allocation gap > 0;
9. >= **20** players have at least 8 valid games;
10. positive within-player association rate >= **0.58**.

The family disposition is `WR_TARGET_DOMINANT_ROLE_SIGNAL_DISCOVERY_PASS` if at least one exact signal passes every gate. Otherwise it is `NO_ACTIONABLE_WR_TARGET_DOMINANT_ROLE_SIGNAL`.

## Interpretation rules
- No threshold lowering after results.
- No alternate windows.
- No combination hunting.
- A near miss remains a failure.
- A discovery pass is not a model win and cannot alter M38.
- The next step after a pass must be a separately frozen multi-season/source-coverage replication followed by the established WR full-stack integration protocol.
