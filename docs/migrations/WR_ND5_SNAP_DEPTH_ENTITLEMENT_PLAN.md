# WR-ND5 — Snap / Depth Target Entitlement Diagnostic

## Status

Frozen research protocol. Diagnostic only. No production change is authorized by this document.

## Lineage

- Exact M38 merge parent: `b98518d97b3038f471aee9ae3201009b2c70bb29`.
- M38 WR hierarchy multipliers remain frozen at `(1.40, 1.14, 0.91, 0.78)`.
- WR-ND1 localized the unresolved false-low / 10+ target opportunity problem to within-WR allocation.
- WR-ND3 result: `NO_ACTIONABLE_DYNAMIC_ENTITLEMENT_SIGNAL`; target-share recency acceleration was a near-miss but did not clear the frozen gate, while simple vacancy signals failed.
- WR-ND4 result: `SNAP_AND_DEPTH_SOURCES_RECOVERED`; prior-game offensive snap share and strictly pregame depth-chart rank/slot are genuinely new, high-coverage information families.
- ND4 canonical corrected run: `34053783657`, job `101541889893`, SHA `d33e9f2b4d80df3eb5f98eea897d5bf8afc27442`.

## Research question

Does genuinely new pregame role information — **offensive snap participation and strictly pregame depth state** — identify when M38 materially under- or over-allocates a WR's targets?

This is a within-WR entitlement diagnostic. It is not an M38 multiplier retune and it does not reopen ND3's target-history window search.

## Exact parent reconstruction

Every canonical run must separately check out exact M38 commit `b98518d97b3038f471aee9ae3201009b2c70bb29`, rebuild 2025 Weeks 1-18 historical inputs using 2024 as prior season, and verify:

- M38 receiving-yard rows = `4647`
- M38 receiving-yard MC MAE = `17.099904733366`
- ND1-comparable WR entitlement evaluation rows = `2130`
- target reconstruction MAE approximately `2.076010432546`

Any parent/casebook drift is an integrity failure.

## Leakage boundary

For target game `G`:

- snap signals may use only completed games strictly before `G`;
- depth-chart state may use only snapshots with timestamp/date strictly before `G`;
- target-week targets, receptions, yards, snap outcomes, or future depth snapshots are prohibited upstream;
- 2025 nflverse play-level participation remains prohibited because the 2023+ files are postseason releases and are not source-time safe for same-season walk-forward;
- sportsbook / player-prop information is prohibited upstream.

## Frozen signal families

### A. `SNAP_ACCEL_1V4`

`last prior same-team offense_pct - mean of last four prior same-team offense_pct values`

Purpose: identify a receiver whose actual field participation has recently expanded or contracted beyond his established role.

High / low groups: top quartile vs bottom quartile among available values.

### B. `SNAP_LEVEL_PRIOR1`

Last prior same-team offensive snap percentage.

Purpose: test whether actual recent field entitlement contains information beyond M38's target-share hierarchy.

High / low groups: top quartile vs bottom quartile among available values.

### C. `DEPTH_TOP2_STATE`

Target-game strictly pregame depth-chart `pos_rank <= 2` versus `pos_rank >= 3`.

Purpose: test current declared depth entitlement. This uses no target-game outcome and no inferred route assignment.

High group: rank `<= 2`.
Low group: rank `>= 3`.

### D. `DEPTH_RANK_PROMOTION`

`previous eligible pregame positional rank - target-game eligible pregame positional rank`

Positive values mean the player moved upward in the depth hierarchy before the target game.

The previous rank must come from the most recent strictly pregame depth snapshot associated with the team's preceding game date. If no prior eligible snapshot exists, the signal is missing rather than imputed.

High group: promotion `> 0`.
Low group: no promotion `<= 0`.

## No combinations in ND5

ND5 may not combine snap and depth signals after seeing results. A later branch may test an interaction only if its component information families independently establish evidence under this frozen protocol.

## Evaluation outcomes

Primary:

`allocation_residual = actual within-WR target share - M38 predicted within-WR target share`

Secondary:

`raw_target_error = actual targets - M38 predicted targets`

Frozen high-entitlement miss tail carried forward from ND1/ND3:

- actual targets `>= 10`, and
- raw target underprediction `>= 3`.

## Signal scoring

For each candidate signal report:

1. coverage rate;
2. Spearman correlation with allocation residual where ordinal/continuous;
3. frozen high-vs-low allocation-residual gap;
4. frozen tail enrichment;
5. W2-18 gap;
6. W13-18 gap;
7. WR1, WR2, WR3 gaps under M38;
8. target-error gap as a secondary descriptive audit.

## Frozen actionable gate

To preserve comparability with ND3, the same evidence standard is reused.

A candidate advances only if **all** are true:

- coverage >= `0.85`;
- Spearman >= `+0.08` for continuous/ordinal signals;
- high-vs-low allocation-residual gap >= `+0.025`;
- W2-18 gap > `0`;
- W13-18 gap > `0`;
- positive gap in at least 2 of WR1/WR2/WR3;
- 10+ / under-by-3 tail enrichment >= `1.20x`.

For binary `DEPTH_TOP2_STATE`, Spearman is computed on the binary indicator and must still clear `+0.08`.

For `DEPTH_RANK_PROMOTION`, the numeric promotion magnitude is used for Spearman while the frozen high/low split remains `>0` vs `<=0`.

No threshold may be lowered after results are visible.

## Frozen disposition

- exactly one signal passes: `<SIGNAL>_ACTIONABLE`
- two or more pass: `MULTIPLE_ROLE_ENTITLEMENT_SIGNALS`
- none pass: `NO_ACTIONABLE_SNAP_DEPTH_ENTITLEMENT_SIGNAL`

A passing diagnostic authorizes only a later frozen predictive integration test. It does not authorize production.

## Anti-duplication / prohibitions

Do not:

- retune M38 hierarchy multipliers;
- retry ND3 with different target-share recency windows;
- revive simple teammate-vacancy features rejected by ND3;
- use target-week snap outcomes;
- use postseason-released play participation as historical pregame truth;
- infer routes from snaps;
- infer WR-CB assignments;
- combine snap/depth features after results;
- fit a supervised model in ND5;
- use sportsbook/player-prop inputs upstream;
- promote any production logic from this diagnostic alone.

## Separate explosive / QB lane

WR explosive-yardage modeling and the future WR-ceiling-to-QB-upper-tail bridge remain separate downstream research lanes. ND5 is strictly about WR mean target entitlement.
