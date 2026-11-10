# Per-cell alt table-support audit

**Date:** 2026-11-10 · **Rows:** `data/research/alt_table_support_audit_2026_wk09_v1.csv`
**Status:** closes the standing REQUIRED gate — *"run the per-cell alt table-support
audit (n, players, the post-hoc flag) before treating any alt row as bettable."*
It had never been run. Everything priced from these tables so far ran ahead of it.

**VERDICT: the tables are bettable only where they UNDERSTATE. That is `rec_yards`
and `pass_yards`. They are not bettable for `rush_yards`, `rush_rec_yards`,
`receptions`, or any rung beyond 1.6x of the line, in any market.**

## 1. Cell support — the clearance table passes

`data/research/alt_line_clearance_v2.csv`, 130 cells over four markets:

| | min | p25 | median | max |
|---|---|---|---|---|
| `n` | **516** | 630 | 712 | 1545 |
| `players` | **43** | 92 | 115 | 250 |

**Zero cells below n=200.** The thin-cell risk this audit was written to catch does
not exist here. The thinnest cells are `rush_rec_yards` quartile 0 (n=516, 156
players) and `rush_yards` quartile 0 (n=579, 97 players).

## 2. The receptions table is post-hoc on every cell, and the restriction did not work

`data/research/alt_line_receptions_absolute_v2.csv`, 48 cells, **all 48 carrying
`post_hoc_line_restriction = True`.**

`PRISM_ALT_LINE_TABLES_V2_RESULTS` is honest about why:

> decile calibration came back **0.0525** against a stated **0.03** bar... Emitted
> cells are therefore restricted to main lines <= 4.5. **That restriction was chosen
> after seeing which cell failed — it is post-hoc.**

2026 is a clean out-of-sample season; the fit was 2024-2025. Tested against 1,260
week 9 receptions rungs (100% matched a cell), the weighted mean |error| over 30
cells is **0.0525** — **the same number that failed the bar in-sample.** Dropping
the cell that exposed the problem did not fix the problem.

Signed error **-0.0177: the table OVERSTATES.** The miss is concentrated in the
tail, and every deep rung came in at exactly zero:

| main line | >= k | table says | week 9 actual |
|---|---|---|---|
| 2.5 | 6 | 0.067 | **0.000** |
| 2.5 | 7 | 0.030 | **0.000** |
| 3.5 | 7 | 0.098 | **0.000** |
| 3.5 | 8 | 0.049 | **0.000** |
| 4.5 | 9 | 0.052 | **0.000** |
| 4.5 | 10 | 0.023 | **0.000** |

Those are the cells that built the week 9 LOTTERY tier, which went 0-for-6.

## 3. The fraction table misses by the same magnitude, and the DIRECTION is what matters

Weighted mean |error| over the fraction table against week 9: **0.0517** — also
above the 0.03 bar. But the signed error splits hard by market, and that split
**predicts realized profit**:

| market | n | table | actual | calibration error | ROI | P&L |
|---|---|---|---|---|---|---|
| `rec_yards` | 1401 | 0.548 | 0.593 | **+0.0449** | **+2.9%** | +40.5u |
| `pass_yards` | 651 | 0.427 | 0.469 | **+0.0413** | -0.7% | -4.4u |
| `receptions` | 1260 | 0.331 | 0.310 | -0.0202 | **-30.0%** | -377.9u |
| `rush_yards` | 673 | 0.539 | 0.502 | -0.0367 | **-18.5%** | -124.2u |
| `rush_rec_yards` | 321 | 0.508 | 0.427 | **-0.0809** | **-33.9%** | -108.9u |

**correlation(calibration error, ROI) = +0.923.** Every market where the table
understates made money. Every market where it overstates lost it.

This is not quite tautological — ROI depends on the offered prices as well as the
probability error, so a table could overstate and still profit against generous
prices. It did not. The mechanism is doing the work.

By rung depth, across all markets:

| zone | n | table | actual | error | ROI |
|---|---|---|---|---|---|
| at or under the line | 2126 | 0.662 | 0.674 | +0.0117 | -4.4% |
| 1.0-1.6x | 1587 | 0.328 | 0.331 | +0.0035 | -9.7% |
| **beyond 1.6x** | 593 | 0.101 | 0.073 | **-0.0286** | **-55.1%** |

`rush_rec_yards` is the worst cell family in the repository: it overstates by
-0.0809 and every frac band from 0.7x upward is negative, to -0.136 at 1.3-1.6x.

## 4. What this changes

The week 9 betting results are now mechanically explained rather than attributed
to variance. `PRISM_ALT_FIRST_PRICED_GRADING_2026_W9` reported the per-market ROI
spread; this audit supplies its cause.

**Gate, effective immediately:**

1. **`rec_yards` and `pass_yards` only.** Both understate, so the table's own
   number is a floor rather than a ceiling.
2. **`rush_rec_yards` is excluded outright** until rebuilt. -0.0809 is the largest
   miss measured and it is optimistic.
3. **`rush_yards` excluded** at table face value (-0.0367).
4. **`receptions` excluded from table-priced selection.** Note this does NOT
   contradict the +43.7% ROI the triple-intersection earned on receptions in week
   9: that filter required the player's own record and PRISM to agree, which is
   what rescued it. The TABLE alone is not usable for this market.
5. **No rung beyond `frac_of_line` 1.6x in any market.** -55.1% ROI.
6. The receptions `>= k` cells at or above the zero rows in §2 are not bettable at
   any price.

## 5. What a rebuilt table must show

Not a lower in-sample error. On a season held out entirely:

1. weighted mean |error| at or under 0.03 **on the holdout**, not after removing
   the cell that failed;
2. **signed** error at or above zero per market — a table that understates is
   usable, a table that overstates is not, and the sign matters more than the
   magnitude;
3. tail cells (p <= 0.10) validated separately, because that is where the current
   table fails completely and where the longest tickets are built.

The post-hoc restriction should be re-derived on a clean split now that 2026
provides a third season, exactly as the original results document asked.
