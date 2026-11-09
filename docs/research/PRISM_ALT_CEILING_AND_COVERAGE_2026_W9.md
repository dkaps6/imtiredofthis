# Why the high-usage plays never reached the board — 2026 week 9

**Date:** 2026-11-08 · **Rows:** `data/research/alt_graded_2026_wk09_v1.csv`
**Status:** research finding. Two distinct causes, one data, one model. Nothing promoted.

A direct challenge was raised: Bijan Robinson, Puka Nacua and CeeDee Lamb were
obvious alt-rushing/receiving plays, and the shipped board contained none of
them. Both halves of that are correct, for two unrelated reasons.

## Cause 1 — alt coverage had a 16% hole, concentrated by kickoff time

| | count |
|---|---|
| players on the main priced board | 231 |
| players with any alt offer | 198 |
| on main board, **no alt ladder at all** | **37 (16%)** |

The 37 are not random. They are whole games:

- **BAL / JAX** — played Thursday. The Sunday capture is after the fact.
- **CIN / ATL** — kicked 09:30 ET in London; the capture ran 09:08 ET, 22
  minutes out, with alt markets already thinning.

Highest-projection players with no alt coverage: Joe Burrow (254.9), Trevor
Lawrence (242.7), Michael Penix Jr. (239.3), **Bijan Robinson (100.7)**, Chase
Brown (75.6), Derrick Henry (70.5), Zay Flowers (65.6), Drake London (63.5),
Ja'Marr Chase (57.3), Tee Higgins (56.5).

**Bijan Robinson was never rejected by any filter. He had no alt ladder to
reject.** This was presented as a full board without the coverage caveat.

Fix: capture per kickoff window rather than once, and emit a coverage manifest
(priced players vs players with alt ladders, grouped by game and kickoff) so a
hole is visible before selections are made, not after.

## Cause 2 — PRISM compresses the high end of the usage distribution

By main-line quartile, week 9 settled games:

**rec_yards (n=127)**

| tier | n | avg main | PRISM | actual | PRISM bias | line bias |
|---|---|---|---|---|---|---|
| bottom | 32 | 9.5 | 15.7 | 16.1 | -0.4 | -6.6 |
| low-mid | 32 | 18.6 | 21.8 | 29.9 | -8.1 | -11.4 |
| high-mid | 31 | 32.8 | 33.5 | 36.9 | -3.4 | -4.1 |
| **TOP** | 32 | 58.3 | 53.1 | 65.6 | **-12.5** | -7.2 |

**rush_yards (n=57)**

| tier | n | avg main | PRISM | actual | PRISM bias | line bias |
|---|---|---|---|---|---|---|
| bottom | 15 | 8.9 | 10.2 | 17.1 | -6.9 | -8.2 |
| low-mid | 14 | 21.7 | 22.6 | 26.2 | -3.6 | -4.5 |
| high-mid | 14 | 39.5 | 37.7 | 36.6 | +1.2 | +3.0 |
| **TOP** | 14 | 67.1 | 51.8 | 59.1 | **-7.3** | **+8.1** |

In the top rushing quartile the market line was 8.1 yards **high** while PRISM
was 7.3 yards **low** — PRISM is 15.4 yards worse than the line precisely where
the biggest tickets live.

Largest misses, all ceiling games: CeeDee Lamb -128.0 (proj 67.0, actual 195),
Matthew Golden -110.8, Christian Watson -105.8, Kayshon Boutte -101.3, Tank
Dell -76.1, Puka Nacua -71.8 (proj 78.2, actual 150), DeMario Douglas -69.3,
Tyler Goodson -60.2 (rush), Kenneth Walker III -48.4 (rush).

This is the **high-end ceiling-compression limitation already documented for
RB P3** in AGENTS.md §2, which that promotion required be "tracked
prospectively." This is the prospective measurement, and it is not confined to
RBs — it is at least as severe for WR `rec_yards`.

Consequence for selection: because `p_prism` is the binding constraint on 35.8%
of `rec_yards` and 54.8% of `rush_yards` rungs, a filter requiring PRISM to beat
the price systematically removes exactly the high-usage OVER rungs that pay.
Puka Nacua produced **zero** eligible legs; every rung on his ladder was killed
by `p_prism`. All 24 of his rungs hit, to >139.5 at +429. Lamb >124.5 at +360
was in the v1 lottery ticket and was removed in v2 on PRISM's 3.4% estimate.
It hit.

The slate was a high-passing outlier — the market line also ran low on
`rec_yards` TOP (-7.2) — so some of this is slate variance. The relative
comparison against the line is the fair one, and PRISM lost it at the top in
both markets.

## Cause 3 (secondary) — no price band is profitable blind

All graded alt OVER rungs priced better than -250: n=3,275, 36.4% hit vs 39.6%
breakeven, **-526.4u, -16.1% ROI**. By band:

| price | n | hit% | breakeven | ROI |
|---|---|---|---|---|
| <=-250 | 1031 | 78.7% | 82.6% | -4.7% |
| -250..-150 | 551 | 63.0% | 65.5% | -3.8% |
| -150..-100 | 676 | 51.9% | 54.1% | -4.0% |
| +100..+200 | 873 | 38.9% | 41.4% | -6.5% |
| +200..+400 | 476 | 20.6% | 26.6% | -22.8% |
| +400..+1000 | 407 | 9.6% | 13.6% | -35.8% |
| >+1000 | 273 | 2.6% | 6.0% | -61.0% |

Every band loses. Price filtering alone is not a strategy. Within the -250 band
`rec_yards` still returned +47.9u (+4.6%), so the signal is market selection,
not price selection.

## What this supports

1. Emit an alt coverage manifest per run; capture per kickoff window. A 16%
   hole must never again be invisible at selection time.
2. Do not let `p_prism` veto OVER rungs for top-quartile-usage players. Either
   bias-correct `model_proj` by usage tier as well as by market, or screen those
   players on own-record and the table alone.
3. Keep the triple-intersection for `receptions`, where it added 73pp of ROI
   over the blind market.
4. Treat ceiling compression as a first-class production defect alongside the
   `model_sd` width defect. Both are the same failure: a distribution fitted to
   the middle and never checked at the edges.
5. One slate. Re-test on week 10 before changing production.
