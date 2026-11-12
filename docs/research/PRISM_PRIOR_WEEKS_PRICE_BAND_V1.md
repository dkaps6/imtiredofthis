# Weeks 1-8: the ladder cannot be rebuilt, but the mechanism can be tested

**Date:** 2026-11-12
**Scripts:** `scripts/research/grade_mainline_board_v1.py`, `scripts/research/analyze_alt_ladder_depth_v1.py`
**Rows:** `data/market_track_record/graded/2026_wk06_wk08_mainline_graded_v1.csv` (6,032 priced bets)

> ## REVISION 2026-11-12: TD PROPS ARE EXCLUDED FROM ALL EVIDENCE
>
> `anytime_td` is **not a market we bet**, so it carries no weight on any decision
> and is struck from the evidence base. It remains in the graded rows as a
> measurement artifact only, and `build_alt_cards_v1.py` has always excluded it.
>
> **This costs a claim this document originally made.** Every price band above
> +200 in weeks 6-8 was **100% `anytime_td`**, because main lines only reach
> **-270 to +299** — there are **21 bets above +150 and zero above +300** on
> bettable markets. So the "amplification law confirmed out-of-sample
> (1.65x -> 21.59x)" in §5 and the "longshot collapse replicated on 825 bets" both
> rested entirely on a market we do not play. **Both are withdrawn as evidence.**
> §5 is rewritten below with what the bettable markets actually support.
>
> **What is unaffected**, because it was computed on yardage/receptions only from
> the start: the +12.1-point overstatement (§2), the +3.48-point edge over a coin
> flip (§3), the disagreement gradient and the 5-6x overdispersion factor (§4),
> and weeks 1-2 (§6).

---

## 0. There is no alt ladder before week 9, and there never was

The week-9 ladder analysis (`PRISM_ALT_LADDER_DEPTH_W9_V1`) **cannot be run for
weeks 1-8.** Three independent confirmations:

1. `data/market_track_record/alt_props/` and `alt_edges/` contain **week 9 only**.
2. Weeks 6-8 `raw_props/` carry exactly six standard main-line markets
   (`player_pass_yds`, `player_reception_yds`, `player_receptions`,
   `player_rush_yds`, `player_rush_reception_yds`, `player_anytime_td`) — one
   `line` per player/market/book. No `_alternate` markets. The boards' "2 distinct
   lines per player-market" is two books quoting the same main line, not a ladder.
3. `full-slate.yml` carries `fetch_alt_lines` as a **separate, costlier input**
   ("~5 MORE MARKETS x EVERY EVENT OF CREDIT; requires fetch_live_odds"), first
   exercised in week 9.

Weeks 3-5 have no market data at all — the documented permanent loss in
`AGENTS.md` §5a. **Nothing is recoverable here and no paid refetch is proposed.**

What *is* testable on prior weeks is the mechanism the ladder analysis uncovered:
that a roughly constant calibration error costs progressively more ROI as the
price lengthens. That needs prices and outcomes, not ladders.

## 1. Two traps that corrupt this data, both found by sanity-checking

**Trap 1 — unpriced rows.** The board stores an OVER and an UNDER row for every
offer, but `anytime_td` has **no UNDER price on 100% of rows** — books do not
offer "will not score." Grading them anyway produces a hit rate of **81.4% and an
ROI of -100% simultaneously**, because pandas averages only the priced losers.
1,748 rows dropped. The grading script now asserts no unpriced row survives.

**Trap 2 — the balanced board is not a result.** All 2,142 two-sided offers have
**exactly one winning side, 2,142 of 2,142.** So every yardage/reception market
grades at exactly **50.0%** and its band ROI is **the book's hold, mechanically** —
nothing to do with PRISM. An earlier pass of this analysis reported those band
ROIs as a replication of the amplification law. That reading was wrong and is
withdrawn here. Every PRISM claim below is computed on **one bet per offer: the
side PRISM itself rates above the price.**

**Trap 3 — the zero-involvement denominator.** `load_player_stats` omits a
player-game with no recorded stat, so a player who played and caught nothing is
missing rather than zero. 450 rows were restored as true zeros via
`load_snap_counts` (`offense_snaps > 0`); 462 with no snaps were dropped as
unsettleable. Treating "missing" as "no bet" would have deleted real losses.

## 2. THE HEADLINE: PRISM claims 64% and delivers 52%, every week

Weeks 6-8, main lines, PRISM's own picked side, **one bet per offer, n=2,142, 42 games**:

| week | n | PRISM claims | delivers | **overstatement** | ROI |
|---|---|---|---|---|---|
| 6 | 662 | 63.5% | 52.1% | **+11.4 pts** | -1.7% |
| 7 | 771 | 64.3% | 53.4% | **+10.9 pts** | +1.0% |
| 8 | 709 | 63.4% | 49.4% | **+14.0 pts** | -6.9% |
| **all** | **2142** | **63.8%** | **51.7%** | **+12.1 pts** | **-2.5%** |

**A +11 to +14 point overstatement, stable across three independent weeks.** This
is the quantified form of the `model_sd` / dispersion defect the handoff has
carried as "open" for several checkpoints. It is no longer a qualitative concern.

Per price band, the overstatement is **flat** — it is not a longshot problem:

| band | n | claims | delivers | err | breakeven | edge | ROI | amp |
|---|---|---|---|---|---|---|---|---|
| -200..-150 | 110 | 68.5% | 59.1% | +9.4 | 62.5% | -3.5 | -5.6% | 1.63x |
| -150..-130 | 137 | 66.1% | 56.9% | +9.1 | 58.0% | -1.1 | -1.8% | 1.69x |
| -130..-110 | 1416 | 65.2% | 52.5% | **+12.7** | 53.2% | -0.8 | -1.5% | 1.88x |
| -110..+100 | 181 | 60.7% | 51.4% | +9.4 | 51.5% | -0.1 | -0.1% | — |
| +100..+130 | 219 | 56.7% | 42.9% | **+13.8** | 46.8% | -3.9 | -8.3% | 2.13x |
| +130..+150 | 43 | 48.5% | 34.9% | +13.6 | 42.1% | -7.2 | -16.8% | 2.34x |

## 3. How much skill is actually there: +3.5 points against a 5.9-point vig

| strategy | n | ROI | 95% CI |
|---|---|---|---|
| random side (exact coin-flip EV) | 2142 | **-5.94%** | [-6.24%, -5.63%] |
| **PRISM-picked side** | 2142 | **-2.46%** | [-8.13%, +3.27%] |
| hindsight-perfect side | 2142 | +88.13% | [+87.52%, +88.74%] |

**PRISM's edge over a coin flip: +3.48 points, 95% CI [-2.16, +9.16], P(>0) = 88.6%.**
The vig it must beat is **5.94 points**. So PRISM's side selection recovers about
**59% of the hold** — real skill, measurable, and **not enough to break even.**

This is the cleanest statement of where the project actually stands. PRISM is not
noise: it beats a coin flip by three and a half points, and the direction of its
disagreement with the market is genuinely informative (§4). It is also not a
winning model: it covers a bit over half the vig.

## 4. The direction is informative; the magnitude is inflated 6x

Weeks 6-8, yardage/receptions, bucketed by PRISM's signed disagreement with the
market (`fair_prob - market_prob`):

| disagreement | n | PRISM says | market says | **actual** | ROI |
|---|---|---|---|---|---|
| -30.2 pts | 429 | 20.3% | 50.5% | 46.6% | **-13.1%** |
| -16.6 | 642 | 33.7% | 50.3% | 47.4% | -11.0% |
| -5.8 | 1071 | 44.0% | 49.8% | 49.2% | -7.5% |
| +5.8 | 1071 | 56.0% | 50.2% | 50.8% | -5.2% |
| +16.6 | 642 | 66.3% | 49.7% | 52.6% | +0.3% |
| +30.2 | 429 | 79.7% | 49.5% | **53.4%** | **+1.6%** |

**ROI rises monotonically across all six buckets**, -13.1% to +1.6%. The sign of
PRISM's disagreement carries real information.

**But read the magnitudes.** When PRISM says **79.7%** and the market says 49.5%,
the truth is **53.4%**. The market was far closer than PRISM. Regressing outcome on
PRISM's deviation:

- against a 50% anchor: **slope 0.199** — the spread must be shrunk to **20%** of its size, i.e. divided by **5.0x**
- against the market as anchor: **slope 0.159** — a claimed 30-point edge is really **4.8 points**
- `corr(fair_prob, market_prob) = 0.166`, mean absolute disagreement **13.9 pts**

So PRISM disagrees with the market constantly and violently, is right about the
*direction* often enough to beat a coin flip, and is wrong about the *size* by
roughly a factor of six.

**Statistically, the supported half is the fade.** Bottom disagreement decile:
**-12.9%, P(>0) = 2.9%.** Top decile: **+1.4%, P(>0) = 57.8%.** PRISM's
demonstrated skill is identifying what *not* to bet. That is consistent with every
prior result: the nine-gate builder works by exclusion (blind -5.3% → gated
+4.4%), the sign rule works by exclusion, and the table audit's surviving verdict
was a list of exclusions.

## 5. On bettable markets, the long end cannot be tested here at all

**Withdrawn:** the `anytime_td` amplification table that stood here, and the
"-33.5% on 825 bets" longshot result. TD props are not a market we play, and
extrapolating their pricing behaviour onto alt yardage rungs was a cross-market
inference this document should not have made.

What the bettable markets (`pass_yards`, `rec_yards`, `receptions`,
`rush_yards`, `rush_rec_yards`) support, on PRISM's picked side, n=2,142:

| band | n | edge | ROI | amp |
|---|---|---|---|---|
| -200..-150 | 110 | -3.5 | -5.6% | 1.63x |
| -150..-130 | 137 | -1.1 | -1.8% | 1.69x |
| -130..-110 | 1416 | -0.8 | -1.5% | 1.88x |
| -110..+100 | 181 | -0.1 | -0.1% | — |
| +100..+130 | 219 | -3.9 | -8.3% | 2.13x |
| +130..+150 | 43 | -7.2 | -16.8% | 2.34x |

Amplification runs **1.63x -> 2.34x and is monotone**, which is consistent with the
law — but over a price range of only -200 to +150, with the top band thin at
n=43. Bands at `<=-200` (n=15) and `>+150` (n=21) are too thin to read.

**The available price range is the binding limit.** Main lines in weeks 6-8 span
**-270 to +299**: 21 bets above +150, **zero above +300**. The long end simply does
not exist on a main-line board.

So **week 9's alt ladder remains the only wide-range measurement of
amplification on a product we actually bet** (1.27x -> 14.34x, `PRISM_ALT_LADDER_DEPTH_W9_V1` §3).
The law is corroborated over a narrow range here and is **not** independently
confirmed. It needs week 10 alt ladders to be tested properly.

Likewise, **week 9's "+150 cliff" remains untestable on prior weeks**, and the
week-9 alt board is still the only evidence for the longshot collapse on a
bettable product (>+300 = -46.5%, >1.6x depth = -53.5%).

## 6. Weeks 1-2, kept separate

805 rows, already archived graded. These hold **only the model's selected side**
(626 UNDER vs 179 OVER), so they are a different population from a full two-sided
board and are **not pooled** above.

Stored `unit_result` ROI: **-5.3%**. By band: <=-150 -2.2% (n=37), -150..-110
-6.3% (n=569), -110..+100 -3.2% (n=79), +100..+150 -4.0% (n=113).

Consistent with weeks 6-8's picked-side -2.5% and week 9's near-even-money -2% to -7%.

## 7. Where the project stands, across every week with data

| week(s) | product | n | result |
|---|---|---|---|
| 1-2 | main lines, model-selected | 805 | -5.3% |
| 6-8 | main lines, PRISM-picked side | 2142 | **-2.5%** (vs -5.9% coin flip) |
| 9 | alt ladders, whole board | 5115 | -15.7% |
| 9 | alt, nine-gate survivors | 76 | +4.4% (CI straddles 0) |

**No positive edge has been established at any price, in any market, in any week,
at statistical significance.** Every significant result is a loss. The two
non-negative pockets — week-9 gated alt legs (+4.4%, P(>0) not established) and
`anytime_td` +150..+300 (+4.7%, P(>0) = 67.6%) — are single-week hypotheses.

What *is* established: PRISM beats a coin flip by 3.5 points, its disagreement
direction is informative, its magnitude is inflated ~6x, and the longshot end is a
graveyard at every price above +300.

## 8. What follows

1. **Shrink `fair_prob` toward the market before any selection.** The factor is
   measured: 0.159-0.199. This is a production change with a number attached, and
   it is the first one these analyses have actually earned. **It must be fit on
   weeks 6-8 and validated on week 10, not fit and evaluated on the same rows** —
   the +10.5% that market-anchored shrinkage at k=0.10 returned on weeks 6-8 is
   in-sample and is not a result.
2. **Stop spending research effort above +300.** This now rests on week 9's 5,115
   alt rungs alone (>+300 = -46.5%), not on TD props. It is a one-week result on
   the only week that has the product.
3. **The HIGH/LOTTERY tier question rests on week 9 alone** after the TD-prop
   withdrawal, not on two independent products. Those tiers lost 25-46% there.
   Week 10 alt ladders would make it two.
4. **Build selection around exclusion**, which is where the demonstrated skill is,
   rather than around finding plus-money winners.
5. Week 10 is the first genuine out-of-sample test of the shrinkage factor, the
   nine gates, and the `rec_yards` near-even-money hypothesis.
