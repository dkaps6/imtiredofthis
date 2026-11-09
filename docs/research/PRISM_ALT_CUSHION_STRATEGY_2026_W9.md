# The cushion rule: stack rungs well below the projection — 2026 week 9

**Date:** 2026-11-08 · **Rows:** `data/research/alt_cushion_pool_2026_wk09_v1.csv`,
`data/research/alt_graded_2026_wk09_v1.csv`
**Status:** strongest result measured so far. **One slate. Not promoted.** Re-test on
week 10 before any production change.

Define **cushion** = `model_proj / alt_threshold`: how far the football projection sits
above the rung being bet. The proposal under test: take rungs comfortably below the
projection at modest favourite prices, and stack them.

## Singles, by cushion (4,306 graded alt OVER rungs, 13 final games)

| cushion | n | hit% | breakeven | P&L | ROI |
|---|---|---|---|---|---|
| <1.0x (rung above projection) | 2401 | 27.3% | 33.3% | -590.3u | -24.6% |
| 1.0-1.25x | 649 | 59.5% | 61.7% | -18.7u | -2.9% |
| 1.25-1.5x | 448 | 67.4% | 71.1% | -16.9u | -3.8% |
| **1.5-2.0x** | 470 | **80.2%** | 76.3% | **+33.0u** | **+7.0%** |
| **2.0-3.0x** | 260 | **83.8%** | 80.2% | **+14.8u** | **+5.7%** |
| >3.0x | 78 | 82.1% | 79.3% | +3.2u | +4.1% |

Restricted to -150..-400 prices:

| cushion | n | hit% | breakeven | ROI |
|---|---|---|---|---|
| 1.0-1.5x | 513 | 64.7% | 69.7% | -6.5% |
| **1.5-2.0x** | 177 | 81.4% | 71.9% | **+13.5%** |
| **2.0-3.0x** | 111 | 78.4% | 71.5% | **+10.8%** |

By market at cushion >=1.5 and price <=-150 (n=720, +4.2% overall):
`rush_rec_yards` +11.2%, `pass_yards` +10.1%, `rush_yards` +8.5%,
`rec_yards` +3.6%, `receptions` -0.6%.

## Parlays — the part that matters

Pool: cushion 1.5-3.0x, price -150..-400, best rung per player/market, one leg per
game. 129 legs, 100 players, 11 games. Singles: 80.6% hit vs 70.2% breakeven,
**+15.6% ROI**. Random samples of 3,000 parlays per size:

| legs | win% | avg odds | ROI |
|---|---|---|---|
| 2 | 65.0% | +104 | **+31.8%** |
| 3 | 52.7% | +193 | **+53.4%** |
| 4 | 43.9% | +317 | **+80.7%** |
| 5 | 35.2% | +500 | **+109.1%** |
| 6 | 29.0% | +755 | **+147.1%** |

The same test on the **shipped** triple-intersection pool (53 legs, singles
**+19.8%** ROI): 2-leg **-19.0%**, 3-leg **-31.7%**, 4-leg **-80.0%**, at average
2-leg odds of **+1780** against the cushion pool's **+104**.

## Why the shipped board failed at parlays despite positive singles

Parlay win probability is the product of leg hit rates. Legs that are +EV at long
prices hit near 50% or less; 0.5^3 = 12% versus 0.81^3 = 53%. Equal single-bet
ROI, opposite parlay behaviour. The shipped tiers optimised per-leg edge and
ignored hit rate, so every tier above MODERATE was built from rungs that cannot
survive being multiplied together. The HIGH and LOTTERY tiers were additionally
drawn from frac > 1.5 depth bands that return -42% and -59% blind.

## Why the edge exists mechanically

Books price an alt ladder as a deterministic transform of the main line. When the
football projection diverges upward from that line, **every rung below it is
mispriced simultaneously**, and the deepest rungs carry the smallest absolute
price error but the highest hit probability. The edge is in the book's mechanical
laddering, not in model brilliance.

This also reframes the ceiling-compression defect
(`PRISM_ALT_CEILING_AND_COVERAGE_2026_W9.md`): compression *shrinks* cushions on
top-usage players rather than eliminating them, pushing them from >3.0x down into
1.5-2.0x, which is still the most profitable band. The players it genuinely kills
are mid-tier, pushed from 1.5x down to 1.2x where returns turn negative. Example:
Bijan Robinson, main 92.5 rushing, PRISM 73.7, actual 100 — a 45-yard rung is a
1.64x cushion on the compressed projection and 2.2x on truth; profitable either
way.

## Caveats, stated plainly

- **One slate, 13 games, 11 with cushion legs.** The parlay table is a resampling
  of a single realized outcome, not an expectation. The magnitudes will not repeat.
- Legs within the pool share slate-level passing environment; one-leg-per-game
  removes same-game correlation only.
- Week 9 was a high-passing outlier — the market line itself ran low on top-tier
  `rec_yards` — which inflates every OVER result including this one.
- `model_proj` enters only as an ordinal comparison against the rung. That is
  precisely why it works: it avoids the miscalibrated `model_sd` width and the
  compressed tail, both documented defects. Do not convert cushion to a
  probability through the current simulated distribution.

## Proposed selection rule for week 10 (test, not promotion)

1. Require `cushion >= 1.5` and price between -150 and -400.
2. Prefer `rush_rec_yards`, `pass_yards`, `rush_yards`; `rec_yards` acceptable;
   exclude `receptions` from the cushion rule (-0.6%).
3. Cap `frac_of_line <= 1.5`.
4. One leg per game, 3-5 legs per ticket.
5. Enforce the injury gate and an alt coverage manifest before selecting.
6. Grade against archived prices and compare directly to the triple-intersection
   on the same slate.

---

## ADDENDUM 2026-11-09 — ex-ante test, and the zero-floor failure mode

The parlay table above samples **winners**. An ex-ante test was run: rank the
strict pool before kickoff with no outcome knowledge, take the top N, and count.

Strict pool, one rung per player: 98 legs, 11 games, **78.6% hit**.

| ranking (pre-kickoff) | 11 legs | 15 | 22 | 30 |
|---|---|---|---|---|
| biggest cushion | 5/11 | 9/15 | 16/22 | 24/30 |
| shortest price | 9/11 | 10/15 | 17/22 | 23/30 |
| own record (shrunk) | 9/11 | 12/15 | **19/22** | 25/30 |
| lowest frac_of_line | 6/11 | 9/15 | 16/22 | 22/30 |

**Every ranking loses every parlay size.** At a 78.6% leg rate, 22 consecutive
is 0.786^22 = 0.5% and 12 consecutive is 5.3%. The "+1,818,000 22-leg ticket"
was selected with hindsight and was not reachable.

### Ranking by cushion alone selects zero-floor players

Five of the six misses in the top-22 by cushion recorded a **literal zero**:
Saquon Barkley (rec yds >4.5, cushion 2.98x, actual 0), Omarion Hampton (2.83x,
0), Kyle Williams (2.78x, 0), Laquon Treadwell (2.76x, 0), Chris Bell (2.70x,
0). Cushion is largest precisely when the rung is small relative to a modest
projection — a low-volume receiver whose floor is zero. This is the 11.4%
`rec_yards` zero rate (20% in the low-projection tercile) landing on the legs
cushion ranks highest.

### A floor screen helps but does not fix it

| rule | pool | hit% | ROI | top-12 |
|---|---|---|---|---|
| cushion>=1.5 | 87 | 72.4% | +0.6% | 8/12 |
| + own_rate>=0.60 | 64 | 76.6% | +3.8% | 7/12 |
| + own_rate>=0.75 | 39 | 76.9% | +3.7% | 9/12 |
| + own_rate>=0.85 | 28 | 75.0% | -0.3% | 9/12 |
| + own_rate=1.00 | 9 | 66.7% | -10.9% | 6/9 |

Even players with perfect 2026 records missed: Quinshon Judkins (7/7, needed
4.5 rec yds, got 2), Devaughn Vele (7/7, needed 19.5, got 17), Tre Harris (6/6,
needed 14.5, got 6). No ex-ante screen reached 12/12.

### Correct conclusion

The per-leg edge is real (+11.3% ROI, 79.5% hit on the strict pool). An n-leg
parlay of independent +EV legs has *higher* expected value than n singles —
1.11^n - 1 — so long parlays are not an EV mistake. They are a **realization**
mistake: 17 slates a season cannot deliver the trials a 0.5%-probability ticket
needs. Small parlays (3-5 legs, 38-48% win at this hit rate) capture most of
the multiplicative edge while cashing often enough to be realized inside a
season.

Uniformity constraints (one leg per game) are **dropped** as a selection rule —
they were a convenience for independence math, not a finding, and they cost
real options. They are retained only where a book's same-game parlay pricing
forces them, which this capture cannot measure.

**Revised week 10 test:** cushion >= 1.5, price -130..-400, frac <= 1.5, exclude
`receptions`, require own_rate >= 0.75 as a floor screen, no per-game quota,
3-5 legs per ticket, plus the injury gate and coverage manifest. Grade against
archived prices alongside the triple-intersection.
