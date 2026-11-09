# Football context on top of the market rule — 2026 week 9

**Date:** 2026-11-09 · **Rows:** `data/research/alt_pool_with_football_context_2026_wk09_v1.csv`
**Status:** research. One slate. Not promoted.

Challenge raised: every refinement in `PRISM_ALT_CUSHION_STRATEGY_2026_W9.md` is
market structure — cushion ratios, price bands, `frac_of_line`, main-line floors.
None of it is football. AGENTS.md orders the reasoning player identity → current
role → **opportunity allocation** → **opponent/environment** → market outcome →
reliability. The cushion work is step 5 with a crude proxy for step 2.

## Opponent strength matters more than any market refinement found so far

Defensive allowance computed from nflverse weekly stats, **strictly prior**
(weeks 1-8 only), as mean yards per game conceded to the relevant position
group, matched per leg: RB rushing for `rush_yards`; WR / TE / RB receiving for
`rec_yards` and `rush_rec_yards` by the player's modal position.

Refined pool (role floor + cushion >= 1.5 + own_rate >= 0.75, no receptions),
n=222 with a known matchup, baseline 87.6% hit / +5.5% ROI:

| opponent tier | n | hit% | breakeven | ROI | yds allowed/g |
|---|---|---|---|---|---|
| toughest D | 59 | 79.7% | 82.6% | **-1.8%** | 42.1 |
| tough | 55 | 87.3% | 80.4% | +7.4% | 80.0 |
| soft | 59 | 81.4% | 80.6% | -2.3% | 125.7 |
| **softest D** | 49 | **95.9%** | 82.4% | **+16.9%** | 157.8 |

A 16-point hit-rate and 19-point ROI spread across legs the market rule treated
as equivalent. **Against the toughest quartile the edge disappears entirely.**
The middle two tiers invert (tough +7.4% vs soft -2.3%), which is noise at
n≈55; the extremes are large and directionally sane.

Offensive pace (team plays/game, strictly prior) is nearly inert: slow 84.5% /
+5.6%, mid 87.4% / +5.3%, fast 89.1% / +4.5%. Opponent is the lever, not tempo.

## Integrated rule, tested ex ante

| rule | n | hit% | ROI | ex-ante top-15 | odds |
|---|---|---|---|---|---|
| cushion + role floor only | 52 | 88.5% | +4.4% | 15/15 | +1,564 |
| **+ opponent >= median allowance** | 32 | **93.8%** | **+10.2%** | **15/15** | +1,042 |
| + opponent top-third allowance | 23 | 91.3% | +8.5% | 14/15 | — |
| **+ exclude toughest quartile only** | 45 | **91.1%** | **+7.2%** | **15/15** | **+1,659** |

Excluding only the toughest quartile is the best operating point: it keeps 45
legs, lifts hit rate 2.6pp and ROI 2.8pp over the market-only rule, and pays
*more* than the median screen because it discards fewer legs.

Matchup does **not** buy odds. Price-ranking inside the soft-D pool goes 12/15
and 13/15 — it breaks the ticket exactly as price-ranking did without the
screen. Football context raises reliability, not payout.

The integrated ex-ante top 15 went 15/15 at +1,042 and reads as a football
card: Watson rec >29.5 vs NE (152.1 allowed, actual 171), Javonte Williams rush
>29.5 vs IND (actual 101), Olave rec >29.5 vs CLE (actual 98), Johnston rec
>14.5 vs HOU (187.4 allowed, actual 58), McMillan, Collins, Tucker, JSN,
Sutton, Fields, Jameson Williams, Hutchinson, Keenan Allen, Michael Wilson,
Dobbins.

## What is still not in the model — explicitly

The +10.2% above came from the crudest possible matchup statistic. These are
unaddressed and each is plausibly worth more than any cushion refinement:

1. **Opportunity allocation** (AGENTS.md step 3). No teammate competition, no
   target-share concentration. A projection inflated by a teammate's absence is
   not the same evidence as earned volume.
2. **Game environment.** No implied team total, spread or game-script read. A
   back in a projected blowout has a different carry distribution than one
   trailing.
3. **The player-state anomaly gate** (AGENTS.md, non-negotiable). Recent games
   were never labeled `HEALTHY_FULL` / `INJURY_LIMITED` / `PARTIAL_GAME` /
   `DNP_INACTIVE`. The `own_rate` screen treats every game as clean evidence,
   which the contract forbids.
4. **Coverage matchup.** The repo has a WR-CB builder; this used team-level
   yards allowed, not who covers whom.
5. **Opponent adjustment of the defensive metric.** Raw yards conceded per game
   is not schedule-adjusted, so a defense that faced three weak offenses reads
   as elite for no reason. This is the most likely source of noise in the
   non-monotonic middle tiers.
6. **Weather.** Not touched.

## Next pass, in priority order

1. Schedule-adjust the defensive metric (expected-yards-allowed vs actual).
2. Add the player-state labels and recompute `own_rate` on clean games only.
3. Add opportunity allocation: team target/carry share and teammate availability.
4. Add implied team total and spread as environment (decision-side only —
   sportsbook variables must not construct the football projection, AGENTS.md
   section 3).
5. Re-test the whole stack on week 10 against the market-only rule on identical
   rows.
