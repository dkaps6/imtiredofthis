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

---

## ADDENDUM 2026-11-09 — snap share, and the Roschon Johnson failure

A shipped SNF leg (Roschon Johnson, CHI, alt receptions >=2 at +294) was faded
by the user on the grounds that he is the third-string back. He is:

| CHI back, weeks 1-8 | snap share | carries/g | targets/g |
|---|---|---|---|
| D'Andre Swift | 63.5% | 17.9 | 3.5 |
| Kyle Monangai | 40.5% | 15.0 | 1.2 |
| **Roschon Johnson** | **15.6%** | **4.8** | 1.8 |

~12 offensive snaps per game, and Monangai's emergence at 15 carries/game had
compressed him further. Every layer of the shipped screen cleared him anyway:

- `own_rate` 4/6 — the denominator counts any game with a recorded stat, so a
  12-snap cameo with one catch for 5 yards registers as a hit. The screen
  measured "did he clear a trivial number", not "is he involved".
- `model_proj` 12.3 rec yards — PRISM over-projected a 15.6%-snap back. An
  opportunity-allocation defect, not a usable read.
- cushion 4.9x — large *because* the rung was trivial, i.e. because the book
  had correctly priced him as irrelevant.

`load_snap_counts` was used for the games-played denominator all session and
never used as a screen. The data was in hand.

### Snap share splits the pool harder than any other variable found

Refined pool (n=258, baseline 87.6% hit / +5.5% ROI), snap share = mean
`offense_pct` weeks 1-8, strictly prior:

| snap share | n | hit% | breakeven | ROI |
|---|---|---|---|---|
| **25-40%** | 11 | **27.3%** | 69.8% | **-66.2%** |
| 40-55% | 18 | 100.0% | 82.9% | +22.1% |
| 55-70% | 77 | 89.6% | 79.9% | +12.0% |
| 70-100% | 150 | 89.3% | 85.0% | +5.2% |

Below 40% snaps the sign flips entirely. n=11 in that band, so the magnitude is
unreliable; the direction is not, and it has an obvious mechanism.

### A 30% snap floor is the only screen all session that cost nothing

| rule | n | hit% | ROI | ex-ante top-15 | odds |
|---|---|---|---|---|---|
| role + cushion + softD | 45 | 91.1% | +7.2% | 15/15 | +1,659 |
| **+ snap >= 30%** | 43 | **93.0%** | **+9.0%** | **15/15** | **+1,653** |
| + snap >= 40% | 42 | 92.9% | +8.2% | 15/15 | +1,245 |
| + snap >= 50% | 40 | 92.5% | +7.2% | 15/15 | +974 |
| + snap >= 60% | 36 | 91.7% | +6.4% | 15/15 | +876 |

+1.9pp hit and +1.8pp ROI for 6 basis points of payout. Every other screen
added tonight cost real odds; this one is free. Roschon at 15.6% is excluded by
any floor at 20%+.

### The deeper point

The role floor (addendum 2 of the cushion doc) and the snap floor encode the
same football fact — **is this player actually on the field** — and the shipped
SNF card had neither, because `receptions` carried a 0.5 main line that no
absolute floor was applied to. `data/roles_ourlads.csv` is a canonical
production artifact described in AGENTS.md section 5 as "current roster/depth
roles" and was never joined at any point in this work. It is not present in a
local checkout (built in CI), but snap counts give the same signal and were
already loaded.

### Rule update

Add to the week 10 test rule, between the role floor and the cushion test:

- **`snap_pct >= 0.30`** (mean `offense_pct`, weeks strictly prior, from
  `load_snap_counts`), and
- recompute `own_rate` over games where the player actually played rather than
  games where he recorded a stat — the current denominator rewards cameos.

Join `data/roles_ourlads.csv` when running in CI, and treat an explicit
third-string depth label as disqualifying regardless of snap share.
