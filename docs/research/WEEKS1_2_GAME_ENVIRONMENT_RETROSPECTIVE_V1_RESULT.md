# Weeks 1-2 2026 Retrospective — the game environment decides the board

Status: **OBSERVATIONAL / NOT PROMOTED**
Date: 2026-10-19
Population: `data/market_track_record/graded/2026_wk01_wk02_graded.csv`, 797 settled
legs (805 rows, 8 VOID), every row `production_decision=BET`

## Why this exists

Written down the same day it was found, because Weeks 3-5 are the cautionary
tale: see `AGENTS.md` §5a. Nothing here is promoted and nothing here authorizes
a production change.

## 1. A RETRACTION, recorded first

During this analysis I found `pass_yards` at 36-16 (69.2%, +15.99u) against
every other market losing, and recommended narrowing the card to QB passing
yards. **That recommendation was wrong and is withdrawn.**

The Week 3 postmortem had already tested and closed it:

- 54 eligible slices, game-cluster-aware BH-FDR q=0.10, **0 survived**
- disposition `NO_SLICE_SURVIVES_MULTIPLE_COMPARISONS_CORRECTION`
- verbatim: *"Nominal pass-yards strength does not authorize a QB/pass-yards
  production carveout"* and *"Do not: add a QB/pass-yards carveout"*

Week 4 then inverted it outright: `pass_yards` 12-12 (-1.36u) while
`rush_yards` was the best market (+4.06u) and TE went from -18.24u through
week 3 to +3.15u. The paired MAE test on the weeks 1-2 window was never
significant anyway (t = -1.46, n=52).

**Lesson for a future session: read the closed-science docs before forming a
recommendation from a fresh window, not after.**

## 2. The finding that does hold: environment, not player precision

Splitting identical bets by what the offense actually did:

| bet | quiet offense | loud offense | swing |
|---|---|---|---|
| `rec_yards` UNDER | 42-20, 67.7%, +17.27u | 19-36, 34.5%, -18.91u | **33 pts** |
| `receptions` UNDER | 39-21, 65.0%, +13.76u | 18-45, 28.6%, -26.73u | **36 pts** |
| `rush_yards` UNDER | 13-13, 50.0% | 18-11, 62.1%, +5.12u | 12 pts |

Nothing about the bet changes. Terciles are on realized team passing yards
reconstructed from priced rows (a proxy, not a box score).

The one mechanically coherent winner is the **pass/run tradeoff**: when a team
threw, its back did not run. Montgomery U65.5 → 10 (HOU threw 353), Gibbs
U87.5 → 52 (DET 327), Cook U76.5 → 57 (BUF 334), Brooks U33.5 → 14 (CAR 361).
Finite snaps, finite plays.

## 3. The environment is not currently knowable pregame

Splitting the same cells by the market's **pregame** estimate of team volume
(the QB's pass-yards line) does not reproduce the separation:

| bet | pre-low | pre-mid | pre-high |
|---|---|---|---|
| `rec_yards` UNDER | 49.1% | 47.0% | 53.8% |
| `receptions` UNDER | 54.4% | 36.4% | 48.6% |

Flat, non-monotonic, and backwards for `rec_yards`. Our own model's
team-volume disagreement does not predict the realized deviation either:
`corr(our_proj - line, actual - line) = +0.200`, n=52, t=+1.44.

So the standing 78%-UNDER posture was not a read on the environment. Half the
games cooperated.

**The two largest "quiet offense" wins were quarterback injuries.** wk1 MIN
threw 18 yards (Murray exited), wk2 NYG threw 20 (Dart exited). Addison 0,
Jennings 0, Nabers 1 catch. Because the board is 78% unders it is implicitly
long "the starting QB gets hurt" — a windfall, not foresight.

## 4. Cushion does not predict outcomes

The alt-line method's premise does not hold in this sample. Cushion measured
in model standard deviations, pooled:

| cushion (sd) | hit% | | cushion (sd) | hit% |
|---|---|---|---|---|
| 0.00-0.10 | 52.2% | | 0.50-0.75 | 48.1% |
| 0.10-0.25 | 44.5% | | 0.75-1.00 | 55.1% |
| 0.25-0.50 | 51.2% | | 1.00+ | 52.8% |

Flat, and within `pass_yards` the two halves are **18-8 and 18-8, identical**.
Already stated independently in the Week 5 handoff: *"alt-line distance is not
calibrated survival probability."*

Buying down to an alt still raises hit probability arithmetically. What has no
support is using cushion to **choose** candidates — which is exactly what the
Week 6 blind card did.

## 5. Where the units came from

Winners' clearing margin: median 9.5 yds, mean 17.2. **37.2% cleared by 15+**;
only 17.5% squeaked by <=1. The money came from the line being badly wrong,
not from precision.

The 78% UNDER posture netted exactly nothing: **310-310, 50.0%, -29.99u**.

## 6. Dispersion test — a hypothesis of mine that FAILED

I expected within-game legs to behave as one correlated bet. They do not:
game hit rates ranged 26.9%-72.7% with stdev 10.1pp, against ~10.0pp expected
for independent legs at p=0.496 — ratio **1.01x**. The environment correlation
is real but modest (r = -0.437, n=22 games, p~0.04), and at this repo's
evidential standard (0-of-50 then 0-of-54 under FDR) a single p=0.04 at n=22
is not actionable.

Recorded because a failed hypothesis of mine is evidence too.

## 7. Two `player_form` role defects found

Both reached a published card before being caught, both still unfixed upstream.

**`depth_role` is an alignment slot, not a rank.** Ourlads WR1/WR2/WR3 name
LWR/SWR/RWR, and `depth_index` separates starter from backup within a slot.
TEN: Calvin Ridley reads `depth_role=WR1` while being `model_role=WR4` on an
11.7% target share (second body at LWR); the room's actual WR1 is Carnell Tate,
who reads `depth_role=WR3` because he lines up at RWR. NO: Kamara is
`model_role=RB1` at a 39.0% rush share carrying `depth_role=RB2`.

**`model_role` ranks quarterbacks by descending ypa.** 6 of 37 QB rows carry
the swap. TEN reads Trubisky QB1 (ypa 8.94, 4 games) over Cam Ward (5.82, 22
games); CHI reads Bagent (9.45, 5 games) over Caleb Williams (7.09, 19). JAX,
KC and DEN hide it only because their ypa order happens to match depth order.

So the two fields fail in opposite directions by position. Rank must be read
per position: `model_role` for RB/WR/TE, `depth_role` for QB.

## 8. What the next session should test

The payoff is already sitting there: a 33-point swing on a bet we already
make. The target is **not** better player MAE — it is whether a team's
realized passing volume can be classified pregame, even coarsely.

The obvious untested signal is the **game total and implied team total**.
It could not be tested here because `odds_game.csv` was expiring with the
artifact; PR #692 fixes that, and `data/market_track_record/game_odds/`
now exists from Week 6 forward.

One cell separated pregame: `rush_yards` UNDER where the market expected low
passing, 18-8, 69.2%, +7.99u. n=26 across ~27 cells examined. Preregister it
for Week 6; do not bet it.
