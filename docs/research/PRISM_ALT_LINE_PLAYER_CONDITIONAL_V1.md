# Alternate-line edges, asked about the named player

**Date:** 2026-11-06 · **Slate:** 2026 Week 9 · **Event:** JAX @ BAL (TNF, already played)
**Status:** RESEARCH FINDING. Nothing promoted. No production behaviour changed.
**Rows:** `data/research/alt_line_player_conditional_2026_wk09_tnf_v1.csv`
**Script:** `scripts/research/audit_alt_line_player_conditional_v1.py`
**Source boundary:** `data/market_track_record/alt_props/2026_wk09_full_edges.csv` (paid,
captured 2026-11-05T20:03Z, archived in git).

## CORRECTION 2026-11-08 — the calibration claim below is RETRACTED

**What this document originally concluded, and what is wrong with it.**

The grading section below reports 2026 Week 9 TNF at a realized 20.2% against a
table prediction of 34.1%, with the 0-10%, 10-20% and 20-30% bands all at 0.0%.
That was written up as evidence the table over-predicts and that the mid
probability band is unusable. **It is not evidence of that.**

Every alternate rung is an OVER, so a low-volume game depresses all of them
together. TNF produced **27 total receptions, a maximum of 4 by any player, and
only 14 of 72 players caught a pass.** One game is one draw of the game
environment, not 84 independent cells. The shortfall was confounded with the
environment and I treated it as a property of the table.

**The out-of-sample test that should have been run first:**
`scripts/research/backtest_alt_clearance_table_v1.py`, rows in
`data/research/alt_clearance_backtest_2026_v1.csv`. The table was fit on
2024+2025; 2026 is a genuine holdout. Archived paid boards for weeks 1, 2, 6, 7
and 8 supply the main-line anchors, nflverse weekly stats supply the outcomes.

**13,990 cells. The table is well calibrated in every band.**

| predicted band | n | predicted | actual | deviation |
|---|---|---|---|---|
| 0-10%   |   159 | 4.8%  | 7.5%  | +2.8pp |
| 10-20%  |   452 | 14.3% | 18.1% | +3.9pp |
| 20-30%  | 1,036 | 26.8% | 26.6% | -0.2pp |
| 30-50%  | 3,438 | 42.9% | 41.2% | -1.7pp |
| 50-70%  | 4,953 | 60.2% | 57.9% | -2.2pp |
| 70-100% | 3,952 | 80.6% | 80.3% | -0.3pp |

Maximum deviation is 3.9pp and the two lowest bands come in *above* prediction.
The >=70% band is stable week to week (83.1 / 79.6 / 80.9 / 76.9 / 81.0 against
80.6 predicted), so TNF's 81.3% was never the anomaly -- the mid-band collapse
was.

By market, over the same 13,990 cells:

| market | n | predicted | actual |
|---|---|---|---|
| rec_yards      | 6,430 | 57.5% | 57.1% |
| receptions     | 2,640 | 55.5% | 55.8% |
| rush_rec_yards | 1,480 | 58.2% | 60.3% |
| rush_yards     | 3,440 | 57.1% | 52.2% |

`rush_yards` is the weakest at -4.9pp and is the one market worth discounting.
Nothing supports excluding the mid band, and nothing supports the 0.9578 or
0.5833 haircuts that were derived from TNF and briefly applied to live tickets.

**Consequences for how this file should be read:**

1. The band table in the grading section is a single game. Use the backtest.
2. `robust_all_weights` is not disqualified by probability band. A mid-band leg
    is as usable as a high-band one.
3. FINDING 2 below -- that the population table and a player's own 4-8 game
    record disagree on ~23% of rows -- still stands and is unaffected. It is a
    statement about small player samples, not about table calibration.
4. The lesson is procedural: one game is a hypothesis, not a calibration. The
    backtest cost nothing and existed to be run at any time.

---

## Why this exists

`scripts/price_alt_line_offers_v1.py` prices alternate rungs off the market-anchored
clearance table, deliberately not off the model — `pricing_v2._prob_over` is a Normal
approximation wrong by up to +0.448 in the tails, which is where every alt line lives.

But the table is a **population aggregate**: *"players whose posted receptions line is 1.5
clear >=1 about 89% of the time."* It does not know which player. That collides head-on with
the AGENTS.md player-projection first principle, and it collides in the direction that costs
money — **the table finds its biggest edges on players the book priced LOW**, which is exactly
the sub-population a league-wide average over-states.

This audit asks each priced row the player-level question, and reports expected value across
the whole shrinkage range rather than at one chosen weight.

## FINDING 1 — the denominator trap, which inverted the top-ranked bet

`nflreadpy.load_player_stats()` **does not emit a row for a player-game with no receiving
involvement.** Counting its rows as games played divides by *games where he caught something*.
The inflation is largest for low-volume players — precisely the population the table is applied
to.

Measured on this event: **Matthew Hibner has 5 `player_stats` rows and 7 games with offensive
snaps** (0.34 / 0.33 / 0.50 / 0.58 / 0.39 / 0.39 / 0.41 share, so a stable role, not a
demotion — weeks 7-8 were target variance and the state is `HEALTHY_FULL`).

| | rate | vs break-even 0.528 at -112 | verdict |
|---|---|---|---|
| inflated `player_stats` denominator | 3/5 = **0.600** | +0.072 | ranked **1st** on the board, +0.277 conservative EV |
| true snap-count denominator | 3/7 = **0.429** | **-0.099** | **negative on his own record** |

The single best-looking row on the board was an artifact of the denominator. Games played must
come from `load_snap_counts()` with `offense_snaps > 0`, receiving stats left-joined, missing
filled with zero. The script does this and the docstring says why.

## FINDING 2 — the population and the player disagree on a quarter of the rows

Of **142** comparable receptions rows, **33 (23%)** are positive-EV on the population table and
**not** positive on the player's own record. Examples:

| row | price | needs | table says | he has done it |
|---|---|---|---|---|
| Hibner >=1 | -112 | 52.8% | 82.0% | 3/7 = 42.9% |
| Bateman >=1 | -350 | 77.8% | 89.0% | 6/8 = 75.0% |
| Cameron >=3 | +482 | 17.2% | 28.7% | 1/7 = 14.3% |
| Henry >=2 | +128 | 43.9% | 53.2% | 3/8 = 37.5% |

**21 of 142** are positive at *every* shrinkage weight from player-only to population-only.
Those are the only ones this audit would defend, and `robust_all_weights` marks them.

## FINDING 3 — the edge, where it exists, is confined to one market

Best conservative EV by market on this event:

| market | best `ev_conservative` |
|---|---|
| `receptions` | **+0.277** |
| `rec_yards` | -0.045 |
| `rush_yards` | -0.074 |
| `pass_yards` | -0.087 |
| `rush_rec_yards` | -0.148 |

Every yards ladder is efficiently priced against our table. Only the integer market, which uses
the absolute-threshold table rather than a fraction-of-line table, shows anything. That is also
the only market where a per-game player rate is directly comparable, which is why this audit
restricts itself to it.

## FINDING 4 — alternate ladders are OVER-only

All 374 priceable rows on this event are `side == OVER`. The provider returns alternate markets
as a ladder of over-rungs, so **there is no under to take**. Any strategy built on alt lines is
structurally long.

## WHAT THIS DOES NOT ESTABLISH

- **No multiplicity control.** 21 robust rows out of 142 were selected using the same small
  samples that compute their EV. Some of that is selection, and this audit does not bound how
  much. A holdout or a cross-slate replication is required before any of it is a strategy.
- **Player samples are 4-8 games.** The shrinkage sensitivity is a robustness check, not a
  substitute for evidence.
- **The 0.5-line cell is thin**: n=122 player-games, 77 players, versus n=1171/248 for the
  1.5-line cell. Hibner's rows rest on the thin one.
- `post_hoc_line_restriction` is `True` on every cell used. The `L <= 4.5` cap was separately
  validated out of sample, but the table's structure was still chosen after looking at data.
- **Nothing here was bet.** The event kicked off at 2026-11-06T01:15Z while this audit was
  being refined, which is its own lesson: an audit that lands after kickoff is a research
  artifact, not a decision.

## NEXT

1. ~~Grade this event at row level once the nflverse week-9 feed refreshes.~~
   DONE, and it produced the retracted claim above. Superseded by the 13,990-cell
   backtest. Keep grading weekly, but never again generalise from one game. 735 alt rows with
   predictions recorded pre-kickoff, including the rows this audit called robust. That is a
   direct test of the audit method, not only of the table.
2. Decide whether the clearance table should be rebuilt **player-conditionally** (role/target
   share as a covariate) rather than applied as a league aggregate. Finding 2 is the argument
   that it should.
3. Do not treat `robust_all_weights` as a bet list until (1) and the multiplicity question are
   answered.
