# PRISM — ALT-LINE DECISION TABLE, AND THE HISTORICAL DATASET I MISSED

> **DEPRECATED 2026-11-04 — do not bet from this table.**
> Its thresholds are expressed as a fraction of the **model projection**, which is compressed by
> 32-54pp across volume, so a fixed fraction means a different real threshold for every player.
> Measured cost on held-out 2025: at `frac_of_proj = 0.6` for `rush_yards` the table promises 70.4%
> and reality runs 60.9% (low volume, overpaying) to 90.4% (high volume, true break-even -943
> against the -238 quoted here). It also fails decile calibration at 8.92pp.
> **Superseded by `data/research/alt_line_market_anchored_table_v1.csv`**, documented in
> `PRISM_ALT_LINE_ANCHOR_RESULTS_V1.md`. Retained as research lineage only.


**Date:** 2026-11-04
**Artifact:** `data/research/alt_line_clearance_curve_2024_2025_v1.csv`
**Source:** `docs/research/overnight/clean_v1_full_stack_vegas_benchmark_detail.csv` —
**17,715 rows, 2024+2025, 445 players**, carrying `mc_proj`, `ml_proj`, `state_proj`,
`ensemble_proj`, `line`, `over_odds`, `under_odds`, `actual`, already joined.

---

## 1. A correction: I said this data did not exist

`PRISM_REC_YARDS_SYNTHESIS_PREREGISTRATION_V1` §4 stated:

> *"Historical component projections do not exist for receivers —
> `data/backtests/full_stack_vegas_benchmark_v1/non_qb_detail.csv` carries no `mc_proj`/`ml_proj`/
> `state_proj` — so a multi-season fit is **not available at any effort short of re-running
> history**."*

**I checked one file and concluded from it.** The `docs/research/overnight/` copy of the same
benchmark carries **all four component projections plus lines, both sides' odds and outcomes, for
two full seasons.**

**This invalidates the central limitation of that study**, which was "three weeks of one season
against M89's three full seasons." **The rec_yards synthesis can and should be refitted on
2024→2025 with 2026 prospective** — the exact M89 design. That is now the highest-value open item
and it was blocked only by my own search.

## 2. The three missing 2026 weeks — what actually happened

**W3, W4 and W5 2026 do not exist in this repository in any form.** `raw_props` and `game_odds`
hold only W6–W8; `boards` holds W1, W2, W6, W7, W8; `graded` holds W1–W2 only.

**And for Week 3 the cause is not retention — it is a mechanical failure.**
`docs/research/RB_PD2_WEEK3_PROSPECTIVE_LOCK_V1_RESULT.md` records run `36330757181` with live
odds authorized, and states: *"The canonical pricing step later failed mechanically, so this run
is **not** a replacement betting-board authority."* **The Week 3 run happened and produced no
board.** A research shadow hook captured 54 rows before the failure; that is all that survives.

**So the runs did happen.** The boards did not survive — one failed in production, and the AGENTS.md
§5a retention defect covers the rest. **Two full prior seasons are a better asset than those three
weeks would have been.**

## 3. The decision table

Clearance of a threshold expressed as a fraction of the model projection, **clustered over
players**, 2024+2025. Full table in the CSV; the operationally useful rows:

| market | frac of proj | cleared | need better than | conservative (CI low) |
|---|---|---|---|---|
| `pass_yards` | 0.70 | 85.5% | **−588** | −479 |
| `pass_yards` | 0.80 | 76.1% | **−319** | −268 |
| `pass_yards` | 1.00 | 57.6% | −136 | −117 |
| `rush_rec_yards` | 0.60 | 82.5% | **−473** | −409 |
| `receptions` | 0.60 | 82.6% | **−475** | −439 |
| `receptions` | 1.00 | 64.5% | −181 | −169 |
| `rec_yards` | 0.60 | 72.6% | **−265** | −246 |
| `rec_yards` | 1.00 | 57.3% | −134 | −125 |
| `rush_yards` | 0.60 | 70.4% | −238 | −206 |

**How to use it:** take the threshold as a fraction of the PRISM projection, read the clearance
probability, and compare the book's price to "need better than." **Use the conservative column** —
it is built from the CI lower bound, i.e. the price you need to be right rather than lucky.

**This needs no API and no fetch.** The probability comes from this table; the price is on the
screen at the book.

## 4. Validated against the actual W8 card — and it lost

Applying the table ex ante to the eight W8 legs that carried **both** a price and a PRISM
projection:

| tier | leg | frac | p(clear) | price | need | verdict | hit |
|---|---|---|---|---|---|---|---|
| SAFE | Warren 2+ rec | 0.60 | 82.6% | **−550** | −474 | **−EV** | N |
| SAFE | Taylor 40+ rush | 0.58 | 71.1% | −225 | −246 | +EV | N |
| SAFE | Shakir 30+ rec | 0.62 | 71.9% | −210 | −256 | +EV | **Y** |
| SAFE | Tucker 4+ rec | 0.99 | 65.0% | −120 | −186 | +EV | N |
| MOD | Hampton 5.5+ rec | 0.41 | 79.7% | −114 | −392 | +EV | **Y** |
| MED | M. Taylor 22.5+ | 0.75 | 66.9% | −114 | −202 | +EV | N |
| HIGH | Taylor 49.5+ rush | 0.72 | 65.6% | −122 | −191 | +EV | N |
| HIGH | Rodgers 218.5+ | 0.94 | 63.3% | −113 | −173 | +EV | N |

**Approved 7, of which 2 hit. Realized −0.5209 per $1.** **The table lost on Week 8.**

**It got one thing clearly right:** it **rejected Warren 2+ at −550**, the card's heaviest
favourite, as −EV — and it missed.

## 5. What the table is, and what it cannot do

**It converts a projection into a fair price. It cannot know when the projection is wrong.**

Week 8's projections were badly off, mostly too high: Taylor **68.61 → 23**, Tucker **4.05 → 0**,
Rodgers **232.3 → 172**, Mason Taylor **29.97 → 19**. When a projection is three times the
realized value, "40+ is 58% of projection" is really 174% of the truth, and the table's 71%
clearance is simply the wrong number.

**So the table is a pricing instrument, not a projection instrument.** On 17,712 historical rows it
is right on average, because average projection quality is baked into it. **In a week where the
projections run hot or cold it is wrong in the same direction they are.** That is not a flaw to
fix — it is the boundary of what a clearance curve can do, and it is why the projection work and
the pricing work are not substitutes.

**7 legs from one week is no validation either way.** The table needs grading across many weeks,
which the ledger format now makes cheap.

## 6. `anytime_td` — removed

**Per user directive, `anytime_td` is excluded from all ticket construction going forward.**

Supporting evidence, all independent: its 80.47% board win rate is an artifact of long-odds
pricing against a ~83% break-even; it hit **42.1%** in the W7 ledger across 38 legs with **no price
recorded on a single one**; and it is **absent from the historical benchmark entirely** — the
2024/2025 dataset covers `rec_yards`, `receptions`, `rush_yards`, `rush_rec_yards`, `pass_yards`
and never priced TDs. **There is no calibration for it and no evidence it was ever gradeable.**

## 7. Tier construction — now supportable, with one requirement

Tiers can be built from measured probabilities rather than judgement. The clearance curve gives
`p` per market per threshold fraction; a ticket's probability is the product across legs (adjusted
for same-game correlation, which this table does **not** measure); and the payout is the book's.
**SAFE should mean a measured joint probability, not a label.**

**The requirement, and it is the binding one: record the price and the PRISM projection on every
leg.** The W8 card did this for 8 of 47 legs, which is why only 8 could be evaluated above. **With
both fields the table prices every leg automatically; without them it prices none.**

No odds fetched. No bet authorized. No production code modified.
