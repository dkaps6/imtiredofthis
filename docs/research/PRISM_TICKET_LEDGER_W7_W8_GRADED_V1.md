# PRISM — TWO WEEKS OF ACTUAL TICKETS: **THE VARIANCE IS THE STORY**

**Date:** 2026-11-04
**Supersedes the framing of** `PRISM_W7_TICKET_LEDGER_GRADED_V1.md`, which concluded "the legs
are good" from Week 7 alone. **Week 8 does not support that.**
**Ledgers:** `docs/research/evidence/2026-10-25_week7_ticket_ledger/ticket_legs.csv` (163 legs),
`docs/research/evidence/2026-11-01_week8_sunday_tickets/ticket_legs.csv` (47 legs, newly
converted from the prose card).
**Grader:** `scripts/research/grade_ticket_ledger_v1.py` — handles overs and unders, voids
non-participants.

---

## 1. The two weeks do not look like the same process

| | legs | **hit rate** | priced legs | **EV/$1** |
|---|---|---|---|---|
| **Week 7** | 163 | **65.6%** | 93 | **+0.1412** |
| **Week 8** | 47 | **31.9%** | 8 | **−0.5808** |
| **combined** | **210** | **58.1%** | 101 | +0.0840 |

**A 33.7-point swing in leg hit rate between consecutive weeks.** The combined EV of +0.0840 is
**92% Week 7 by priced-leg count** — it is the Week 7 number with a rounding correction, not two
weeks of agreeing evidence.

## 2. The tier labels are not calibrated — and this is the finding

| tier | Week 7 leg hit% | Week 8 leg hit% |
|---|---|---|
| **SAFE** | **93.3%** | **25.0%** |

**The tier called SAFE went from the best cell in Week 7 to near the worst in Week 8.** Week 8's
SAFE legs were Warren 2+ receptions at **−550** (caught **1**), Taylor 40+ rush (**23**), Shakir
30+ rec (**64** ✓), Tucker 4+ receptions (**0**). **One of four, including a −550 favourite.**

**A label that conveys confidence it does not have is worse than no label**, because it drives
stake sizing. Until a tier's historical hit rate is measured and the label is set from it, the
ladder is naming, not calibration.

## 3. Week 8, leg by leg — the grading is verifiable

| leg | threshold | actual | |
|---|---|---|---|
| Jaylen Warren receptions | 2+ (−550) | **1** | ✗ |
| Jonathan Taylor rush | 40+ (−225) | **23** | ✗ |
| Khalil Shakir rec | 30+ (−210) | 64 | ✓ |
| Tre Tucker receptions | 4+ (−120) | **0** | ✗ |
| Aaron Rodgers pass | 218.5 (−113) | **172** | ✗ |
| Daniel Jones pass | 150+ | **144** | ✗ (by 6) |
| Ja'Marr Chase rec | U79.5 | **141** | ✗ |
| Keon Coleman rec | 29.5+ | **4** | ✗ |
| Drake Maye pass | 175+ | 199 | ✓ |

**`rush_yards` went 0 for 5. `rush_rec_yards` 0 for 1.** Consistent with both being the
worst-measured markets in today's board work.

## 4. What survives from the Week 7 document

**Only the arithmetic, which needs no sample:**

- **The leg-count gradient.** `0.571^8 ≈ 1.1%`. An eight-leg ticket is hopeless at realistic leg
  hit rates whatever the edge per leg. Week 8 confirms it from the other direction: MODERATE went
  **4 legs, 4 hits, 1 winner**; LOTTERY went **11 legs, 2 hits**.
- **`anytime_td` is a trap.** 42.1% in Week 7, 38 legs, and **no price recorded on a single one**.
  Independently shown today to be break-even at ~83% while winning 80%.

**What does not survive:** "+14% EV on the legs." That was one week, on a priced subset excluding
the worst market, and Week 8 came in at **−0.58** on the eight legs that carried a price.

## 5. The measurement problem that blocks everything

**Only 101 of 210 legs carry a price — 48%.** Week 8 recorded prices on **8 of 47**.

**EV cannot be computed without the price, and EV is the only number that decides anything.** A
ledger without prices records what was chosen, not whether choosing it was right. **This is the
single cheapest fix available and it requires no model work, no API and no study: write the price
on every leg at the time it is taken.**

## 6. A flaw found in my own grader, and a caveat

The first pass treated **presence in the weekly stats table as proof of playing**, which counted a
scratched player's zeros as an **UNDER win**. Fixed: a leg is now **VOID** unless the player
recorded real volume (attempts + targets + carries > 0).

**Remaining caveat:** the `Lamar Jackson U237.5` leg still grades as a hit on 0 passing yards.
There is a defensive back of the same name, so this is plausibly a **name collision** rather than
a quarterback who played and threw for nothing. It sits in the explicitly-entertainment `FUN_1M`
tier and touches no conclusion above, but **name collision is a real grading risk in any
player-keyed ledger and should be resolved with team or position matching, not name alone.**

## 7. What two weeks actually supports

**Nothing about edge.** 210 legs across two weeks, with a 33.7-point swing and 48% of legs
unpriced, cannot distinguish a good process from a lucky week.

**Three things it does support, all free:**

1. **Record the price on every leg.** Without it this stays unmeasurable forever.
2. **Stop using tier labels for stake sizing** until each tier's hit rate is measured. SAFE at
   93.3% then 25.0% is not a confidence level.
3. **Cut `anytime_td`, and cut the long tickets.** Both conclusions are arithmetic, both hold
   across the two weeks, and neither needs another study.

**Keep grading every week in this format.** Two weeks is an anecdote with error bars; ten weeks
would be a measurement. The grader and the ledger schema now exist, so the marginal cost per week
is one CSV.

No odds fetched. No bet authorized. No production code modified.
