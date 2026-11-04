# PRISM — GRADING THE ACTUAL BETS: THE LEGS ARE GOOD, THE PARLAYS LOSE

**Date:** 2026-11-04
**Source:** `docs/research/evidence/2026-10-25_week7_ticket_ledger/ticket_legs.csv` — 163 legs,
30 tickets, 11 tiers, Week 7 2026.
**Status:** MEASUREMENT on one week. **Not an edge claim.** See §4 for why the headline number is
weaker than it looks.

---

## 0. The thing I got wrong all session

**Every leg in this ledger is an ALT LINE** — `200+`, `30+`, `40+`, `2+ receptions`, `U4.5`.
Median price **−168**, range **−500 to +152**.

**Everything I measured today was main-line side-picking.** The whole board study, the market EV
decomposition, the `pass_yards` durability test, the situational cells — **all of it graded
`sign(model_proj − vegas_line)` on main lines, which is not the activity being performed.**

That is why the measurements said "no edge" while the user's experience said "we hit bets." **I
was grading a different game.** This document grades the right one.

## 1. Leg level — positive

| | |
|---|---|
| legs graded | **163 / 163** |
| **leg hit rate** | **65.6%** (107/163) |
| **EV per $1, priced legs bet straight** | **+0.1412** (n=93) |

| market | n | hit% | median threshold | median price | EV/$1 |
|---|---|---|---|---|---|
| **rec_yards** | 67 | **77.6%** | 30+ | −180 | **+0.1025** |
| **pass_yards** | 47 | **70.2%** | 200+ | −136 | **+0.2101** |
| rush_yards | 11 | 54.5% | 50+ | −225 | +0.0544 |
| **anytime_td** | 38 | **42.1%** | — | **unpriced** | — |

**The three priced markets are all positive as single legs.** `rec_yards` at 77.6% on a 30+
threshold is consistent with the validated alt-line curve (69–79% at ~50% of projection).

## 2. Ticket level — 4 of 30

| tier | tickets | won | avg legs | leg hit% |
|---|---|---|---|---|
| **SAFE** | 4 | **3** | 3.8 | **93.3%** |
| **MODERATE** | 4 | **1** | 4.8 | 84.2% |
| MEDIUM | 4 | 0 | 5.8 | 69.6% |
| HIGH | 4 | 0 | 7.5 | 70.0% |
| **LOTTERY** | 6 | **0** | **8.2** | 57.1% |
| AGGRESSIVE | 2 | 0 | 4.0 | 25.0% |
| others | 6 | 0 | 3.0 | 33–67% |

**30 tickets, 4 winners (13.3%). 9 of 30 lost by exactly one leg.**

**The gradient is monotone in leg count.** 3.8 legs at a 93.3% hit rate won 3 of 4. 8.2 legs at
57.1% won 0 of 6. `0.571^8 ≈ 1.1%` — a lottery ticket at those hit rates is arithmetically
hopeless regardless of how good the legs are.

## 3. The arithmetic that matters

**If legs were truly +0.1412 EV and independent, parlays would MULTIPLY the edge, not destroy
it:** parlay EV + 1 = ∏(1 + EV_leg), so three +14% legs would be **+48%**. **Parlays of
genuinely +EV legs are more +EV, not less.**

They did not behave that way. **Three candidate explanations, and they are not mutually
exclusive:**

1. **The legs are not independent.** Same-game stacks (Maye + Henry, Jones + Strange) correlate,
   and SGP pricing charges for correlation the bettor does not get credit for.
2. **The +14% is one week on 93 legs.** W7 was also the best week on the main board (+0.0260),
   so this may be a hot sample.
3. **70 of 163 legs carry no recorded price, including all 38 `anytime_td` legs — the worst
   market at 42.1%.** The +14% is computed on a **subset that excludes the worst-performing
   market entirely.**

## 4. So the headline number is not an edge claim

**+0.1412 on 93 legs from one week, with the worst market excluded by missing prices, is not
evidence of a +14% edge.** It is evidence that **the leg-selection process in this ledger
produced a positive result in Week 7 on the markets that were priced.**

**What is more solid than the EV figure:**

- **`anytime_td` at 42.1% is a genuine problem and was independently flagged today** — the board
  study found its apparent 80% win rate was an artifact of long-odds pricing, break-even ~83%.
  **38 of 163 legs, nearly a quarter of the book, in the one market measured as a trap twice over.**
- **The leg-count gradient is arithmetic, not sample noise.** It does not need more weeks.

## 5. What this actually supports

**Testable, and the evidence points at it rather than at a new model:**

1. **Drop `anytime_td` from the tickets.** 42.1% hit, worst market, independently identified as
   mispriced-against. No study required.
2. **Collapse the tier ladder toward fewer legs.** SAFE (3.8 legs) went 3/4; LOTTERY (8.2 legs)
   went 0/6, and 0.571^8 says that is structural. **If the legs are +EV, 2–3 leg combinations
   capture the multiplication; 8-leg tickets convert it into a raffle.**
3. **Prefer `rec_yards` and `pass_yards` alt thresholds**, which hit 77.6% and 70.2%.
4. **Record the price on every leg.** 70 of 163 unpriced makes the book ungradeable, and the one
   number that decides everything — EV — cannot be computed without it.

## 6. What this does NOT establish

**One week. 93 priced legs. No holdout.** The W8 card
(`WEEK8_SUNDAY_0748_EST_SIX_TICKET_ENTER_CARD_V1.md`) exists but is prose, not a machine-readable
ledger, so it is not graded here.

**A prior session already measured PRISM's Week 7 Brier at 0.280 against the market's 0.249** —
the model is **worse calibrated than the market**, which is consistent with everything measured
today and is a reason to expect the leg edge to be **selection and threshold choice**, not model
superiority.

**The honest next step is to grade W8 and every future week in this format**, and to require the
price field. **Two weeks is not proof either, but it is the difference between a measurement and
an anecdote.**

No odds fetched. No bet authorized. No production code modified.
