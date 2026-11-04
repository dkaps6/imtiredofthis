# PRISM — VACANCY UNDERREACTION, WIDENED: PREREGISTRATION V1

**Frozen:** 2026-11-04, **before any widened statistic is computed.**
**Status:** PREREGISTRATION. No result. **No bet authorized.**
**Follows:** `PRISM_SITUATIONAL_EDGE_RESULTS_V1.md` §1, where H1 was **untestable** (80 rows,
10 players).

---

## 0. The integrity problem with this study, stated first

**I have already seen the narrow result's point estimates, and they leaned positive**
(+0.0549 on the model side, **+0.1772** on the directional form, control negative). **Re-testing
the same five weeks with a new definition is a second look at data I have partially seen.**

Three controls, all binding:

1. **The definitions below are fixed now and are the complete set.** No definition may be added,
   relaxed or re-cut after any result is seen. If none reaches testable size, the answer is
   "still untestable", not "try another cut".
2. **Multiplicity:** six cells are tested, so G1 uses a **Holm-corrected** 95% family-wise
   threshold, not six independent 95% intervals. Six independent looks at a null produce a
   nominal "significant" result about 26% of the time.
3. **The four gates do not move.** They are the gates `pass_yards` failed.

**If a cell passes, it carries less weight than a first look would**, and the honest disposition
is a prospective test, not a conclusion.

## 1. Why the narrow test failed, and what widening legitimately fixes

The narrow definition required the inactive player to be a **top-2 target-getter** and the bet to
be on a **same-position teammate**. Two defects:

- **a rank threshold is arbitrary** where a **share threshold** is the football quantity;
- **it tested only target vacancies.** An inactive **RB1 frees roughly 50–60% of team carries**,
  against a WR1's ~20–25% of targets. **The larger and cleaner opportunity shift was never
  tested at all.** This is the substantive omission, not the sample size.

## 2. Fixed definitions — 3 vacancy × 2 beneficiary = 6 cells

**Vacancy (inactive player, `OUT` or `DOUBTFUL` on the pregame report), by prior-weeks share:**

- **V-STRICT** — held ≥25% of prior team targets, or ≥35% of prior team carries.
- **V-MID** — ≥15% of targets, or ≥25% of carries.
- **V-LOOSE** — ≥10% of targets, or ≥20% of carries.

**Beneficiary:**

- **B-SAME** — teammate in the same position group.
- **B-SKILL** — any WR/TE/RB teammate. *(Football basis: the production redistribution rule
  itself assumes a WR1 vacancy spills 60% to WR2, 30% to slot/TE, 10% to RB/FB.)*

**Market matching is fixed by mechanism, not chosen:** a **target** vacancy is tested on
`rec_yards`, `receptions`, `rush_rec_yards`; a **carry** vacancy on `rush_yards`,
`rush_rec_yards`. A vacancy never scores a market it cannot mechanically affect.

## 3. Method — today's three corrections are mandatory

**EV per $1 at posted odds** (never win rate) · **cluster bootstrap over PLAYERS** (never rows) ·
**all shares and tiers from PRIOR WEEKS ONLY**. Market data is permitted: selection is
market-assisted under AGENTS.md §3.

## 4. Gates — unchanged, all four required

- **G1** EV cluster CI lower bound above 0, **Holm-corrected across the 6 cells.**
- **G2** EV positive in **≥4 of 5 weeks** (or ≥3 of 4 if a week has no qualifying rows).
- **G3** dropping the single best player leaves the CI lower bound above 0.
- **G4** **n ≥ 200** rows and **≥ 25 distinct players.**

## 5. Controls — the thing that decides what any pass means

- **C1 — always-OVER control.** EV of blindly taking the over on every qualifying row, ignoring
  the model. **If C1 ≈ the model-side cell, the market misprices vacancies and the model adds
  nothing.** That is still usable information, but it is a **market fact**, and must be reported
  as such.
- **C2 — non-vacancy baseline** in the same markets and weeks, so any effect is measured against
  the right comparison rather than against the whole board.
- **C3 — placebo.** The same test run on teammates of players designated **QUESTIONABLE who
  PLAYED** (measured retention 0.8535 target / 0.5718 rush). **A real vacancy effect should be
  absent or much weaker here. If the placebo looks as good as the treatment, the effect is not
  vacancy-driven.**

## 6. Prediction

**V-LOOSE × B-SKILL is the only cell likely to reach G4**, and I expect it to fail G1 after Holm
correction. **Carry vacancies will show a larger point estimate than target vacancies**, because
the freed share is 2–3× bigger and rushing volume is less noisy (dispersion 0.548 against 0.915).

**I expect C1 ≈ the model-side cell** — i.e. if anything is there, it is the market underreacting,
not the model seeing something. The model is structurally blind to inactives, so it cannot be the
source of a vacancy edge.

**Named risks:**
- **I want this to work**, which §0 exists to contain. **A cell passing G1 and G4 while failing
  G2 or G3 is `pass_yards` again and must be reported as a negative.**
- **The placebo is the real test.** Injury designations correlate with team quality, game script
  and opponent. If C3 looks like the treatment, any apparent edge is confounded by whatever makes
  a team have injured starters, not by the vacancy.

My record: **10 hits, ~21 misses**; the named risk has landed in nearly every study.

## 7. Scope

2026 W1/W2/W6/W7/W8 committed boards. **No odds fetched.** **No bet authorized regardless of
outcome** — a pass licenses a prospective W9+ test only. No production code modified. Alt lines
remain open and out of scope, pending a separate authorization decision.
