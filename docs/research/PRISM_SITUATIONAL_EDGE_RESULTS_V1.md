# PRISM — SITUATIONAL EDGE: H2 REFUTED, **H1 UNTESTABLE**, AND THE REMAINING ANGLE NAMED

**Date:** 2026-11-04
**Preregistration:** `PRISM_SITUATIONAL_EDGE_PREREGISTRATION_V1.md`, frozen first.
**Verdict: K1 fires — no cell passes all four gates.** But the two hypotheses fail in
**different ways**, and only one of them is actually refuted.
**Artifacts:** `data/research/situational_edge_v1.csv`,
`scripts/research/evaluate_situational_edge_v1.py`.

---

## 1. H1 — vacancy underreaction is **UNTESTABLE**, not refuted

| cell | n | players | win% | EV/$1 | 95% CI (clustered) |
|---|---|---|---|---|---|
| vacancy rows, model side | **80** | **10** | 57.50% | +0.0549 | **[−0.3129, +0.4347]** |
| vacancy + model says OVER | **22** | **4** | 63.64% | +0.1772 | **[−0.7301, +0.8853]** |
| control: always OVER on vacancy | 80 | 10 | 50.00% | −0.0788 | [−0.5506, +0.4394] |
| no-vacancy baseline | 7706 | 284 | 52.50% | −0.0219 | [−0.0692, +0.0258] |

**80 rows from 10 players across 3 weeks. G4 fails on size and the intervals are 0.76 wide.**
**Nothing can be concluded.** The point estimates lean positive (+0.0549, and +0.1772 on the
directional form) and the control leans negative, which is the *shape* H1 predicted — but at
n=10 players that is indistinguishable from noise.

**This is recorded as UNTESTED. It must not be cited as a negative, and must not be cited as
promising.** My vacancy definition was strict — the inactive player had to be a **top-2
target-getter** on his team by prior-weeks usage, and the bet had to be on a **same-position
teammate**. A broader definition, or more weeks, is what this needs.

## 2. H2 — the alpha-tier side is **REFUTED**, and the contradiction is resolved

| cell | n | players | win% | EV/$1 | 95% CI |
|---|---|---|---|---|---|
| top quartile + model UNDER | 1028 | 65 | **50.19%** | **−0.0655** | [−0.1999, +0.0652] |
| top quartile + model OVER | 458 | 52 | 51.09% | −0.0581 | [−0.2583, +0.1575] |
| bottom quartile + model UNDER | 840 | 143 | 51.43% | −0.0390 | [−0.1846, +0.1089] |
| **bottom quartile + model OVER** | 776 | 120 | **46.65%** | **−0.1006** | [−0.2566, +0.0580] |

**All four negative. G4 passes on size for every cell, so this is a real refutation, not a
power failure.**

**The earlier contradiction is resolved: neither figure was an edge.** The 46.80% came from
outcome-contaminated tiering; the 56.62% came from a different volume definition and **does not
reproduce** under within-market pregame quartiles and EV — it lands at **50.19%**. Both were
win-rate artifacts of how the tier was drawn.

### The one mechanism-backed pattern, and what it is worth

**Bottom quartile + model says OVER is the worst cell on the board: 46.65%, −0.1006.** That is
exactly what `D3` predicts — the model over-projects low-opportunity players (2.40 phantom
receiving yards at zero targets), says OVER on them, and loses. **Mechanism and betting record
agree.**

**But avoiding it does not create an edge.** Those 776 rows are 10% of the board; removing them
moves the board from **−0.0211 to about −0.0123.** Still negative. **Stopping a leak is not an
edge** — it is worth saying because it is true, not because it solves anything.

## 3. What this does and does not establish

**Refuted (do not revisit):** the alpha-tier side as a selection key, in either direction, at
either end of the volume distribution.

**Untested (do not describe as closed):**
- **H1 vacancy underreaction** — 80 rows is not a test;
- **alt lines** — V21 §4's curve is the only **out-of-sample-validated positive** in this
  programme and today's work never touched it;
- **correlated parlays / SGPs** — EV structure differs entirely; needs SGP prices;
- **live / in-game** — never measured.

## 4. The honest statement about selection, stated carefully

Three of today's negatives share one property: **they tested whether a pregame key can rank
MAIN-LINE rows by expected profit.** Confidence, `edge_pct`, disagreement size, market cell,
price, volume tier, side — **all fail.** That is a consistent and well-evidenced conclusion about
*that* question.

**It is not a conclusion about whether plays can be found each week.** What has been measured is
a **mechanical rule applied to every row of a board.** It does not measure:

- selecting a **small number** of spots rather than ranking 1,500;
- **alt-line thresholds**, where the only validated positive result in this programme lives;
- **information that arrives after the board is built** — late inactives, weather, beat reports;
- **football judgment about a specific situation**, which is not a column in any file here.

**A bettor using those can be profitable while every mechanical key on the board fails.** Nothing
in today's work contradicts that, and nothing in it measures it.

## 5. The remaining angle, named concretely

**Alt lines are the best remaining shot, and the reason is evidence rather than hope.**

`PRISM_LIVE_SIDE_GRADING_CORRECTION_V1` §4 trained a threshold curve on W1/2/6 and tested it on
W7. **All nine cells transferred directionally**, with a systematic 6–10pp optimism that is itself
now quantified and correctable:

| | clears |
|---|---|
| threshold at **~50% of model projection** | **69–79%** |
| threshold at **75–80% of projection** (the market's popular tier) | **51–60%** |

**The market's popular alt tier clears barely more than a coin flip.** If it is priced as though
it clears more — which is the usual shape of recreational alt pricing — that is a real,
structural, repeatable inefficiency, and it is the opposite of everything tested today because it
does **not** depend on the model out-picking the market on a main line. It depends on the
**distribution's shape**, which is the one thing this programme has measured well.

**What it requires: alt-line prices, which are not on the committed boards.** That is a paid
OddsAPI fetch, and betting research is CLOSED pending explicit authorization. **This is a decision
for the user, not a research question for me.**

## 6. Prediction scored

- **"H1 passes G4 and fails G1"** — **WRONG in an important way.** H1 failed **G4** (size), so it
  was never testable. I expected 200+ vacancy rows and got 80.
- **"H2 resolves toward the pregame figure and still fails on EV once clustered"** — **HALF
  RIGHT.** It fails, but it does **not** resolve toward 56.62%; it lands at 50.19%, so the pregame
  figure was itself an artifact.
- **Named risk — "the temptation is to accept a cell that passes two gates and call it
  promising"** — **binding and honoured.** H1's +0.1772 on 22 rows is precisely the number that
  temptation would seize on. It is reported as untestable.

## 7. Scope

No odds fetched. **No bet is authorized by this study.** No production code modified. Alt lines,
parlays and live betting remain **untested and explicitly open.**
