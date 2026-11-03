# PRISM — RECEIVING MASS BIAS, V1 RESULT — **UNDERPOWERED, NO GATE READ**

**Date:** 2026-11-03
**Preregistration:** `PRISM_RECEIVING_MASS_PREREGISTRATION_V1.md`, frozen in commit `c4a78c86`.
**Status:** STOPPED AT THE SAMPLE CHECK. **No arm was scored. No gate was read.**

---

## What happened

The preregistration specified:

- fit δ (D0) and w (D2) on **weeks 1–2**, evaluate on **weeks 6–8**;
- per-player bias computed over players with **≥3 graded weeks**;
- **minimum 100 such players**, else *"report underpowered and read no gate."*

Evaluation window (weeks 6–8), receptions:

| graded weeks in eval window | players |
|---|---|
| exactly 1 | 63 |
| exactly 2 | 83 |
| **exactly 3** | **44** |
| **≥3 (the preregistered bar)** | **44** |

**44 < 100. The minimum is not met.** No arm was computed.

---

## The design error

The §2 sample table quoted **124 players with ≥3 weeks** — but that count was across **all five
graded weeks (1, 2, 6, 7, 8)**. Once a fit/eval split is imposed, the evaluation window holds
only **three** weeks, so clearing a ≥3-week bar requires a player to be priced in **every single
evaluation week**. Only 44 are.

Those two requirements are close to mutually exclusive, and I did not notice when writing the
design. The sample-size check was placed before the arms specifically because the Week 8 grading
failed by reading an undersized cell, and it did its job here.

**It would have been easy to read "≥3 graded weeks" against the all-weeks figure of 124 and
proceed.** That reading is available in the text. It is not the one the design means, and
taking it would be choosing an interpretation after seeing that the strict one blocks the study.

---

## What was NOT done

No arm (D0, D1, D2) was computed. No bias, breadth, MAE or side-accuracy number from this design
exists. Nothing from V1 may be cited as a result.

---

## Next

A V2 preregistration replaces the fit/eval split with **leave-one-week-out cross-fitting**:
for each week W, hyperparameters are fit on all *other* graded weeks and applied to W. Every row
is then evaluated out-of-sample, every player accumulates up to five weeks, and the ≥3-week bar
becomes reachable (124 players).

**V2's design was written without computing any arm**, so the change is driven by the power
failure alone and not by any observed outcome. That is what keeps it legitimate rather than a
rewrite toward a preferred answer.
