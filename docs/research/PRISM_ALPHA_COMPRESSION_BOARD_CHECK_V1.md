# PRISM — BOARD CHECK: THE COMPRESSION IS CONFIRMED, THE OBVIOUS FIX FAILS

**Date:** 2026-11-03
**Tests the prediction committed in** `PRISM_IRREDUCIBLE_FLOOR_AND_ALPHA_HAIRCUT_V1` §9,
**before this data was examined.** Confirmatory, not exploratory.
**Sample:** W1, W2, W6, W7, W8 priced boards (W3–5 permanently lost, AGENTS.md §5a).
5,114 graded receiving rows after pushes. Graded on `sign(model_proj − vegas_line)`, **never**
the `side` column, which agrees only 39.6% of the time.

---

## 1. The prediction was right

> *"The W1–8 board should show our model leaning UNDER on high-target-share receivers more often
> than it should, and losing disproportionately on those unders."*

| target share | n | model says UNDER | actual went UNDER | lean |
|---|---|---|---|---|
| <5% | 500 | 46.0% | 58.0% | −12.0pp |
| 5–12% | 1670 | 42.0% | 53.8% | −11.7pp |
| 12–20% | 1792 | 62.7% | 48.3% | +14.4pp |
| **20%+ alpha** | **1152** | **89.4%** | **46.9%** | **+42.5pp** |

| target share | n under | UNDER win% | n over | OVER win% |
|---|---|---|---|---|
| <5% | 230 | **71.30%** | 270 | 53.33% |
| 5–12% | 702 | **59.54%** | 968 | 50.41% |
| 12–20% | 1124 | 51.25% | 668 | 56.59% |
| **20%+ alpha** | **1030** | **46.80%** | 122 | 52.46% |

Both halves confirmed. **89.4% of all alpha signals are unders, and they lose** (46.80% against a
53.5% break-even at −115).

## 2. Third independent measurement of the same 16%

Measured **against the market**, which absorbs how the season actually ran, the compression is
**stable every week — it is not an early-season prior artifact**:

`model_proj` vs `vegas_line`, relative, `rec_yards`:

| tier | W1 | W2 | W6 | W7 | W8 |
|---|---|---|---|---|---|
| <5% | −13.0% | −7.8% | **+30.2%** | **+14.3%** | **+7.7%** |
| 5–12% | −4.4% | +5.4% | **+25.7%** | **+11.9%** | **+20.7%** |
| 12–20% | −10.5% | −8.6% | −3.2% | −1.8% | +3.7% |
| **20%+ alpha** | **−20.0%** | **−13.5%** | **−14.3%** | **−19.0%** | **−14.9%** |

**This is the compression signature, visible in pricing:** the model runs **over** on bench
receivers and **−15% to −20% under** on alphas, in every single week. Mean alpha deviation
**−16.3%** — matching the allocator haircut (16.4%) and the true-rate study (16.2%) to within a
point, from three independent datasets.

## 3. The obvious fix fails — and this is the important part

A per-tier multiplicative de-compression, fitted on held-out weeks and applied out of sample:

| holdout | n | baseline | de-compressed | |
|---|---|---|---|---|
| fit W1,W2,W6 → apply W7,W8 | 2000 | 57.10% | **52.80%** | **−4.30pp** |
| fit W6,W7,W8 → apply W1,W2 | 2248 | 49.64% | 51.87% | +2.22pp |
| fit W1,W2 → apply W6,W7,W8 | 2866 | 55.76% | **50.73%** | **−5.02pp** |

**Two of three holdouts are materially negative.** The alpha tier flipped from 92% under to 13%
under and accuracy fell 53.12% → 46.43%.

**Why it fails:** the multiplier was fitted as `mean(actual) / mean(model_proj)`, so it absorbed
**both** the model's compression **and** the fact that alpha receivers beat their lines in this
particular season. Those are different quantities, and only one of them is ours:

| | alpha `rec_yards` |
|---|---|
| mean `vegas_line` | 59.41 |
| mean **actual** | **66.01** |
| **the market itself was low by** | **−10.0%** |

The fitted multiplier came out **1.369** — a 37% uplift — against a model defect of 16%. The
other 21% was the 2026 passing environment, which is not ours and need not repeat.

**Correcting a real projection bias made the side worse.** Projection accuracy and side accuracy
are not the same objective, and this is a measured instance of them pulling apart.

## 4. Two facts that must not be merged

**(a) A model defect — stable, structural, ours to fix.** The receiving distribution is
compressed toward the middle: bench receivers over-projected +8% to +30%, alphas under-projected
−15% to −20%, every week, three independent confirmations at ~16%. This makes **89% of alpha
signals unders by construction**, so alpha unders carry an artifact where information should be.

**(b) A season regime — not ours, may not repeat.** The market under-set alpha receiving lines by
10% across W1–8 2026. **That, not our model, is the main reason alpha unders lost**, because in
this sample *any* alpha under lost.

Conflating them produces the 1.369 multiplier and loses 4.30pp. Keep them separate.

## 5. What is actually actionable

**Do not fire "model says under on a 20%+ target-share receiver."** n=1030, 46.80%, Wilson 95%
≈ [43.7%, 49.9%] — excludes break-even. Removing that one category lifts the graded receiving
board **53.07% → 54.65%.** The justification is §4(a): the signal is produced by a structural
artifact, not a read. **The 46.80% figure itself is not a forward estimate**, because §4(b) means
it is partly regime; per week it ran 33.03 / 41.51 / 54.26 / 59.63 / 46.39 — W6 and W7 were
*profitable*. Treat this as suppressing a known-contaminated signal, not as a discovered edge.

**Do not deploy a multiplicative de-compression.** §3 falsifies it out of sample. The fix belongs
at source — the group-mean shrinkage that compresses the distribution — and must be validated
against **true rate** and **side accuracy separately**, since §3 shows improving one can damage
the other.

## 6. A correction to the record on the weekly trend

Receiving-only side accuracy by week: **51.44 → 47.66 → 52.66 → 59.29 → 54.55**.

The improvement from W1–2 (≈49.7%) to W6–8 (≈55.4%) is real and substantial. But it is **not
monotone** — W2 fell below W1 and W8 fell below W7. "Up every week" overstates it, and the
alpha subset is more volatile still (35.0 / 42.2 / 55.8 / 58.2 / 47.1). Worth stating plainly
because bankroll sizing should not be built on a trend that is partly noise.

## 7. Next

The root cause is now triangulated: group-mean shrinkage (`GROUP_STRENGTH = 3.0` toward
`DEFAULTS["tgt_share"]["WR"] = 0.16`) compresses the share distribution, which both inflates
bench players — creating the 16.4% wasted mass — and haircuts the alpha. One mechanism explains
the allocator finding, the true-rate finding and the pricing finding.

A fix must be preregistered with **both** objectives scored, and must beat the §3 null. No
sportsbook data entered any model computation; this document uses market lines for **grading
only**, per AGENTS.md §3.

---

# ADDENDUM V1a — §5's BETTING RULE IS WITHDRAWN (LOOKAHEAD IN THE TIERING)

**Added:** 2026-11-03, same day, before any part of §5 was acted on.

§1–§5 tiered players by their **season-long** level — target share computed from all of W1–W8,
**including the week being graded.** A player who had a big game was therefore sorted into the
top tier *for that same week*, enriching the top tier with players who had just gone off. That
biases every "unders lose at the top" figure.

Re-tiering with **prior weeks only** (the player's level from weeks strictly before the graded
week) reverses the conclusion:

| | contaminated tiering | **pregame-only tiering** |
|---|---|---|
| top-tier + model says UNDER | 46.80% (n=1030) | **56.62% (n=1194)** |
| the rest of the board | 54.65% | 51.80% |
| WR `rec_yards` top-tier unders | — | **57.52%** (n=226) |
| WR `receptions` top-tier unders | — | 51.46% (n=206) |

**§5 is withdrawn in full. Do not suppress alpha unders.** On a clean pregame split that
category is the **best** cell on the board, not the worst. Win rate by pregame tier and side:

| pregame tier | n under | UNDER win% | n over | OVER win% |
|---|---|---|---|---|
| Q1 low | 546 | 55.31% | 960 | 49.17% |
| Q2 | 682 | 53.67% | 714 | **55.74%** |
| Q3 | 940 | 51.91% | 508 | **44.88%** |
| **Q4 top** | **1194** | **56.62%** | 202 | 51.49% |

Overall 52.80% (n=5746). The one losing cell is **Q3 overs at 44.88%**, which no prior document
identified and which this correction found only because the tiering was redone.

**This is the same class of error as grading on the `side` column and as quoting a CI that
covered the wrong quantity: a number that looked actionable because its construction leaked the
answer.** It is the second betting recommendation withdrawn in this session. The rule going
forward: **any tier, filter or threshold used for a betting decision must be computed from
information available before kickoff, and that must be stated explicitly in the document that
proposes it.**

## What survives

**The compression finding is unaffected**, because `model_proj − vegas_line` contains no
outcome. Re-tiered pregame-only, **10 of 10 role × market cells still show it**, monotone in
every one. §2's weekly stability, §4's separation of model defect from season regime, and §3's
falsification of the multiplicative fix all stand. Only the betting rule in §5 is withdrawn.
