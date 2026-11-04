# PRISM — THE IRREDUCIBLE FLOOR, AND WHY EVERY SIGNAL STUDY CAME BACK NULL

**Date:** 2026-11-03
**Status:** DIAGNOSTIC. Nothing promoted. **Corrects figures in three earlier documents.**
**This is the most consequential measurement in the program to date.**

---

## 1. The question nobody had asked

Every study in this program measured **our** error. None measured **how much error is
physically irreducible.** A WR drawing 8 of ~30 team targets has a realized share that bounces
on multinomial luck *even when his true rate is known exactly*. If that floor sits near our
error, the lane is closed and the nulls were inevitable.

Method: take each player's **8-week** target share as a proxy for his true rate, re-simulate
W7/W8 from the real team target totals (mean 30.1), and ask what a **model with perfect
knowledge of the true rate** still scores.

## 2. There is a ceiling, and we are most of the way to it

| | M1 |
|---|---|
| C1 production | 0.05436 |
| **E1 (current best)** | **0.04203** |
| **irreducible floor — perfect knowledge of true rate** | **0.03245** `[0.02905, 0.03592]` |
| ORACLE_BOTH as reported in `ALLOCATOR_SHAPE_RESULTS_V2` | 0.02513 |
| **ORACLE_RANK re-run on purely SIMULATED data** | **0.02505** |

**Real headroom above the floor: +0.00958.** Not +0.01690.

**The last two rows are the correction.** ORACLE_BOTH scored 0.02513 — *below the physical
floor*, which no real model can do. Re-running the same oracle against simulated data where it
can only be fitting noise returns **0.02505, statistically identical.** So ORACLE_BOTH's margin
under the floor was **entirely realization-fitting.** `ALLOCATOR_SHAPE_RESULTS_V2` §4 flagged
those oracles as "inflated upper bounds"; this quantifies it — roughly **43% of that +0.01690
was fake.**

## 3. Three of the four tiers are already AT or BELOW the luck floor

Per-row, with the floor computed on the same selection:

| tier | n | E1 \|bias\| | luck floor | real room |
|---|---|---|---|---|
| <5% | 463 | 0.0228 | 0.0170 | +0.0058 |
| **5–12%** | 121 | **0.0375** | **0.0362** | **+0.0013** |
| **12–20%** | 98 | **0.0481** | **0.0444** | **+0.0037** |
| **20%+** | 87 | **0.1013** | 0.0739 | **+0.0274** |

**This explains the entire history of null results.** Five estimator corrections (prior
anchoring, ensemble reweighting, additive offset, recency, snap decomposition) and four signal
studies (receiving yards, target share, QB quality, rushing NGS) were aimed at a pooled metric
in which **600 of 769 rows were already at the information limit.** They were not bad studies.
**They were studies of a solved problem**, and the solved majority outvoted the one tier that
isn't.

It also explains why *"a naive opportunity count beat every sophisticated arm"* in three of four
studies. In a tier at its noise floor, the naive estimator is correct and sophistication can only
add variance.

## 4. The selection artifact — a second correction

Binning on *actual* share selects rows that got lucky, so part of the top tier's apparent
under-allocation was regression to the mean. Replicating that selection inside the simulation:

| tier | E1 bias | **artifact** | real room |
|---|---|---|---|
| 20%+ | −0.1009 | **−0.0632** | +0.0274 |

**63% of the −0.1009 reported in `MC_LEAK_FIX_RESULTS_V1` and `ALLOCATOR_SHAPE_RESULTS_V2` was
a binning artifact, not model error.** Those tables should be read with this correction.

The gap survives it. Binning on *predicted* share instead gives +0.0269 — same answer.

## 5. The one real defect, measured without realization or selection

Scoring E1 against each player's **true rate** directly — no single-game noise, no outcome
conditioning:

| player's **true** rate | n | E1 − TRUE | \|E1 − TRUE\| |
|---|---|---|---|
| <5% | 452 | +0.0043 | **0.0068** |
| 5–12% | 149 | +0.0044 | **0.0172** |
| 12–20% | 107 | −0.0066 | **0.0212** |
| **20%+** | **61** | **−0.0392** | **0.0440** |

**The model is excellent for 93% of players and systematically under-rates each team's true
alpha by −0.0392 — 16% of his rate, about 1.1 targets per game.**

## 6. The mechanism, confirmed to the fourth decimal

**Mass spent on players with near-zero true rates is, under a normaliser that only scales down,
a proportional haircut on everyone.** It is invisible on a bench player (−16% of 0.006 = −0.001)
and decisive on an alpha (−16% of 0.24 = −0.039).

| | |
|---|---|
| wasted mass fraction per team-week | **0.1640** |
| alpha's true rate | 0.2414 |
| E1 gives him | 0.2021 |
| **predicted bias from the haircut alone** | **−0.0390** |
| **observed bias** | **−0.0393** |
| unexplained residual | **−0.0002** |

A pooled regression of bias on `−waste × true_rate` across all 769 rows gives slope +0.376, not
1.0 — **and that is expected**, because bench players are the *source* of the haircut, not its
victims: their bias is **positive** (+0.0043). The mechanism is specific to the top of the
distribution, which is exactly where it was derived to apply.

## 7. A third correction — the hindsight gate is the wrong fix

`ALLOCATOR_SHAPE_RESULTS_V2` §7 named a participation gate as the next instrument. Tested
against **true rates** rather than realizations, perfect weekly gating **makes the model worse
overall**:

| | \|E1 − TRUE\|, all rows |
|---|---|
| E1 | **0.01374** |
| perfect weekly gate | **0.02358** |

It halves the alpha's gap (−0.0392 → −0.0198, share 0.2021 → 0.2190) but damages everyone else,
because it zeroes players who hold a genuine true rate and merely caught nothing that week.

**So the instrument is not "predict who draws zero this week"** — that is both unknowable and
harmful. It is **"stop granting mass to players whose true rate is ~0 at all"**: a *stable role*
property, not a weekly coin flip. 452 of 769 rows carry a true rate under 5%. The universe
averages 14.2 players where roughly 7–8 hold a real role.

## 8. What this means for the program

**The ceiling is real and close — for almost everyone.** Pooled MAE on this market is within
0.0096 of physically unbeatable, and no amount of matchup, efficiency, NGS or game-script work
will move it, which is precisely what the record already shows: matchup sensitivity not
measurable (p=0.395), defense shape non-persistent (r=+0.017), game script not forecastable.

**One defect remains and it is worth more than everything already attempted**: the alpha
haircut, −0.0392 of share on 61 player-weeks, with a mechanism confirmed to ±0.0002 and a
named cause in the normaliser rather than in football.

This is the PLAYER PROJECTION FIRST PRINCIPLE vindicated as a *measurement* claim, not a
preference: the position aggregate was at its floor while the named alpha was off by 16%, and
**the aggregate metric concealed it for the entire program.**

## 9. Authorized next step — and a falsifiable market prediction

The fix is a **stable-role universe filter** applied before allocation, replacing a weekly
zero-prediction. It must be preregistered separately, scored against **true rate** as well as
realization (§7 is why), and must not be assumed to help the 5–20% tiers, which are at their
floor and can only be harmed.

**A prediction that connects this to money and can be checked against the existing graded
board, at no cost:** under-rating alphas by 16% of target share means systematically
under-projecting receptions and yards for exactly the players carrying the most liquid props.
**The W1–8 board should therefore show our model leaning UNDER on high-target-share receivers
more often than it should, and losing disproportionately on those unders.** If that pattern is
absent, §5–§6 do not transmit to pricing and this finding is narrower than it appears.

No sportsbook data entered any computation here.
