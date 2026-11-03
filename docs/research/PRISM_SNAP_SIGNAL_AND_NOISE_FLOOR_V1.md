# PRISM — SNAP SIGNAL CHECK + THE USAGE-SHARE NOISE FLOOR V1

**Date:** 2026-11-03
**Status:** NEGATIVE RESULTS + a measurement that redirects the lane. Nothing promoted.
**Follows:** `PRISM_TGT_SHARE_RECONCILIATION_V1.md` §6, which proposed snap share as the
missing role signal and gated it on publication timing and identity coverage.

---

## 1. Snap counts: both gates, then the test

**Timing — PASSES.** `nflreadpy.load_snap_counts(2026)` carries **weeks 1–8** as of 2026-11-03,
one day after the Week 8 Monday night game. A Thursday production run would have week W−1 snaps
in hand. This is walk-forward safe by construction: the feature is prior-week snaps, never
week-W snaps.

**Coverage — PASSES, better than expected.** Snap counts key on `pfr_player_id` + name string;
`load_player_stats` keys on GSIS `player_id`. There is **no shared identifier**, so the join is
by normalised name + team + week. That join resolves **99.3%** of 2,425 offensive player-weeks;
only 5 distinct players fail. (The normaliser must strip suffixes — the source spells our own
motivating case `"Luther Burden"` against `"Luther Burden III"` in player stats.)

**The test — FAILS.** Decomposing target share into target-rate × snap-share and renormalising
within team, predicting Week 8 from weeks 1–7 (227 players, 27 teams):

| predictor | MAE |
|---|---|
| A: season-to-date target **share** | **0.04547** |
| B: target-rate × snap-share (renormalised) | 0.04751 |

Gain from using snaps **−0.00145**, CI [−0.00513, +0.00193]. No improvement; point estimate
negative.

**And it gets the motivating case backwards.** Chicago W1–7 mean `offense_pct`:
Loveland 0.851, **Odunze 0.850**, Raymond 0.619, **Burden 0.606**. Snap share would have ranked
Odunze *further above* Burden — the opposite of the correction it was proposed to make. Burden
is a high-target-rate player on fewer snaps, not a snap-volume player.

---

## 2. The measurement that matters: there is almost no headroom

Comparing against an **oracle** that knows each player's true full-season mean share — which
uses week-W data and is therefore unbeatable, not achievable. Both computed on **identical
rows** (1,392 player-weeks, 284 players with ≥5 games):

| | MAE |
|---|---|
| **oracle** (player's own full-season mean) | **0.04332** |
| **production C1** | **0.05100** |
| naive C0 (season-to-date) | 0.05608 |

**Total headroom from production to an unbeatable oracle: 0.00768.**

Every candidate tested today moved the needle by 0.0002–0.0015, mostly in the wrong direction.
They were not failing to find a large effect — **there is no large effect to find.** The
shipped posterior already captures most of the available signal in these inputs.

### Correction to an earlier figure

An initial version of this measurement reported headroom of **0.039** by comparing the oracle
on all players with ≥5 games against the model MAE on the eligible prediction set. Those are
different populations — the all-player set is dominated by near-zero-share players whose weekly
deviations are tiny, which depresses the floor. On matched rows the headroom is **0.00768**.
The 0.039 figure is withdrawn.

---

## 3. Where the headroom actually is: the alpha tier

| actual week-W share | n | oracle | production C1 | headroom |
|---|---|---|---|---|
| <5% | 264 | 0.03369 | 0.04851 | **+0.01482** |
| 5–12% | 476 | 0.03340 | 0.03644 | +0.00304 |
| 12–20% | 328 | 0.04265 | 0.04093 | **−0.00172** |
| **20%+** | **324** | **0.06643** | **0.08464** | **+0.01821** |

Production is **at or past the oracle for mid-share players (12–20%)** — there is nothing left
to win there. The headroom is concentrated in the two tails, and the larger, more valuable one
is the **20%+ tier: the alpha receivers.**

**This converges with V21.** `PRISM_LIVE_SIDE_GRADING_CORRECTION_V1.md` §6 found the model
projecting Ja'Marr Chase at 4.49 receptions and Justin Jefferson at 4.12 against markets
implying 6+. That is the same tier, reached by a completely independent route — market pricing
there, oracle comparison here.

---

## 4. What this closes, and what it opens

**Closed — do not revisit without new inputs.** Five candidate improvements to the general
usage-share estimator, all tested and all failed:

1. prior weight by time decay (A1) — preregistered, failed
2. prior weight by prior/current disagreement (A2) — preregistered, failed
3. dropping the prior for team-changers — worse
4. recency / within-season trend weighting — worse
5. target-rate × snap-share decomposition — worse

Combined with §2, the conclusion is not "these ideas were poorly chosen." It is that
**the general estimator is within 0.008 of an unbeatable oracle and is not the problem.**

**Opened.** The research question is no longer *how do we weight the inputs* but:

> **Why is production 0.018 worse than the oracle specifically on 20%+ target-share players,
> when it is at the floor for the 12–20% tier?**

That is a narrow, player-level, falsifiable question about the top of each team's receiving
hierarchy, it is where the money is in both projections and markets, and two independent lines
of evidence now point at it.

It should be preregistered with the alpha tier as the primary population — not as a subgroup
of a position-wide test, which is exactly the mistake the Week 8 grading made.
