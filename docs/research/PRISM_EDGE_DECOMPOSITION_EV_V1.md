# PRISM — EDGE DECOMPOSITION: THE BOARD IS **NEGATIVE EV**, AND ONE MARKET CARRIES EVERYTHING

**Date:** 2026-11-04
**Status:** MEASUREMENT. Overturns the working assumption that the model has a betting edge.
**Sample:** W1, W2, W6, W7, W8 2026 priced boards. **7,786 graded rows with usable posted odds.**
Graded on `sign(model_proj − vegas_line)`, **never** the `side` column.

---

## 1. The correction that drives everything: win rate was the wrong metric

Every edge figure this programme has cited — 52.96%, 53.07%, "trending up each week" — is a
**win rate**. The boards carry `vegas_over_odds` and `vegas_under_odds`, so **actual EV is
computable and was never computed.**

| | |
|---|---|
| median posted price on the side we would take | **−114** |
| break-even win rate at that price | **53.27%** |
| our actual win rate | **52.50%** |
| **shortfall** | **−0.77pp** |

**The model wins slightly less often than it needs to.** Measured as realized profit per $1
staked on the full board:

| scope | n | win% | **EV/$1** | 95% CI |
|---|---|---|---|---|
| all markets | 7786 | 52.56% | **−0.0211** | **[−0.0418, −0.0002]** |
| excl. `anytime_td` | 7406 | 52.50% | **−0.0215** | **[−0.0429, −0.0006]** |

**Both CIs exclude zero. Betting this board flat loses roughly 2% per unit.**

**Context that matters:** flat −110 two-way vig is about −4.5%, so **the model does carry real
skill** — it halves the house edge. It just does not cross it.

## 2. One market is profitable, and it is the one we built properly

| market | n | win% | median odds | **EV/$1** | 95% CI | |
|---|---|---|---|---|---|---|
| **`pass_yards`** | 554 | 57.40% | −113 | **+0.0824** | **[+0.0042, +0.1605]** | **PROFITABLE** |
| `rush_yards` | 1328 | 52.86% | −114 | −0.0008 | [−0.0519, +0.0498] | flat |
| `receptions` | 2516 | 53.97% | −129 | −0.0167 | [−0.0531, +0.0201] | — |
| `rec_yards` | 2534 | 51.30% | −114 | −0.0328 | [−0.0699, +0.0038] | — |
| **`rush_rec_yards`** | 474 | 44.30% | −114 | **−0.1665** | **[−0.2502, −0.0830]** | **LOSING** |
| `anytime_td` | 380 | 53.68% | −132 | −0.0124 | [−0.1126, +0.0886] | unmeasurable¹ |

**`pass_yards` is the only market with a CI excluding zero on the profitable side — and it is the
only market routed through a dedicated player-level synthesis (M89/M90).** That is the third
independent corroboration of the same pattern: `PRISM_ROLE_SHRINKAGE_FIX_PREREGISTRATION_V1` §2
found it the only market without compression; now it is the only one that makes money.

**`rush_rec_yards` loses 16.7% per unit with a CI excluding zero** — and it is independently the
worst projection cell in the programme (−13.7% to −23.6% bias). **Stop pricing it.** That is the
single clearest actionable result here.

¹ only 380 of 6,728 `anytime_td` rows carry usable odds, so this market is effectively unmeasured.

## 3. The `anytime_td` trap — an 80% win rate that loses money

Graded on win rate alone, `anytime_td` shows **80.47%** across 5,520 rows and would have been the
headline finding. It is an artifact:

- `vegas_line` is **always 0.5**;
- `model_proj` is a probability — mean **0.187**, median 0.162;
- so **the model says UNDER on 94% of rows**;
- median `vegas_over_odds` is **+500**, putting the under near −500/−600, where **break-even is
  83–86%.**

**We win 80.47% on a bet that needs ~84%. The 80% "edge" is a losing position.** Any future
report that cites a win rate without the matching price is not reporting an edge.

## 4. There is no weekly trend, and the edge is not in large disagreements

**By week** — the "improving every week" reading does not survive pricing:

| wk | win% | EV/$1 | 95% CI |
|---|---|---|---|
| 1 | 52.44% | −0.0191 | [−0.0649, +0.0270] |
| 2 | 50.30% | **−0.0532** | **[−0.0985, −0.0074]** |
| 6 | 52.49% | −0.0330 | [−0.0816, +0.0170] |
| 7 | 54.83% | **+0.0260** | [−0.0194, +0.0713] |
| 8 | 52.88% | −0.0265 | [−0.0742, +0.0193] |

**No week is significantly positive.** W7 was the good week; W8 gave it back. Win rate rose
W2→W7, which is what produced the "trending up" impression, but **EV did not follow.**

**By disagreement size** — my own hypothesis from the previous exchange is **dead**:

| bin | mean gap | win% | EV/$1 | |
|---|---|---|---|---|
| Q1 closest | 3.0% | 52.96% | −0.0038 | |
| Q2 | 9.3% | 49.66% | **−0.0661** | **LOSING** |
| Q3 | 17.3% | 53.38% | −0.0094 | |
| Q4 | 26.7% | 52.16% | −0.0303 | |
| Q5 furthest | 87.8% | 54.32% | +0.0019 | flat |

**Non-monotone, and the furthest bin is flat, not profitable.** I proposed last exchange that the
edge was carried by large disagreements with Vegas and that fixing the projections would shrink
exactly the disagreements that pay. **That explanation is now falsified.** Q2 is the significant
loser; Q5 is break-even.

**By side:** OVER −0.0333 `[−0.0675,+0.0008]`, UNDER −0.0137 `[−0.0412,+0.0134]`. Neither clears.

## 5. The strategic reversal — the A4 protection was guarding a liability

I advised holding the shrinkage fix (A4) because its **−1.13pp side-accuracy cost** might break
the betting edge. That advice rested on an edge that measures **negative**.

**And A4 provably cannot touch the one profitable market.** A4 changes the `tgt_share` and
`rush_share` anchors; `pass_yards` is priced from `ypa` through the M89/M90 synthesis, and
`PRISM_ROLE_SHRINKAGE_FIX_PREREGISTRATION_V2`'s **G5 verified `ypa` stays bit-identical under
every arm (0.000e+00)**.

| | EV/$1 | n |
|---|---|---|
| `pass_yards` — **A4 cannot affect it** | **+0.0824** | 554 |
| `rec_yards` + `receptions` — what A4 affects | **−0.0248** | 5050 |

**A4's entire side-accuracy cost falls on markets that already lose money, and it is structurally
incapable of damaging the market that makes money.** The P-versus-S tension I treated as a reason
for caution is much weaker than I presented it.

**This does not make A4 shippable.** It makes the *objection* weak. A4 still needs the W9 holdout
and a production validation run. But "it might break the edge" is no longer a reason to hold it.

## 6. What this does and does not say about actual betting results

**It says the board, bet flat and in full, is −2.1% per unit.**

**It does not say the user has been losing.** Real bets were a **selected subset** chosen with
reasoning, concentrated in specific plays — not the full board bet blind. A selective bettor can
be positive while the underlying board is negative, and nothing here measures that. **Any claim
about realized personal results needs the actual bet log, which this study does not use.**

## 7. What follows

1. **Stop pricing `rush_rec_yards`** — −16.7% per unit, CI excluding zero, worst projection cell.
   No study needed.
2. **Report EV, never win rate, from here on.** Every historical edge claim in this repository was
   computed without prices and should be treated as unvalidated until recomputed.
3. **`pass_yards` is the template.** The only market with a dedicated player-level synthesis is the
   only market without compression and the only one that profits. **That is the strongest argument
   yet for the player-identity direction** — and it suggests the next build is a `rec_yards`
   synthesis in the M89/M90 mould, not another estimator tweak.
4. **Revisit A4 after W9** (§5), on its merits rather than against a phantom edge.

No odds were fetched — all prices came from the committed boards. No production code changed.
