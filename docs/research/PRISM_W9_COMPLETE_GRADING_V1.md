# Week 9 2026 — complete grading of the alt board and the shipped tickets

**Date:** 2026-11-10 · **Status:** closed. Every week-9 game is final.

**Rows (durable, committed — not artifacts):**
- `data/market_track_record/graded/w9_2026_alt_board_graded_complete_v1.csv` — 5,115 graded rungs
- `data/market_track_record/graded/w9_2026_gated_survivors_graded_v1.csv` — the 76 gate survivors

**Coverage.** The captured alt board covered **13 of the slate's 15 games**. Two games
never had alt coverage at capture time (the Thursday game and the 09:30 London
window, which kicked before the 09:08 capture). All 13 covered games are now graded.
An earlier pass in this chat graded 11 of them with SNF and MNF still pending; this
document supersedes those numbers.

---

## 1. The two late games made every conclusion worse

| | 11 games | **all 13 covered** |
|---|---|---|
| rungs | 4,306 | **5,115** |
| ROI | -13.4% | **-15.7%** |

The 809 rungs from TB@CHI and BUF@MIN graded at **-28.1%**, roughly double the
board's loss rate. Both are low-total defensive games, which is the environment the
board handles worst: `pass_yards` -37.7% and `rush_yards` -38.6% inside those two
games alone.

This is the second time in this project that a conclusion drawn on a partial slate
moved materially once the slate completed. Partial-slate numbers are not to be
written into a document as findings again.

## 2. Per-market, complete board

Signed error is `actual - table`; **positive means the table UNDERSTATES.**

| market | n | table says | actual | signed | ROI | units |
|---|---|---|---|---|---|---|
| `rec_yards` | 1651 | 0.548 | 0.581 | **+0.0329** | **-0.2%** | -4.0u |
| `pass_yards` | 781 | 0.429 | 0.458 | **+0.0298** | -6.8% | -53.3u |
| `rush_yards` | 818 | 0.537 | 0.482 | -0.0557 | -22.0% | -180.2u |
| `rush_rec_yards` | 379 | 0.508 | 0.441 | **-0.0673** | -28.8% | -109.0u |
| `receptions` | 1486 | 0.331 | 0.308 | -0.0225 | -30.7% | -455.8u |
| **all** | **5115** | | **0.457** | | **-15.7%** | **-802.4u** |

**`corr(signed error, ROI) = +0.860`** across the five markets. The ranking rule from
`PRISM_ALT_TABLE_SUPPORT_AUDIT_V1` survives the complete board intact: the two
markets where the table understates are the two least-bad, and all three markets
where it overstates are ruinous.

**What does not survive is the profitability claim.** `rec_yards` was recorded at
+2.9% on the partial slate and is **-0.2%** complete; `pass_yards` was -0.7% and is
**-6.8%**. At table face value, *no market on this board returned a profit.* The sign
rule orders markets correctly. It does not certify any of them as bettable. See the
addendum appended to that audit.

## 3. Depth and price: the two cliffs are confirmed

| `frac_of_line` | n | hit | ROI |
|---|---|---|---|
| <0.85 | 1671 | 74.7% | -4.3% |
| 0.85-1.15 | 1403 | 49.0% | -7.3% |
| 1.15-1.5 | 1293 | 26.4% | -18.7% |
| 1.5-2.0 | 322 | 12.4% | **-44.2%** |
| >2.0 | 426 | 4.5% | **-57.3%** |

Rungs beyond 1.6x: **-53.5%** (n=701), against -55.1% cited in the audit. Rule holds.

| price | n | hit | ROI |
|---|---|---|---|
| <=-300 | 1031 | 80.2% | -5.0% |
| -300..-150 | 846 | 64.7% | -4.1% |
| -150..0 | 807 | 48.9% | -9.5% |
| 0..+150 | 641 | 43.7% | -2.6% |
| +150..+300 | 772 | 26.8% | -19.2% |
| **>+300** | **1018** | **7.9%** | **-46.5%** |

**ROI is negative in every price band.** The board has no profitable region at face
value — only differently-bad ones. The long end is where it dies: >+300 rungs hit
7.9% against a 25% breakeven at +300.

## 4. Decile calibration is actually decent — and that is the point

| predicted | n | pred | actual | err |
|---|---|---|---|---|
| 0.0-0.1 | 495 | 5.2% | 1.6% | +3.6 |
| 0.1-0.2 | 442 | 14.8% | 12.2% | +2.6 |
| 0.2-0.3 | 583 | 26.9% | 23.0% | +3.9 |
| 0.3-0.4 | 433 | 35.1% | 32.1% | +3.0 |
| 0.4-0.5 | 851 | 45.8% | 45.5% | +0.4 |
| 0.5-0.6 | 700 | 54.6% | 56.6% | -2.0 |
| 0.6-0.7 | 608 | 65.2% | 66.4% | -1.3 |
| 0.7-0.8 | 596 | 74.8% | 78.5% | -3.7 |
| 0.8-0.9 | 356 | 85.0% | 84.6% | +0.4 |
| 0.9-1.0 | 51 | 91.9% | 88.2% | +3.6 |

No decile is off by more than 4 points. The probabilities are **not** wildly broken.
The board loses 15.7% anyway, because a 3-to-4-point overstatement at the bottom of
the distribution is fatal when it is sold at +300 and longer: predicted 5.2% against
an actual 1.6% is a 3.6-point error and a ruinous bet. **Small calibration errors in
the tail do all the damage, and that is exactly where long tickets are built.**

## 5. The nine-gate builder: the one thing that worked

Measured on the same pool, FanDuel only, inside the builder's own price window, so
this is a paired comparison on offers that were genuinely takeable:

| | n | hit | ROI |
|---|---|---|---|
| blind (book + price only) | 1162 | 64.4% | **-5.3%** |
| **gated (all nine gates)** | **76** | **88.2%** | **+4.4%** |

| market | blind n | blind ROI | gated n | gated ROI | gated hit |
|---|---|---|---|---|---|
| `rec_yards` | 528 | +0.5% | 53 | **+6.2%** | 88.7% |
| `rush_yards` | 247 | -15.8% | 23 | **+0.3%** | 87.0% |
| `pass_yards` | 108 | -2.6% | 0 | — | — |
| `receptions` | 279 | -8.0% | 0 | — | — |

Cluster bootstrap by game (13 games, 20k draws — legs inside a game are correlated,
so the game is the resampling unit):

- gated ROI **+4.4%**, 95% CI **[-1.8%, +10.5%]**
- blind ROI **-4.9%**, 95% CI [-16.7%, +3.7%]
- gap **+9.3%**, 95% CI **[-2.1%, +23.7%]**, **P(gap>0) = 93.8%**

**Read this honestly.** +4.4% on n=76 over one week is **not** distinguishable from
zero, and neither is the +9.3% gap at the 95% level. What one week supports is:
the gates move the board about nine points in the right direction, 94% of
bootstrap draws agree on the sign, and the gated set hit 88.2% against the ~86%
its prices required. That is encouraging and it is not an established edge. It needs
weeks 10+ before any of it is promotable.

**No market-exclusion change is warranted.** Both emitting markets are >= 0 gated.
Excluding `rush_yards` — which the table audit disqualifies at face value, and which
is -15.8% blind — would have discarded a non-negative. The gate list stays as it is;
only the cited numbers in the builder docstring are refreshed.

## 6. The structural problem the gate exposes

**The nine gates emit zero plus-money legs.** All 76 survivors priced between -225
and -1800, median **-590**. A three-leg card built from median gated legs pays about
**+60**.

The tier structure in use — SAFE / MODERATE / MEDIUM / HIGH / LOTTERY, with HIGH and
LOTTERY required to be >+600 — **cannot be served by the validated gate at all.**
Cushion >= 1.50 plus the main-line floor mathematically selects heavy favorites;
plus-money rungs sit above the projection by construction, which is precisely what
gate G3 exists to reject.

So the tiers split cleanly in two, and this must be said out loud on every future
card rather than discovered again:

- **SAFE and MODERATE can be gate-certified.** They will pay roughly +60 to +250.
- **HIGH and LOTTERY cannot be certified by anything validated in this repository.**
  They are built from the >+300 band that returned **-46.5%** on 1,018 rungs and from
  the >1.6x depth band that returned **-53.5%**.

A lottery ticket is not a worse version of a good bet. On this evidence it is a
different activity, and it should be labeled as uncertified speculation whenever one
is produced, with the -46.5% attached.

## 7. The shipped week-9 tickets, final

| tier | legs | result | price |
|---|---|---|---|
| SAFE | 3/3 | **WON** | +178 |
| MODERATE | 3/4 | LOST | +567 |
| MEDIUM | 3/4 (+1 void) | LOST | +1930 |
| HIGH RISK | 3/5 | LOST | +97200 |
| LOTTERY | 1/6 | LOST | +1337191 |

**1u on each of the five: -2.22u.**

The failure pattern is one leg, repeated. **Roschon Johnson killed MODERATE, MEDIUM
and HIGH** — three of the four losing tickets — on a player who is CHI RB3 at a
**15.6% snap share** and who finished with **0 receiving yards and 0 receptions**.
He is the reason gates G1 (snap floor), G4 (real alternate) and the no-shared-legs
rule exist. The gate as now written rejects every one of those three legs.

LOTTERY went 1/6 with five of its six legs in the >+300 band. That is the -46.5%
band behaving exactly as §3 says it does.

## 8. The legs recommended in chat, graded

Flat 1u, ten legs, **-0.46u**:

| leg | price | actual | result |
|---|---|---|---|
| Aaron Jones rush >30.5 | -113 | 16 | **LOST** |
| Aaron Jones rush >39.5 | +154 | 16 | LOST |
| Jordan Addison rec >39.5 | -113 | 28 | **LOST** |
| Jordan Addison rec >49.5 | +148 | 28 | LOST |
| Emeka Egbuka rec >=5 | +162 | 3 | **LOST** |
| Dawson Knox rec >=1 | -280 | 4 | WON |
| Rome Odunze rec >=3 | -194 | 4 | WON |
| Bucky Irving rush >79.5 | +178 | 97 | WON |
| Tez Johnson rec >=1 | -400 | 2 | WON |
| Kalif Raymond rec >=3 | +164 | 4 | WON |

**Both legs of the Monday ticket lost.** Aaron Jones and Jordan Addison were offered
as a two-leg card at +226 with a stated ~41% chance to cash; Jones rushed for 16
against a >30.5 line and Addison had 28 against a >39.5 line. Neither was close.
The user declined the ticket on football grounds — a low-total defensive game — and
was right. That is the third time in week 9 a user fade beat a model recommendation
(Roschon, the Egbuka rung, this card).

Note that **Addison's >19.5 rung was a gate survivor and won**, while the >39.5 rung
recommended in chat lost. Same player, same game: the gate's rung was takeable and
the chat's rung was not. The gate was available and was not used for that pick.

## 9. What this closes and what it opens

**Closed:**
- Week 9 is fully graded; rows are committed, not left in expiring artifacts.
- The table audit's ranking rule and depth rule are confirmed out-of-sample.
- The table audit's profitability claim is withdrawn (addendum appended there).
- The builder gate's market list is confirmed as-is on the complete board.

**Open:**
- The gate has **never run on a live board.** Week 9 is a retrospective fit-check on
  the week it was designed from. Weeks 10+ are the real test.
- +4.4% is one week, n=76, CI straddling zero. Not promotable.
- The tier structure needs an explicit decision: shrink the card set to what can be
  certified, or keep HIGH/LOTTERY as labeled speculation. This is a user call, not a
  modeling one.
- `model_sd` remains a live production defect (0-10% band hits 39.5%, 70-100% band
  hits 49.3%, everything collapsing toward 46%). §4 here shows the *alt* probabilities
  are better calibrated than `model_sd` is; the production dispersion is the weaker
  object and still needs widening to the measured residual spread.
