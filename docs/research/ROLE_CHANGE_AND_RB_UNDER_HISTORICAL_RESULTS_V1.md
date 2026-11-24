# Role-change rule + RB-under rule on unseen 2024-2025 lines: RESULTS V1

**Date:** 2026-11-20.

**Preregistration:** `ROLE_CHANGE_AND_RB_UNDER_HISTORICAL_PREREGISTRATION_V1.md`, frozen at 933786d.

**Script:** `scripts/research/evaluate_role_change_rb_under_historical_v1.py`.

**Rows:** `data/research/historical_rules/`.

**Data:** 17,786 graded 2024-2025 prop lines, 93.7% matched to gsis.

## Test A: the frozen RB-under rule FAILS on unseen seasons

| | bets | hit | ROI | 95% CI (game-clustered) |
|---|---|---|---|---|
| 2024 | 113 | 60.2% | +12.6% | |
| **2025** | 111 | **47.7%** | **-9.8%** | |
| 2024-2025 | 224 | 54.0% | +1.5% | [-10.8%, +13.5%] |

- The preregistered bar was above breakeven in **both** seasons. **2025 fails.**
- By flag count (line ≥ 30):

  | flags | hit |
  |---|---|
  | 0 | 46.6% |
  | 1 | 52.8% |
  | 2 | 56.7% |
  | 3 | **43.2%** |

  The count does not rank outcomes monotonically.
- **The 2026 discovery result (66%, 83 bets) did not generalize. Out of sample, the rule is
  about breakeven.** The forward tracker keeps running, with the expectation reset to "no
  demonstrated edge".

## Test B: the role-change rule FAILS

| | bets | hit | ROI |
|---|---|---|---|
| RB rising role → OVER | 62 | 46.8% | -11.7% |
| RB falling role → UNDER | 25 | 48.0% | -9.7% |
| WR/TE rising targets → OVER | 60 | 48.3% | -9.4% |
| WR/TE falling targets → UNDER | 21 | 42.9% | -18.3% |

**Recent role swings do not beat the line,** even though the line weights them only about
0.04.

## Test C: overs lose in every market, in both seasons

| market | OVER hit / ROI | UNDER hit / ROI |
|---|---|---|
| rush_yards (2772) | 46.7% / **-11.7%** | 53.3% / -0.8% |
| receptions (5712) | 47.7% / -9.3% | 52.3% / -3.0% |
| rec_yards (5863) | 49.4% / -7.3% | 50.6% / -4.7% |
| pass_yards (813) | 48.5% / -9.0% | 51.5% / -2.9% |
| rush_rec_yards (2626) | 49.5% / -7.3% | 50.5% / -5.4% |

With 2026 (`WHAT_WE_KNOW_2026_W11.md` §1), that makes **three seasons in which blind overs lose
7-12% and blind unders lose about the vig.** It is the most robust market fact in the program.

## Prediction scored

- **A "positive in both seasons":** wrong. 2025 was -9.8%.
- **B "falling-role under positive":** wrong.
- **Receivers "no edge":** right.
- **C "overs lose in both":** right.
- **Named risk** (DraftKings vs FanDuel, name matching): 93.7% matched, so matching is not
  the explanation.
