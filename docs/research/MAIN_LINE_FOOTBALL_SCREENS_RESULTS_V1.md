# Do football screens pick the winning main-line bets within each category? RESULTS V1

**Date:** 2026-11-22.

**Preregistration:**
- `MAIN_LINE_FOOTBALL_SCREENS_PREREGISTRATION_V1.md`;
- frozen in commit 56c3b45 before scoring.

**Script:** `scripts/research/evaluate_main_line_football_screens_v1.py`.

**Rows:** `data/research/main_line_screens/`.

**Verdict:**
- **Rushing UNDER: HOLDS** (the only cell that does).
- Every other market x side: **NO EFFECT**. Within receiving yards, receptions and passing
  yards, the six football screens did not pick out winners.

## The result

Each cell compares **SCREENED** (all six screens pass) with every model pick in that cell. ROI
is at the posted price.

| market / side | 2026 all | 2026 screened | 2024-25 all | 2024-25 screened | 2024 / 2025 screened |
|---|---|---|---|---|---|
| **rush_yards UNDER** | 57.6%, +8.2% (n=283) | **69.8%, +31.0% (n=43, 5/5 weeks)** | 0.0% (n=1854) | **54.5%, +2.2% (n=246)** | -0.5% / +4.9% |
| rush_yards OVER | -13.8% | +4.0% (n=9) | -11.2% | +11.8% (n=35) | too few bets to judge |
| receptions UNDER | -1.6% | -2.0% (n=94, 1/5 weeks) | -1.2% | +3.3% (n=577) | +0.8% / +6.5% |
| receptions OVER | -8.9% | -7.9% (n=30) | +0.6% | -4.9% (n=67) | |
| rec_yards UNDER | -3.7% | **-14.1%** (n=70) | -3.2% | +0.7% (n=561) | |
| rec_yards OVER | -6.4% | **-21.7%** (n=60) | -3.9% | -3.0% (n=135) | |
| pass_yards UNDER | -0.2% | -29.5% (n=8) | -4.2% | -9.4% (n=135) | |
| pass_yards OVER | +6.1% | -17.8% (n=16) | -14.6% | +10.7% (n=5) | |

**How to read it:**
- **Rushing unders:** the screens add something real. They are strongly positive this season,
  and they lift a flat historical cell to +2.2%, positive or near zero in both past seasons.
  Expect the historical number, not 70%.
- **Receiving and passing:** the screened bets did **worse** than the unscreened ones in 2026.
  - Receptions unders were positive on 2024-25 (+3.3%) but not this season, so that is a mixed
    result, not a rule.
  - There is no evidence that any combination of these screens picks winners in those
    categories.
- **Rushing overs** look good screened, but with 9 and 35 bets that is not a finding.

## Applied to 2026 week 11 Sunday (FanDuel, captured 07:50 ET)

Of 390 FanDuel main lines, **50 pass all six screens**. Only the rushing-under cell has
evidence behind it:

| player | line | proj | team spread |
|---|---|---|---|
| Bhayshul Tuten (JAX) | U70.5 | 48.4 | JAX -6 |
| Chuba Hubbard (CAR) | U75.5 | 57.2 | CAR -3 |
| Jahmyr Gibbs (DET) | U84.5 | 58.6 | DET -3.5 |
| J.J. McCarthy (NYG, QB) | U16.5 | 11.0 | NYG +6 |

**What the screens reject** (the screens are frozen; this is what they say, not a judgment):
- Derrick Henry (F6): CAR's run defense is soft.
- Saquon Barkley (F1): 39% of snaps.
- Jordan Mason (F1).
- Cam Skattebo (F2).

The 50 screened legs in the other categories are listed in the rows file. They carry **no**
evidence of an edge.
