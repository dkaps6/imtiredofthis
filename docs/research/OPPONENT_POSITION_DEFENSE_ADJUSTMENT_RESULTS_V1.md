# Opponent defense by position: results V1

**Date:** 2026-11-22.

**Preregistration:**
- `OPPONENT_POSITION_DEFENSE_ADJUSTMENT_PREREGISTRATION_V1.md`;
- frozen in commit c8cdb98.

**Script:** `scripts/research/evaluate_opponent_position_defense_v1.py`.

**Rows:** `data/research/opponent_position_defense/`.

## 1. The preregistered test (the model's real projections, 17,740 rows)

| market / group | fold A (2024 → 2025) MAE gain | fold B (→ 2026) MAE gain | P(improve), B | verdict |
|---|---|---|---|---|
| **receptions / RB** | +0.000 (b 0.28) | **+0.008** (b 0.19) | **0.988** | **PROMOTE** |
| rush_yards / RB | -0.184 | +0.368 | 0.988 | no: flips sign |
| rush_yards / QB | +0.064 (2026 only, n=75) | | 0.686 | no: too few rows |
| rec_yards / TE | +0.062 | -0.123 | 0.24 | no |
| rec_yards / WR | -0.011 (b < 0) | +0.002 | 0.56 | no |
| rec_yards / RB | +0.038 | -0.006 | 0.43 | no |
| receptions / TE | +0.006 | -0.011 | 0.12 | no |
| receptions / WR | -0.000 | +0.000 | 0.59 | no |
| pass_yards / QB | +0.756 | -0.574 | 0.14 | no |

- **Only RB receptions passes all three gates,** with a small gain (MAE 1.534 → 1.526).
- `b` is positive almost everywhere (a weaker defense means more yards), but the gain does not
  hold out of sample. The ensemble and the market already carry most of it.
- **Promoted:** `receptions | RB`, `b = 0.1932`, shrink k = 4
  (`promoted_coefficients_v1.json`). Production integration is the next step.

## 2. QB rushing, the case that triggered this (every QB start, 2022-2026)

Only 75 QB-rushing projections had lines, so it was tested directly:
- **Starts:** ≥ 15 attempts.
- **Baseline:** his own prior rushing (current season plus last season at half weight, at least
  3 games).
- **Fit:** 2022-2024, 1,331-1,349 starts.
- **Test:** 2025-2026, 767-773 starts.

| measure | test MAE gain | corr with residual (test) |
|---|---|---|
| raw QB rushing yards allowed by the opponent | **-0.06** (worse) | |
| opponent effect on QBs' own norms (QB-adjusted) | **-0.02** (worse) | **-0.01** |
| team spread | | -0.04 |
| game total | | -0.00 |

**QBs facing the stingiest fifth of defenses** (QB-adjusted) ran 2.0 yards below their own norm.
**Against the softest fifth:** 2.2 below. There is no pattern. MIN entered week 11 as the
2nd-stingiest by the QB-adjusted measure, and that measure does not predict.

**What does predict QB rushing is the QB's own norm:**

| own norm (yds/start) | hit 10+ | 0 or less |
|---|---|---|
| < 8 | 24% | 40% |
| 8-15 | 46% | 19% |
| 15-25 | 63% | 11% |
| 25+ | 87% | 2% |

## 3. What changes

1. **RB receptions:** the opponent adjustment is promoted into the model. Implementation,
   tests and a pull request are next.
2. **QB rushing:**
   - The opponent's defense against QB runs is **not** a usable signal (2,100+ starts).
   - A QB-rushing alt leg is decided by the QB's own norm and the rung's distance below it.
   - A 15-25-yard-norm QB posts 0 or less about 1 time in 9, and no pregame measure tested here
     flags those games.
   - The G7 position-group fix (commit 246d8f6) keeps comparisons like-for-like. For QB rushing
     it is not predictive.
3. **Every other cell:** opponent yards allowed by position adds nothing reliable to the
   model's projections. Do not reopen it without a different measure (for example
   per-snap/per-route allowed, or coverage scheme).
