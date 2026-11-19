# RB allocative prior on a pregame universe: RESULTS V1

**Date:** 2026-11-19.

**Preregistration:** `RB_ALLOCATIVE_PRIOR_PREGAME_PREREGISTRATION_V1.md`, frozen at 20b2b1a.

**Script:** `scripts/research/evaluate_rb_allocative_prior_pregame_v1.py`.

**Rows:** `data/research/rb_allocative_prior_pregame_v1{,_tables,_rows}.csv`.

**Verdict: K3 FIRED. Not a pass. The candidate is mis-specified as an RB-only change.** The
primary result is large and clean. The mechanism behind K3 is identified, and it points
directly at the corrected design.

## Primary: expected RB carries vs actual, pregame universe, production chain

| | 2025 (wk 4-18) | 2026 (wk 4-10) |
|---|---|---|
| RB player-games | 1347 | 600 |
| MAE, C0 production | 3.256 | 3.457 |
| **MAE, C1 allocative** | **2.812** | **2.904** |
| **gain [95% CI]** | **+0.444 [+0.371, +0.520]** | **+0.553 [+0.415, +0.690]** |
| C2 = C1 + recency (reported) | 2.740 (+0.516) | 2.833 (+0.624) |

This is a **14-16% cut in RB carry error**, about six to ten times the recency gain.

| subset | 2025 C0 → C1 MAE (gain CI) | bias C0 → C1 | 2026 C0 → C1 MAE (gain CI) | bias C0 → C1 |
|---|---|---|---|---|
| lead back (k=1) | 4.78 → 3.96 [+0.67, +0.95] | +3.74 → +2.13 | 5.23 → 4.27 [+0.74, +1.18] | +4.12 → +2.32 |
| committee (K≥3, k≥2) | 2.48 → 2.21 [+0.19, +0.35] | -0.20 → +0.59 | 2.55 → 2.20 [+0.21, +0.49] | -0.06 → +0.95 |

**Gates:**
- **K1, not fired.**
- **K2, not fired:** committee backs improve.
- **K3, FIRED:** team RB mass error |projected - actual| **worsens** in both holdouts:
  - 2025: 2.55 → 3.08;
  - 2026: 2.75 → 3.41.

## Why K3 fired: measured, not guessed (2026 wk 4-10, per team-week)

| position | raw posterior sum, C0 → C1 | projected carries, C0 → C1 | actual |
|---|---|---|---|
| QB | 0.239 → 0.239 | **4.97 → 5.64** | 3.65 |
| RB | 0.922 → 0.823 | **20.38 → 19.41** | 22.43 |
| WR/TE | 0.035 → 0.035 | ~0 → ~0 | 0.80 |
| team total | 1.196 → 1.096 | | |

- The QB block is already over-allocated in production: the QB prior of 0.14 applies to
  every rostered QB, so the block posterior is 0.239 against an actual share of about 0.135.
- C1 makes **only the RB block** coherent, at 0.814 of team carries.
- The team total stays above 0.95, so the cap still scales. The RBs' share of the capped
  budget falls, and **carries move from backs to quarterbacks.**

This is the mass-budget study's "team conservation" law
(`PRISM_RB_RUSH_SHARE_MASS_BUDGET_V1.md` §5) showing up as a measured failure. **An allocative
prior must be allocative for every position block at once.** Fixing one block inside a shared,
capped budget moves the error to the others.

## Fitted table (training 2019-2025, applied to 2026)

Within-RB share by depth K and pregame rank k:
- 2 backs: 0.669 / 0.331.
- 3 backs: 0.671 / 0.247 / 0.082.
- 4 backs: 0.618 / 0.263 / 0.089 / 0.030.

`rb_block` = 0.814 of team carries. The box-score study's top-1 values were 0.706 / 0.628 /
0.563 at K = 2/3/4. The pregame table puts **more** on the top back at K ≥ 3, because the
extra pregame backs are mostly backs who never play.

## Prediction scored

- "C1 wins clearly, well outside noise": **right.**
- "Lead-back bias falls by more than half": **wrong**, it falls about 43-44%.
- "Team RB mass moves toward actual": **wrong.** It moves away, for the reason above.
- **Named risk** (non-playing backs dilute the gain to noise): **did not happen.** The gain
  was larger than feared. The real risk was a different one: the cross-block budget.

## Disposition

**Not passed. Nothing promoted. No production code changed.**

The corrected design is the **allocative prior for every position block**: QB, RB, WR and TE
each get depth × rank tables and block masses fitted the same way. It needs its own frozen
preregistration (`ALLOCATIVE_PRIOR_ALL_BLOCKS_PREGAME_PREREGISTRATION_V1.md`), and it must keep
the same K1-K3, so K3 tests the very failure found here. V1 is not rescued by tuning: the
corrected design adds the same structure to the other blocks and no new free parameters.

No sportsbook variable entered any arm.
