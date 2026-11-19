# RB carry-share recency: RESULTS V1

**Date:** 2026-11-19.

**Preregistration:** `RB_CARRY_SHARE_RECENCY_PREREGISTRATION_V1.md`, frozen at commit caa2234
before any arm ran.

**Script:** `scripts/research/evaluate_rb_carry_share_recency_v1.py`.

**Rows:** `data/research/rb_carry_share_recency_v1.csv`.

**Verdict: PASS as preregistered. Candidate only, not promoted.**

## Result

The best weight on the last 3 games, fitted on 2023-2024 only, is **a = 0.85**.

| holdout | n | PlayerForm today (C0) | + last-3 (C1) | gain | 95% CI (game-clustered) | oracle |
|---|---|---|---|---|---|---|
| **2025** | 1030 | 0.1133 | 0.1090 | **+0.0043** | **[+0.0016, +0.0070]** | 0.0962 |
| **2026 wk1-10** | 402 | 0.1326 | 0.1271 | **+0.0055** | **[+0.0022, +0.0088]** | 0.1028 |

| subset | 2025 gain [CI] | 2026 gain [CI] |
|---|---|---|
| lead backs (C0 ≥ 0.40) | +0.0018 [-0.0025, +0.0063] | **+0.0070 [+0.0013, +0.0130]** |
| role change (\|last-3 - season-to-date\| ≥ 0.15) | +0.0180 [-0.0146, +0.0522], n=52 | **+0.0560 [+0.0186, +0.0925]**, n=16 |
| steady roles | **+0.0036 [+0.0014, +0.0059]** | **+0.0034 [+0.0005, +0.0061]** |

**Gates:**
- **K1, not fired:** the gain CI excludes 0 in both holdouts.
- **K2, not fired:** lead backs are not worse in either holdout. They improve significantly
  in 2026.

**Headroom.** The oracle is the player's full-season mean, which uses future games, so it only
shows how much error could be removed. C1 captures about **25%** of the gap between C0 and the
oracle in 2025 (0.0043 of 0.0171) and about **18%** in 2026 (0.0055 of 0.0298).

**Team sums.** C1's summed shares sit closer to the actual team total (2025: 0.771 vs actual
0.771, C0 0.754; 2026: 0.703 vs 0.745, C0 0.690). Recency does not leak opportunity.

## Validity checks run after the first pass

- **Name-join defect, found and fixed before this write-up.**
  - The first run joined snap counts to stats by name.
  - That zeroed 22 RB season-teams, including Kenneth/Kenny Gainwell, Brian Robinson Jr.,
    Ollie Gordon II and Chase Brown.
  - Re-joined on pfr → gsis IDs (`load_players`, 99.8% coverage): 1 residual zero.
  - The verdict did not change: a went from 0.90 to 0.85, and both CIs still exclude 0.
- **Strictly prior.**
  - Season-to-date and last-3 shift one game.
  - The prior-season share comes from the previous season.
  - The oracle is reported only.
- **No tuning on holdouts.** `a` was fitted on 2023-2024 only, and the grid was preregistered.

## Prediction scored

- "C1 wins": **right**.
- "a lands in 0.4-0.7": **wrong**, it is 0.85. Recent games deserve even more weight than
  predicted.
- "Gain concentrated in role-change rows, small overall": **half right.**
  - Role-change rows gain the most per row.
  - Steady rows also gain, with CIs that exclude 0 in both holdouts.
  - **The named risk** (steady roles add noise and wash out the gain) **did not happen.**

## What this does and does not establish

- PlayerForm's RB carry share should include a recent-games component. This is measured
  against production's own formula, on two blind holdouts.
- **It does not establish the end-to-end effect** on rush-yards projections. Two things sit
  downstream of PlayerForm:
  - the Bayesian layer, which over-shrinks lead backs by about 9-11pp
    (`PRISM_RB_CARRY_CONCENTRATION_RESULTS_V1.md`);
  - the simulation's cap and residual bucket (`PRISM_RUSH_ALLOCATION_RESULTS_V1.md`).

  Both can absorb or distort an input improvement.
- **Next, if pursued:** an end-to-end test of C1 inside the projection, combined with the
  carry-concentration fix, which ships INERT. The test is projected carries and rush yards
  vs actual, graded per player (AGENTS.md player-first principle). It needs its own
  preregistration.
- No sportsbook variable was used anywhere.
