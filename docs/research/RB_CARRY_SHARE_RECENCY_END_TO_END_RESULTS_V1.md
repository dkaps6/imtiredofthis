# RB carry-share recency, end to end: RESULTS V1

**Date:** 2026-11-19.

**Preregistration:** `RB_CARRY_SHARE_RECENCY_END_TO_END_PREREGISTRATION_V1.md`, frozen at commit
8e494d8 before any arm ran.

**Script:** `scripts/research/evaluate_rb_recency_end_to_end_v1.py`.

**Rows:**
- `data/research/rb_recency_end_to_end_v1.csv` (summary)
- `data/research/rb_recency_end_to_end_v1_rows.csv`

**Verdict: PASS as preregistered.** The gain survives Bayes, top-5 and the 0.95 cap. **It is
small.** Candidate only, not promoted.

## Result: expected RB carries vs actual carries, a = 0.85 frozen from the parent

| | 2025 (wk 4-18) | 2026 (wk 4-10) |
|---|---|---|
| RB player-games | 1347 | 600 |
| MAE, C0 production chain | 3.256 | 3.457 |
| MAE, C1 with the last-3 input | 3.183 | 3.411 |
| **gain [95% CI, team-game clustered]** | **+0.073 [+0.035, +0.112]** | **+0.046 [+0.013, +0.078]** |
| lead back MAE, C0 → C1 | 4.683 → 4.583 | 5.157 → 5.051 |
| lead back gain CI | [+0.030, +0.169] | [+0.044, +0.170] |
| lead back bias (actual - projection), C0 → C1 | +3.36 → +3.29 | +3.76 → +3.75 |
| shape: within-team RB share MAE, C0 → C1 | 0.1328 → 0.1287 | 0.1356 → 0.1337 |
| team RB carries: projection C0 / C1 / actual | 18.6 / 18.8 / 22.1 | 18.5 / 18.7 / 22.4 |

**Gates:**
- **K1, not fired:** both CIs exclude 0.
- **K2, not fired:** lead backs improve in both holdouts, CIs excluding 0.

## The offline chain is faithful to production where it can be checked

It reproduces the production level defects that earlier studies measured on the real
simulator:
- **Team RB mass:** about 3.5 carries per team too low. `PRISM_RB_RUSH_SHARE_MASS_BUDGET_V1.md`
  measured 19.77 vs 23.15 on the W7 production universe.
- **Lead backs:** under-projected by about 3.4-3.8 carries.

The deficit has the same size and direction, which supports this as a faithful reproduction,
not a different model.

## Size: why "small" is the honest word

- **Relative to error:**
  - At the input level (parent), the gain was 3.8% of MAE.
  - End to end, it is **2.2% (2025) and 1.3% (2026)**.
  - In yards, about 0.2-0.3 yards of rushing MAE per back.
- **Why it shrinks:** Bayes gives the current-season input weight equal to its game count,
  while the position prior gets 3 and last season up to 6. The window change is diluted, as
  predicted.
- **The level defects are an order of magnitude larger** than anything this fixes:
  - team mass is -3.5 carries;
  - lead-back bias is +3.4.
  - Recency does not fix them, and was not designed to.

## Prediction scored

- "C1 still wins, by less than at the input level": **right** (3.8% → 2.2% / 1.3%).
- **Named risk** (dilution makes it noise, K1): **did not happen.** The CIs exclude 0 in both
  holdouts.

## What this means for the lane

1. Recency is a **valid, small** improvement that survives the production chain.
   - It should ride along with the larger fixes, not ship alone.
   - Promotion would need a real production replay with the Ourlads universe and the INERT
     concentration transform in place.
2. **The prize is still the budget and shrinkage defects:** mass -3.5 carries per team, lead
   backs +3.4. The two coupled changes designed in `PRISM_RB_RUSH_SHARE_MASS_BUDGET_V1.md`
   Addendum 3 are allocative priors and evidence-precision shrinkage weights. **They have
   never been preregistered or tested.** This offline chain gives them a validated harness.
3. No sportsbook variable was used. No production code changed.
