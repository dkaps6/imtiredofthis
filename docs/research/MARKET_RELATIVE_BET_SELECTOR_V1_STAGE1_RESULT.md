# Market-Relative Bet Selector V1 — Stage 1 Result

**STATUS: STAGE 1 COMPLETE — NO MARKET PASSED — DO NOT ADVANCE THESE MARKETS TO STAGE 2.**

Canonical execution:
- run: `36800239299`
- head: `a7e271e43eb8d2aa770ce0ce079baf5c908aa963`
- artifact: `11135730047`
- digest: `sha256:1a48ca7c1b333a735b5348cc87a49cbd385e9d80e983bb87613858130f09976c`

Frozen design:
- `MARKET_RELATIVE_BET_SELECTOR_V1_PLAN.md`
- `MARKET_RELATIVE_BET_SELECTOR_V1_PRE_RESULT_AMENDMENT.md`

No 2026 Weeks 1-3 outcomes were used to fit this study. No Week-4+ outcomes were opened.
No paid data or OddsAPI acquisition was used.

## Source / execution integrity

All source and mechanics gates passed:
- exact QB authority artifact hash verified;
- exact TE-R5P authority artifact hash verified;
- leakage-safe 2024/2025 schedules rebuilt;
- exact authority source rows built;
- free historical DK/FD market archive rebuilt;
- Stage-1 mechanics tests passed;
- both holdout directions executed once;
- evidence bundle uploaded.

Matched scoreable rows:
- QB pass yards: 2024 n=407, 2025 n=364;
- TE receiving yards: 2024 n=679, 2025 n=667;
- TE receptions: 2024 n=637, 2025 n=679.

WR remained source-blocked by the frozen WR-R15 2025-confirmation prohibition.
RB remained source-blocked because no 2024-2025 promoted retrospective authority exists.

## Results

### QB passing yards

Fit 2024 -> test 2025:
- beta raw/constrained: **0.41619**
- market baseline MAE: **51.2060**
- candidate MAE: **51.2465**
- paired improvement: **-0.0404 yards**
- game-cluster bootstrap 95% CI: **[-0.9439, +0.8447]**
- disposition: **DIRECTION_FAIL**

Fit 2025 -> test 2024:
- beta raw/constrained: **0.18443**
- market baseline MAE: **55.2396**
- candidate MAE: **54.8465**
- paired improvement: **+0.3931 yards**
- bootstrap 95% CI: **[+0.0056, +0.7780]**
- disposition: **DIRECTION_PASS**

Because both directions were required, QB terminal Stage-1 disposition is:
**NO_VERIFIED_INCREMENTAL_MODEL_LEVEL_SIGNAL_V1**.

### TE receiving yards

Fit 2024 -> test 2025:
- beta raw: **-0.12829**
- constrained beta: **0.0**
- candidate equals market anchor;
- disposition: **DIRECTION_FAIL**

Fit 2025 -> test 2024:
- beta raw: **-0.07014**
- constrained beta: **0.0**
- candidate equals market anchor;
- disposition: **DIRECTION_FAIL**

Terminal:
**NO_VERIFIED_INCREMENTAL_MODEL_LEVEL_SIGNAL_V1**.

The negative raw coefficients are recorded but may not be inverted or rescued under the frozen plan.

### TE receptions

Fit 2024 -> test 2025:
- beta raw: **-0.11856**
- constrained beta: **0.0**
- disposition: **DIRECTION_FAIL**

Fit 2025 -> test 2024:
- beta raw/constrained: **0.06674**
- market baseline MAE: **1.64992**
- candidate MAE: **1.64607**
- paired improvement: **+0.00385**
- bootstrap 95% CI: **[-0.00208, +0.00961]**
- disposition: **DIRECTION_FAIL**

Terminal:
**NO_VERIFIED_INCREMENTAL_MODEL_LEVEL_SIGNAL_V1**.

## Interpretation

The simple downstream fair-line architecture is not supported.

On the exact scoreable current-authority historical populations, the model-market
level disagreement does not add stable two-directional held-out information
beyond the market consensus line.

This directly means:
- do not use `model projection - Vegas line` magnitude as a trustworthy betting
  confidence variable;
- do not create a market-relative fair line from a fitted scalar beta for these
  markets;
- do not advance these markets to Stage 2 probability/EV construction;
- do not rescue QB from the one favorable holdout direction;
- do not invert TE's negative beta;
- do not tune a new threshold.

This is a selector result, not a claim that the football model has no value.
It says the tested scalar level disagreement is not a stable betting edge beyond
the market.

The broader bet-selection problem remains open, but the next architecture must
be genuinely different and must pass anti-retest review before any outcome
fitting.
