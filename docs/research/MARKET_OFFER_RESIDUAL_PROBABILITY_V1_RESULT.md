# Market Offer Residual Probability V1 — Result

**STATUS: COMPLETE — TERMINAL NULL FOR THIS ARCHITECTURE.**

Canonical scientific execution:
- run: `36801531063`
- head: `14ea9e77a9301fccb7b8523e1b01d064641c8b07`
- artifact: `11136326900`
- digest: `sha256:fca5a66648047e2c17aaf02a01de7d449d12b39effb2d478fd72e18843cc2695`

Frozen plan:
- `docs/research/MARKET_OFFER_RESIDUAL_PROBABILITY_V1_PLAN.md`
- freeze commit: `14bd0c18629a4b76620e7a0c90d2d8288f2b8659`

Two earlier workflow attempts were mechanical-only and are not scientific
results. The final execution passed tests, exact authority verification,
historical schedule construction, football-source construction, market archive
construction, the frozen two-direction study, and evidence upload.

No 2026 outcome was used to fit this study. No paid OddsAPI or paid data was
used.

## Frozen architecture

One model per market:
- StandardScaler;
- L2 LogisticRegression, C=1.0, lbfgs, max_iter=1000;
- five fixed offer-level features only:
  - signed model gap;
  - absolute model gap;
  - offer line minus cross-book median line;
  - cross-book line range;
  - book no-vig OVER probability minus 0.5;
- one player-market-game total sample weight regardless of number of books;
- exact book+line OVER/UNDER target;
- baseline = same book+line no-vig OVER probability;
- 2024 -> 2025 and 2025 -> 2024;
- 10,000 game-cluster bootstrap resamples, seed 20261001.

Both holdout directions had to pass every gate.

## QB passing yards

### Fit 2024 -> test 2025
- train offers: 809; test offers: 704
- train identities: 407; test identities: 364
- test game clusters: 240
- baseline log loss: **0.693179**
- candidate log loss: **0.689756**
- improvement: **+0.003423**
- baseline Brier: **0.250016**
- candidate Brier: **0.248372**
- baseline AUC: **0.499883**
- candidate AUC: **0.509039**
- bootstrap 95% CI for log-loss improvement:
  **[-0.006208, +0.012948]**
- disposition: **DIRECTION_FAIL**

### Fit 2025 -> test 2024
- train offers: 704; test offers: 809
- baseline log loss: **0.693116**
- candidate log loss: **0.692108**
- improvement: **+0.001008**
- baseline Brier: **0.249984**
- candidate Brier: **0.249367**
- baseline AUC: **0.500326**
- candidate AUC: **0.533275**
- bootstrap 95% CI:
  **[-0.015702, +0.017567]**
- disposition: **DIRECTION_FAIL**

Terminal:
`NO_VERIFIED_MARKET_OFFER_RESIDUAL_PROBABILITY_SIGNAL_V1`.

The point metrics are mildly encouraging, but uncertainty is much too large to
claim a repeatable betting probability advantage.

## TE receiving yards

### Fit 2024 -> test 2025
- baseline log loss: **0.694106**
- candidate log loss: **0.694853**
- improvement: **-0.000748**
- baseline Brier: **0.250472**
- candidate Brier: **0.250842**
- baseline AUC: **0.471213**
- candidate AUC: **0.517073**
- bootstrap 95% CI:
  **[-0.007690, +0.006070]**
- disposition: **DIRECTION_FAIL**

### Fit 2025 -> test 2024
- baseline log loss: **0.692961**
- candidate log loss: **0.705074**
- improvement: **-0.012114**
- baseline Brier: **0.249907**
- candidate Brier: **0.255402**
- baseline AUC: **0.513154**
- candidate AUC: **0.528242**
- bootstrap 95% CI:
  **[-0.023852, -0.000233]**
- disposition: **DIRECTION_FAIL**

Terminal:
`NO_VERIFIED_MARKET_OFFER_RESIDUAL_PROBABILITY_SIGNAL_V1`.

This is a substantive negative, including significant log-loss harm in the
2025-fit -> 2024-test direction.

## TE receptions

### Fit 2024 -> test 2025
- train offers: 1,175; test offers: 1,291
- test identities: 678
- baseline log loss: **0.682629**
- candidate log loss: **0.677012**
- improvement: **+0.005616**
- baseline Brier: **0.244768**
- candidate Brier: **0.242099**
- baseline AUC: **0.581031**
- candidate AUC: **0.599087**
- bootstrap 95% CI:
  **[-0.002938, +0.014052]**
- disposition: **DIRECTION_FAIL**

### Fit 2025 -> test 2024
- train offers: 1,291; test offers: 1,175
- baseline log loss: **0.685971**
- candidate log loss: **0.682361**
- improvement: **+0.003610**
- baseline Brier: **0.246482**
- candidate Brier: **0.244638**
- baseline AUC: **0.567452**
- candidate AUC: **0.592184**
- bootstrap 95% CI:
  **[-0.005530, +0.012566]**
- disposition: **DIRECTION_FAIL**

Terminal:
`NO_VERIFIED_MARKET_OFFER_RESIDUAL_PROBABILITY_SIGNAL_V1`.

Receptions improved every point metric in both directions, but the preregistered
cluster-bootstrap intervals include zero in both directions. That is an
interesting descriptive lead, not validated actionability.

## Terminal interpretation

All three markets finish as:

`NO_VERIFIED_MARKET_OFFER_RESIDUAL_PROBABILITY_SIGNAL_V1`

Therefore:
- do not deploy this offer-level logistic selector;
- do not tune C;
- do not add/remove features;
- do not build interactions;
- do not use side/role/week/edge-bin subgroup rescue;
- do not convert the receptions point-metric improvement into a betting rule;
- do not proceed to the proposed sparse-actionability stage from this V1.

This reinforces the prior Market-Relative Bet Selector V1 null: a downstream
selection layer cannot currently establish stable historical edge from the
tested model-market information. The next high-value work should improve or add
genuinely new football information and continue prospective betting evidence
capture, rather than iterate historical selector variants on the same exposed
seasons.
