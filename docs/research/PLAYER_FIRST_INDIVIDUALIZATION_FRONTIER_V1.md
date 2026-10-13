# PLAYER-FIRST INDIVIDUALIZATION FRONTIER V1

Date: 2026-10-13

Status: **AUDIT COMPLETE / RESEARCH DIRECTION CORRECTED / NO PRODUCTION CHANGE**

## Why this note exists

The model is an individual-player projection system. Position-level aggregates remain useful as global safety guards, but a position-level MAE result is not sufficient evidence that an individual-player mechanism is exhausted.

Canonical prior audit:
- `PLAYER_INDIVIDUALIZATION_PARTIAL`
- run `37558838148`
- 600 distinct players / 40,464 player-week-metric rows.

That audit established the current architecture as:

`individual player history + material position shrinkage + promoted player specialists + mostly shared weekly environment transforms`

The present audit reconciles later work against that standard.

## What is genuinely individualized already

- QB passing mean: M89/M90 individual history plus team/opponent environment.
- WR opportunity: M38 anchor plus WR-R15 individual entitlement.
- TE opportunity: TE-R5P individual entitlement.
- RB: PlayerForm/Bayesian individual history; Week-1 specialist lineage; non-Week-1 room allocation remains less specialized.
- Opportunity uncertainty: same-player share volatility is replicated; RB carry and WR target allocation uncertainty have qualified historical mechanisms.
- RB rushing-yard uncertainty: same-player projection difficulty is a qualified individual distribution signal.

## Material position pooling still present

Bayesian v2 uses fixed position-pool prior strengths plus prior-player and current-player evidence.

Representative Week-5 direct position-pool weights from the canonical audit:
- WR target share 23.1%; WR YPT 29.4%.
- RB rush/target share 23.1%; RB YPC/YPT 29.4%.
- TE target share 25.0%; TE YPT 31.25%.

These are not evidence of a bug. They are regularization. But they prove that an established player's projection is not yet purely individual.

## Individual signals that survived even when an implementation failed

### RB same-player error / difficulty persistence

Historical same-player persistence replicated across 2021-2024 and the original 2025 diagnostic.

The separately frozen mean-neutral uncertainty implementation then qualified historically:
- run `35039152022`;
- disposition `RB_YARD_DIFFICULTY_MC_WIDTH_QUALIFIED`;
- 4,550 rows / 209 players;
- pooled CRPS improvement 1.218%;
- highest-difficulty player slice CRPS improvement 2.624%;
- interval coverage and fixed football-threshold Brier metrics improved;
- point mean was exactly unchanged;
- every frozen gate passed.

This is direct evidence that individual player identity/history can improve the model without a position-wide mean correction.

### WR same-player error persistence

The original WR-R3 diagnostic found:
- signed error persistence;
- individual difficulty persistence;
- extreme-miss persistence;
- all three effects positive across six seasons.

The later combined candidate did **not** pass its frozen promotion contract and remains closed as an implementation. However:
- all six seasons improved MAE;
- WR1/WR2/WR3 slices improved;
- median qualifying-player MAE delta improved;
- 2025 MAE/RMSE/bias/correlation all moved in the favorable direction.

Therefore the exact candidate is closed, but individual WR state is not scientifically exhausted.

### WR/TE target-share trajectory

Strictly-prior named-player target-share trajectory is a real historical allocation signal. Its exact eligibility contract remains protected. Prospective evidence is still limited.

## Exact individual mechanisms closed

Do not rescue or relabel:
- receiver drop/catchability conversion V1;
- generic individual efficiency-volatility V1;
- direct WR/TE target-depth x opponent-depth YPT multiplier;
- direct RB run-location x opponent run-location YPC multiplier;
- WR direct bias-shrink formulas already scored;
- RB PD3/PD4/PD5 mean-correction formulas;
- generic Q90 upper-tail hinge;
- previously closed generic matchup-transmission formulas.

A failed formula closes that formula, not every individual-player hypothesis.

## Open architectural seam

The current rules layer still applies major environment responses uniformly to teammates after their individual baselines are established.

Examples:
- receiving efficiency: `rules_ypt = base_ypt * pass_eff_mult`;
- rushing efficiency: `rules_ypc = base_ypc * rb_rush_eff_mult`.

Thus two players can start from different personal baselines but still receive the same response to the same team/opponent efficiency environment.

The prior matchup program primarily tested:
- whether defensive variables explain residuals conditional on role/opportunity; and
- direct formulas that map matchup state into YPT/YPC.

It did **not** test whether the **same named player's own strictly-prior residual response to the model's existing environment adjustment** predicts his next residual.

That is a distinct player x environment question.

## Correct evaluation hierarchy going forward

For a player-specific hypothesis:

1. hypothesis grain = stable player identity;
2. feature history = strictly prior same-player evidence;
3. primary scientific evaluation = player-game + per-player heterogeneity;
4. game and player dependence must both be respected;
5. aggregate position metrics = non-harm guardrails, not the sole definition of success;
6. position may regularize/fallback for sparse players but may not erase identity;
7. no sportsbook input upstream;
8. no 2026 outcome tuning;
9. no production change from a diagnostic.

## Next frozen diagnostic

`PLAYER_X_ENVIRONMENT_RESPONSE_V1`

Question:

> Conditional on the current full-stack player projection and the existing shared football environment multiplier, does a player's own strictly-prior response history predict the sign/magnitude of his next projection residual?

This is diagnostic-only. A positive result may authorize a separately frozen integration candidate. A null closes this exact response-state construction; it does not reopen old matchup formulas.
