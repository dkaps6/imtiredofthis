# Football Matchup Transmission V1 — Integration Candidate Contract

Status: **FROZEN BEFORE CANDIDATE SCORING — RESEARCH ONLY — NO PRODUCTION CHANGE**

Branch: `research-football-matchup-integration-candidates-v1`

Parent diagnostic branch: `research-football-matchup-transmission-v1`

Parent Phase B/C authority:

- run `37514137803` — SUCCESS
- artifact `11436786668`
- digest `sha256:ccc0a3962f5505bdb0465fb8be8ff0f158f3781064cbffabc875361ed99db8ef`
- source SHA `a8cf0731675228e2eea8c25411dae5f8b2f187fc`
- 62 frozen Phase B/C specifications tested
- 38,258 skill-position player-market rows
- 894 QB control rows
- zero sportsbook inputs
- zero candidate models fit
- Phase A was not rerun

This contract is frozen **after the diagnostic gate and before any implementation is scored**.

## 1. Exact candidates allowed to advance

Only the three Phase B specifications that met the frozen two-season / two-cluster / same-live-semantics gate may advance.

### FMT-RB1 — RB rushing environment transmission

- cohort: `RB_RUSH`
- market: `rush_yards`
- feature: opponent `pass_rate_faced`
- production semantic: M89/M90 corrected public-football context
- favorable direction: **lower** opponent pass-rate-faced
- oriented feature:
  `weakness_z = -1 * week_z(opponent_pass_rate_faced)`
- mechanism claim: opponent run/pass environment may leave residual rushing-volume information untransmitted after the existing player/role projection.
- This is **not** defensive rushing quality and does not reopen M95A/M95B.
- Parent diagnostic:
  - 2024: rho `0.081609`, game CI `[0.022101, 0.138108]`, player CI `[0.028800, 0.133956]`
  - 2025: rho `0.071750`, game CI `[0.019807, 0.124488]`, player CI `[0.021614, 0.120301]`

### FMT-WR1 — WR offensive PROE transmission

- cohort: `WR_REC`
- market: `rec_yards`
- feature: offense `true_proe`
- production semantic: M89/M90 corrected public-football context
- favorable direction: **higher** offense true PROE
- oriented feature:
  `weakness_z = week_z(offense_true_proe)`
- mechanism claim: team pass-intent information may be under-transmitted into WR receiving-yard means after the existing player/role projection.
- Parent diagnostic:
  - 2024: rho `0.051808`, game CI `[0.006289, 0.098462]`, player CI `[0.004143, 0.098167]`
  - 2025: rho `0.046536`, game CI `[0.004813, 0.088848]`, player CI `[0.000813, 0.090140]`

### FMT-TE1 — TE defensive pass-success transmission

- cohort: `TE_REC`
- market: `rec_yards`
- feature: opponent `def_pass_success_allowed`
- production semantic: M89/M90 corrected public-football context
- favorable direction: **higher** defensive pass success allowed
- oriented feature:
  `weakness_z = week_z(opponent_def_pass_success_allowed)`
- mechanism claim: opponent pass-success vulnerability may be under-transmitted into TE receiving-yard means after the existing player/role projection.
- Parent diagnostic:
  - 2024: rho `0.062807`, game CI `[0.001478, 0.123042]`, player CI `[0.011052, 0.112411]`
  - 2025: rho `0.104347`, game CI `[0.048856, 0.158724]`, player CI `[0.036988, 0.170256]`

## 2. Explicit exclusions

No other Phase B/C signal may enter candidate scoring.

In particular:

- `TE_REC × def_te_ypt_allowed` replicated diagnostically but is **blocked** because historical official-stat YPT and live Sharp YPT do not have proven same-source semantics.
- Sharp `dl_stuff_rate` and `dl_ybc_per_rush` remain source-blocked.
- outside/slot YPT remain source-blocked.
- box and coverage historical proxies remain diagnostic-only because current live provider parity is unproven.
- `def_rush_epa` and generic RB role × run-defense quality remain blocked by the M95A/M95B anti-retest ledger.
- QB matchup families remain closed; QB was control only.
- no combination of the three advancing candidates is authorized in this stage.

## 3. Candidate functional form

Each candidate is scored **individually**.

For a target player-game:

`candidate_projection = baseline_projection + beta_train * weakness_z`

Rules:

1. `weakness_z` is the same leakage-safe weekly cross-sectional oriented z-score frozen in Phase B/C.
2. `beta_train` is a single matchup-transmission coefficient fit only on the frozen training season described below.
3. The fit has **zero intercept**. The candidate is not allowed to repair generic mean bias.
4. No threshold, top-N rule, bellcow/WR1/TE1 carveout, target-game usage, sportsbook feature, or second matchup feature is allowed.
5. If the fitted coefficient is non-positive, the candidate fails its sign gate and closes without reinterpretation.
6. Coefficients are never re-fit by evaluation season.
7. No production code path is changed during this experiment.

## 4. Temporal design

The Phase B/C discovery seasons were 2024 and 2025. Therefore the candidate stage must not pretend that scoring those seasons is independent confirmation.

### Coefficient training

Fit `beta_train` using **2022 Weeks 2-18 only**, with:

- 2021 available only as strict-prior history;
- current 2022 evidence strictly before the target week;
- the same football-only historical projection machinery;
- the same M89/M90 corrected team-context semantics;
- zero sportsbook data.

### Primary out-of-selection confirmation

Evaluate the frozen 2022 coefficient on **2023 Weeks 2-18**.

2023 outcomes were not used by Phase B/C to select these three features and are the primary confirmation season for this candidate contract.

### Secondary consistency replay

Without changing the 2022 coefficient, also replay on:

- 2024 Weeks 2-18
- 2025 Weeks 2-18

These seasons are **not** independent confirmation because their residuals motivated candidate selection. They are consistency / full-stack materiality checks only.

## 5. Baseline authority and parity

For all seasons, baseline means are the canonical football-only walk-forward projection produced by the current historical harness with frozen ensemble weights.

Before interpreting the 2024/2025 secondary replay, the workflow must prove exact identity and numerical parity against the parent right-tail baseline authority for the overlapping candidate market rows.

If 2024/2025 baseline parity fails, scoring fails closed.

For 2022/2023, no sportsbook information may be used to construct or modify the baseline.

## 6. Frozen training estimator

For each candidate separately, on supported 2022 rows:

- target: `actual - baseline_projection`
- predictor: `weakness_z`
- estimator: zero-intercept ordinary least squares

`beta_train = sum(weakness_z * residual) / sum(weakness_z^2)`

No regularization strength, clipping threshold, intercept, feature selection, or hyperparameter is searched.

Training support requires at least:

- 200 player-market rows
- 50 distinct games

If support fails, the candidate closes.

## 7. Frozen evaluation metrics

For baseline and candidate, report by season:

- n
- distinct games
- distinct players
- MAE
- RMSE
- bias
- median absolute error
- Pearson correlation
- 75+ yard absolute-error count
- 100+ yard absolute-error count

Also report:

- mean absolute adjustment
- 95th percentile absolute adjustment
- maximum absolute adjustment
- pre/post residual Spearman correlation to the candidate `weakness_z`

Paired uncertainty:

- 5,000 game-cluster bootstrap replicates
- fixed seed `20261006`
- statistic: `MAE_baseline - MAE_candidate`

## 8. Frozen pass/fail gates

A candidate is labeled `INTEGRATION_CANDIDATE_CONFIRMED` only if **all** are true:

### Training sanity
- 2022 support passes;
- `beta_train > 0`.

### Primary 2023 confirmation
- candidate MAE < baseline MAE;
- candidate RMSE <= baseline RMSE;
- game-cluster bootstrap probability that MAE improves >= `0.80`;
- 75+ yard absolute-error count does not increase;
- 100+ yard absolute-error count does not increase;
- absolute residual Spearman relationship to the candidate matchup variable is smaller after correction than before correction.

### 2024/2025 consistency
- frozen coefficient is unchanged from 2022;
- MAE is non-worse in **both** 2024 and 2025;
- pooled 2024+2025 MAE improves;
- pooled 75+ and 100+ yard absolute-error counts do not increase.

A failure on any required gate closes that candidate. Do not tune the coefficient, gate, cohort, sign, or feature after scoring.

## 9. Promotion boundary

Even a confirmed candidate is **not** authorized for production promotion by this contract.

If one or more candidates confirm:

1. freeze a separate production-order integration/shadow contract;
2. wire only the confirmed mechanism(s) into a non-production historical/live shadow;
3. prove conservation / ordering / unaffected-market invariance;
4. require forward 2026 evidence before production promotion unless a separately frozen governance decision explicitly authorizes otherwise.

No paid OddsAPI pull is required or authorized by this candidate score.
