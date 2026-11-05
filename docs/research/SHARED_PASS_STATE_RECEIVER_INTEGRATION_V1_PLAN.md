# Shared Pass-State Receiver Integration V1 — Frozen Historical Plan

Date: 2026-09-24

Status: **FROZEN BEFORE OUTCOME SCORING**

## 1. Scientific question

The current production stack has a confirmed split-state contradiction:

- selected QB pass-yard distributions are generated from the frozen C2 completed-pass / receiving process;
- production WR/TE/RB receiving arrays remain the separate canonical process;
- current 2026 Week-3 read-only audit showed median player array correlation only `0.0399`, Q4 receiver median p90 draw disagreement `49.53 yd`, and canonical zero-reception/positive-yard states on a median `16.42%` of player draws.

V1 asks one bounded question:

> Does installing the **already-frozen C2 completed-pass receiver state** into the receiver Monte Carlo layer improve receiver prediction/distribution quality under the current promoted entitlement + downstream ensemble stack?

This is not C1 or C3 and introduces no receiver feature search, target-share retune, position router, or sportsbook input.

## 2. Frozen candidate

### Baseline

For each historical week, reconstruct the current leakage-safe receiver stack:

1. canonical historical football context;
2. M38 explicit finite target entitlement;
3. fold-safe TE-R5P entitlement;
4. fold-safe WR-R15 entitlement where its OOS authority permits it;
5. canonical state-capturing Monte Carlo;
6. current fixed ML/state components;
7. current ensemble weights;
8. current RB Rush+Receiving Conservation V2 downstream.

### Candidate receiver state

Generate the exact current C2 completed-pass receiver process from the same canonical team states and the same final entitlement shares:

- target allocation from the same finite team pass-attempt state;
- receptions sampled from target count and the existing frozen catch-rate inputs;
- receiving yards generated from completed receptions using the existing C2 `YPT / catch_rate` YPR translation;
- existing C2 YPR bounds: `3.0 .. 35.0`;
- existing C2 residual catch rate: `0.64`;
- existing C2 residual YPT: `7.5`;
- existing C2 residual receiving bucket;
- one constant team C2 scale anchored to the canonical raw QB MC mean;
- no new coefficients or fitted parameters.

For an integrated team, replace only:
- `receptions`;
- `rec_yards`;
- `rush_rec_yards = unchanged rush_yards + integrated rec_yards`.

Do **not** change:
- target entitlement;
- team pass attempts;
- rushing attempts;
- rushing yards;
- anytime-TD arrays;
- QB point-mean authority;
- ML/state component models;
- ensemble weights.

The existing QB C2 distribution is treated as frozen authority. This experiment does **not** seek another QB improvement.

### Downstream production semantics

After the candidate MC arrays are produced, apply the current downstream ensemble/mean-alignment behavior exactly as production does. V1 therefore tests whether the **shared latent passing state** helps under the current full stack; it does not claim that final independently mean-aligned market arrays are yet a fully conserved accounting identity.

If V1 qualifies, final production integration must be frozen separately and must explicitly audit final adjusted-outcome cross-market conservation.

## 3. Historical routing without selector leakage

The persisted production selector was final-fit on 2024-2025 and may **not** be replayed in-sample as validation.

Therefore V1 has two frozen views.

### A. Mechanism-stability view — ALL-C2

Apply the exact receiver integration to every eligible team-game in:
- 2024 W1-18, prior=2023;
- 2025 W1-18, prior=2024.

Purpose: test whether the completed-pass/conservation mechanism is stable across two independent seasons, without using the final fitted selector.

This is mechanism evidence only, not the production routing gate.

### B. Production-routing view — 2025 Phase-J OOS

Use only the frozen **walk-forward 2025 Phase-J `use_c2` decisions** from:
- run `34151640191`;
- artifact `10029560958`;
- digest `sha256:e456130a131ef9c5fcd78e598ca538085dbe5cb111a116558a56d7848d8dd249`;
- casebook `phase_j_qb_distribution_casebook.csv`.

That artifact contains 440 OOS 2025 QB team-games and 412 frozen `use_c2=True` decisions.

For aligned OOS team-games:
- `use_c2=True`: candidate installs the integrated C2 receiver state;
- `use_c2=False`: candidate remains exact baseline.

No final-fit production-selector coefficient is used to choose a historical row.

## 4. Frozen specialist authority

Historical construction:
- TE-R5P fold-safe authority: run `34152797603`;
- WR-R15 fold-safe authority: run `34238301577`;
- WR-R15 is consumed only in seasons/folds allowed by its own frozen OOS contract;
- current fixed ensemble weights;
- current RB Rush+Receiving Conservation V2 downstream.

Monte Carlo:
- `2000` draws per week;
- baseline seed `42 + week`;
- C2 receiver seed uses the already-frozen C2 integration seed family and is fixed before scoring;
- no sportsbook inputs.

## 5. Scoring

Target-game outcomes are loaded **only after** baseline and candidate arrays are complete.

Score WR / TE / RB receiver rows separately and macro-average the three families.

### Receiving yards

For baseline and candidate:
- full-stack final mean MAE;
- RMSE;
- bias;
- median / p75 / p90 absolute error;
- 30+ and 40+ yard miss rates;
- sample CRPS from the final mean-aligned distribution;
- 80% and 90% interval coverage error.

### Receptions

For baseline and candidate:
- full-stack final mean MAE;
- RMSE;
- bias;
- p90 absolute error;
- sample CRPS.

### RB rush+receiving yards

After current RB V2:
- MAE;
- p90 absolute error;
- 30+ yard miss rate.

## 6. Frozen integrity gates

Every item must pass:

1. fitted parameters = `0`;
2. candidate variants scored = `1`;
3. sportsbook inputs upstream = `0`;
4. target-game outcomes used upstream = `0`;
5. C1 used = false;
6. C3 used = false;
7. target entitlement baseline vs candidate exact;
8. rush-attempt arrays exact;
9. rush-yard arrays exact;
10. anytime-TD arrays exact;
11. baseline-vs-candidate QB pass-yard arrays exact in the routed production view;
12. unselected 2025 Phase-J OOS team receiver arrays exact;
13. selected integrated state has zero `receptions == 0 && rec_yards > 0` draws;
14. raw C2 selected-team pass/receiver+residual identity max gap <= `1e-10`;
15. RB V2 pathwise identity gap <= `1e-10`;
16. 2025 historical routing identities match the frozen Phase-J casebook exactly.

Any integrity failure means **mechanical/integrity failure**, not scientific failure. Only bounded plumbing repair is allowed.

## 7. Frozen scientific gates

### A. ALL-C2 mechanism stability

For **each** of 2024 and 2025:

1. macro WR/TE/RB receiving-yard MAE is nonworse;
2. macro receiving-yard CRPS is nonworse;
3. macro receiving-yard p90 absolute error is nonworse;
4. macro receiving-yard 40+ miss rate is nonworse;
5. no individual position-family receiving-yard MAE regresses by more than `0.50 yd`;
6. macro receptions MAE is nonworse;
7. macro receptions CRPS is nonworse;
8. RB rush+receiving MAE is nonworse;
9. RB rush+receiving p90 is nonworse.

Across pooled 2024-2025:
10. macro receiving-yard MAE must **strictly improve**;
11. macro receiving-yard CRPS must **strictly improve**.

### B. 2025 Phase-J OOS routed production view

On the full aligned 440-team-game OOS routing cohort, with only frozen `use_c2=True` teams changed:

1. macro receiving-yard MAE must **strictly improve**;
2. macro receiving-yard CRPS must **strictly improve**;
3. macro receiving-yard p90 is nonworse;
4. macro receiving-yard 40+ miss rate is nonworse;
5. no position-family receiving-yard MAE regresses by more than `0.50 yd`;
6. macro receptions MAE is nonworse;
7. macro receptions CRPS is nonworse;
8. RB rush+receiving MAE is nonworse;
9. RB rush+receiving p90 is nonworse.

## 8. Disposition

Only if **every integrity gate and every scientific gate** passes:

`SHARED_PASS_STATE_RECEIVER_INTEGRATION_V1_QUALIFIED`

Otherwise:

`SHARED_PASS_STATE_RECEIVER_INTEGRATION_V1_FAILED_CLOSED`

A qualified result authorizes only a separately frozen production-integration/certification plan.

## 9. Stopping rule

After scoring, do not rescue V1 with:
- C1/C3;
- a different selector threshold;
- final-fit selector replay on 2024/2025;
- position carveouts;
- WR-only / TE-only / RB-only routing;
- player-volume thresholds;
- alternate residual catch/YPT;
- alternate YPR caps;
- per-position scale factors;
- sportsbook-conditioned routing;
- outcome-fitted mean blending;
- 2026 result fitting.

A failed V1 may generate future hypotheses, but this exact candidate closes.
