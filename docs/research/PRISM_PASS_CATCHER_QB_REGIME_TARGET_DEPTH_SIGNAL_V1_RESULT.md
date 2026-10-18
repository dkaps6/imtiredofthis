# PRISM Pass-Catcher × QB-Regime Target-Depth Signal V1 — Result

Date: 2026-10-17

Status: **COMPLETE / FAILED CLOSED**

Frozen plan:
`docs/research/PRISM_PASS_CATCHER_QB_REGIME_TARGET_DEPTH_SIGNAL_V1_PLAN.md`

Source authority:
- run `39087349029` — SUCCESS;
- artifact `12120839869`;
- disposition `PRISM_PASS_CATCHER_QB_REGIME_TARGET_DEPTH_STATE_V1_SOURCE_READY_ALL`.

## Canonical scoring authority

- workflow: `Research PRISM Pass Catcher QB Regime Target Depth Signal V1`
- run: `39087599967` — **SUCCESS**
- exact head: `1c6595bf3f7926e46e6ec451442cd41166f36ddd`
- artifact: `12121267285`
- artifact name: `prism-pass-catcher-qb-regime-target-depth-signal-v1-39087599967`
- digest: `sha256:a1f72e22d8bc9101f08e8fb592d259e80baf7c749c4ec6b8079cd4a43d690aa2`
- disposition:
  `PRISM_PASS_CATCHER_QB_REGIME_TARGET_DEPTH_SIGNAL_V1_CLOSED`

All workflow steps passed:
- source-authority download;
- frozen safeguards;
- exact frozen signal execution;
- integrity certification;
- strict repository audit;
- Issue #535 result record;
- artifact upload.

## Frozen candidate

Baseline:
`prior8_mean_air_yards`

Candidate:
`same_qb_mean_air_yards`

Target entitlement was never loaded or changed.

No coefficient, blend, shrinkage, alternate window or threshold was fit.

## Results

### WR

2024 development:
- rows: **1,512**
- players: **160**
- MAE: **4.7101 -> 4.7864** (**+0.0763 worse**)
- RMSE: **6.3724 -> 6.4702** (worse)
- p90 AE: **10.1483 -> 10.0000** (slightly better)
- candidate closer: **29.365%**
- median equal-player MAE delta: **+0.0000**

2025 independent confirmation:
- rows: **1,452**
- players: **169**
- MAE: **4.8793 -> 4.9888** (**+0.1095 worse**)
- RMSE: **6.7291 -> 6.8493** (worse)
- p90 AE: **10.5118 -> 10.6213** (worse)
- candidate closer: **27.686%**
- median equal-player MAE delta: **+0.0054**
- player-cluster 95% CI: **[+0.05578, +0.16550]**
- team-week-cluster 95% CI: **[+0.04652, +0.17665]**

WR qualified: **FALSE**.

The 2025 paired uncertainty is entirely on the harmful side.

### TE

2024 development:
- rows: **742**
- players: **83**
- MAE: **3.6007 -> 3.7044** (**+0.1038 worse**)
- RMSE: **5.0416 -> 5.1429** (worse)
- p90 AE: **7.3564 -> 7.6206** (worse)
- candidate closer: **28.976%**
- median equal-player MAE delta: **+0.0000**

2025 independent confirmation:
- rows: **754**
- players: **83**
- MAE: **3.5792 -> 3.6146** (**+0.0354 worse**)
- RMSE: **4.8152 -> 4.8651** (worse)
- p90 AE: **7.5790 -> 7.6542** (worse)
- candidate closer: **28.117%**
- median equal-player MAE delta: **+0.0082**
- player-cluster 95% CI: **[-0.01117, +0.08323]**
- team-week-cluster 95% CI: **[-0.02284, +0.09482]**

TE qualified: **FALSE**.

## Interpretation

The source is excellent. The mechanism is not.

For both WR and TE, replacing the named player's ordinary strictly-prior target-depth state with his strictly-prior same-QB target-depth state does **not** improve next-game target-depth prediction.

The exact family is therefore closed:

> Current QB identity, represented as raw same-QB target-depth history under the frozen V1 support contract, does not add replicated target-depth value beyond the named player's ordinary recent target-depth identity.

This does **not** imply QB identity can never affect target quality through a genuinely different football object.

It does imply that same-QB raw target-depth replacement is exhausted and may not be rescued.

## Anti-rescue / closure

Do not:
- blend ordinary and same-QB target depth;
- fit a shrink coefficient;
- use >=3-only same-QB history;
- change the eight-game window;
- lower any source/support gate;
- carve out QB changes;
- carve out WR1/TE1;
- pool WR+TE;
- substitute deep-target rate or target-depth SD;
- translate this failed signal into YPT/YPR/receiving yards;
- add opponent variables to rescue it;
- use 2026 outcomes or sportsbook information.

Any later QB-conditioned opportunity-quality mechanism must use genuinely new pregame football information, not another transform of the same raw QB-pair depth history.

## Integrity

- candidate variants: 1;
- chronology violations: 0;
- target entitlement loaded: false;
- target entitlement changed: false;
- receiving yards scored: false;
- receptions scored: false;
- touchdowns scored: false;
- market translation scored: false;
- coefficients fit: 0;
- thresholds searched: 0;
- sportsbook inputs: 0;
- OddsAPI calls: 0;
- 2026 outcomes: 0;
- production changed: false.
