# PRISM Pass-Catcher × QB-Regime Allocation Signal V1 — Result

Date: 2026-10-17

Status: **FAILED CLOSED / NO PRODUCTION CHANGE**

Model:
**PRISM — Player Role & Individual Situation Model**

Frozen contract:
`docs/research/PRISM_PASS_CATCHER_QB_REGIME_ALLOCATION_SIGNAL_V1_FREEZE.md`

Canonical execution:
- run: `39067914161` — **SUCCESS**
- exact head: `87e57f4341c1ca6ae79d585a9b52bbc80a9a29b2`
- artifact: `12113942923`
- digest: `sha256:8019c23d69db0a9fb2b8c68b180277bcbdd17bdd99c3e9f393ce51286c329fbf`

Disposition:

`PRISM_PASS_CATCHER_QB_REGIME_ALLOCATION_SIGNAL_V1_CLOSED`

Neither WR nor TE qualified.

## Question

Does a named pass catcher's strictly-prior target-allocation history with the
pregame-selected QB regime add next-game target-share information beyond a
strong ordinary recent-role baseline?

Frozen candidate:
- baseline = ordinary strictly-prior recent target-share state;
- signal = same-QB prior mean target share minus overall prior mean target share;
- no fitted coefficient;
- candidate applied only with >=2 same-QB prior games;
- WR/TE room mass conserved exactly;
- 2024 development / 2025 independent confirmation;
- target share is the primary endpoint.

## WR result

### 2024 development

Support:
- rows: **2,141**
- players: **219**
- team-weeks: **454**

Target-share MAE:
- baseline: **0.05535084**
- candidate: **0.05628118**
- delta: **+0.00093034** — worse

P90 absolute error:
- baseline: **0.13196481**
- candidate: **0.13371954** — worse

Target-count MAE:
- baseline: **1.71803147**
- candidate: **1.74831779** — worse

Candidate strictly closer:
- **12.10%**
- ties: **72.63%**

Room conservation:
- max gap: **2.22e-16** — PASS

The candidate failed the development performance gates.

### 2025 confirmation

Support:
- rows: **2,094**
- players: **221**
- team-weeks: **446**

Target-share MAE:
- baseline: **0.05851975**
- candidate: **0.05889262**
- delta: **+0.00037287** — worse

P90 absolute error:
- baseline: **0.13333333**
- candidate: **0.13378971** — worse

Target-count MAE:
- baseline: **1.75630909**
- candidate: **1.76791387** — worse

Candidate strictly closer:
- **13.94%**
- ties: **71.49%**

Paired 2025 inference on candidate-minus-baseline absolute error:
- player-cluster 95% CI: **[-0.00004699, +0.00077793]**
- team-week-cluster 95% CI: **[-0.00013527, +0.00092016]**

Both confirmation CIs cross zero and the point estimate is worse.

WR disposition:
**CLOSED**.

## TE result

### 2024 development

Support:
- rows: **1,336**
- players: **127**
- team-weeks: **454**

Target-share MAE:
- baseline: **0.04207719**
- candidate: **0.04259334**
- delta: **+0.00051616** — worse

P90 absolute error:
- baseline: **0.10466124**
- candidate: **0.10705190** — worse

Target-count MAE:
- baseline: **1.30488040**
- candidate: **1.32161369** — worse

Candidate strictly closer:
- **11.53%**
- ties: **71.71%**

Room conservation:
- max gap: **5.55e-17** — PASS

The candidate failed the development performance gates.

### 2025 confirmation

Support:
- rows: **1,296**
- players: **129**
- team-weeks: **446**

Target-share MAE:
- baseline: **0.04254674**
- candidate: **0.04285709**
- delta: **+0.00031034** — worse

P90 absolute error:
- baseline: **0.10321564**
- candidate: **0.10389296** — worse

Target-count MAE:
- baseline: **1.26042009**
- candidate: **1.27165826** — worse

Candidate strictly closer:
- **13.50%**
- ties: **69.83%**

Paired 2025 inference:
- player-cluster 95% CI: **[+0.00000800, +0.00066599]**
- team-week-cluster 95% CI: **[+0.00004013, +0.00057359]**

Both CIs are entirely **above zero**: the frozen same-QB candidate is
statistically worse than the ordinary-role baseline for TE confirmation.

TE disposition:
**CLOSED**.

## Interpretation

The source exists and is high quality. The predictive signal does not.

Knowing that a named WR/TE historically received a different target share with
the same QB regime does **not**, under this frozen no-fit finite-room
construction, improve next-game target allocation beyond ordinary recent role.

This matters because the football story was plausible:
- QB identity can matter;
- the historical same-QB sample is real;
- support is large;
- chronology is clean.

But source availability is not predictive value.

The result therefore rejects the direct rule:

> shift the player's current target allocation toward his historical same-QB
> target-share state.

Do not rescue this result by changing the history window, requiring >=3 games,
fitting a shrink factor, isolating QB changes, selecting WR1/TE1, pooling
families, or moving the same signal downstream to receiving yards.

## Protected implications

This failure does not close:
- QB effects on opportunity **type** or target quality;
- player-conditioned opponent interaction;
- exact teammate competition state if a new legal source becomes available;
- jointly coherent finite allocation using genuinely new player-state inputs.

It does close this exact:
`same-QB target-allocation delta -> current target-share adjustment`
family.

WR-R15, TE-R5P, Stage-1 participation, qualified Stage-2 cells, WR Stage-2
source block, raw recent-share closure and finite-allocation conservation remain
unchanged.

## Integrity

- candidate variants: **1**
- fitted coefficients: **0**
- threshold searches: **0**
- chronology violations: **0**
- target-game outcomes used in predictors: **0**
- 2026 outcomes: **0**
- sportsbook inputs: **0**
- OddsAPI calls: **0**
- receiving yards scored: **false**
- market translation scored: **false**
- production changed: **false**
