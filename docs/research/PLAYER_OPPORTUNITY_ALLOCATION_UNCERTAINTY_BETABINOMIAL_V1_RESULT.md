# PLAYER OPPORTUNITY ALLOCATION UNCERTAINTY — BETA-BINOMIAL V1 RESULT

Date: 2026-10-12

Status: **RB CARRY SHARE AND WR TARGET SHARE PASS THREE-SEASON HISTORICAL DISTRIBUTION GATES. RB TARGET SHARE FAILS CLOSED. NO PRODUCTION CHANGE.**

Frozen contract:
- `docs/research/PLAYER_OPPORTUNITY_ALLOCATION_UNCERTAINTY_BETABINOMIAL_V1_FREEZE.md`

Canonical authority:
- workflow run: `38296726460`
- exact head: `b94e2188b6fd0c20b41263b6dc9c8161b151d5da`
- artifact: `11768483266`
- artifact digest: `sha256:403360d3245e78507cfe7646022291a676535fe6bf90d607ba85f89f900eae4c`
- workflow conclusion: `SUCCESS`

Superseded first scoring artifact:
- run `38296371417`
- its CRPS calculations were valid, but all-row NLL averages were singular
  because unchanged mean-share boundary rows can assign zero probability;
- it is superseded by the paired log-score convention frozen before the
  authoritative rerun.

Overall disposition:
`BETA_BINOMIAL_ALLOCATION_UNCERTAINTY_HAS_PASSING_CELLS`

Evaluation:
- seasons 2023, 2024, 2025
- weeks 5-18
- rows: **15,540**
- players: **525**
- candidate-changed rows: **8,177**
- no fitted coefficients
- no thresholds
- no sportsbook / OddsAPI
- production unchanged
- max expected-count gap: **0.0**
- max reconstructed-share gap: **4.91e-13**

## Candidate definition

Baseline marginal allocation:

`Y ~ Binomial(N, prior_share_mean)`

Candidate when V2 excess variance is positive and mathematically valid:

`P ~ Beta(alpha, beta)`

`Y | P ~ Binomial(N, P)`

with alpha/beta derived algebraically from:
- frozen prior mean share;
- frozen V2 `excess_share_var`.

The candidate and baseline have exactly the same expected count `N * mean`.

No widening coefficient was fit.

## RB carry share — HISTORICAL PASS

### 2023
- rows: 1,464
- changed: 950 (**64.89%**)
- baseline CRPS: **2.24985**
- candidate CRPS: **2.06841**
- CRPS improvement: **+0.18144**
- game-cluster CI: **[+0.14482, +0.22052]**
- player-cluster CI: **[+0.13285, +0.23759]**
- mean paired NLL improvement: **+0.94078**
- 80% coverage: **65.37% -> 79.44%**
- 90% coverage: **71.99% -> 85.86%**

### 2024
- rows: 1,468
- changed: 1,035 (**70.50%**)
- CRPS improvement: **+0.14365**
- game-cluster CI low: **+0.11205**
- player-cluster CI low: **+0.10477**
- mean paired NLL improvement: **+0.81666**
- 80% coverage: **62.74% -> 76.57%**
- 90% coverage: **71.73% -> 84.06%**

### 2025
- rows: 1,493
- changed: 890 (**59.61%**)
- CRPS improvement: **+0.10333**
- game-cluster CI low: **+0.07805**
- player-cluster CI low: **+0.06686**
- mean paired NLL improvement: **+0.66058**
- 80% coverage: **71.13% -> 80.17%**
- 90% coverage: **75.42% -> 85.26%**

Disposition:
`BETA_BINOMIAL_ALLOCATION_UNCERTAINTY_HISTORICAL_PASS`

## WR target share — HISTORICAL PASS

### 2023
- rows: 2,283
- changed: 1,170 (**51.25%**)
- baseline CRPS: **1.25178**
- candidate CRPS: **1.23335**
- improvement: **+0.01843**
- game-cluster CI: **[+0.01180, +0.02558]**
- player-cluster CI: **[+0.01019, +0.02778]**
- mean paired NLL improvement: **+0.11116**
- 80% coverage: **76.87% -> 81.78%**
- 90% coverage: **84.23% -> 88.22%**

### 2024
- rows: 2,175
- changed: 1,161 (**53.38%**)
- CRPS improvement: **+0.02530**
- game-cluster CI low: **+0.01514**
- player-cluster CI low: **+0.01365**
- mean paired NLL improvement: **+0.14989**
- 80% coverage: **76.09% -> 82.25%**
- 90% coverage: **83.08% -> 87.59%**

### 2025
- rows: 2,232
- changed: 1,196 (**53.58%**)
- CRPS improvement: **+0.02773**
- game-cluster CI low: **+0.02018**
- player-cluster CI low: **+0.01825**
- mean paired NLL improvement: **+0.15044**
- 80% coverage: **76.34% -> 81.68%**
- 90% coverage: **83.87% -> 88.22%**

Disposition:
`BETA_BINOMIAL_ALLOCATION_UNCERTAINTY_HISTORICAL_PASS`

## RB target share — FAIL CLOSED

2023 and 2024 pass the frozen CRPS/inference/log-score gates.

2025:
- rows: 1,493
- changed: 487 (**32.62%**)
- CRPS point improvement: **+0.00203**
- game-cluster 95% CI: **[-0.00132, +0.00548]**
- player-cluster 95% CI: **[-0.00159, +0.00600]**
- mean paired NLL improvement remains positive, but log score cannot rescue the
  failed CRPS replication gate.

Disposition:
`BETA_BINOMIAL_ALLOCATION_UNCERTAINTY_HISTORICAL_FAIL_CLOSED`

Do not tune RB target-share variance or pool it with RB carry.

## Interpretation

This is the first downstream uncertainty implementation in the current
individual-opportunity roadmap to improve a proper probabilistic allocation
score with three-season replication while preserving the mean exactly.

Supported mechanisms:
- **RB carry-share uncertainty**
- **WR target-share uncertainty**

Unsupported under this exact family:
- RB target-share uncertainty
- TE target-share uncertainty (excluded upstream by V2)

This does not alter the mean allocator.

## Next legal move

For RB carry and WR target only:

1. freeze `excess_share_var` prospectively before Week-6 outcomes;
2. run a 2026 prospective conditional allocation shadow;
3. design a separately frozen **joint team-conserving** simulator mechanism;
4. preserve player means, team opportunity totals, and semantic RNG isolation;
5. do not production-promote without prospective evidence.

The marginal historical PASS does not itself authorize replacing canonical
multinomial allocation.
