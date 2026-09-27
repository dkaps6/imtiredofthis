# Bayesian Current-State Transmission V1 — Frozen Result

Disposition: **BAYESIAN_CURRENT_STATE_TRANSMISSION_SYSTEMIC_MISMATCH_CONFIRMED**

Production changed: **false**

## Authority

- parent main at freeze: `37deff5b51b5ec48117556b91d12dba1d4ca1fad`
- research branch: `research-bayesian-current-state-transmission-v1`
- frozen plan commit: `a740e2b489abbcf1619de3c27a78e447dc2a5a3b`
- implementation head/run head: `52f4619164af81582ba25b8efcb2b26c88e8b158`
- authoritative run: `36330210860` = **SUCCESS**
- artifact: `10935860713`
- digest: `sha256:0423d3e2e4044880d8894297c154319b26dc9f07d41ed05da3a5851708228293`
- scored rows: `6,785`
- candidate variants: `0`
- parameters fit: `0`
- sportsbook inputs: `0`
- 2026 outcomes: `0`

Focused frozen-contract tests passed before scoring.

## Question

Current-Season State Persistence V1 already showed that the exact production
PlayerForm four-game pseudo-prior blend improves next-game opportunity state.

Production then feeds the split prior/current evidence through a second
empirical-Bayes layer. For target/rush share the exact production posterior adds:

- position-group prior strength = `3.0`;
- prior-player cap = `6.0`;
- current evidence weight = `current_games`.

For a veteran with >=6 prior games and two completed current games:

- PlayerForm current-state weight = `2/(4+2) = 0.333333`;
- production Bayes current-state weight = `2/(3+6+2) = 0.181818`.

This audit asked whether that downstream posterior improves or degrades the
already-validated opportunity update.

## Integrity

Independent artifact review after the run:

- duplicate scored player/week/metric rows = **0**;
- scored temporary identities = **0**;
- nonfinite PlayerForm predictions = **0**;
- nonfinite Bayesian predictions = **0**;
- nonfinite actuals = **0**;
- 2024 scored rows = **3,444**;
- 2025 scored rows = **3,341**.

Target-week outcomes were joined only after both predictions were frozen.

## Primary result

Positive paired AE delta means the Bayesian posterior has larger absolute error
than PlayerForm.

| Position / metric | 2024 PlayerForm MAE | 2024 Bayes MAE | Delta | 2025 PlayerForm MAE | 2025 Bayes MAE | Delta | Frozen status |
|---|---:|---:|---:|---:|---:|---:|---|
| RB rush share | 0.124324 | 0.135835 | +0.011511 | 0.112230 | 0.125110 | +0.012881 | PLAYERFORM_BETTER |
| WR target share | 0.062149 | 0.064143 | +0.001994 | 0.061479 | 0.063904 | +0.002425 | PLAYERFORM_BETTER |
| TE target share | 0.048414 | 0.050345 | +0.001931 | 0.045134 | 0.046843 | +0.001709 | PLAYERFORM_BETTER |

Relative MAE improvement of PlayerForm versus Bayes:

- RB rush share: **8.47% in 2024**, **10.30% in 2025**;
- WR target share: **3.11% in 2024**, **3.79% in 2025**;
- TE target share: **3.83% in 2024**, **3.65% in 2025**.

Player-clustered paired-bootstrap 95% intervals for the Bayes-minus-PlayerForm
AE delta are entirely positive in all six season/metric cells.

## Two-completed-game / Week-3 analogue

| Position / metric | n | PlayerForm MAE | Bayes MAE | Relative PF gain | AE delta | 95% clustered CI | P(PF better) |
|---|---:|---:|---:|---:|---:|---|---:|
| RB rush share | 182 | 0.116184 | 0.126933 | 8.47% | +0.010749 | [0.002179, 0.019754] | 0.9928 |
| WR target share | 294 | 0.055930 | 0.059806 | 6.48% | +0.003876 | [0.001159, 0.006594] | 0.9988 |
| TE target share | 172 | 0.038839 | 0.042238 | 8.05% | +0.003399 | [0.000815, 0.005900] | 0.9962 |

The frozen systemic gate therefore passes exactly.

## Veteran two-game slice

For `current_games == 2` and `prior_games >= 6`:

- RB rush share remains PlayerForm-better by `+0.008810` AE;
- WR target share remains PlayerForm-better by `+0.002599` AE;
- TE target share remains PlayerForm-better by `+0.001511` AE.

The WR/TE veteran-slice bootstrap intervals cross zero narrowly, so those slices
are supporting diagnostics, not the classification authority. The all-player
two-game frozen cohort remains positive and strongly supported for all three.

## Interpretation

The result is not a generic claim that Bayesian shrinkage is bad.

It is a production-order finding for these three fast-moving opportunity
metrics:

1. production PlayerForm already performs a validated historical/current blend;
2. the downstream empirical-Bayes layer re-combines the same evidence with an
   additional position-group prior and a larger effective old-information mass;
3. for RB rush share, WR target share and TE target share, that second
   recombination is less accurate out of sample in both tested seasons.

This is therefore a concrete weighting/recombination contradiction, not a
sportsbook result and not a Week-3 outcome fit.

## What this does NOT authorize

Do **not**:

- retune Bayes group strengths;
- retune prior-player caps;
- introduce position-specific fit weights;
- create RB/QB/depth/top-N exceptions;
- modify Rush Pool Evidence Guard or revive that closed family;
- change Bayesian efficiency metrics such as YPC/YPT/YPA;
- use Week-3 outcomes to choose a repair;
- mutate production from this audit alone.

## Authorized next step

Freeze a separate **production-order opportunity-authority candidate** before
historical full-stack scoring.

The candidate may test only this structural question:

> For RB rush share and WR/TE target share, should downstream rules consume the
> already-validated PlayerForm opportunity blend instead of re-shrinking those
> same metrics through the production Bayesian posterior?

All other Bayesian metrics and all downstream football rules must remain
unchanged. The candidate requires full-stack historical validation before any
production proposal.
