# Opportunity State Conflict Uncertainty V1 — Result

Status: **COMPLETE — NULL — CLOSED**  
Disposition: **OPPORTUNITY_STATE_CONFLICT_UNCERTAINTY_NULL**  
Production changed: **false**

## Authority

- branch: `research-opportunity-state-conflict-uncertainty-v1`
- frozen plan commit: `7e2f87b70f651c85a062f2063a0d3acda2a21f8f`
- optimized implementation head: `22cbd552342caeecf27a87438cc7f2c689863b9b`
- canonical run: `37492425193` = SUCCESS
- artifact: `11426127514`
- digest: `sha256:51ef028a66ffffa01608b6c5dee76429b80d6bdf9a718915b5a352e747984f4a`
- source historical panel rows: 6,785
- sportsbook inputs: 0
- 2026 outcomes: 0
- fitted parameters: 0
- mean changes: 0

Parent authority remains:
`BAYESIAN_CURRENT_STATE_TRANSMISSION_SYSTEMIC_MISMATCH_CONFIRMED`.

The already-tested Opportunity Authority Priority V1 mean replacement remains
FAILED CLOSED and was not reopened.

## Frozen question

Does:

`abs(PlayerForm opportunity state - Bayes opportunity state)`

identify player-weeks where the existing Bayesian opportunity authority is less
reliable?

The preregistered requirement was positive monotone association plus a positive
Q4-vs-Q1 absolute-error difference with player-cluster bootstrap lower CI > 0
in both 2024 and 2025.

## Results

### RB rush share

2024:
- n = 1,036
- Spearman conflict vs Bayes AE = **-0.1364**
- Q4 minus Q1 Bayes AE = **-0.04192**
- 95% player-cluster CI = **[-0.06288, -0.02190]**

2025:
- n = 957
- Spearman = **-0.0897**
- Q4 minus Q1 Bayes AE = **-0.02568**
- 95% CI = **[-0.04660, -0.00611]**

The direction is the opposite of the uncertainty hypothesis in both seasons.

### WR target share

2024:
- n = 1,551
- Spearman = **-0.0153**
- Q4 minus Q1 Bayes AE = **-0.00412**
- 95% CI = **[-0.01109, +0.00258]**

2025:
- n = 1,547
- Spearman = **-0.0357**
- Q4 minus Q1 Bayes AE = **-0.00289**
- 95% CI = **[-0.01071, +0.00478]**

No replicated positive uncertainty relationship.

### TE target share

2024:
- n = 857
- Spearman = **-0.0520**
- Q4 minus Q1 Bayes AE = **-0.00398**
- 95% CI = **[-0.01353, +0.00516]**

2025:
- n = 837
- Spearman = **-0.1080**
- Q4 minus Q1 Bayes AE = **-0.01149**
- 95% CI = **[-0.01973, -0.00350]**

Again, no positive uncertainty relationship.

## Interpretation

The known PlayerForm-vs-Bayes opportunity mismatch is real, but the magnitude of
that disagreement is **not** a useful uncertainty-width signal.

Therefore do not:
- widen distributions when PlayerForm and Bayes disagree;
- create a disagreement threshold;
- invert the signal;
- use low disagreement as a new betting selector;
- revive the failed mean-replacement candidate;
- create RB-only / TE-only rescue variants.

This exact uncertainty idea is CLOSED.

The Week-4 postmortem's larger right-tail finding must be pursued through a
different mechanism.
