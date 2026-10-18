# PRISM WR/TE Player Signed Component Bias V1 — Result

Date: 2026-10-18

Status: **COMPLETE / PARTIAL QUALIFICATION / RESEARCH ONLY**

Model: **PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_WR_TE_PLAYER_SIGNED_COMPONENT_BIAS_V1_SIGNAL`

Frozen-before-scoring commit:
`dcf70236709d0f586c72bcaa4e3550dbefe31efb`

Implementation head:
`ac31b82907c416f76ffb078e9b2af8b6100a9ea6`

Run:
`39109423266`

Artifact:
`12128983909`

Digest:
`sha256:bb14d6474778d0afd53ca74bb60237b381aa4e973195bab3d09d54ed87ef2490`

Issue #535 result comment:
`6247355342`

Disposition:
`PRISM_WR_TE_PLAYER_SIGNED_COMPONENT_BIAS_SIGNAL_QUALIFIED_PARTIAL`

## Executive result

The player-specific architecture produced a real split:

- **WR opportunity: QUALIFIED**
- **TE opportunity: QUALIFIED**
- WR efficiency: CLOSED
- TE efficiency: CLOSED

The useful memory is therefore not generic “this player is reliable” or “this
player's mechanism is stable.”

The useful memory is more specific:

> For WR and TE opportunity, a named player's strictly-prior signed component
> error contains stable information about the direction/magnitude of his next
> opportunity-component error beyond a target-player-excluded position prior.

That statement did **not** survive for efficiency.

This is consistent with the PRISM direction that player-specific mechanisms
may legitimately differ by component rather than inheriting one permanent
position-wide rule.

## Frozen signal

For each target row:

Broader prior:
- same position;
- same season;
- strictly earlier NFL weeks only;
- excludes the target player's own history;
- same-week outcomes excluded.

Player evidence:
- same player's strictly-prior signed component error;
- source-certified prior 4-8 rows, max prior 8.

Fixed shrinkage:
- exactly four broader-prior pseudo-observations.

`hierarchical_signed_bias = (n * player_prior8_signed_bias + 4 * position_prior_mean) / (n + 4)`

Baseline meta-prediction:
`position_prior_mean`

Candidate meta-prediction:
`hierarchical_signed_bias`

Target:
the next signed component error.

No coefficient was fit.
No pseudo-count was searched.
No projection correction was applied.

## Sign convention

Opportunity error is:

`(predicted_targets - actual_targets) * predicted_YPT`.

Therefore:
- positive opportunity error = PRISM overprojected opportunity;
- negative opportunity error = PRISM underprojected opportunity.

Efficiency error is:

`actual_targets * (predicted_YPT - actual_YPT)`.

This result does not authorize an efficiency correction.

## WR opportunity — QUALIFIED

Development:
2023.

Support:
- 748 rows;
- 130 players.

Performance:
- mean meta-AE gain: **+0.193690**
- RMSE: **22.538774 -> 22.328533**
- Spearman: **-0.0367 -> +0.1596**

Independent confirmation:
2024.

Support:
- 1,190 rows;
- 145 players.

Performance:
- mean meta-AE gain: **+0.903962**
- player-cluster 95% CI lower bound: **+0.430131**
- team-week-cluster 95% CI lower bound: **+0.539409**
- equal-player 95% CI lower bound: **+0.365994**
- named-player improved share: **56.14%**
- median named-player improvement: **+0.188076**
- p10 named-player improvement: **-1.912722**
- sign accuracy diagnostic: approximately **60.00% -> 59.83%**

All frozen qualification gates passed.

Disposition:
`WR_OPPORTUNITY_PLAYER_SIGNED_COMPONENT_BIAS_SIGNAL_QUALIFIED`

## WR efficiency — CLOSED

Development meta-AE gain:
**-0.672676**

Independent confirmation meta-AE gain:
**-0.871068**

Confirmation player-cluster CI lower bound:
**-1.130509**

Named-player improved share:
**28.95%**

Median named-player improvement:
**-0.711722**

The efficiency signal failed materially, not marginally.

Disposition:
`WR_EFFICIENCY_PLAYER_SIGNED_COMPONENT_BIAS_SIGNAL_CLOSED`

Do not rescue with:
- alternative shrinkage;
- different history windows;
- rank splits;
- sign-only thresholds;
- player subsets;
- volatility/trust transforms.

## TE opportunity — QUALIFIED

Development:
2024.

Support:
- 609 rows;
- 78 players.

Performance:
- mean meta-AE gain: **+1.022134**
- RMSE: **19.206849 -> 18.111011**
- Spearman: **-0.1732 -> +0.2612**

Independent confirmation:
2025.

Support:
- 627 rows;
- 85 players.

Performance:
- mean meta-AE gain: **+0.760106**
- player-cluster 95% CI lower bound: **+0.385476**
- team-week-cluster 95% CI lower bound: **+0.376867**
- equal-player 95% CI lower bound: **+0.456058**
- named-player improved share: **63.49%**
- median named-player improvement: **+0.560161**
- p10 named-player improvement: **-0.930635**

All frozen qualification gates passed.

Disposition:
`TE_OPPORTUNITY_PLAYER_SIGNED_COMPONENT_BIAS_SIGNAL_QUALIFIED`

## TE efficiency — CLOSED

Development meta-AE gain:
**-0.270804**

Independent confirmation meta-AE gain:
**-0.401925**

Confirmation player-cluster CI lower bound:
**-0.623114**

Named-player improved share:
**28.57%**

Disposition:
`TE_EFFICIENCY_PLAYER_SIGNED_COMPONENT_BIAS_SIGNAL_CLOSED`

Do not rescue.

## Architectural meaning

This result is evidence for player-specific rule learning, but in a constrained
form.

It supports:
- named-player opportunity-bias memory;
- strict-prior hierarchical shrinkage;
- player-specific mechanism eligibility;
- component-specific personalization.

It does **not** support:
- unrestricted per-player models;
- generic player “trust” scores;
- generic efficiency memory;
- treating a qualified meta-signal as permission to change projections
  directly.

The next question is not whether the signal exists.

The next question is whether the signal can be translated through the actual
finite-opportunity allocation mechanisms without harming individual players.

## Downstream bridge state

Source/topology bridge work is separately frozen.

WR requires:
- exact WR-R15 authority recovery;
- real R15 anchor/secondary hierarchy;
- separate anchor-vs-secondary deployment eligibility.

TE requires:
- exact TE-R5P pool conservation.

No projection action is authorized by this result document.

## Integrity

- component means changed: **0**
- projection means changed: **0**
- coefficients fit: **0**
- thresholds searched: **0**
- 2026 outcomes used: **0**
- Week-6 2026 outcomes used: **0**
- sportsbook inputs used: **0**
- OddsAPI calls: **0**
- production changed: **false**
