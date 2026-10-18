# PRISM WR/TE Player Component-Dominance Routing V1 — Result

Date: 2026-10-18

Status: **CLOSED ALL / RESEARCH ONLY / NO PROJECTION CHANGE**

Model: **PRISM — Player Role & Individual Situation Model**

Frozen contract:
`f72e8c644c8cac0280fddb39ba4e02cc96c30931`

Authoritative execution:
- run `39153337388` — SUCCESS;
- exact head `85762e646514ba5aa09ad8dff15da1228e1a976c`;
- artifact `12145318782`;
- artifact name `prism-wr-te-player-component-dominance-routing-v1-39153337388`;
- digest `sha256:65f429583508247c3e644a5cc810b2892ae396e693655bf9577d5965485d53d4`;
- Issue #535 result comment `6253408739`.

Disposition:

`PRISM_WR_TE_PLAYER_COMPONENT_DOMINANCE_ROUTING_V1_CLOSED_ALL`

## Question

Does the named player's own strictly-prior opportunity-vs-efficiency dominance
history improve prediction of which receiving-yard error component will dominate
his next game, relative to a target-player-excluded position prior?

This was a player-specific mechanism-routing question only.

It did not change a projection mean or width.

## WR

Development 2023:
- rows: 748;
- players: 130;
- Brier gain: **-0.026438**;
- log-loss gain: **-0.043161**;
- equal-player Brier gain: **-0.033335**.

Independent confirmation 2024:
- rows: 1,190;
- players: 145;
- Brier gain: **-0.045366**;
- log-loss gain: **-0.076063**;
- player-cluster 95% CI lower bound: **-0.056594**;
- team-week-cluster 95% CI lower bound: **-0.058736**;
- equal-player 95% CI lower bound: **-0.062156**;
- median named-player Brier improvement: **-0.049313**;
- improved players: **22.07%**;
- worsened players: **77.93%**;
- p10 named-player improvement: **-0.123767**.

WR failed both the development-direction and independent-confirmation gates.

## TE

Development 2024:
- rows: 609;
- players: 78;
- Brier gain: **-0.028756**;
- log-loss gain: **-0.054421**;
- equal-player Brier gain: **-0.021882**.

Independent confirmation 2025:
- rows: 627;
- players: 85;
- Brier gain: **-0.015709**;
- log-loss gain: **-0.030151**;
- player-cluster 95% CI lower bound: **-0.030726**;
- team-week-cluster 95% CI lower bound: **-0.034686**;
- equal-player 95% CI lower bound: **-0.029176**;
- median named-player Brier improvement: **-0.033737**;
- improved players: **40.00%**;
- worsened players: **60.00%**;
- p10 named-player improvement: **-0.106052**.

TE also failed both development and independent confirmation.

## Interpretation

The result is not ambiguous:

> a named WR/TE's recent component-dominance fingerprint did not add useful
> next-game routing information beyond the frozen position prior.

The candidate worsened proper scoring in both development and confirmation for
both positions.

This closes the exact family:
- 1.5x opportunity/efficiency/mixed dominance classes;
- same-season latest-8 / minimum-4 player state;
- four position-prior pseudo-observations;
- component-class prediction as a routing signal.

Do not rescue with:
- another dominance ratio;
- entropy/Gini;
- recent-4 variants;
- WR1/WR2+/TE1 subsets;
- pooled WR+TE scoring;
- signed-bias thresholds;
- high-error-only cohorts;
- sportsbook information.

The prior CLOSED mechanism-instability trust result remains closed.

The upstream signed-opportunity results remain exactly as previously adjudicated.

## Next frontier

With:
- WR signed-opportunity bridge blocked by immutable source identity;
- TE signed-opportunity bridge blocked by the frozen combined recovery gate;
- exact historical WR/TE uncertainty integration source-blocked;
- component-dominance routing now CLOSED ALL;

the separate QB passing-TD market-specific source lane is a valid distinct
frontier.

Its source has already been certified:

`PRISM_QB_PASS_TD_MARKET_SPECIFIC_SOURCE_READY`

No downstream QB passing-TD predictive result is implied by this WR/TE closure.

## Integrity

- production projection means changed: **0**;
- distributions changed: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 outcomes used: **0**;
- Week-6/Week-7 outcomes used: **0**;
- sportsbook inputs used: **0**;
- OddsAPI calls: **0**;
- production changed: **false**.
