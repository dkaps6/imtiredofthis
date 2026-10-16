# Player High-Value Opportunity Persistence V1 — Result

Date: 2026-10-15

Status: **COMPLETE / CLOSED / NO PRODUCTION CHANGE**

Frozen plan:
`docs/research/PLAYER_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_V1_FREEZE.md`

Canonical run:
- run `38848538570` — SUCCESS
- exact scored head `89b92b56d9b9786a3cc8802a3d2b645caeca9ba5`
- artifact `12024037594`
- digest `sha256:2b02f5a6c7a8a05962517f72a9d62ba287d06ffe5ced37abfd70b623a5120afe`

Parent source authority:
- run `38848080453`
- artifact `12023852453`
- digest `sha256:1d8a22bf6240b4d4443ab7432d3224b63eaef97439639e7b47f536aed174adda`
- disposition `PLAYER_HIGH_VALUE_SCORING_OPPORTUNITY_SOURCE_READY`.

## Overall disposition

`PLAYER_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_CLOSED`

No channel cleared every frozen gate.

This closes only the exact V1 claim:

> last-4 same-team named-player high-value opportunity share, min-3 games and
> min-4 team high-value opportunities, directly forecasts next-game
> high-value share better than ordinary overall opportunity share.

Do not generalize this result to "scoring-area usage is useless."

## RB rushing — CLOSED

`NO_ACTIONABLE_RB_RUSH_HIGH_VALUE_OPPORTUNITY_PERSISTENCE`

Primary high-value pool:
inside-10 team rushing attempts.

2022:
- rows 977 / players 144
- overall-share baseline MAE 0.17985
- high-value-share candidate MAE 0.18446
- gain -0.00461 — worse.

2023:
- 1,232 / 148
- 0.19532 -> 0.19489
- gain +0.00043.

2024:
- 1,262 / 133
- 0.20020 -> 0.19977
- gain +0.00043.

2025:
- 1,288 / 141
- 0.20597 -> 0.20160
- gain +0.00437.

But 2025 dependence-aware confirmation did not clear:
- player-cluster 95% CI [-0.00353,+0.01270]
- game/team-week 95% CI [-0.00415,+0.01249].

Pooled gain was only +0.00046 and both pooled clustered intervals crossed zero.

2025 median per-player gain was barely positive:
+0.000046 across 118 players with >=4 scored games.

Interpretation:
recent inside-10 rushing share did not provide a robust independent improvement
over the named RB's ordinary overall rushing share.

## RB receiving — favorable but CLOSED

`NO_ACTIONABLE_RB_RECEIVING_HIGH_VALUE_OPPORTUNITY_PERSISTENCE`

This was the only channel with directionally favorable point estimates in all
four seasons.

2022:
- 1,367 rows / 151 players
- MAE 0.07171 -> 0.07018
- gain +0.00152.

2023:
- 1,528 / 149
- 0.08154 -> 0.07682
- gain +0.00472.

2024:
- 1,528 / 137
- 0.07152 -> 0.07103
- gain +0.00049.

2025:
- 1,546 / 145
- 0.07522 -> 0.07321
- gain +0.00201.

Pooled:
- 5,969 rows / 247 RBs
- gain +0.00220
- player-cluster CI [+0.00011,+0.00419]
- game/team-week CI [+0.00090,+0.00357].

But independent 2025 confirmation remained uncertain:
- player-cluster CI [-0.00175,+0.00611]
- game/team-week CI [-0.00044,+0.00449].

2025 median per-player gain:
+0.00091; positive-player share 51.59%.

Frozen V1 therefore fails closed.
Do not rescue with a different window, band or RB subset.

## WR receiving — CLOSED / worse

`NO_ACTIONABLE_WR_RECEIVING_HIGH_VALUE_OPPORTUNITY_PERSISTENCE`

High-value red-zone share was worse than ordinary target share in every season:
- 2022 gain -0.00299
- 2023 -0.00265
- 2024 -0.00303
- 2025 -0.00190.

Pooled:
- 8,883 rows / 368 WRs
- gain -0.00263
- player-cluster CI [-0.00461,-0.00049]
- game/team-week CI [-0.00418,-0.00109].

This is strong evidence against the exact V1 WR formulation.

## TE receiving — CLOSED / clearly worse

`NO_ACTIONABLE_TE_RECEIVING_HIGH_VALUE_OPPORTUNITY_PERSISTENCE`

High-value red-zone share was worse than ordinary target share in every season:
- 2022 gain -0.00407
- 2023 -0.00840
- 2024 -0.00432
- 2025 -0.00590.

Pooled:
- 5,502 rows / 198 TEs
- gain -0.00571
- player-cluster CI [-0.00768,-0.00379]
- game/team-week CI [-0.00739,-0.00402].

2025 median per-player gain:
-0.00346; only 37.29% of supported players improved.

This exact TE high-value-share persistence hypothesis is terminal.

## What this teaches us about the Dobbins example

The result does NOT contradict the football reasoning used pregame.

It says something narrower:

**recent scoring-area share by itself is not a robust replacement for ordinary
named-player usage when forecasting who gets the next game's high-value
opportunities.**

The manual Dobbins reasoning contained more information:
- current workload;
- workload trajectory;
- current lead-back status;
- teammate competition;
- high-value usage;
- whether Denver was likely to create scoring opportunities at all.

The frozen V1 intentionally isolated only one piece:
high-value share persistence.

That piece alone is insufficient.

This is exactly why player identity must be multi-dimensional rather than a
single universal coefficient.

## Anti-rescue

Do not:
- change RB inside-10 to inside-20 or inside-5 to rescue;
- change receiving inside-20 to inside-10/end-zone;
- change last-4/min-3;
- lower the prior high-value support;
- select only lead backs/WR1s/TE1s after seeing results;
- pool channels;
- fit a coefficient;
- add TD outcomes to rescue;
- use sportsbook prices.

## Boundaries

- sportsbook inputs 0
- OddsAPI calls 0
- 2026 outcomes 0
- predictive models fit 0
- production unchanged.

Current generic ATD remains uncertified.

Canonical model status remains:
`PLAYER_INDIVIDUALIZATION_PARTIAL`.
