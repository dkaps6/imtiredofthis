# PRISM Player Mechanism-Instability Trust V1 — Historical Source Audit

Date: 2026-10-17

Status: **FROZEN SOURCE AUDIT / NO RELIABILITY SCORING YET**

Model:
**PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_PLAYER_MECHANISM_INSTABILITY_TRUST_V1_SOURCE_AUDIT`

## Question

The Week-4/5 player-first replay showed that the same named player can fail through
opportunity one week and efficiency the next. Before testing whether that
mechanism switching predicts future projection fragility, verify that an
independent historical panel preserves exact named-player weekly component
identity.

This source audit asks only:

> Do we have source-valid historical same-player rows where final receiving-yard
> error is exactly decomposed into opportunity and efficiency components, with
> enough chronology to construct a strictly-prior mechanism-switching state?

No trust score is evaluated here.

## Exact historical authorities

### WR
- run `37634044900`
- artifact `11487359185`
- digest `sha256:13bb2df4d239f9f2832370520c3d31a08ed4f39c782766fbfb96295edc11c4e3`
- seasons: 2023 development / 2024 independent confirmation
- parent disposition: `WR_PLAYER_PERSISTENCE_MIXED_MECHANISM`

Required row fields:
- season / week / team / player_clean_key
- prior_games
- current_opportunity_error
- current_efficiency_error
- current_total_error
- prior8_abs_opportunity_error_mean
- prior8_abs_efficiency_error_mean

### TE
- run `37631011196`
- artifact `11486082927`
- digest `sha256:1b208531a8cf34735760081b7a232c4cd164a551c325f518d33428a50d7da8ab`
- seasons: 2024 development / 2025 independent confirmation
- parent disposition: `TE_PLAYER_PERSISTENCE_MIXED_MECHANISM`

Same required component fields apply.

## Exact identity requirement

For every source row:

`current_total_error = current_opportunity_error + current_efficiency_error`

Maximum absolute gap must be <= 1e-10.

Player-season-week identity must be unique.

No same/future observation may be used in the eventual pregame state.

## Chronology support check

The eventual V1 candidate, if source-ready, will require:
- same player;
- same season;
- prior rows only;
- latest up to 8 prior component-state rows;
- minimum 4 prior component-state rows.

This source audit may count rows satisfying that chronology support, but may not
score whether their future errors are larger or smaller.

Frozen support gate per position-season:
- at least 250 target rows with >=4 strictly-prior component rows;
- at least 50 distinct players.

These gates are source/readiness gates only and may not be lowered after audit.

## Position boundary

WR and TE are the only V1 candidate families.

RB:
- current project has strong RB player-difficulty and allocation science;
- no exact parallel historical opportunity+efficiency mechanism-persistence panel
  has been certified for this V1;
- therefore RB is `SOURCE_NOT_PROVEN_FOR_THIS_V1`, not approximated.

QB:
- excluded by anti-retest;
- prior QB internal-disagreement / opportunity-efficiency reliability work is
  already closed or exhausted;
- this V1 must not reopen QB reliability.

## If source-ready

Freeze a separate mechanism-instability trust test before scoring.

The source audit does **not** authorize:
- a mean change;
- a distribution-width change;
- a betting selector;
- a threshold;
- a production trust label.

## Integrity

- Week-4/5 rows used to score candidate: 0
- 2026 outcomes used: 0
- sportsbook inputs: 0
- OddsAPI calls: 0
- models fit: 0
- thresholds searched: 0
- production changed: false

Disposition at freeze:
`PRISM_PLAYER_MECHANISM_INSTABILITY_TRUST_V1_SOURCE_AUDIT_FROZEN`.
