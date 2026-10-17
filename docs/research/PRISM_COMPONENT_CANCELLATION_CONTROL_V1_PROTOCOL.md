# PRISM Component-Cancellation Control Audit V1 — Frozen Diagnostic Protocol

Date: 2026-10-17

Status: **FROZEN BEFORE IMPLEMENTATION / DIAGNOSTIC ONLY / NO PREDICTIVE PROMOTION**

Model:
**PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_COMPONENT_CANCELLATION_CONTROL_AUDIT_V1`

## 1. Player-first question

PRISM is judged on the named player in his actual football situation.

A small final projection error does not necessarily prove that PRISM understood
the player's opportunity and efficiency correctly. Opposing component errors can
cancel.

This audit asks:

> Among player-game-market rows with exact opportunity × efficiency
> decomposition, how often does a small final-output error conceal large,
> opposing component errors?

This is an evaluation/control-validity audit. It does not create a new mean,
reliability score, selector, or production rule.

## 2. Authorities

Primary exact component authority:
- 2026 Weeks 1-4 player-output decomposition;
- run `37777319154`;
- artifact `11550446909`;
- digest `sha256:7acfc453258a49f3aadf0d0ea386b13827eb715f1dfb055b96505f733320d204`.

Supplemental Week-4/5 matched-control authority:
- run `39060246734`;
- artifact `12111007952`;
- digest `sha256:0f120e2002452af548fbea20f0ff1355ec04eb926e53b5845ca663ea2e98a5be`.

The Week-4/5 replay remains forensic. Week-6 evidence is not read.

## 3. Exact decomposition

For every component-eligible row preserve the already-established Shapley
identity:

`opportunity_contribution = (actual_opp - predicted_opp) * (pred_eff + actual_eff) / 2`

`efficiency_contribution = (actual_eff - pred_eff) * (pred_opp + actual_opp) / 2`

Define the downstream/non-factor contribution only as the exact identity residual:

`downstream_contribution = (actual_output - baseline_projection) - opportunity_contribution - efficiency_contribution`

No fitted attribution is allowed.

## 4. Cancellation metric

Let:

`G = |opp_contribution| + |eff_contribution| + |downstream_contribution|`

`E = |actual_output - baseline_projection|`

For `G > 1e-12`:

`cancellation_fraction = 1 - E / G`

Otherwise set the cancellation fraction to 0.

Properties:
- 0 means component errors do not materially cancel;
- values near 1 mean large opposing component errors net to a small final error;
- this is an algebraic diagnostic, not a prediction.

Do not fit a threshold from outcomes.

## 5. Required summaries

Report:
1. all exact component rows;
2. cancellation distribution by week / position / market;
3. cancellation distribution by within-position-market final-error decile;
4. bottom-final-error rows with their component magnitudes for inspection;
5. the replay's matched-correct controls where exact component authority exists;
6. named examples only after the population summaries.

The central comparison is descriptive:

> Does low final error systematically imply low component error?

No position-average MAE optimization follows from this audit.

## 6. Anti-retest boundary

QB M89/M90 already documented strong opportunity/efficiency cancellation.

This V1 does **not** reopen or retune QB mean science. QB rows may remain in the
control-validity population only to test whether final-error-only evaluation is
generally trustworthy.

Do not:
- convert cancellation into fitted weights;
- create a betting selector;
- create a reliability threshold;
- fit player-specific corrections;
- change Stage-1 or Stage-2;
- reopen raw recent-share replacement;
- change efficiency authority;
- use sportsbook information upstream.

## 7. Scientific use

If low-final-error rows frequently have large cancellation:
- future PRISM postmortems must distinguish **final-number correctness** from
  **mechanism correctness**;
- matched controls should expose component coherence rather than assuming a
  small final error means the player was understood;
- this may generate a separately testable reliability question later, but this
  audit itself cannot promote one.

If cancellation is rare among low-error rows, final-error matched controls remain
a stronger mechanism-validity proxy.

## 8. Integrity

At freeze:
- Week-6 outcomes read: 0;
- sportsbook inputs: 0;
- OddsAPI calls: 0;
- models fit: 0;
- thresholds searched: 0;
- projection means changed: 0;
- production changed: false.

Disposition:
`PRISM_COMPONENT_CANCELLATION_CONTROL_AUDIT_V1_FROZEN`.
