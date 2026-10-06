# Opportunity State Conflict Uncertainty V1 — Frozen Plan

Status: **FROZEN BEFORE SCORING — RESEARCH ONLY — NO PRODUCTION CHANGE**

Branch:
`research-opportunity-state-conflict-uncertainty-v1`

Parent scientific authority:
- `BAYESIAN_CURRENT_STATE_TRANSMISSION_SYSTEMIC_MISMATCH_CONFIRMED`
- authoritative run `36330210860`
- artifact `10935860713`
- digest `sha256:0423d3e2e4044880d8894297c154319b26dc9f07d41ed05da3a5851708228293`

## Why this exists

The parent audit established that production contains two distinct, leakage-safe
opportunity-state estimates:

1. the faster PlayerForm four-game pseudo-prior blend; and
2. the slower downstream empirical-Bayes posterior.

For RB rush share, WR target share and TE target share, PlayerForm was more
accurate than Bayes in both 2024 and 2025. The obvious production-order mean
replacement was then tested separately in Opportunity Authority Priority V1 and
**FAILED CLOSED**. That exact mean candidate remains closed and may not be
rescued by an RB-only or TE-only version.

Week-4 postmortem work later observed the same architectural disagreement in
live pregame state, but no Week-4 outcome may be used to fit or tune this study.

This V1 asks a genuinely different question:

> Does the magnitude of disagreement between the fast PlayerForm opportunity
> state and the slower Bayes opportunity state identify player-weeks where the
> existing production opportunity estimate is more uncertain?

The hypothesis concerns **uncertainty**, not which mean should replace which.

## Frozen signal

For each row in the exact historical panel from the parent audit:

`state_conflict = abs(playerform_value - bayes_value)`

No sign is used. No sportsbook line, odds, edge, model price or target-game
information enters the signal.

## Frozen target

Primary error target:

`bayes_abs_error = abs(bayes_value - actual_value)`

This deliberately evaluates uncertainty around the existing downstream
production opportunity authority. It does not retroactively substitute
PlayerForm as the mean.

## Exact scope

Same three opportunity states as the parent audit:

1. RB `rush_share`
2. WR `tgt_share`
3. TE `tgt_share`

Historical seasons are scored independently:
- 2024 Weeks 2-18, with 2023 prior
- 2025 Weeks 2-18, with 2024 prior

No 2026 outcome is allowed.

## Primary tests

For each position/metric and each season independently:

### Test A — monotone uncertainty association
Spearman correlation between:
- `state_conflict`
- `bayes_abs_error`

Required sign: **> 0**.

### Test B — high-conflict error separation
Within each season/metric, assign deterministic quartiles from the empirical
distribution of `state_conflict`.

Compare:
- Q4 highest-conflict mean Bayes absolute error
- Q1 lowest-conflict mean Bayes absolute error

Primary effect:

`Q4_minus_Q1_bayes_AE`

Required sign: **> 0**.

### Test C — player-clustered uncertainty
Bootstrap players, preserving all player-weeks inside each sampled player
cluster.

- reps: 10,000
- seed: 20261006
- statistic: Q4 minus Q1 mean Bayes absolute error

Required: **95% CI lower bound > 0**.

## Support floor

A position/metric/season cell is scoreable only with:
- >= 100 rows total;
- >= 20 unique players;
- nonzero variance in `state_conflict`;
- nonempty Q1 and Q4.

Otherwise it is `INSUFFICIENT_SUPPORT`.

## Frozen disposition

A position/metric is:

`STATE_CONFLICT_UNCERTAINTY_REPLICATED`

only if, in **both 2024 and 2025**:
1. support floor passes;
2. Spearman > 0;
3. Q4-minus-Q1 Bayes AE > 0;
4. player-cluster bootstrap 95% CI lower bound > 0.

Otherwise:

`STATE_CONFLICT_UNCERTAINTY_NOT_REPLICATED`

Overall:
- if >=1 position/metric replicates: `OPPORTUNITY_STATE_CONFLICT_UNCERTAINTY_SIGNAL_CONFIRMED`;
- if none replicate: `OPPORTUNITY_STATE_CONFLICT_UNCERTAINTY_NULL`;
- any lineage/leakage/cohort failure: `OPPORTUNITY_STATE_CONFLICT_UNCERTAINTY_INTEGRITY_FAILURE`.

The per-metric classification is frozen in advance. A positive RB result and
negative WR/TE result may be reported as such; that is not a post-hoc carveout.

## Anti-retest / closed-family protection

This study MUST NOT:
- replace Bayes means with PlayerForm means;
- retune `GROUP_STRENGTH` or `PRIOR_PLAYER_CAP`;
- create an RB-only version of Opportunity Authority Priority V1;
- revive Rush Pool Evidence Guard;
- modify top-N/depth/share allocation rules;
- fit a sportsbook selector;
- use Week-3/Week-4 outcomes to select a threshold;
- globally widen every market's distribution;
- search multiple disagreement formulas or thresholds.

Only the one absolute-disagreement signal above is tested.

## What a PASS would authorize

A replicated signal does **not** authorize production.

It authorizes exactly one separately frozen follow-up:

> a mean-neutral distribution/uncertainty candidate that uses the preregistered
> state-conflict signal without changing the football mean.

That future candidate must be frozen before it is scored, must preserve the
closed mean-replacement result, and must pass genuine out-of-sample/full-stack
distribution gates.

## What a FAIL means

If no metric replicates across both seasons, this exact state-conflict
uncertainty idea closes. Do not tune the conflict formula, quartile cut,
position subsets, or Bayes constants to rescue it.
