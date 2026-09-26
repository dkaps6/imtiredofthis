# Discrete Count Mean Alignment V1 — Production Integration Frozen Plan

Date frozen: 2026-09-26

Status: **FROZEN BEFORE INTEGRATION OUTPUT**

Research authority:
- `docs/research/DISCRETE_COUNT_MEAN_ALIGNMENT_V1_RESULT.md`
- authoritative replay run `36276366140`
- result artifact `10916528395`
- digest `sha256:0563d974f8c25de290aabec9c92665c8e509fce99c2b83c0ec10e9c5a5c69102`
- disposition `DISCRETE_COUNT_MEAN_ALIGNMENT_V1_QUALIFIED_FOR_INTEGRATION_TEST`

This plan does not reopen the research formula.

## Objective

Test whether the exact qualified integer-preserving count alignment can be inserted into canonical production pricing without changing:
- football means;
- ensemble weights;
- simulation inputs;
- rules;
- player universe;
- non-count distributions;
- specialist routing;
- sportsbook isolation;
- zero-MC behavior.

## Exact candidate

In `scripts/run_pricing_v2.py`, replace the generic post-ensemble alignment statement with one versioned helper.

For every market, first preserve the existing production guard:

If:
- `mc_proj > 0`, and
- final `target_mean` is finite,

construct the current continuously aligned array:

`z = base_outcomes * max(0, target_mean / mc_proj)`

Then:

### `receptions` and `rush_att`

Apply the exact qualified V1 largest-remainder projection:

1. `floor_i = floor(z_i)`
2. `K = round(sum(z)) - sum(floor)`
3. add one count to the `K` draws with largest fractional remainder;
4. stable draw index is the deterministic tie-breaker.

### Every other market

Return `z` unchanged, exactly matching current continuous production semantics.

### Production guard no-op

If current production cannot align because MC mean is non-positive/nonfinite or target is nonfinite:
- return `base_outcomes` unchanged for every market;
- do not invent count support;
- do not repair the separate zero-MC ensemble-transmission issue.

## Version / audit fields

Count-market output rows should expose:
- `discrete_count_alignment_applied`
- `discrete_count_alignment_version = DISCRETE_COUNT_MEAN_ALIGNMENT_V1`
- `discrete_count_alignment_pre_fractional_rate`
- `discrete_count_alignment_post_integer_max_gap`
- `discrete_count_alignment_target_mean_gap`

Rows outside qualified scope:
- applied = 0;
- version empty;
- numerical projections/probabilities must remain bit-identical to baseline.

## Required integration A/B

Baseline:
current `main` production pricing semantics.

Candidate:
exact V1 helper, with no other model changes.

### Historical exact-parity gate

Reuse immutable authority from run `36275245917` and research result run `36276366140`.

The production helper applied to the exact historical arrays must reproduce the qualified A1 arrays/statistics within numerical tolerance.

No refit and no second science test.

### Production-path invariance gates

Required:
1. `pass_yards` distributions/output exact baseline;
2. `rush_yards` exact baseline;
3. `rec_yards` exact baseline;
4. `rush_rec_yards` exact baseline, including RB Conservation V2;
5. `anytime_td` exact baseline;
6. QB M89/M90 routing unchanged;
7. RB P3 routing unchanged;
8. RB Rush+Receiving Conservation V2 pathwise identity unchanged;
9. ensemble means/weights unchanged;
10. ML/State/Bayesian/rule inputs unchanged;
11. player/team universe unchanged;
12. sportsbook inputs added upstream = 0;
13. zero-MC rows exact baseline no-op;
14. Week-1 protected routes unchanged except qualified count-market alignment if those markets are priced.

### Count-market mechanics gates

For applied `receptions` / `rush_att` rows:
- output arrays nonnegative integer;
- target mean gap <= `0.5 / N + 1e-12`;
- production helper output exact research helper output;
- deterministic repeatability under same input;
- no extra candidate variant.

## Current production verification

Use the strongest available replay/current authority that does **not** require a new paid OddsAPI acquisition.

If an old paid artifact has expired, do not purchase/refetch it just to satisfy this integration test.

Repo CI must pass.

A no-live-odds current Full Slate may be used for upstream/invariance certification, but it is not a substitute for count-pricing-path exercise. Historical immutable arrays are the primary exact count-path authority.

## Promotion bar

Integration passes only if:
- exact research parity passes;
- every non-scope invariance gate passes;
- count mechanics pass;
- Repo CI passes;
- no unrelated production drift is introduced.

Disposition if all pass:
`DISCRETE_COUNT_MEAN_ALIGNMENT_V1_INTEGRATION_PASS_READY_FOR_PROMOTION`

Otherwise:
`DISCRETE_COUNT_MEAN_ALIGNMENT_V1_INTEGRATION_FAILED_CLOSED`

## No rescue

Do not:
- alter the largest-remainder rule;
- add stochastic rounding;
- add position/volume carveouts;
- change the zero-MC guard;
- tune coverage;
- rescale interval width;
- alter betting thresholds;
- add a separate receptions vs rush-att method;
- use Week-3 outcomes;
- combine with the zero-MC transmission repair.

No production promotion occurs from this plan itself.
