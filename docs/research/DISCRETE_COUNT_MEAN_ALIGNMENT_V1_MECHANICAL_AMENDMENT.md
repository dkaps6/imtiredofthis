# Discrete Count Mean Alignment V1 — Mechanical Contract Amendment

Date: 2026-09-26

Status: **MECHANICAL CLARIFICATION ONLY — FROZEN SCIENCE UNCHANGED**

Parent plan:
`docs/research/DISCRETE_COUNT_MEAN_ALIGNMENT_V1_PLAN.md`

Authoritative first run:
- run `36275245917`
- exact historical rebuild completed through the frozen ensemble / market archive
- Step 14 stopped before scoring with:
  `RuntimeError: A0 failed exact production mean alignment: 0.0534815119650126`

## What the stop exposed

The frozen plan correctly defined A0 as current production-style mean alignment, but its mechanical gate implicitly assumed every finite ensemble target can be reached by rescaling the MC distribution.

Production `scripts/run_pricing_v2.py` has a stricter guard:

```python
if np.isfinite(mc_proj) and mc_proj > 0 and np.isfinite(target_mean):
    adjusted_outcomes = base_outcomes * max(0.0, target_mean / mc_proj)
else:
    adjusted_outcomes = base_outcomes
```

Therefore an MC distribution with mean exactly zero is intentionally left unchanged even when ML/State produce a nonzero calibrated ensemble mean.

The historical authority contains such rows, especially in `rush_att`.

## Amendment

A0 must reproduce **exact production semantics**, including the zero/nonfinite-MC no-op guard.

For rows where:
- `mc_mean > 0`, and
- target mean is finite,

the original frozen contract remains unchanged:
- A0 = exact multiplicative alignment;
- A1 = fixed integer-preserving largest-remainder projection of A0;
- A0 must equal the target mean within numerical tolerance;
- A1 target-mean error must be <= `0.5 / N + tolerance`.

For rows where production itself cannot align because `mc_mean <= 0` or is nonfinite:
- A0 must remain exact raw MC;
- A1 must also remain exact raw MC;
- the row remains in football-distribution scoring identically in both arms;
- the row is excluded only from the target-mean-alignment mechanical bound;
- no support is invented and no nonzero mass is injected.

## Why this is not a science redesign

Unchanged:
- candidate rounding algorithm;
- count-market scope;
- 2024/2025 historical cohort;
- CRPS gates;
- coverage diagnostics;
- receptions Brier/log-loss gates;
- no sportsbook-upstream rule;
- no parameter fitting;
- no threshold search;
- no subgroup rescue;
- no Week-3 outcomes.

This amendment only makes A0 faithful to the code it is supposed to reproduce.

## Separate finding — not repaired here

The stop also exposed a distinct systems-integrity issue:

> some `rush_att` rows have `mc_proj == 0` while ML/State imply a nonzero calibrated `ensemble_proj`, but production's zero-MC guard prevents that ensemble mean from reaching the final priced distribution.

That issue is **not** repaired by Discrete Count Mean Alignment V1.

It will remain a separate diagnostic/architecture lane because injecting new support into a zero-MC distribution is a different intervention from preserving integer support during ordinary mean alignment.

No production/model/weight/threshold change.
