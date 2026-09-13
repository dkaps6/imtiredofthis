# Decision-Gate (STRONG/LEAN) Mechanics Diagnosis V1

**STATUS: RESEARCH ONLY. NO PRODUCTION/MODEL/WEIGHT/THRESHOLD CHANGE.**

## Background

GPT-5.6's checkpoint 22 (Issue #535) found `grade_full_stack_vegas_benchmark_v1.py`'s
STRONG_EDGE gate (`EV >= 0.05 AND prob_edge >= 0.03`) fires on ~87-93% of graded
rows even under the corrected empirical translator, and that this rate does not
meaningfully vary across `component_sd` quartiles — ruling out "some rows are
more miscalibrated than others" as the explanation. Checkpoint 38 pinned finding
the actual mechanical cause (independent of football-model quality) as an
approved next step.

## Mechanism

`grade_full_stack_vegas_benchmark_v1.py` computes the model's fair probability as:

```
p_over = norm.cdf((proj - line) / component_sd)
component_sd = std(mc_proj, ml_proj, state_proj)  # cross-component disagreement
```

`component_sd` measures how much three internal, correlated sub-model estimates
of the *same* underlying quantity disagree with each other — not how much the
*actual outcome* varies from any single best estimate. Those are different
quantities: real player-prop outcomes have large irreducible game-to-game
variance (injuries, game script, weather, opponent) that three internal models
built from largely the same inputs will never disagree about, because they all
lack that information the same way.

If `component_sd` systematically understates the true outcome spread, the
`Normal(proj, component_sd)` used for `p_over` is too narrow. A too-narrow
normal pushes `p_over`/`p_under` toward 0/1 for almost any `proj` vs `line` gap
— which is exactly what a market maker's genuinely calibrated ~50/50-ish
no-vig probability is not doing — manufacturing a large apparent model-vs-market
edge on nearly every row, regardless of whether the underlying football
projection is any good.

## Empirical result

`scripts/research/diagnose_strong_gate_component_sd_v1.py`, run against the
already-committed, already-graded
`docs/research/overnight/clean_v1_full_stack_vegas_benchmark_detail.csv`
(17,715 rows, no new simulation, no new data pull):

| market | component_sd (median) | empirical resid SD (std of proj-actual) | ratio | STRONG rate |
|---|---:|---:|---:|---:|
| pass_yards | 18.70 | 74.72 | **0.25** | 88.8% |
| rec_yards | 7.29 | 28.47 | **0.26** | 91.5% |
| receptions | 0.69 | 2.08 | **0.33** | 93.8% |
| rush_rec_yards | 12.57 | 39.15 | **0.32** | 94.5% |
| rush_yards | 8.59 | 28.94 | **0.30** | 88.8% |

`component_sd` is **3-4x too small relative to true empirical residual
spread, in every single market.**

Quartile breakdown (full detail: `docs/research/overnight/decision_gate_component_sd_diagnosis_v1.csv`)
confirms this is not a "low-disagreement rows are worse" pattern — it holds at
every quartile, including the *highest*-disagreement one:

| market | Q1 (lowest component_sd) ratio | Q4 (highest component_sd) ratio |
|---|---:|---:|
| pass_yards | 0.10 | 0.46 |
| rec_yards | 0.15 | 0.44 |
| receptions | 0.18 | 0.52 |
| rush_rec_yards | 0.17 | 0.51 |
| rush_yards | 0.18 | 0.49 |

Even the highest-disagreement quartile is still undersized by roughly 2x. This
is exactly why GPT-5.6's quartile analysis found no clean gradient in STRONG
rate: the gate over-fires at every quartile because `component_sd` is
uniformly, not selectively, too narrow.

## Interpretation

This is a **purely downstream, decision-layer defect**, independent of the
quality of the underlying MC/ensemble football projection: the STRONG gate's
false-positive rate is driven by feeding it the wrong notion of uncertainty
(cross-component agreement) rather than a true outcome-uncertainty estimate.
This is the same distinction the empirical-translator fidelity work (PR #546,
#548, #549, #551) has been drawing all session between the legacy
`Normal(mean, component_sd)` approximation and the actual simulated MC outcome
distribution — the simulated distribution's spread is a measure of real
projected outcome variance; `component_sd` never was.

**No fix is proposed or implemented here.** The natural remediation — replacing
`component_sd` with the empirical MC outcome distribution's own spread (already
validated and reused throughout tonight's fidelity work) as the gate's
uncertainty input — would change live decision-gate behavior and is exactly the
kind of production/threshold change that requires explicit owner approval
before any implementation is attempted. This diagnosis stops at "here is the
mechanical cause, quantified," per the task's scope.

## Reproducibility

```
python scripts/research/diagnose_strong_gate_component_sd_v1.py \
  --detail docs/research/overnight/clean_v1_full_stack_vegas_benchmark_detail.csv \
  --out docs/research/overnight/decision_gate_component_sd_diagnosis_v1.csv
```

Tests: `tests/test_diagnose_strong_gate_component_sd_v1.py` (synthetic fixture
constructed to mirror the real undersize pattern; verifies the diagnostic
computes ratios/quartiles correctly and fails closed on missing columns).
