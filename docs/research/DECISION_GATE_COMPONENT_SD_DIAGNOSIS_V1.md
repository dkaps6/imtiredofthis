# Decision-Gate (STRONG/LEAN) Mechanics Diagnosis V1

**STATUS: RESEARCH ONLY. NO PRODUCTION/MODEL/WEIGHT/THRESHOLD CHANGE.**

## Background

GPT-5.6's checkpoint 22 (Issue #535, run `34712931786`, artifact
`10304625925`) quartiled `legacy_component_sd` and compared STRONG coverage
under two translators. The **legacy** `Normal(mean, component_sd)` translator
fires flat and extreme in every quartile (e.g. rec_yards 93.3% Q1 down to only
89.6% Q4 — if anything *higher* in the low-disagreement quartile), ruling out
"only the high-disagreement rows are miscalibrated" as the explanation for
*that* translator. The **empirical MC** translator, by contrast, shows a real
gradient (rec_yards 74.4% Q1 up to 90.5% Q4) — but its aggregate STRONG
coverage still sits at 86.81% non-QB, and high-variance-quartile rows still
clear STRONG at 90-96% even when the uncertainty measure is the real simulated
distribution, not a proxy. Checkpoint 38 pinned finding the mechanical cause
of the *legacy* translator's flat over-firing as an approved next step; this
diagnosis addresses that part only (see "Scope and limits" below for what it
does not address).

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
quality of the underlying MC/ensemble football projection: the *legacy*
translator's STRONG gate false-positive rate is driven by feeding it the wrong
notion of uncertainty (cross-component agreement) rather than a true
outcome-uncertainty estimate. This is the same distinction the
empirical-translator fidelity work (PR #546, #548, #549, #551) has been
drawing all session between the legacy `Normal(mean, component_sd)`
approximation and the actual simulated MC outcome distribution — the
simulated distribution's spread is a measure of real projected outcome
variance; `component_sd` never was.

**No fix is proposed or implemented here.** Replacing `component_sd` with the
empirical MC outcome distribution's own spread would change live decision-gate
behavior and requires explicit owner approval before any implementation is
attempted, regardless of the scope question below.

## Scope and limits of this diagnosis (per GPT-5.6 checkpoint 28's correction)

This diagnosis fully explains the *legacy* translator's flat, uniform
over-firing: `component_sd` is undersized at every quartile, not just the low
end, so the gate manufactures extreme probabilities regardless of row
disagreement. **It does not explain the residual over-firing that survives
under the empirical MC translator.** GPT-5.6's checkpoint 22 already measured
that residual directly (86.81% non-QB STRONG coverage in aggregate, 90-96% in
the highest-variance quartiles) using the real simulated-outcome distribution
— not a proxy — so `component_sd`'s undersizing cannot be the explanation for
that remaining share. An earlier version of this document's Interpretation
section overstated the empirical-MC remediation's expected effect; corrected
here.

The correct, narrower taxonomy, per checkpoint 28:
- `component_sd`-as-outcome-variance defect (legacy translator only): **REPRODUCED and quantified by this diagnosis**.
- `component_sd` as the complete explanation for STRONG over-firing under the corrected empirical translator: **not supported — a second, independent gate-mechanics issue remains**.

The remaining audit checkpoint 22 calls for — isolating the no-vig/EV/`prob_edge`
gate algebra and its joint distribution on the empirical-MC cohort itself (not
the legacy-translator cohort this diagnosis used) — requires the
`historical-fair-probability-reconstruction-v1` artifact GPT-5.6 produced
(run `34712931786`, artifact `10304625925`), which this diagnosis did not have
access to. That continuation is not attempted here.

## Reproducibility

```
python scripts/research/diagnose_strong_gate_component_sd_v1.py \
  --detail docs/research/overnight/clean_v1_full_stack_vegas_benchmark_detail.csv \
  --out docs/research/overnight/decision_gate_component_sd_diagnosis_v1.csv
```

Tests: `tests/test_diagnose_strong_gate_component_sd_v1.py` (synthetic fixture
constructed to mirror the real undersize pattern; verifies the diagnostic
computes ratios/quartiles correctly and fails closed on missing columns).
