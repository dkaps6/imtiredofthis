# Receiving Rule Semantics Integrity V1 — Week 3 Prospective Result

Status: COMPLETE — PROSPECTIVE EVIDENCE ONLY  
Date: 2026-09-29  
Frozen Stage-1 authority: run `36276736046` / artifact `10917254062`  
Frozen artifact digest: `sha256:c7ff51d17adfaf0f96e18808cec5948fa314f126f5c8b0392c8bed5568fa66fa`  
Postgame grade authority: run `36629181044`  
Postgame artifact: `11061636348`  
Postgame digest: `sha256:bbe8a30cdeb68b2b1b4df952aa6dc8632c74c1588df7f86a1b2c0b1476b1ceeb`

## Disposition

`WEEK3_PROSPECTIVE_EVIDENCE_ONLY_CONTINUE_UNCHANGED_CELLS`

No Week-3 production qualification is authorized.

All four frozen cells were scored without changing the pregame football state:
- A0B0 — current semantics
- A1B0 — middle_open percent->fraction normalization only
- A0B1 — slot-alignment carry only
- A1B1 — both repairs

The exact frozen explicit-entitlement simulator remained byte-identical to the
Stage-1 authority and was run at 25,000 draws, seed 42. No sportsbook input or
parameter fitting was used.

## A1B0 — middle_open semantic normalization

Targeted TE cohort, 87 rows:

Target share:
- MAE 0.057831 -> 0.057413, improvement -0.000418
- p90 0.108359 -> 0.108359, unchanged

Receptions:
- MAE 1.575768 -> 1.564234, improvement -0.011535
- p90 2.670208 -> 2.722872, worse +0.052664

Receiving yards:
- MAE 17.672488 -> 17.569587, improvement -0.102902 yd
- p90 28.522119 -> 28.654294, worse +0.132175 yd

Thus all three targeted TE MAEs improved, but the frozen targeted p90 non-worse
gate failed for receptions and receiving yards.

Pooled WR+TE:
- target-share MAE improved 0.064546 -> 0.064384
- receptions MAE improved 1.546000 -> 1.545167
- receiving-yards MAE worsened slightly 20.226991 -> 20.229284
- receiving-yards p90 worsened 41.552929 -> 42.495345

Therefore the pooled non-regression gate also does not clear cleanly.

## A0B1 — slot-alignment carry

Targeted frozen SWR cohort, 56 rows:

- target-share MAE 0.064436 -> 0.064674, worse
- receptions MAE 1.195809 -> 1.282276, worse
- receiving-yards MAE 18.646441 -> 19.634319, worse

Pooled WR+TE:
- target-share MAE improved only 0.000029
- receptions MAE worsened +0.011430
- receiving-yards MAE worsened +0.205074

Week-3 evidence is adverse to this exact candidate.

## A1B1 — both repairs

On the exact downstream changed-row cohort:
- target-share MAE improved only 0.000034
- receptions MAE worsened +0.004527
- receiving-yards MAE worsened +0.132387
- all three targeted p90 deltas were worse

Pooled WR+TE receiving-yards MAE worsened +0.192666 and p90 worsened +2.467953.

Week-3 evidence does not support the combined cell.

## Identity grading notes

Three postgame name-form seams were resolved without altering pregame model
identity:
- Matt Hibner -> Matthew Hibner via the existing verified current alias registry
- Drew Ogletree -> Andrew Ogletree via research-only official-source evidence
- Hollywood Brown -> Marquise Brown via research-only official-source evidence

These changes affect postgame outcome attachment only, not any pregame feature,
projection, rule, or production identity configuration.

## Boundary

- parameters fit: 0
- sportsbook inputs: 0
- production mutations: 0
- no rescue variants
- no threshold retuning
- no partial A1B0 promotion from one week

The exact frozen cells may continue prospectively if additional observation
weeks were part of the original research program, but Week 3 alone does not
justify production integration of A1B0, A0B1, or A1B1.
