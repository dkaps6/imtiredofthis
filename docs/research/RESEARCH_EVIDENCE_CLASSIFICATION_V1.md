# Research Evidence Classification V1

**STATUS: METHODOLOGY DRAFT FROZEN BEFORE WR-R18 RESULTS. NO MODEL SCIENCE CHANGE.**

## Purpose

Future research results in this repository must distinguish:

1. **Experiment disposition** — whether the exact preregistered experiment cleared its frozen gates.
2. **Scientific evidence disposition** — what repeatable, directional, seasonal, subgroup, or mechanistic information was honestly learned even when the experiment did not clear promotion gates.
3. **Production eligibility** — whether evidence is strong, replicated, stable, pregame-observable, and integration-tested enough to alter production science.

A failed experiment does not automatically imply that every measured component is scientifically useless. Conversely, a positive descriptive component does not override a failed preregistered gate.

## Core rules

- Never relabel a preregistered FAIL as PASS after inspecting a holdout or alternate season that the original contract said should remain sealed.
- Preserve meaningful sub-findings separately from the experiment disposition.
- A post-result subgroup or seasonal clue may generate a new hypothesis only if that hypothesis is frozen before any new/untouched evidence is exposed.
- Do not call a failed season an `ANOMALY` merely because other seasons pass.
- An anomaly/regime explanation requires either a prospectively identified pregame-observable football state or later independent replication supporting that explanation.
- Threshold misses near a gate and hard directional reversals are not the same scientific outcome and should not be summarized identically.
- Historical market results remain downstream and cannot rescue failed football mechanism evidence.

## Experiment disposition labels

Use the experiment's own frozen PASS/FAIL/DATA labels as primary authority. At minimum distinguish:

- `PASS`: all frozen development/holdout gates required by the plan cleared.
- `FAIL`: one or more frozen scientific gates failed.
- `DATA_BLOCKED`: source, identity, support, or leakage constraints prevented an honest test.
- `MECHANICALLY_INVALID`: implementation/run failure occurred before valid football evidence was produced.

Project-specific result docs may use more precise labels, but must map cleanly to one of these categories.

## Scientific evidence labels

These labels supplement but never replace experiment disposition.

### `NO_DIRECTIONAL_EVIDENCE`
The scored evidence is near-null, incoherent, or does not move consistently with the preregistered mechanism.

### `PARTIAL_DIRECTIONAL_EVIDENCE`
One or more meaningful components move in the hypothesized direction, but the complete frozen gate stack does not pass.

Examples:
- coherent subgroup direction but pooled effect below threshold;
- near-threshold correlation and residual gap with correct sign but inadequate tail support;
- a mechanistically relevant descriptive component survives while another required protection gate fails.

### `DIRECTIONAL_CONTRADICTION`
Material evidence moves opposite the preregistered mechanism. This is stronger negative evidence than a near-null.

### `DEVELOPMENT_ONLY_SIGNAL`
Development gates pass, an untouched holdout is legitimately unlocked, but the holdout does not replicate.

### `REPLICATED_SIGNAL`
Development and all required untouched confirmation evidence replicate under the frozen contract.

### `MEDIATED_OR_CONFOUNDED_EVIDENCE`
A raw signal exists but substantially disappears/reverses after a predeclared role, team, opportunity, or mechanism-specific robustness check. Preserve it as a clue, not as direct causal/independent support.

### `REGIME_UNSTABLE`
Multiple honestly scored seasons show materially different behavior, including sign reversals or strong pass/fail heterogeneity, and no pregame-observable regime explanation has yet been validated.

### `MAJORITY_SEASON_REPLICATION`
For a prospectively multi-season study with three or more seasons, a majority of independently scored seasons replicate in the preregistered direction while at least one season does not.

This is not automatically production-eligible. The failed season must be characterized before deciding whether the mechanism is broadly stable, context-dependent, or too fragile.

### `ALL_SEASONS_REPLICATED`
Every prospectively scored season clears the required directional/replication contract.

### `SINGLE_SEASON_SIGNAL`
Only one of multiple honestly scored seasons shows meaningful support. Treat as weak/unstable evidence unless a prospectively testable regime mechanism explains the concentration.

## How to interpret 2-of-3 seasons

A two-pass / one-fail pattern is **not** automatically a universal FAIL and is **not** automatically evidence that the failed season was anomalous.

The result doc must report at least:
- exact per-season N/coverage;
- effect sign and magnitude by season;
- gate distances by season, not only binary pass/fail;
- whether the failed season is a marginal threshold miss, near-null, or directional reversal;
- whether the same player/role cohorts drive the positive seasons;
- whether any pregame-observable football regime was frozen before the next independent test.

Interpretation examples:
- two strong positive seasons + one marginal same-sign miss -> `MAJORITY_SEASON_REPLICATION`, potentially sampling-sensitive;
- two strong positive seasons + one strong opposite-sign season -> `REGIME_UNSTABLE` until a real regime mechanism is validated;
- two borderline passes + one hard fail -> weak evidence despite the nominal 2-of-3 count.

## Holdout protection

If a frozen plan says a holdout remains sealed unless development gates pass, a development failure ends that exact experiment.

The sealed season may only be used later if:
1. a materially new/narrower hypothesis is formulated from already-visible evidence;
2. the new hypothesis, cohort, features, direction, gates, support, and stop rules are frozen first; and
3. the new use does not silently relabel the old failed experiment as a pass.

This is how useful development clues can be preserved without contaminating the holdout.

## Result-document minimums

Every substantive research result doc should include separate sections for:
- `EXPERIMENT_DISPOSITION`
- `SCIENTIFIC_EVIDENCE_DISPOSITION`
- per-season/component gate table
- positive findings worth preserving
- negative findings / closed rescue paths
- whether any untouched holdout remains sealed
- what exact new information would be required to reopen or narrow the mechanism
- `PRODUCTION_ELIGIBILITY` (normally `NO` unless separately integration-tested)

## Production rule

Scientific evidence labels are research-memory tools, not permission to change production.

Production eligibility still requires the project's applicable replication, protection, integration, regression, and operational gates. No descriptive or seasonal salvage finding can bypass those protections.
