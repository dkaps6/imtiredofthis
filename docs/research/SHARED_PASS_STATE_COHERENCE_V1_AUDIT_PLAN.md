# Shared Pass-State Coherence V1 — Read-Only Audit Plan

Date: 2026-09-24

Status: FROZEN BEFORE NEW SCORING

## Purpose

Determine whether the current production stack materially describes two different passing games on the same Monte Carlo draw:

1. the QB C2 shadow completed-pass / receiving process that generates selected QB pass-yard distributions; and
2. the canonical WR/TE/RB receiving arrays that remain in production.

This is a structural audit first. It is not a production candidate and it does not authorize any receiver-output replacement.

## Why this is genuinely open

Current production C2 explicitly preserves receiver outputs:
- `receiver_outputs_replaced = 0`
- selected QB pass-yard distributions may consume `C2_QB_MEAN_NEUTRAL_DISTRIBUTION_V1`
- WR/TE/RB receiver arrays remain canonical.

Therefore selected QB pass-yard draws and final production receiver draws are not currently required to come from the same completed-pass realization.

The old shared conservation lineage was NOT scientifically failed at full stack:
- scientific source run `34081764151` disposition `CONSERVATION_ONLY_SUPPORTED`
- exact 884 QB games: QB mean MAE unchanged, CRPS improved by about 1.29 yd, 80% coverage improved materially, conservation gap 0
- full-stack rerun `34139757238` / artifact `10025733640` ended `MECHANICAL_OR_INTEGRITY_FAILURE`
- that result had `receiver_rows = 0`; receiver scientific gates were therefore never evaluated.

C1 group target-mass calibration and C3 joint combination remain closed and are NOT part of this audit.

## Frozen scope

Research-only, no sportsbook input, no target-game outcomes.

Use the current production ordering and current promoted authorities:
- M89/M90 QB mean anchor;
- C2 QB distribution selector;
- M38 + WR-R15 target entitlement;
- TE-R5P entitlement;
- current RB receiving/rushing state;
- RB Rush+Receiving Conservation V2 remains untouched.

Historical/current replays must use leakage-safe pregame inputs only.

## Phase A — structural code/provenance proof

Record and verify:
1. C2 selected QB pass-yards are generated from the C2 shadow receiving process.
2. C2 shadow receiver outputs are not installed into production.
3. current production receiver arrays remain canonical after C2 selection.
4. WR-R15 and TE-R5P entitlements are consumed before the C2 shadow state.
5. zero sportsbook inputs reach any upstream state.
6. no target-game outcome is used.

If any of these are false, stop and document the actual architecture.

## Phase B — no-outcome Monte Carlo coherence audit

For every eligible C2-selected QB team-game, capture on identical seeds/iterations:
- final production QB pass-yard array;
- canonical production receiver arrays by player;
- C2 shadow receiver arrays by player;
- C2 residual receiving bucket;
- team pass attempts and shared pass-efficiency state;
- player entitlement shares and position family.

Report, without fitting or tuning:

### Team/draw coherence
- correlation of final QB pass yards with canonical modeled receiving total;
- correlation of final QB pass yards with C2 shadow receiving total;
- p10/p25/p50/p75/p90 of QB-minus-canonical-modeled-receiving gap;
- rate of opposing-tail draws: QB >= p75 while canonical receiving total <= p25, and vice versa;
- mean/SD ratio of canonical modeled receiving total vs C2 shadow modeled receiving total.

### Player-level state divergence
For each WR/TE/RB receiver:
- canonical-vs-shadow array correlation;
- mean gap;
- SD ratio;
- p90 absolute draw gap.

Summarize by WR / TE / RB and by entitlement quartile.

### Invariance
- QB mean anchor gap;
- target-entitlement identity;
- rushing arrays identity;
- RB Rush+Receiving V2 identity;
- no receiver production outputs changed by the audit itself.

## Interpretation rule

This audit does NOT pass or fail a candidate.

It answers only:
- whether the current split QB/receiver state is materially divergent;
- where the divergence is concentrated;
- whether a separately frozen one-pass-state production candidate is justified.

No output from this audit may be used to tune C1/C3, choose arbitrary thresholds, select players, or modify production.

## If material divergence is confirmed

Freeze a separate candidate plan BEFORE any outcome scoring.

That candidate must:
- use one shared passing realization;
- preserve M89/M90 mean authority unless separately replaced;
- consume current WR-R15 / TE-R5P entitlement;
- avoid C1 and C3;
- avoid sportsbook inputs;
- preserve all rushing and RB-V2 invariants;
- be evaluated under independent predeclared historical/full-stack gates.

## Stopping rule

Do not branch into:
- C1/C3 resurrection;
- QB mean retuning;
- receiver threshold searches;
- player/position carveouts;
- sportsbook-conditioned routing;
- post-result parameter search.

If the split state is small/immaterial, close this exact lane and move on.
