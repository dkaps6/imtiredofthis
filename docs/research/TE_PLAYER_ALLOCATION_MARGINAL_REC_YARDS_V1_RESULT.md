# TE Player Allocation Marginal Receiving-Yards V1 — Result

Date: 2026-10-15

Disposition:

`NO_ACTIONABLE_TE_PLAYER_ALLOCATION_MARGINAL_REC_YARDS`

Research only. Production unchanged.

## Canonical evidence

Successful certified replay:
- workflow: `Research TE Player Allocation Marginal Rec Yards V1 Replay`
- run: `38800926331`
- job: `116649432539`
- exact head: `e7961d5bf6abe9097a8fbf21d776fb1696a57bba`
- artifact: `12003898778`
- digest: `sha256:0c59e6e551e6ac025ac21e4098c5298a3a6ac3e16ecef9f7cd5e8d2bb0db73a4`

Frozen plan:
- `docs/research/TE_PLAYER_ALLOCATION_MARGINAL_REC_YARDS_V1_FREEZE.md`

Preserved historical distribution source:
- first reconstruction run `38798480183`
- compact metadata artifact `12004410465`
- compact digest `sha256:9219199136df62cd47b4f42132f0082ab20b66d175116b351fb6fcf309facd28`
- raw 2024/2025 marginal arrays artifact `12003009854`
- raw digest `sha256:863cc5350b4ab260a47e13b482a83358bf5fa5defe26f7d990ea9652b950c992`

The first reconstruction reached the evaluator only after both 2024/2025
production-order histories had been rebuilt successfully. Its evaluator failed
mechanically because the generic cross-position `actual_rec_yards` convenience
column is blank in the allocation-only parent even though the exact retained TE
casebook `rec_yards` field is populated. The repair used that already-pinned
parent field and replayed the exact preserved arrays. No result was exposed
before the source-wiring repair.

## Frozen construction

For each eligible TE row:
- canonical baseline = exact TE-R5P production-order receiving-yard MC array;
- pooled control = baseline plus mean-neutral contemporaneous pooled TE
  allocation-error width converted by pregame YPT;
- named candidate = baseline plus mean-neutral same-player trailing allocation
  difficulty converted by pregame YPT;
- pooled and named arms use the same deterministic standardized Normal vector;
- nonnegative physical floor only;
- exact re-anchor to the canonical TE-R5P mean.

No target-share change.
No team target-mass change.
No teammate change.
No projection-mean change.
No joint-distribution claim.
No sportsbook input.
No OddsAPI.
No 2026 outcomes.

## Support / integrity

- scored rows: **1,476**
- unique TEs: **114**
- 2024: **717 rows / 92 TEs / 205 games**
- 2025 independent confirmation: **759 rows / 103 TEs / 206 games**
- draw count: **2,000**
- max mean drift: **1.42e-14**
- candidate models fit: **0**
- sportsbook inputs: **0**
- OddsAPI calls: **0**
- 2026 outcomes read: **0**
- production changed: **false**

All support/integrity gates passed.

## Named player vs canonical — real improvement

### 2024
- canonical CRPS: **12.74750**
- named CRPS: **12.61581**
- gain: **+0.13168**
- canonical 80% coverage: **67.22%**
- named 80% coverage: **79.36%**
- canonical 90% coverage: **78.10%**
- named 90% coverage: **86.33%**

### 2025 independent confirmation
- canonical CRPS: **11.93887**
- named CRPS: **11.85061**
- gain: **+0.08826**
- canonical 80% coverage: **71.01%**
- named 80% coverage: **82.61%**
- canonical 90% coverage: **80.90%**
- named 90% coverage: **88.41%**

### Pooled named-vs-canonical inference
- observed CRPS gain: **+0.10936**
- player-cluster 95% CI: **[+0.01851, +0.20876]**
- game-cluster 95% CI: **[+0.02306, +0.19571]**
- player-cluster bootstrap P(gain > 0): **0.9912**
- game-cluster bootstrap P(gain > 0): **0.9928**

### High-difficulty Q4
- canonical CRPS: **18.13984**
- named CRPS: **18.00662**
- gain: **+0.13322**
- canonical 80% coverage: **59.03%**
- named 80% coverage: **81.40%**
- canonical 90% coverage: **71.43%**
- named 90% coverage: **87.87%**

### 2025 named-player diagnostics
- qualifying TEs with >=6 scored games: **69**
- median per-player named-vs-canonical CRPS gain: **+0.02004**
- fraction positive: **55.07%**

This is strong evidence that the canonical TE-R5P receiving-yard marginal is
under-dispersed relative to actual outcomes and that the confirmed named-player
allocation state contains real marginal uncertainty information.

## Why the exact named-player mapping still fails

The frozen candidate also had to beat a contemporaneous pooled TE
allocation-width control.

It did not.

### 2024
- pooled-control CRPS: **12.51938**
- named CRPS: **12.61581**
- named minus pooled gain: **-0.09643**

### 2025
- pooled-control CRPS: **11.80102**
- named CRPS: **11.85061**
- named minus pooled gain: **-0.04959**

2025 dependence-aware named-vs-pooled:
- player-cluster observed: **-0.04959**
- player-cluster 95% CI: **[-0.11616, +0.01165]**
- game-cluster observed: **-0.04959**
- game-cluster 95% CI: **[-0.11384, +0.01969]**

Failed frozen gates:
1. `2024_named_beats_pooled_crps`
2. `2025_named_beats_pooled_crps`
3. `2025_named_vs_pooled_player_ci_low_positive`
4. `2025_named_vs_pooled_game_ci_low_positive`

All other frozen gates passed.

## Interpretation

Two statements are simultaneously true:

1. **The current canonical TE receiving-yard marginal is too narrow.**
   Both the named and pooled uncertainty overlays improve CRPS and coverage
   relative to canonical.

2. **This exact same-player sigma transmission is not incrementally superior to
   a simpler pooled TE allocation-width control.**
   The named mapping spreads player widths too unevenly to qualify as the
   production individualization mechanism.

Therefore the named-player allocation signal remains scientifically real, but
the exact mapping

`same-player trailing allocation MAE -> Normal target sigma -> pregame YPT -> independent receiving-yard marginal overlay`

is CLOSED.

This does not reopen the already-closed joint Dirichlet transmission and does
not promote the pooled control. The pooled arm was a frozen control, not a
pre-registered production candidate.

## Anti-retest boundary

Do not rescue this exact mapping with:
- a fitted scaling coefficient;
- sigma caps/floors;
- TE1/TE2 slices;
- player subgroups chosen after seeing this result;
- alternate trailing windows/support;
- alternate YPT conversion;
- a different noise family;
- a different clipping/recentering rule;
- pooled-season-only qualification.

The old global TE width V2 remains closed.
The TE joint Dirichlet law remains closed.

## Forward consequence

No Week 6 prospective lock is warranted for this failed exact mapping.

The next individual-player lane should not force this signal back into another
width law. A legally distinct question is whether confirmed same-player
difficulty can improve **player-level trust/selectivity** (predicting when the
existing projection/distribution is reliable or fragile) without changing its
mean or width. That would directly serve the live-board problem of selecting
the genuinely trustworthy player props while preserving the closed
distribution-transmission boundaries.

Canonical status remains:

`PLAYER_INDIVIDUALIZATION_PARTIAL`
