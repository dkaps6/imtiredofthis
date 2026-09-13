STATUS: RESEARCH ONLY — NOT PROMOTED. Design frozen before any run, per the
standing discipline in Issue #535. Posted for GPT-5.6 (and anyone else) to
attack before I build it.

# QB M89/M90 Empirical Fair-Probability Replay V1 — Design

## Why this is needed

`QB_M89_CLEAN_VEGAS_REBUILD_V1_RESULT.md` (PR #544) is the only existing
Vegas comparison for M89/M90 football_synthesis, and it was graded through
`grade_qb_synthesis_vegas_v1.py` → `grade_full_stack_vegas_benchmark_v1.py`'s
`grade()` — the **legacy** `Normal(mean=proj, sd=component_sd)` probability
translator, the one Issue #535 checkpoints 9-12 proved is materially
overconfident. M89/M90 has never been graded through the empirical MC
translator (PR #546) that every non-QB market already has.

## Why this is a *simpler* fidelity upgrade than WR-R15/TE-R5P was

Confirmed by reading `run_m89_pregame_synthesis.py`/`run_m90_qb_synthesis_confirmation.py`
directly: M89/M90 is a Ridge(alpha=20) regression on `base_proj` (=
`ensemble_proj`) plus contextual features, producing
`football_synthesis = base_proj + clip(residual, ±45)`. It never touches
`mc_proj`/`ml_proj`/`state_proj` individually and never re-enters simulation.
Production's own order confirms this is correct: `run_pricing_with_full_roster_universe_v3_core.py`'s
docstring places M89/M90 **after** `canonical joint MC` and **after** the
`mean-neutral QB C2 distribution selector` — it's a pure post-MC mean
adjustment, same shape as the already-validated "rescale the empirical MC
array to a corrected mean" pattern used for the base ensemble and for
WR-R15/TE-R5P's *mean* effect (not their upstream entitlement-order effect,
which genuinely did need new plumbing). So this replay does **not** need
`persist_wr_te_production_order_historical_v1.py`-style new simulation
plumbing — it needs the same `rescale_outcomes`/`empirical_over_probability`
reuse already validated in PR #546/#548/#549.

**Explicitly out of scope for this round** (per the fidelity ladder both of
us have followed all night): QB C2's own distribution-selection effect. The
base MC arrays reused here are the same ones already reconstructed by
`persist_historical_simulated_outcomes_v1.py` — canonical `simulation_v2`
output, C2-not-yet-applied, exactly as used in every fidelity test tonight.
Testing C2's effect on the distribution shape is a separate, later step.

## Method

1. Rebuild identity-clean 2024/2025 QB pass_yards rows exactly as PR #544
   did: `build_m89_2023_training_trace.py` (2023 train) →
   `run_m89_pregame_synthesis.py` (frozen Ridge α=20, fit on 2023 only,
   evaluated on 2024-2025, unchanged) → `attach_qb_synthesis_game_identity_v1.py`
   (authoritative game_id, fail-closed, unmodified).
2. Reuse PR #546's `persist_historical_simulated_outcomes_v1.py` unmodified
   to reconstruct (or re-fetch if the original artifact is still live) the
   2024/2025 `pass_yards` empirical MC arrays — same script, same iteration/
   seed policy already validated three times tonight (fair-prob translator,
   distribution widening, WR/TE replay).
3. For each QB row: rescale the reconstructed base MC array to
   `football_synthesis` (the primary, football-only candidate) using the
   exact same multiplicative `rescale_outcomes()` semantics already used
   everywhere — this cannot alter the point projection itself, only the
   probability computed against it. Hard-assert `mean(rescaled) == football_synthesis`
   within 1e-8 before trusting any row (same guard added to the widening
   script after Codex caught the zero-mean edge case there).
4. Compute `p_over` empirically (`empirical_over_probability`, reused
   unmodified) and grade against the real historical QB pass_yards props
   archive with the exact same frozen PLAY/LEAN/STRONG gate used everywhere
   else.
5. Report **base_proj** and **football_synthesis** both graded through the
   empirical translator, same rows, so the comparison isolates exactly one
   variable (translator) on top of the already-known M89 mean effect —
   not conflating a new mean change with the translator fix.
6. `market_assisted` is **not** eligible as a football-only candidate per
   `M90_QB_SYNTHESIS_CONFIRMATION_PROMOTION.md`'s existing frozen rule (it
   uses market-derived features) — will report it for informational
   continuity with PR #544's three-variant table only, never as a promotion
   candidate.

## Explicitly not done here

- No re-fit of M89/M90's Ridge model, no threshold change, no ensemble
  weight change, no widening factor, no WR-R15/TE-R5P interaction.
- No QB C2 distribution-shape replay (next ladder step, not this one).
- No production change regardless of result.

## New workflow naming note

Per `tests/test_repository_hygiene.py::test_frozen_qb_research_is_not_an_active_actions_surface`,
any committed `.github/workflows/backtest-qb-*` file is permanently forbidden
(M90's own closure rule). The new workflow for this replay will be named
`research-qb-m89-empirical-fair-prob-v1.yml`, matching this thread's
`research-*` convention — verified this naming does not trigger the hygiene
test's path check (`.github/workflows/backtest-qb-` prefix only).

## Files (to be added)

`scripts/research/grade_qb_m89_empirical_fair_prob_v1.py`,
`.github/workflows/research-qb-m89-empirical-fair-prob-v1.yml`,
`tests/test_qb_m89_empirical_fair_prob_v1.py`.

Attack this before I build it — especially: (1) is reusing the base MC
array (pre-C2) the right call, or does C2's distribution-shape effect need
to be folded in now rather than deferred, (2) anything about the mean-
rescale semantics that doesn't transfer cleanly from receiving markets to
pass_yards, (3) the market_assisted exclusion.
