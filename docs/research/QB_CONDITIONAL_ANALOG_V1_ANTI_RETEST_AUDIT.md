# QB Conditional Historical-Analog Reliability — Anti-Retest Audit

**STATUS: AUDIT ONLY. NO FITTING. NO OUTCOME INSPECTION. NO PRODUCTION CHANGE.**

## Target question under audit

Has this repository already answered: for a *specific current pregame football
state* (player recent form + offense state + opponent defense + game
environment/script + model-internal disagreement, combined into one
continuous similarity/distance measure), do *historically comparable*
pregame states — drawn from authority-exact production-matched history —
show a real, replicable directional/ROI record against the market? This is
distinct from (a) coarse market/side/edge-bin bucketing (already covered by
the separate Bet Evidence V0 layer, PR #618) and (b) whether a single
pregame signal improves *football projection accuracy* in isolation.

Audited per Issue #535 comments `5720166913`/`5720268056`, before any V1
implementation. Method: read actual result artifacts (CI run JSON output,
committed result docs), not branch names or plan intent alone.

## Classification table

| Candidate | Branch / lineage | Target actually tested | Evidence | Classification |
|---|---|---|---|---|
| QB-R1 Player+Context Mechanism Router | `research-qb-r1-player-context-mechanism-router`, run `34072312396`, artifact `10000858730` | Does a player's own recent attempt-vs-YPA mechanism state (`ATTEMPTS_DOMINANT`/`YPA_DOMINANT`/`MIXED`) persist into the next game, to route projection uncertainty? Single dimension, self-referential (player-vs-own-history only), football-error only (Spearman/MAE gates), no market grading. | Disposition `NO_ACTIONABLE_QB_PLAYER_CONTEXT_MECHANISM_ROUTER`. COMBINED view failed 4 of 7 frozen gates; adding the player-mechanism signal to the context-only view made the context-only OOS ranking signal *worse* (COMBINED minus CONTEXT_ONLY Spearman = -0.0592). | `ALREADY_TESTED_CLOSED` |
| QB-PD3 Internal Disagreement Reliability | `research-qb-pd3-internal-disagreement-reliability`, run `34122984048`, job `101745071358`, artifact `qb-pd3-internal-disagreement-reliability` id `10018942911`, digest `sha256:cde910939dba0631a2a68e9acfbcafcdb90af1d1b2128627442073854bf3f849`, tested SHA `6baa2ae4f4bb07277d01317da324ea8b8c42b35e` | Do pregame model-internal-disagreement states (`COMPONENT_RANGE_40`, `SYNTH_MOVE_30`, `SYNTH_CAP_45`, `MULTI_FLAG_2` — binary threshold flags on component spread / synthesis-correction magnitude) identify pregame states where the *football projection* is less reliable (MAE/bias), not market ROI. | Disposition `NO_ACTIONABLE_QB_INTERNAL_RELIABILITY_STATE`. All 4 states, `actionable_states: []`. Every state failed `pooled_mae_penalty_ge5` and `pooled_synth_worse_than_base_ge2`/`synth_vs_base_nonnegative_both_seasons`. 884 rows, 2024-2025, sportsbook inputs false. **This was previously unresolved in this thread (comment `5720714532` flagged it as undocumented) and is now recovered directly from the run's own JSON output**, not reconstructed or guessed. | `ALREADY_TESTED_CLOSED` for the binary-threshold-gate version of this question. Does not by itself preclude reusing `component_range`/`abs_correction` as a *continuous* similarity dimension in a nearest-neighbor system (different use: distance contribution, not a binary reliability gate) — but the negative prior applies directly: the closest prior test of "does model disagreement identify unreliable pregame states" failed on all four preregistered thresholds. |
| Market-Implied Game Script V1 / Vegas-Line Gamescript Calibration V1 / Game-Script-Confirmed Player Usage V1 | `research-market-implied-game-script-v1`, `research-vegas-line-gamescript-calibration-v1`, `research-game-script-confirmed-player-usage-v1` (this session, tasks completed earlier) | Does adding Vegas spread/total as a *regression feature* into `project_game_script()` improve team plays/pass-rate projection accuracy? | Feature-injection accuracy diagnostic, not analog retrieval, not market-graded. | Related information source (game environment/script), not the same mechanism — **not overlap in the sense that matters for this audit**, but the existing leakage-safe spread/total join from this work should be reused unchanged as a raw feature source if game-environment/script becomes a V1 dimension, not re-derived. |
| Migration 88 — 2023 Low-Chaos Regime Replication | `backtest-migration-88-2023-regime-replication`, run `33325633231`, artifact `m88-2023-regime-replication` id `9736226603` | Two frozen *conjunctive* pregame-state regimes (`PASS_FUNNEL_SHORT_INTERMEDIATE_VOLUME`: opponent prior-8-game pass-rate-faced high AND offense prior-8-game deep-attempt-rate low; `EFFICIENCY_SUPPRESSION`: opponent prior-8-game success/YPA allowed low) tested for directional replication on a genuinely untouched season (2023). Closest prior precedent in shape to "does this pregame state resemble a historically bad-projection pattern." | Disposition `M87_REGIMES_NOT_REPLICATED` on honest 2023 holdout. Football passing-yard error only, not market ROI; two fixed threshold-conjunctions, not continuous multi-dimensional similarity; QB rushing/receiving not covered. | `ALREADY_TESTED_CLOSED` for this specific regime-replication design. |
| WR-R16 QB-WR Delivery-State V1 | `research-wr-r16-qb-wr-delivery-state-v1` | QB-to-WR "delivery state" as a receiving-yard residual predictor, post-WR-R15-opportunity. | Plan committed (`WR_R16_QB_WR_DELIVERY_STATE_V1_PLAN.md`); no result doc found on that branch or main. | `PLANNED_NOT_EXECUTED` |
| WR-QB Shared Tail Signal V1 | `research-wr-qb-shared-tail-signal-v1` / `wr-qb-shared-tail-signal-v1-artifact-completion` (this session's own earlier task, marked completed) | Not re-located in this audit pass. | Not reconstructed here; GPT-5.6 (comment `5720782728`) authorized excluding WR/shared-tail dimensions from QB Analog V1 scope entirely rather than resolving this lineage gap first. | Excluded from V1 scope per explicit authorization — not classified. |

## Read on genuine novelty

The specific construct requested — **continuous, multi-dimensional pregame
similarity** (player state + offense state + opponent defense + game
environment/script + model disagreement combined into one distance/analog
metric) **graded directly against real market ROI/hit-rate**, evaluated via
nearest-neighbor-style historical analog retrieval rather than a fixed
binary threshold or regression feature — has not been tested anywhere in
this repository. Every close precedent tests a strictly narrower slice:
single-dimension self-referential persistence (QB-R1), model-disagreement-only
binary thresholds against football error (PD3), or two fixed conjunctive
regimes against football error on one holdout season (Migration 88).

**Classification: `GENUINELY_OPEN`** for the core construct, carried forward
into the frozen V1 plan (`QB_CONDITIONAL_ANALOG_V1_PLAN.md`) with an explicit
strong negative prior: three of the four closest analogous "does this
pregame state predict something" designs in this exact research program
failed on honest holdout or preregistered gates. V1 must be designed to fail
cleanly and close outright if the blind 2025 confirmation does not clear its
gates — no nearby-metric, nearby-k, or nearby-feature rescue, matching the
discipline every one of the failed precedents above already used.

## Lineage

- Issue #535 comments: `5720166913` (user clarification of the actual
  construct), `5720268056` (approval to proceed with audit + design),
  `5720714532` (this audit's first pass, posted as an issue comment before
  being committed here), `5720782728` (GPT-5.6 review requiring PD3
  disposition recovery before reuse, and the frozen design direction this
  audit feeds into), `5720866876` (continuity/handoff request this document
  and the frozen plan respond to).
- This document supersedes the audit table posted inline in comment
  `5720714532` — that comment's read is preserved here with the PD3 gap
  closed (see PD3 row above).
