# QB-PD3 and TE-R1 — Recovered Results

**STATUS: RESEARCH ONLY — NOT PROMOTED.** Both runs already executed successfully on their frozen research branches on 2026-09-07 and were never written up. This recovers the actual output from the GitHub Actions job logs (not re-executed — same run, verified via run/job/artifact IDs below) so the findings aren't lost.

## QB-PD3 — Internal Disagreement Reliability

- Branch: `research-qb-pd3-internal-disagreement-reliability`
- Run: `34122984048`, Job: `101745071358`, Artifact: `10018942911` (`qb-pd3-internal-disagreement-reliability`), completed 2026-09-07T12:39:17Z, conclusion: success
- Cohort: exact M89 2024-2025 synthesis trace, 884 rows
- Sportsbook inputs used: false. Production changed: false.

**Disposition: `NO_ACTIONABLE_QB_INTERNAL_RELIABILITY_STATE`.**

Tested 4 candidate internal-disagreement/correction-magnitude states (`COMPONENT_RANGE_40`, `MULTI_FLAG_2`, `SYNTH_CAP_45`, `SYNTH_MOVE_30`) as predictors of *when the synthesis correction becomes unreliable* — i.e., hunting for a state where the M89 synthesis actively hurts accuracy vs. the base ensemble, so it could be flagged/gated. None of the 4 states passed: `pooled_mae_penalty_ge5` and `pooled_synth_worse_than_base_ge2` both failed for all 4. Reading the actual numbers, 3 of the 4 states show the synthesis *helping* (negative `synth_minus_base_abs_mean`, i.e., synthesis MAE below base MAE) even within these flagged sub-populations — there's no evidence the synthesis correction breaks down anywhere these diagnostics can see.

**This is a mildly reassuring null result, not a failure to worry about**: no internal-disagreement or correction-magnitude signal identifies a state where the promoted QB synthesis should be distrusted. Combined with QB-PD2's earlier finding (player-history persistence isn't actionable either), this closes out the "QB reliability/calibration" diagnostic lane the team opened after PD2 — both come back clean/null. No further action authorized from this pair; if you want QB reliability work to continue, it needs a genuinely different angle than internal disagreement or player history.

## TE-R1 — Individual Mechanism Decomposition

- Branch: `research-te-r1-individual-mechanism-decomposition`
- Run: `34123413402`, Job: `101746420399`, Artifact: `10019112418` (`te-r1-individual-mechanism-decomposition`), completed 2026-09-07T12:44:02Z, conclusion: success
- Cohort: exact Joint V1 pass/receiving conservation casebook (source run `34081764151`), 6,371 scoreable rows, 2020-2025, 108 players with ≥20 games
- Sportsbook inputs used: false. Production changed: false.

**Disposition: `TE_MECHANISM_DECOMPOSITION_ACTIONABLE`.** All 5 frozen gates passed (`each_season_ge500`, `mechanism_concentration`, `players20_ge40`, `scoreable_ge4000`, `shapley_reconstruction_le1e_6` — max Shapley reconstruction gap `5.68e-14`, i.e., exact).

Shapley-decomposed pooled absolute error mass across the three mechanisms (targets, catch-rate, YPR):

| Mechanism | Share of error mass | Dominant-game count (highest-error quartile) |
|---|---:|---:|
| TARGETS | 45.2% | 1,068 |
| YPR | 29.5% | 327 |
| CATCH_RATE | 25.3% | 198 |

**TARGETS (entitlement/volume) dominates TE error, by a wide margin, consistent across all 6 seasons.** This is the TE-side confirmation of exactly what the WR side found independently in `WR_ND1_POST_M38_RESIDUAL_DECOMPOSITION_RESULTS.md` (YPT/targets-family dominance) and matches why the R2→R5 TE lineage correctly prioritized building the entitlement layer (now TE-R5P in production) before touching efficiency. No surprises here, but it's now a real, gate-passed, documented result rather than an implicit assumption — and it's independent confirmation that the TE-R5P entitlement-first strategy was the right call. It also quantifies the remaining opportunity: YPR + CATCH_RATE (efficiency) together are still ~55% of the error mass, i.e., slightly *more* than targets alone, so efficiency work is not a minor residual — it's a comparably-sized open problem, same conclusion the TE_GAP_FINDINGS.md doc already reached from the R2-R6 branch reading, just now with an exact number behind it.

## What this changes in the master findings doc

- QB-PD3 is no longer "unfinished" — it's a completed null result. Remove it from the "shovel-ready" list; the real open QB lane remains the public-intent thread only.
- TE-R1 is no longer "unfinished" — it's a completed, gate-passed decomposition that quantifies (doesn't newly discover) the TE efficiency gap already identified. Strengthens the case for the Coverage v2 efficiency-signal proposal, since we now know efficiency is ~55% of TE error mass, not a rounding error.
