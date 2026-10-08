# PLAYER TARGET SHARE TRAJECTORY V1 — HISTORICAL INTEGRATION / OOS GATE

Frozen: 2026-10-08.
Parent: `PLAYER_TARGET_SHARE_TRAJECTORY_SHADOW_V1_CONTRACT.md` and immutable Week-5 lock run 37654382316 / artifact 11497153776.
Decision: **Gate source-provenance first, score nothing until the gate is resolved.**

## Historical target seasons and target weeks

Predeclared compatibility population: 2023, 2024, 2025 regular seasons, target-team games with at least four strictly prior same-season completed team games (usually Week 5+, adjusted for BYEs, schedule, team changes and canceled games). All WR and TE players in the **historically reconstructed pregame active universe**, never outcome-selected players or sportsbook-offered players. Player target feature excludes target game and future games; identify target histories with official GSIS IDs and canonical roster aliases only. Maintain an explicit row disposition on identity/conflicting target counts.

**These seasons are not automatically OOS**: production TE-R5P and WR-R15 frozen assets both declare `training_seasons = [2022, 2023, 2024, 2025]`. Exact contemporary-model integration back into those seasons overlaps fitting data. No assertion of independent historical predictive success is allowed absent separately verifiable as-of/out-of-fold specialists. The model version's training dates, not just the trajectory feature dates, govern OOS status.

## Static Gate 0 — implemented before any grading

Read only two already-committed production JSON assets:
- `data/models/te_r5p_production_model_v1/te_r5p_production_model_v1.json` — expected `TE_R5P_PRODUCTION_MODEL_V1`
- `data/models/wr_r15_production_model_v1/wr_r15_production_model_v1.json` — expected `WR_R15_PRODUCTION_MODEL_V1`.

Record SHA256, `training_seasons`, model versions and per-target-season overlap. Fail closed on missing, corrupt or drifted models.

A historical season is `OOS_BY_DECLARED_FIT_MEMBERSHIP` only if **neither** model reports that season in its training set; this is necessary but **not sufficient** for valid point-in-time OOS, which additionally requires historical source availability, specialist training chronology and authentic as-of team/player inputs.

If any candidate season overlaps either model, report `HISTORICAL_INTEGRATION_DIAGNOSTIC_ONLY__SPECIALIST_TRAINING_OVERLAP`. This forbids promotion and OOS claims; it does not prohibit a clearly labeled mechanical compatibility study.

No sportsbook inputs; no target outcomes; no Week-5 2026 outcomes; zero fitting.

## Exact candidate (already frozen; NO new fit)

Baseline ordered chain: PlayerForm/Bayes/rules -> M38 -> TE-R5P -> WR-R15 -> explicit simulation. Trajectory uses completed same-season, same-team team games: last 2 vs all earlier (at least 2 earlier). `delta = player_recent2_targets/team_recent2_targets - player_earlier_targets/team_earlier_targets`; missing feature means `delta=0`.

Frozen shadow: `weight_i = baseline_entitlement_i * exp(delta_i)`. Normalize within the exact protected TE room or WR2+ room to preserve total modeled target entitlement. WR1 M38 anchor fixed. Team receiving target mass, RB/QB/other market means and per-opportunity efficiency remain unchanged. The original shadow computes zero *feature adjustment* for ineligible rows; room normalization can change their final share when eligible teammates move, so no assertion of exact player-row no-change from a zero feature.

Do not change exponent coefficient, windows, eligibility, fallback, room partition, clipping or mask based on retrospective results.

## Later historical integration gates (unscored now)

- **Source/time gate:** a target-game pregame universe and team/roster status timestamp contract comparable to live, actual zero opportunity not used to select participants; team schedule and original in-game IDs validated; strict pre-target chronological feature construction; no mistaken `ACT + INA` replay semantics.
- **Parent parity:** match exact frozen baseline M38/TE-R5P/WR-R15 entitlements and source model versions; artifact hash-lock source paths/commit and row universe before loading grade data. Assert TE pool, WR2+ pool, WR1 anchor and full team modeled target mass within `1e-12`. No TE/WR baseline refit.
- **Grading isolation:** completed-game nflverse PBP target counts used after parent freeze only; stats source conflicts excluded, not forced; fixed row universe. Exact target-share and final receiving-count/yards scores comparing same player-game baseline vs shadow, with the promoted baseline per-target efficiency held invariant.
- **Interpretation:** target share/yardage improvement on historical overlapping training seasons is *descriptive compatibility evidence*, not an independent OOS confirmation or a production promotion. The prospective 2026 Week-5+ outcome-blind locks and previously frozen prospective sample/pass gates remain authoritative.

## Stop conditions

No scoring if source chronology, historical cohort, starter/availability parity, identity, parent version or conservation cannot be certified. No back-application to 2026 W1–W4; no premature Week-5 2026 grading. No paid OddsAPI. No production edits.

Gate 0 output schema:
- `status`
- `research_only`, `production_changed`, `sportsbook_inputs_used`, `target_outcomes_read`, `week5_2026_outcomes_read`, `parameters_fit`
- `candidate_seasons`
- each asset: path, sha256, expected/observed version, declared training seasons
- each historical target season: training-overlap list, eligible for clean OOS claim (always false at this gate; further as-of verification necessary)
- follow-up disposition.

Any failure in file/version contract is a hard error, never silently treated as eligibility.
