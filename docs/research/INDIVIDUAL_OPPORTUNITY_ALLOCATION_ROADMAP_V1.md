# INDIVIDUAL OPPORTUNITY ALLOCATION — BOUNDED EXECUTION ROADMAP V1

Frozen 2026-10-08. Branch: `research-individual-opportunity-roadmap-2026-10-08`.

## Decision

`OPPORTUNITY_DOMINANT_ACROSS_ALL_PRIMARY_MARKETS__INDIVIDUAL_ROLE_SHARE_ALLOCATION_NEXT`

This is a **reconciliation and next-execution gate**, not a new predictor, scorecard, or production promotion. Production science and all immutable prospective Week-5 locks remain unchanged. No paid OddsAPI, sportsbook upstream, new thresholds, outcome-selected groups, or hidden Week-5 grades.

**Canonical latest evidence:** Player Output Component Decomposition V1 run 37777319154 / exact science head cb1e511adcb58e40fc41d0949306083a304e83dd / artifact 11550446909 / digest sha256:7acfc453258a49f3aadf0d0ea386b13827eb715f1dfb055b96505f733320d204. The opportunity oracle removes more MAE than the efficiency oracle in all eight tested position/market cells. Oracle diagnostics are not deployable predictors and efficiency-subset percentages are not population-identical.

## Existing studies: DO NOT REDO

1. All-player 2026 W1-W4 replay: 37683439543, 4,488 point rows, workload compression confirmed.
2. Player Opportunity Allocation Audit V1: 37686839645, historical ACT+INA availability mismatch discovered.
3. Historical Availability Parity Diagnostic V1: 37687979574, explicit ACT-only reconstruction; production availability already promoted; inactive-replay failures must NOT be called live production failures.
4. Corrected volume-vs-share decomposition: 37694474836, 2,106 ACT-only opportunity rows. **QB team volume dominates; RB/WR/TE player share dominates**. The separately reported 37689289742 version differs in volume-denominator semantics; prefer the corrected 37694474836 authority.
5. Player Share Input Coverage and Residual Audit V1: 37695773433, focal share shrinkage concentrated at Bayesian posterior; WR-R15 and TE-R5P are helpful corrections, not defects to discard; raw-history global replacement failed to earn authority.
6. Opportunity-to-existing-shadows crosswalk: branch research-player-opportunity-decomposition-to-existing-shadows-v1. Maps WR/TE trajectory and RB carry share to already-frozen prospective mechanisms; QB same-data volume avenues exhausted.
7. RB Receiving Share State Coverage V1: 37702111845; strict-prior RB room share exists but was not generically consumed.
8. RB Receiving Room Share W1-W4 Impact V1: 37703522415; within-room strict-prior allocation improves final RB targets 7.60%, receptions 3.83%, rec yards 2.89%, rush+rec 1.66%; rushing unchanged. Week-5 immutable receiving lock 37705464974.
9. RB carry/snap Week-5 allocation lock: 37560824479; frozen 50/50 recent carry share and snap fraction; preserve M96 retrospective stop.
10. WR/TE Target Share Trajectory V1 signal: 37638269235; Week-5 immutable shadow 37654382316 / artifact 11497153776. Four previous same-season target-team games mandatory; **zero eligible 2026 W1-W4 rows**.
11. Player Landscape Transmission Audit V1: 37716318460, individual core confirmed with incomplete transmission; historical route volume / YPRR source-parity blocked.
12. WR-R3 combined calibration 34726509088 CLOSED; generic team coverage near-null; WR-CB historical assignment source blocked; generic Vegas/game-script no actionable state; FMT V1 three integrations CLOSED; M95A/B CLOSED; universal symmetric target-depth transform W1-W4 CRPS worse and unpromoted.

## Three distinct probability/volume/share layers

| Layer | QB | RB | WR | TE | Verdict now |
|---|---|---|---|---|---|
| A. Participation / active-role probability | Current availability and starter authority already promoted; historical ACT/INA identity parity corrected | Current availability already promoted; latent usage zeros remain diagnostic | Same | Same | **No new zero-state threshold**. Require a previously unavailable, pregame-valid *active-role* observable and a validated as-of historical source before new mechanism. Do not classify players using realized zeros. |
| B. Team opportunity volume | **Dominant**: actual team-volume oracle removes 60.3% of QB attempt MAE (corrected ACT-only run 37694474836) | Secondary for carries/targets | Secondary for targets | Secondary for targets | **QB research gate closed to recycled same-data/game-script/57:43 retunes**. Reopen only with materially new pregame game-plan/intent observation and as-of source provenance; keep M89/M90/C2 protected. |
| C. Room share / player allocation | QB share secondary (24.7% oracle MAE removed) | **Dominant** carry (64.5%) and targets (76.5%) | **Dominant** targets (68.6%) | **Dominant** targets (77.1%) | Validate current frozen share mechanisms; do not create four new allocators. |

**Most important distinction:** a pregame-player universe is not an observed snap roster; explicit statuses and uncertain playing time require separate contracts. The original historical ACT+INA reconstruction has been corrected diagnostically. Do not call a pregame probabilistic zero-state model validated merely because it can exclude postgame zero participants.

## Per-position remaining legal work

**QB — gate, not new formula.** Prior QB volume research M40-M42/M16-M21, M63-M65, M67-M69, M73, shared-opportunity work and same-data retunes have terminal dispositions. Keep a source-acquisition watch only for a *new* legitimate pregame intent / starting role / team pass-volume input. No generic YPA, game-script, Vegas, fresh positional mean, or attempt model fit without this new source gate.

**RB carries — existing lock.** Week-5 50/50 carry/snap allocation already frozen; do not retro-apply it to W1-W4 or reopen M96. Accumulate prospectively under original acceptance and injury/identity contract. Do not use realized carries to choose a focal subgroup.

**RB receiving — existing confirmed mechanism.** W1-W4 strict-prior within-RB-room receiving share improved final projections; Week-5 lock already frozen. Preserve total RB+FB receiving target mass, R26 vacancy ordering, R22 tail logic, team pass volume, and fixed per-target efficiency. No second RB receiving proxy/threshold search.

**WR / TE — next executable research focus.** Keep M38 WR1 fixed; redistribute only WR2+ within their preserved pool; preserve full TE room. Validate the already-frozen `weight_i = baseline_entitlement_i * exp(trajectory_delta_i)` within-room normalized transform. Exact trajectory eligibility: two most recent vs all earlier completed same-season, same-team team games, >=4 prior target-team games and >=2 earlier games. No 2026 W1-W4 rescue.

## WR/TE historical Week-5+ integration: sequence and scientific gate

See `docs/research/PLAYER_TARGET_SHARE_TRAJECTORY_HISTORICAL_INTEGRATION_V1_GATE.md` in this branch.

The precise 2026 production TE-R5P and WR-R15 deployed JSON models report `training_seasons = [2022,2023,2024,2025]`. **Therefore their direct 2023-2025 historical replay is NOT independent out-of-sample confirmation**. It may be used as a *retrospective compatibility / deterministic transmission diagnostic only*, unless exact as-of/out-of-fold specialist authorities for each historical target season can be proven. This rule is frozen BEFORE any historical scoring. No after-the-fact relabeling as OOS or promotion.

Gate 0 (execute first): static frozen-model provenance and eligibility check; compare candidate historical seasons against model training seasons, verify exact protected asset versions; emit a terminal descriptive/OOS eligibility verdict. This gate reads no target outcomes.

Gate 1: prove historical M38 -> TE-R5P -> WR-R15 entitlement reconstruction is source-parity safe, time-ordered, and calibrated without target-season training leakage for any claim of independent OOS. If impossible, mark `HISTORICAL_INTEGRATION_DIAGNOSTIC_ONLY`, not clean validation. Never mutate frozen production fitted coefficients.

Gate 2: if appropriate, build a separately versioned historical shadow evaluator replicating the exact Week-5 formula and protected room/WR1/team invariants, using only strictly prior PBP for features and later PBP for **grading only**. Freeze source, rows, identities, exclusions, scoring, and baseline parity BEFORE scoring. Report target/share MAE and receiving-yard impact with exact frozen per-target efficiency. Historical descriptive results can never replace the four-week prospective confirmation gate.

Gate 3: continue accumulating the immutable **prospective** Week-5+ locks. Original threshold gates remain: >=4 locked future weeks, >=400 scoreable WR/TE player-games, >=120 distinct identities, >=80 rooms; pooled target MAE improvement, non-worse RMSE, WR & TE both improve, 3/4 weeks non-worse, rank non-worse, bootstrap P(positive improvement)>=0.80, and zero leakage/conservation violations. No Week-5 outcomes are graded under this plan.

## Operational / stopping rules

- No automatic production promotion, no paid OddsAPI, no prop/game odds used as predictor, no GSIS private raw data.
- No new player mean, YPA/YPT/YPC, coverage, defense-vs-position, Vegas or generic matchup multiplier.
- Do not rerun full W1-W4 replay, decomposition, landscape audit, historical availability parity, or share coverage.
- Do not call an Action running without current GitHub Actions status.
- Live main at branch creation: `5ce513f23a48952bca69616eccee66f06a38a87d` (documentation-only commits since handoff).
- Latest Full Slate on that head was completed/failure in QB C2 football-only distribution state context (37778827288), while Repo CI passed (37778827423); keep production blocker separately visible, do not blame or modify this research lane.
- If model provenance fails Gate 0, do **not** score purported clean historical OOS. Record why and proceed only under an explicitly labeled retrospective diagnostic with original prospective test unchanged.

Disposition: `INDIVIDUAL_OPPORTUNITY_NEXT_STEPS_RECONCILED__TRAJECTORY_HISTORICAL_OOS_PROVENANCE_GATE_NEXT`.
