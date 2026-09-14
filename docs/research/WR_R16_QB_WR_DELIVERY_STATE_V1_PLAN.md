# WR-R16 QB–WR Delivery-State V1 — Frozen Plan

**STATUS: FROZEN BEFORE ANY WR-R16 RESULT. RESEARCH ONLY. NO PRODUCTION CHANGE.**

## Purpose

Continue the WR receiving-yard research frontier without reopening closed target-entitlement or generic efficiency-history lanes.

Known state entering this experiment:
- WR-R15 / M38 target entitlement remains production authority and is not retuned here.
- WR receptions/opportunity is materially healthier than receiving-yard translation.
- WR-R7 rejected persistent player explosive/YAC/air-yard history and simple defense allowance as explanations for game-level YPR misses, and explicitly redirected future work toward richer route/coverage/tracking/QB-delivery information.
- WR-R11 rejected a generic strict-prior NGS target correction and explicitly required a genuinely different mechanism such as role/depth/route archetype or matchup-dependent efficiency.
- WR-R3 combined calibration was later executed and failed frozen promotion gates; do not reopen it.
- C1/C3 shared receiver/QB recalibration failed. The later canonical C2/QB-tail -> WR1 right-tail diagnostic also failed. Do not retune that family.
- Historical player-level WR/CB assignments are not defensibly reconstructable in the repo; no fabricated Coverage-v2 assignments.

This study asks a narrower new question:

> Does strictly-prior **QB-to-WR target-quality / delivery state** contain reproducible information about the next-game WR receiving-yard residual after the already-authorized WR-R15 opportunity projection?

This is not a target-share model, not a generic player-trait persistence model, and not a QB passing-yard tail proxy.

## Authority / cohort

Use the exact WR-R15 OOS authority artifact only:
- run `34238301577`;
- artifact `10061328722` (`wr-r15-wr1-anchor-participation-v1`);
- digest `sha256:8df31b5e136621d959272daf0422dfc665593cd0da2eb0892b4aa69c1417f3ce`;
- variant exactly `WR_R15_WR1_ANCHORED_PARTICIPATION`;
- target seasons exactly 2023 and 2024;
- expected authority rows exactly 4,193 = 2,076 (2023) + 2,117 (2024).

The authority projection fields remain immutable. `mc_rec_yards`, `mc_receptions`, `pred_targets`, `entitlement_tgt_share`, and `wr_rank` are read-only baseline outputs.

Target outcome for the diagnostic is `receiving_yard_residual = actual_rec_yards - mc_rec_yards`.

No sportsbook line, price, odds, target-game result feature, or target-game PBP field may enter a predictor.

## Historical PBP source / cutoff

Receiver/PBP context may be reconstructed from nflverse/nflreadpy regular-season PBP for history seasons 2022-2024.

For each scored WR row `(season S, week W, team, player)`:
- only games with chronological cutoff strictly before `(S, W)` may feed features;
- the target game's PBP is forbidden as a feature source;
- target-game PBP may be used only after feature construction for identity/outcome audit if needed, never to choose the expected passer or any feature value;
- receiver identity must map by canonical player identity plus team with a preserved unmatched/ambiguous audit; no fuzzy post-result rescue.

## Frozen feature families

The goal is to test **delivery / target quality**, not recycle WR-R7's raw persistent efficiency proxies.

All histories use the last 8 eligible prior games, with recent-state deltas using the last 3 eligible prior games when available. Minimum history for a primary feature row is 4 prior target-bearing games.

### A. Receiver target-quality state
1. `wr_target_cpoe_mean8`: mean CPOE on passes targeted to this WR.
2. `wr_completed_air_yards_per_target8`: completed air yards divided by targets.
3. `wr_target_depth_sd8`: standard deviation of air yards on targets.
4. `wr_cpoe_recent3_minus8`: recent-3 target CPOE minus prior-8 mean.
5. `wr_completed_air_recent3_minus8`: recent-3 completed-air-yards/target minus prior-8.

These are distinct from WR-R7's rejected raw `PLAYER_AIR_PER_TARGET_PRIOR8`, YAC, and explosive-rate persistence because the primary mechanism here is **quality of delivered/completed targets and change state**, not static player archetype.

### B. Team/QB delivery ecosystem state
Construct strictly-prior team passing delivery context from official pass attempts:
1. `team_cpoe_mean8`.
2. `team_air_per_attempt_mean8`.
3. `team_deep15_completion_rate8` for attempts with air_yards >= 15.
4. `team_completed_air_per_attempt8`.
5. recent-3 minus prior-8 deltas for CPOE and completed-air/attempt.

No target-game starting-QB identity may be inferred from the target game. The team-level delivery ecosystem is used deliberately so QB changes do not require outcome-informed starter selection.

### C. Predeclared interaction signals
No feature search is allowed. Test exactly these four interaction scores:
1. `DELIVERY_CPOE = wr_target_cpoe_mean8 + team_cpoe_mean8`.
2. `COMPLETED_AIR = wr_completed_air_yards_per_target8 + team_completed_air_per_attempt8`.
3. `DEEP_DELIVERY = wr_completed_air_yards_per_target8 * team_deep15_completion_rate8`.
4. `DELIVERY_MOMENTUM = wr_cpoe_recent3_minus8 + team_cpoe_recent3_minus8`.

Standardize component variables using the 2023 development sample only before interaction scoring where scale comparability is required. Freeze those 2023 location/scale parameters before 2024 is read for scientific results.

## Development / holdout separation

- 2023 = development / signal-existence season only.
- 2024 = untouched confirmation holdout.
- Do not pool seasons to choose a threshold, signal, sign, or coefficient.

Stage A is diagnostic only and uses 2023 to determine whether this family merits a fixed confirmation test. Stage B must be frozen before revealing 2024 outcomes.

## Stage A frozen diagnostics (2023)

For each of the four interaction signals, on rows with valid history:
- Spearman correlation with receiving-yard residual;
- Q4-Q1 mean residual gap using 2023-only quartiles;
- Q4/Q1 rate ratio for actual 100+ receiving-yard games where supported;
- Q4/Q1 rate ratio for absolute yardage miss >= 30 yards;
- coverage and sample size.

A signal is `DEVELOPMENT_SUPPORTED` only if all are true:
1. coverage >= 60% of otherwise authority-eligible 2023 rows;
2. abs(Spearman) >= 0.08 with mechanism-consistent sign;
3. abs(Q4-Q1 receiving-yard residual gap) >= 5.0 yards with same sign;
4. either 100+ rate ratio >= 1.20 in the mechanism-consistent direction OR 30+ miss rate ratio >= 1.20;
5. the signal is directionally coherent in both WR1 and WR2+ slices when each slice has >= 150 valid rows.

At most one signal may advance: choose the first passing signal in this frozen priority order: `DEEP_DELIVERY`, `COMPLETED_AIR`, `DELIVERY_CPOE`, `DELIVERY_MOMENTUM`. If none pass, disposition is `NO_ACTIONABLE_WR_DELIVERY_STATE_SIGNAL` and 2024 is not used to rescue the family.

## Stage B holdout contract (must be frozen before 2024 results)

If exactly one signal advances from Stage A, use the 2023 Q25/Q75 thresholds and 2023 standardization parameters unchanged on 2024.

No model fitting or coefficient search is allowed in V1. This is a signal-existence study.

Holdout confirmation requires all:
1. coverage >= 60% of authority-eligible 2024 rows;
2. Spearman same sign and abs >= 0.06;
3. Q4-Q1 residual gap same sign and abs >= 4.0 yards;
4. 100+ yard rate ratio >= 1.15 OR 30+ miss rate ratio >= 1.15 in the same direction;
5. same-sign residual-gap direction in WR1 and WR2+ slices where each has >= 150 rows;
6. no sportsbook inputs and zero target-game feature leakage.

PASS disposition: `WR_QB_DELIVERY_STATE_REPLICATED_SIGNAL`.
FAIL disposition: `NO_ACTIONABLE_WR_DELIVERY_STATE_SIGNAL`.

A PASS is **not** a production model. It authorizes only a separate preregistered candidate-integration experiment that must protect WR-R15 target/reception accuracy and receiving-yard aggregate/individual error.

## Stop rules

Do not after results:
- lower thresholds;
- swap quartiles/tertiles/percentiles;
- add/remove features from the four frozen signals;
- change last-8 / recent-3 history windows;
- infer target-game QB from target-game PBP;
- fit a model on pooled 2023+2024;
- convert a failed correlation into a tail-only rescue;
- reopen WR-R7 raw player air/YAC/explosive-history proxies;
- reopen WR-R11 generic NGS target correction;
- reopen R3 combined calibration;
- use market lines/odds upstream;
- alter M38/R15 entitlement.

If identity/source coverage is insufficient, disposition is `WR_DELIVERY_STATE_DATA_BLOCKED`, not an invitation to fabricate or loosen identity rules.

## Collaboration requirement

Before any real-data Stage A result is run, Claude must independently review this plan for:
- duplication with prior WR work;
- leakage / identity risk;
- feature-family novelty;
- adequacy of the frozen gates;
- any hidden post-result degrees of freedom.

Any accepted amendment must be committed before results and must not be informed by candidate output.