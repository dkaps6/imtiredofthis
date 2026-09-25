# WR-R19 Receiver-Specific Catchability V1 — Frozen Plan

**STATUS: FROZEN PRE-RESULT PLAN. RESEARCH ONLY. NO WR OUTCOME HAS BEEN SCORED UNDER R19. 2024 HOLDOUT SEALED. NO PRODUCTION CHANGE.**

## 1. Question

Does strictly-prior **receiver-specific catchable-target rate**, measured from FTN charting and conditioned on the already-promoted M38 / WR-R15 opportunity projection, contain next-game WR receiving-yard residual information that is not merely team throw quality, target depth, or role?

Primary football mechanism:

> A WR can systematically receive a different quality of delivered targets than the same offense generally produces because route timing, alignment, defensive attention, QB-WR chemistry and throw placement change whether passes to that receiver are charted as catchable. `is_catchable_ball` is a direct charted process observable upstream of completion outcome, unlike realized CPOE/YPT/YAC.

Frozen expected direction: **positive**. Higher prior receiver catchable-target rate should associate with more positive next-game receiving-yard residuals after M38/R15 opportunity is already fixed.

## 2. Canonical base and authority

Branch base:
- `main` = `de6aed84d474867d81427f4e8277219868ac9d50`

WR authority remains unchanged:
- M38 + `WR_R15_PRODUCTION_MODEL_V1`
- exact WR-R15 confirmation run `34238301577`
- artifact `10061328722`
- artifact name `wr-r15-wr1-anchor-participation-v1`
- digest `sha256:8df31b5e136621d959272daf0422dfc665593cd0da2eb0892b4aa69c1417f3ce`
- authority variant `WR_R15_WR1_ANCHORED_PARTICIPATION`
- 2023 development rows = `2,076`
- 2024 untouched holdout rows = `2,117`

Primary outcome:

`yard_residual = actual_rec_yards - mc_rec_yards`

No sportsbook data may enter feature construction, cohort selection, gates, or interpretation.

## 3. Why this is materially new enough to test

Pre-result anti-retest audit established:
- WR-R7 tested historical player air/YAC/explosive persistence, not charted target catchability;
- WR-R17 tested intended target-depth distribution, not delivered-ball process quality;
- WR-R18 tested receiver-attributed CPOE, an outcome/model-derived completion-over-expectation quantity, not charted catchability independent of catch result;
- M81 used FTN `is_catchable_ball` only inside QB/offense/opponent `THROW_DECISION_QUALITY`, never receiver-specific history;
- whole-repo search found no prior receiver-specific `is_catchable_ball` history experiment.

Claude independently agreed the receiver-specific catchability question is genuinely open rather than closed by M81 or R18.

This novelty claim does **not** imply expected success. The accumulated negative WR efficiency evidence is itself informative and should temper expectations.

## 4. Source contract — already audited before outcomes

Canonical source-only audit:
- branch `research-wr-yard-efficiency-feature-audit-v1`
- head `13c571af78368238419fe37076fdd3995813f30b`
- run `34907348510`
- artifact `10373021925`
- digest `sha256:eb0c50110070691cc552027619a430112d60d5a564b61b97a6222f9192c5a924`

The audit loaded no WR outcomes or WR projection artifact.

Exact results:
- 2022: 17,325 receiver target rows, 100% exact FTN->PBP play join, 100% catchability populated;
- 2023: 17,558 receiver target rows, 100% exact join, 100% catchability populated;
- 2024: 17,103 receiver target rows, 100% exact join, 100% catchability populated;
- regular weeks 1-18 complete in FTN and PBP for all three seasons.

Feature-event receiver identity contract:

`FTN nflverse_game_id + nflverse_play_id -> PBP game_id + play_id -> receiver_player_id (GSIS)`

No name-based receiver event assignment and no fuzzy matching.

## 5. Authority-row receiver identity

WR-R15 authority rows contain no stable player ID. Reuse the already-audited R17/R18 identity-only bridge:

1. canonical authority full-name key;
2. strictly-prior nflverse weekly-roster evidence only;
3. resolve to stable GSIS ID;
4. authority team may disambiguate only when it produces exactly one stable ID;
5. no target-week roster evidence;
6. no fuzzy identity rescue;
7. unresolved/ambiguous rows fail closed.

Feature history is then keyed by the resolved GSIS ID against PBP `receiver_player_id`.

## 6. Frozen primary signal

Only one primary candidate is authorized:

### `WR_TARGET_CATCHABLE_RATE8`

For each target WR-game:

1. use only receiver target events from games strictly before the target `(season, week)`;
2. define a target-bearing game as a prior game with at least one official pass attempt targeted to that resolved GSIS receiver ID;
3. choose the **last 8 prior target-bearing games first**;
4. within those eight games, use the FTN `is_catchable_ball` value attached by exact FTN/PBP game-play join;
5. signal = arithmetic mean of the 0/1 catchable labels across those receiver target events.

Support floor:
- at least **4** prior target-bearing games;
- at least **16** receiver target events inside the selected last-8-game window.

The 16-event floor is frozen before outcomes and matches R18's minimum target-event discipline.

No alternate windows, recency deltas, EWMA, thresholds or transforms are allowed.

## 7. Controls — not competing candidate signals

Controls are evaluated only for receiver-specific robustness after a raw Stage-A pass.

### `TEAM_TARGET_CATCHABLE_RATE8`

For the target offense/team:
- last 8 prior team games, selected before event aggregation;
- official pass attempts with a stable targeted receiver ID;
- mean FTN `is_catchable_ball` across those target events;
- minimum 4 prior team games and 40 target events for a usable control value.

### `MEAN_AIR_YARDS_PER_TARGET8`

For the same receiver target events used by the primary signal:
- mean PBP `air_yards` across available values;
- descriptive/robustness control only.

### Frozen opportunity/role controls

From the target WR-R15 authority row:
- `entitlement_tgt_share`;
- WR-rank bucket (`WR1` vs `WR2+` for the frozen robustness regression).

These controls may not compete for candidate selection and may not be used to invent interactions after results.

## 8. 2023 Stage A — raw signal gates

Stage A scores **2023 only**. The 2024 holdout must not be loaded/scored for candidate selection.

A raw Stage-A signal is supported only if **all** gates pass:

1. valid-signal coverage >= **60%** of all 2,076 development rows;
2. Spearman(`WR_TARGET_CATCHABLE_RATE8`, `yard_residual`) >= **+0.08**;
3. Q4-Q1 receiving-yard residual gap >= **+5.0 yards**;
4. mechanism-consistent tail evidence: either
   - Q4/Q1 actual-100+ receiving-yard rate ratio >= **1.20**, or
   - Q4/Q1 residual >= +30-yard rate ratio >= **1.20**;
5. WR1 Q4-Q1 residual-gap direction positive when WR1 n >=150;
6. WR2+ Q4-Q1 residual-gap direction positive when WR2+ n >=150;
7. identity, source, temporal and leakage audits all pass.

Quartile boundaries use linear interpolation and are frozen descriptive mechanics, not tunable thresholds.

The Spearman gate remains the primary monotonic-protection gate. It is **not** replaced merely because R18 showed tail/extreme separation with weak monotonicity.

If any raw gate fails:

`NO_ACTIONABLE_WR_RECEIVER_CATCHABILITY_SIGNAL`

2024 remains sealed.

## 9. Receiver-specific mediation / robustness gate

Run only if every raw Stage-A gate passes.

Residualize the primary signal using a fixed OLS design:

`WR_TARGET_CATCHABLE_RATE8 ~ TEAM_TARGET_CATCHABLE_RATE8 + MEAN_AIR_YARDS_PER_TARGET8 + entitlement_tgt_share + WR1_indicator`

The OLS residual is `receiver_specific_catchability`.

Receiver-specific robustness passes only if:
- Spearman(`receiver_specific_catchability`, `yard_residual`) >= **+0.06**;
- receiver-specific Q4-Q1 residual gap >= **+4.0 yards**.

If raw Stage A passes but this robustness gate fails:

`WR_CATCHABILITY_TEAM_DEPTH_ROLE_MEDIATED`

2024 remains sealed.

Interpretation caveat is frozen: a failed control/mediation check may mean confounding or may over-control a real delivery pathway; it is not causal disproof. It still blocks holdout exposure under this experiment.

If both raw and receiver-specific gates pass:

`WR_RECEIVER_CATCHABILITY_DEVELOPMENT_SUPPORTED`

Only then may the 2024 confirmation stage be implemented.

## 10. 2024 Stage B — untouched confirmation

2024 may be scored only after a valid 2023 `WR_RECEIVER_CATCHABILITY_DEVELOPMENT_SUPPORTED` result and a separate adversarial pre-result implementation review of the Stage-B code.

The signal definition, source semantics, support floor, controls, direction and gates are frozen unchanged.

Stage B uses the same raw gate thresholds as Stage A:
- coverage >=60%;
- Spearman >=+0.08;
- Q4-Q1 residual gap >=+5.0 yd;
- one frozen tail ratio >=1.20;
- positive WR1 and WR2+ direction where n>=150;
- all identity/source/leakage audits pass.

The same receiver-specific robustness thresholds also must pass:
- receiver-specific Spearman >=+0.06;
- receiver-specific Q4-Q1 gap >=+4.0 yd.

If 2023 passes but 2024 fails:
- experiment is not production eligible;
- scientific evidence is recorded under the evidence-classification framework with exact per-gate distances rather than flattened to a generic fail;
- no third-season rescue is available from the WR-R15 authority contract.

If both seasons pass:

`WR_RECEIVER_CATCHABILITY_REPLICATED_2_OF_2`

Even that does **not** directly change production. A separate integration experiment would be required to determine whether/how to alter receiving-yard means or distributions without damaging M38/R15 opportunity performance.

## 11. Descriptive outputs required regardless of outcome

Preserve separately:
- n and coverage;
- Spearman;
- Q1/Q2/Q3/Q4 boundaries and counts;
- signed residual bias and MAE by quartile;
- actual-100+ rate by quartile;
- residual >=+30 rate by quartile;
- WR1 and WR2+ gaps;
- support-count distributions;
- team-control availability;
- exact identity/source audit;
- exact FTN/PBP source hashes used;
- zero target-game feature leakage assertion;
- zero sportsbook-input assertion;
- explicit `holdout_2024_scored` boolean.

These descriptive outputs cannot rescue a failed frozen gate.

## 12. Stop rules

Do not rescue a failure through:
- WR1-only or WR2+-only modeling;
- catchability thresholds learned after seeing 2023;
- tail-only modeling;
- alternate windows (3, 5, 6, 10 games; recent-minus-long; EWMA);
- changing the 16-target / 4-game support floor;
- replacing Spearman after observing its result;
- adding `is_contested_ball`, `is_created_reception`, `is_drop`, `read_thrown`, `is_interception_worthy`, route or coverage fields;
- interactions with defense, QB, role or target depth;
- generic CPOE/YPT/YAC/explosive-history rescue;
- model-zoo substitution;
- sportsbook information upstream.

Any later mechanism must be separately justified as materially new before any additional outcome exposure.

## 13. Evidence classification

Experiment disposition, scientific-evidence disposition and production eligibility remain distinct.

A failed Stage A can retain `PARTIAL_DIRECTIONAL_EVIDENCE` only when preregistered components genuinely survive; that label never overrides frozen gates or opens 2024.

Do not call a failed season an anomaly without a prospectively identifiable football regime and separate preregistration.

## 14. Collaboration gate before implementation

Before any WR-R19 outcome is scored, Claude must independently review this exact frozen plan and specifically attack:
1. novelty versus M81/R18 despite receiver-specific aggregation;
2. whether `is_catchable_ball` semantics support the stated positive mechanism;
3. 4-game / 16-target support floor;
4. last-8 target-bearing-game construction;
5. team catchability control and whether QB-specific control is required;
6. unchanged Spearman / Q4-Q1 / tail gates;
7. mediation design and over-control risk;
8. authority-row GSIS bridge and feature-event PBP receiver identity;
9. holdout/stop-rule protections;
10. whether any materially better open mechanism should replace R19 before results.

Return `REVIEW_PASS` or concrete pre-result amendments.

No implementation/result run is authorized until that collaboration review is reconciled.

## 15. Production disposition

`production_actionable = false`

No paid Full Slate. No production science change. No RB work.
