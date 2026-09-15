# WR-R20 Early / No-Extended Progression V1 — Frozen Plan

**STATUS: FROZEN PRE-RESULT PLAN. RESEARCH ONLY. NO WR-R20 OUTCOME HAS BEEN SCORED. 2024 HOLDOUT SEALED. NO PRODUCTION CHANGE.**

## 1. Question

Does strictly-prior receiver-specific **early / no-extended-progression target share**, conditioned on the already-promoted M38 / WR-R15 opportunity projection, contain next-game WR receiving-yard residual information distinct from team scheme tendency, target depth, opportunity volume and WR role?

Primary signal:

`EARLY_NO_EXTENDED_SHARE8`

Frozen expected direction: **negative**.

Higher prior early/no-extended share is hypothesized to associate with **more negative** next-game receiving-yard residuals after WR-R15 target opportunity is fixed.

Football rationale frozen before outcomes:
- designed/quick/no-extended-progression usage often prioritizes touch certainty and timing over route-development depth;
- conditional on similar target opportunity, a receiver whose target diet is more heavily concentrated in this usage may have less receiving-yard ceiling per target than WR-R15's opportunity projection alone can represent;
- a competing positive "designed touches imply trust" story is acknowledged but rejected prospectively for sign selection;
- the sign may not be changed after any outcome is observed.

The target-depth control below is mandatory precisely because some or all of this negative mechanism may be mediated by air-yard depth. A mediation failure is informative and blocks holdout exposure.

## 2. Canonical authority

Clean branch base:
- `main` = `de6aed84d474867d81427f4e8277219868ac9d50`

WR authority remains unchanged:
- M38 + `WR_R15_PRODUCTION_MODEL_V1`
- exact WR-R15 run `34238301577`
- artifact `10061328722`
- artifact name `wr-r15-wr1-anchor-participation-v1`
- digest `sha256:8df31b5e136621d959272daf0422dfc665593cd0da2eb0892b4aa69c1417f3ce`
- authority variant `WR_R15_WR1_ANCHORED_PARTICIPATION`
- 2023 development rows = `2,076`
- 2024 holdout rows = `2,117`

Primary outcome:

`yard_residual = actual_rec_yards - mc_rec_yards`

No sportsbook information may enter feature construction, cohort selection, gates or interpretation.

## 3. Why R20 is materially distinct enough to test

Closed neighboring lanes:
- WR-R17: target-depth **distribution shape** (`DEPTH_SD8`, `DEPTH_IQR8`, `DEEP15_TARGET_SHARE8`), not QB progression state / designed-target process;
- WR-R18: receiver target CPOE, not play-progression state;
- WR-R19: receiver catchability, not read/design progression state;
- M81: FTN decision/charting variables at QB/offense/opponent aggregation, not receiver-specific prior target history against WR receiving-yard residual;
- contested-target exposure is explicitly rejected by overlap with M72/M75 and sparse base rate.

R20 does not claim novelty merely because the feature is orthogonal to WR-R15. It is authorized for testing because it is a distinct process observable at receiver level and survived the source/semantic/redundancy screens below.

## 4. Pre-result source / semantics lineage

### V2 semantic + source/redundancy audit

- branch: `research-wr-read-priority-source-audit-v1`
- semantic contract: `docs/research/WR_READ_PRIORITY_SEMANTIC_RESOLUTION_V2.md`
- result: `docs/research/WR_READ_PRIORITY_SOURCE_REDUNDANCY_V2_RESULT.md`
- workflow run `34912772828`
- artifact `10375186236`
- digest `sha256:2632b456d166d517f293090637388e39466a1ee4c3600929ef76a0adec6293cb`
- disposition `READ_PRIORITY_R20_PLAN_ELIGIBLE_V2`

Source-only V2 result:
- 2023 supported coverage `76.0116%`;
- Spearman vs `entitlement_tgt_share` `-0.0393056`;
- Spearman vs `pred_targets` `-0.0385956`;
- R² from `entitlement_tgt_share + pred_targets + WR1` `0.0181922`;
- no WR receiving-yard outcomes loaded;
- no 2024 WR-R15 projection/outcome fields parsed.

### V3 required-control feasibility audit

- branch: `research-wr-read-priority-source-audit-v1`
- result: `docs/research/WR_READ_PRIORITY_CONTROL_FEASIBILITY_V3_RESULT.md`
- workflow head `06cc7c10a14bc9f2db79b8b59b06af51fff55192`
- workflow run `34913439449`
- artifact `10375002416`
- digest `sha256:80b82ccb910e643868da019e0ea7941b8a9e03ac281b15476f52554230734fe1`
- disposition `READ_PRIORITY_R20_CONTROL_SET_ELIGIBLE`

V3 source-only result:
- primary-supported rows `1,578`;
- team-control coverage `100%`;
- target-depth-control coverage `100%`;
- Spearman primary vs team control `+0.231844`;
- Spearman primary vs mean air-yards/target `-0.512643`;
- expanded R² with opportunity + role + team scheme + depth `0.348071`;
- no WR receiving-yard outcomes loaded;
- no 2024 WR-R15 projection/outcome fields parsed.

The moderate depth relationship is expected and is not hidden; it is why depth is a mandatory mediation control.

## 5. Frozen FTN progression semantics

Use the maintainer-approved nflreadr / FTN key without relabeling values based on observed performance.

### `EARLY_NO_EXTENDED`
- 2022: raw `NA` (documented uncoded primary) or `DES`;
- 2023+: raw `0` or `DES`.

### `EXTENDED_PROGRESS`
- all seasons: raw `1` or `2`.

### Excluded from the binary denominator
- `CHK` checkdown;
- `SD` scramble drill;
- any unknown code (unknown codes also fail source audit expectations).

No later experiment may reinterpret `CHK`, `SD`, `DES`, `0`, `1` or `2` after seeing outcomes.

## 6. Receiver identity and event contract

Authority-row identity reuses the audited R17-R19 bridge:
1. canonical authority full-name key;
2. strictly-prior nflverse weekly-roster evidence;
3. stable GSIS ID;
4. team disambiguation only when exactly one stable ID remains;
5. no target-week roster evidence;
6. no fuzzy identity rescue;
7. unresolved/ambiguous rows fail closed.

Feature-event assignment:

`FTN nflverse_game_id + nflverse_play_id -> PBP game_id + play_id -> receiver_player_id (GSIS)`

No name-based event assignment.

## 7. Frozen primary signal construction

For each target WR-game:

1. use only receiver target events from games strictly before target `(season, week)`;
2. define prior target-bearing games from official pass targets to the resolved receiver GSIS ID;
3. select the **last 8 prior target-bearing games first**;
4. only after the eight-game window is selected, classify progression events under Section 5;
5. denominator = `EARLY_NO_EXTENDED + EXTENDED_PROGRESS` events only;
6. signal = `EARLY_NO_EXTENDED / classifiable progression targets`.

Support floor:
- at least **4** prior target-bearing games;
- at least **16 classifiable progression targets** inside the selected last-eight-game window.

No alternate windows, recency deltas, threshold transforms, EWMA or category-specific variants are authorized.

## 8. Frozen controls — not competing candidates

Controls run only after every raw Stage-A gate passes.

### `TEAM_EARLY_NO_EXTENDED_SHARE8`
For the target offense:
- last 8 strictly-prior team target-bearing games;
- same V2 progression classification;
- minimum 40 classifiable team target events;
- control only.

### `MEAN_AIR_YARDS_PER_TARGET8`
For the same resolved receiver:
- last 8 strictly-prior receiver target-bearing games;
- mean PBP `air_yards` across non-null target events;
- minimum 12 valid air-yard targets;
- control only.

### WR-R15 opportunity / role controls
- `entitlement_tgt_share`;
- `WR1_indicator` (`wr_rank == 1`).

`pred_targets` is reported descriptively for redundancy auditing but is not added alongside entitlement in the frozen mediation equation to avoid unnecessary opportunity collinearity.

## 9. 2023 Stage A — raw signal gates

Stage A scores **2023 only**. 2024 remains sealed unless Sections 9 and 10 both pass.

Quartiles are by `EARLY_NO_EXTENDED_SHARE8`; Q4 = highest early/no-extended share and Q1 = lowest.

Raw Stage A passes only if **all** gates pass:

1. coverage >= **60%** of all 2,076 development rows;
2. Spearman(`EARLY_NO_EXTENDED_SHARE8`, `yard_residual`) <= **-0.08**;
3. Q4-Q1 receiving-yard residual gap <= **-5.0 yards**;
4. at least one direction-consistent tail gate:
   - **Q1/Q4 actual-100+ receiving-yard rate ratio >= 1.20**, or
   - **Q4/Q1 residual <= -30-yard rate ratio >= 1.20**;
5. WR1 Q4-Q1 residual gap is negative when WR1 n >=150;
6. WR2+ Q4-Q1 residual gap is negative when WR2+ n >=150;
7. identity, source, temporal and leakage audits pass.

If any raw gate fails:

`NO_ACTIONABLE_WR_EARLY_NO_EXTENDED_SIGNAL`

2024 remains sealed.

Descriptive positive findings may still be classified under `RESEARCH_EVIDENCE_CLASSIFICATION_V1`; they do not override the experiment disposition.

## 10. Receiver-specific / scheme-depth-role robustness gate

Run only if **every** raw Stage-A gate passes.

Fixed OLS residualization:

`EARLY_NO_EXTENDED_SHARE8 ~ TEAM_EARLY_NO_EXTENDED_SHARE8 + MEAN_AIR_YARDS_PER_TARGET8 + entitlement_tgt_share + WR1_indicator`

The OLS residual is `receiver_specific_early_no_extended`.

Because the primary direction is frozen negative, robustness passes only if:
- Spearman(`receiver_specific_early_no_extended`, `yard_residual`) <= **-0.06**;
- receiver-specific Q4-Q1 receiving-yard residual gap <= **-4.0 yards**.

If raw Stage A passes but robustness fails:

`WR_EARLY_NO_EXTENDED_TEAM_DEPTH_ROLE_MEDIATED`

2024 remains sealed.

Interpretation is frozen: mediation failure can represent genuine confounding or over-control of a real pathway; it is not causal disproof, but it blocks holdout exposure under R20.

If both raw and robustness gates pass:

`WR_EARLY_NO_EXTENDED_DEVELOPMENT_SUPPORTED`

Only then may 2024 Stage B be implemented after a separate adversarial implementation review.

## 11. 2024 Stage B — untouched confirmation

Stage B is forbidden unless 2023 is validly `WR_EARLY_NO_EXTENDED_DEVELOPMENT_SUPPORTED`.

Signal definition, semantics, identity, support floors, controls, negative direction and thresholds remain frozen unchanged.

Stage B uses the same thresholds as Stage A:
- coverage >=60%;
- Spearman <=-0.08;
- Q4-Q1 residual gap <=-5.0 yd;
- one frozen negative-direction tail ratio >=1.20;
- negative WR1 and WR2+ direction when n>=150;
- all source/identity/leakage audits pass;
- receiver-specific Spearman <=-0.06;
- receiver-specific Q4-Q1 gap <=-4.0 yd.

2023 pass + 2024 fail is not production eligible and is classified transparently under the evidence framework.

If both seasons pass:

`WR_EARLY_NO_EXTENDED_REPLICATED_2_OF_2`

Even a 2-of-2 pass authorizes only a separate integration experiment, not direct production modification.

## 12. Required descriptive outputs

Preserve regardless of disposition:
- n and coverage;
- Spearman;
- Q1-Q4 boundaries/counts;
- signed residual bias and MAE by quartile;
- actual-100+ rate by quartile;
- residual <=-30 rate by quartile;
- WR1 and WR2+ gaps;
- progression category counts and support distributions;
- team-control and air-yards-control availability;
- correlations with team progression share and target depth;
- exact identity/source hashes;
- zero target-game feature leakage;
- zero sportsbook input assertion;
- explicit `holdout_2024_scored` boolean.

## 13. Stop rules

Do not rescue a failure through:
- changing the frozen negative direction;
- replacing Spearman after seeing results;
- reopening `CHK` or `SD` into the denominator;
- testing `0`, `DES`, `1`, or `2` as separate candidate signals;
- thresholding the primary share post hoc;
- alternate history windows, recency deltas or EWMA;
- changing the 4-game / 16-classifiable-target support floor;
- WR1-only or WR2+-only modeling;
- tail-only modeling;
- interactions with defense, QB, target depth, catchability or CPOE;
- combining R17/R18/R19 features to rescue R20;
- model-zoo substitution;
- sportsbook information upstream.

Any later mechanism must be separately justified and frozen before untouched evidence is used.

## 14. Collaboration gate before implementation

Before any WR-R20 outcome is scored, Claude must independently review this exact frozen plan and specifically attack:
1. the frozen **negative** direction and football rationale;
2. the direction-specific tail definitions;
3. the 4-game / 16-classifiable-target support floor;
4. exclusion of `CHK` / `SD`;
5. `TEAM_EARLY_NO_EXTENDED_SHARE8` construction and 40-event floor;
6. `MEAN_AIR_YARDS_PER_TARGET8` construction and 12-event floor;
7. raw gates and unchanged monotonicity protection;
8. fixed mediation design and over-control risk;
9. identity/source/temporal protections;
10. 2024 seal and stop rules;
11. whether accumulated R17-R19 negative evidence makes R20 too adjacent to justify despite V2/V3 novelty screens.

Claude should return `REVIEW_PASS` or concrete pre-result amendments.

No evaluator implementation or WR outcome run is authorized until that review is reconciled.

## 15. Production disposition

`production_actionable = false`

No production model change. No paid Full Slate. No RB work.
