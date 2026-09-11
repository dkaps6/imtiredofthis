# NFL HANDOFF — 2026-09-11 — SEASON-LONG PARITY PASS + WR/TE RUNTIME AUDIT

Status: `STACK2_FROZEN_PARITY_PASS_SEASON_LONG_RUNTIME_AUDIT_ACTIVE`

Read parent recovery checkpoint first for full STACK2/STACK3/ND2A/ND2B lineage:
- `docs/handoffs/NFL_HANDOFF_2026-09-11_SEASON_LONG_RB_PHASE_A_RECOVERY_CHECKPOINT.md`

Branch: `production-season-long-readiness-2026`

## Governing objective

One stable production system for the entire 2026 regular season. Weekly runs refresh only pregame football state. Completed weeks are immutable grading evidence and may motivate prospectively frozen calibration changes only after predefined gates.

## STACK2 frozen allocation parity — PASS

Productionization code:
- `scripts/operations/freeze_rb_stack2_allocation_v1.py`
- implementation commit `472be112a7acf99a1d89aced50c1bc9655dd022f`

Workflow:
- `.github/workflows/rb-stack2-freeze-parity-v1.yml`
- workflow commit `05050776473118558f91ce8de789418300f24b87`

Authoritative verification:
- run `34659593197`
- job `103459130419`
- conclusion `SUCCESS`
- artifact `10286921484`
- artifact name `rb-stack2-freeze-parity-v1`
- artifact digest `sha256:28b8fa73dbe249e55388d108c0e022ab781bb343c1b8952c2c08b131097ffa94`

Result disposition:
- `RB_STACK2_FROZEN_ALLOCATION_PARITY_PASS`
- scientific search performed: `false`
- sportsbook inputs used: `false`

Exact parity evidence versus the archived 2025 STACK2 casebook:
- training rows: `2102` exactly
- holdout rows: `1393` exactly
- parity tolerance: `1e-10`
- max absolute `alloc_full_score` difference: `1.1102230246251565e-16`
- max absolute `alloc_full_share` difference: `1.1102230246251565e-16`

This is effectively exact numerical reproduction of the validated historical STACK2 allocation learner.

Frozen hashes from the passing artifact:
- training matrix SHA256: `5e871fd673cdbfb0f03c82b454adf65ff726109195dd4dcd0c8b2e51969b492e`
- serialized allocation learner SHA256: `d33960cbfae8a61af0dfa92dc61a97b96027c0a1a8036bf4308734442c8bdd9b`
- scikit-learn version: `1.4.2` (also pinned in repository requirements)

Learner remains the exact historical `HistGradientBoostingRegressor` contract:
- loss `squared_error`
- learning_rate `0.05`
- max_iter `160`
- max_leaf_nodes `15`
- min_samples_leaf `30`
- l2_regularization `1.0`
- random_state `17`

No hyperparameter search or retuning occurred.

## 2026 roster metadata contract — PASS

The same run explicitly tested `nflreadpy.load_rosters_weekly(2026)` for the STACK2 metadata fields.

Observed Week-1 source state:
- rows: `114` RB/FB rows
- teams: `32/32`
- weeks currently published: `[1]`
- `status_raw` coverage: `100%`
- `years_exp` coverage: `100%`
- `rookie_year` coverage: `100%`
- `entry_year` coverage: `100%`
- `draft_number` coverage: `78.070175%`

Missing `draft_number` is expected for undrafted players and is already represented by the frozen STACK2 feature semantics (`drafted_flag=0`, `draft_number_log` uses the frozen missing-value fallback). Do not invent draft picks.

Therefore the previously unresolved 2026 roster-metadata source family is available for season-long production.

## STACK2 artifact strategy

The passing artifact contains:
- `rb_stack2_frozen_training_matrix_v1.csv`
- `rb_stack2_allocation_model_v1.pkl.b64`
- `rb_stack2_2025_prediction_parity_v1.csv`
- `rb_stack2_2026_roster_metadata_audit.csv`
- result/contract JSONs.

Do not retrain from changing upstream historical sources in each weekly production run.

The production route must consume an immutable frozen allocation authority. Acceptable implementation must bind to the passing hashes above and the exact FULL feature order. If the serialized learner is committed/embedded, verify its SHA before use. If a deterministic rebuild from an immutable committed training matrix is chosen, verify matrix SHA and output-model SHA before use. Weekly data refresh may only create the target-week feature frame; it may not alter the frozen training set.

## WR-R15 / TE-R5P season-long runtime audit — IMPORTANT FINDING

The fitted WR/TE science is **not** Week-1-only.

Both adapters are written around generic target `season` / `week` rows and enforce strict-prior participation observations with ordinal `< target season*100+week`.

However, both currently share a runtime snap loader whose source season list is fixed to:
- `2020, 2021, 2022, 2023, 2024, 2025`.

TE-R5P owns `_load_snaps()` with that fixed list. WR-R15 imports the same loader from TE-R5P.

Consequence:
- 2026 Week 1 behavior is fine because no 2026 regular-season snap history exists yet;
- for 2026 Week 2+, the current runtime would **not** ingest Week-1 2026 participation;
- for later weeks it would continue using only through-2025 snap history even though the strict-prior logic itself is capable of safely consuming 2026 rows.

This is a **runtime source-freshness / season-long productionization defect**, not evidence that WR-R15 or TE-R5P coefficients are invalid.

### Frozen repair direction before implementation

Do not change WR-R15 or TE-R5P coefficients, feature definitions, conservation contracts, or fitted artifacts.

Repair only the snap-source horizon:
1. historical fitted source seasons remain frozen;
2. runtime feature acquisition must include the current target season in addition to the historical source seasons;
3. strict-prior filtering remains unchanged and must exclude target/same/future week rows;
4. Week-1 2026 output must be numerically invariant to current production because no earlier 2026 regular-season snaps can qualify;
5. a Week-2+ test must prove that a prior 2026 snap observation can enter the feature frame while a target-week observation cannot;
6. sportsbook inputs remain forbidden.

This should be treated as a production/runtime data fix, not new science.

## R26 / R22 season-long status

### R26 receptions

Current version is explicitly `RB_R26_WEEK1_RECEPTIONS_PRODUCTION_V1` and hard-fails outside 2026 Week 1.

Its Week-1 room state is a static offseason vacancy classification: 31 vacancy-active teams and CIN as the control. That classifier must **not** be carried through the whole season unchanged.

The R26 identity/residual mechanism may contain reusable science, but a season-long W2-18 router requires dynamic current-week role/vacancy state and separate prospective qualification. Until then, the protected baseline entitlement path is the safe future-week authority.

### R22 receiving-yard tail

Current version is explicitly `RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_V1` and hard-fails outside 2026 Week 1.

It owns receiving-yard tail/distribution shape, not the receiving-yard mean. Its strict-prior identity features and R19 residual pools may be transportable, but W2-18 transport must be prospectively qualified. Do not simply remove the Week-1 guard.

Until transport is qualified, future-week receiving distributions must use the protected baseline distribution rather than an unqualified Week-1 tail adapter.

## Current season-long readiness verdict

What is now established:
- P3 central Weeks 2-18 science already exists;
- STACK2 allocation learner is reproducible to machine precision;
- its 2026 roster metadata source exists across all 32 teams;
- current depth/status, injury, official inactive and active-role plumbing already exists and is certified;
- PlayerForm already enforces current-season `week < target_week` evidence;
- WR-R15 and TE-R5P fitted logic is generic, with a specific fixed-source-horizon runtime bug identified;
- R26 and R22 remain genuine Week-1 production adapters and require explicit future-week authority/fallback contracts.

What is NOT established yet:
- generic live W2-18 `enriched_att` builder wired into Full Slate;
- immutable STACK2 authority stored in the production tree and hash-validated at runtime;
- WR/TE current-season snap ingestion repair and invariance tests;
- R26 dynamic W2-18 transport qualification;
- R22 W2-18 transport qualification;
- audit of QB runtime for hidden Week-1/source-horizon assumptions;
- generic target-week no-odds Full Slate certification;
- immutable 2026 postgame grading/calibration harness.

## Exact next work order

1. Freeze/store the exact passing STACK2 allocation authority in-repo with hash validation.
2. Build one generic target-week STACK2 feature builder for Weeks 2-18 from current depth/availability/roster + lagged weekly stats/snaps.
3. Feed its `enriched_att` into existing P3 composition; preserve Week-1 route unchanged.
4. Replace Week-1-only P3 downstream validators with target-week routing invariants.
5. Repair WR/TE runtime snap-source horizon with Week-1 invariance and Week-2 prior-row inclusion tests.
6. Audit QB runtime source horizons / week assumptions.
7. Decide R26/R22 W2-18 authority using protected baseline unless transport earns prospective qualification.
8. Run generic no-paid-odds target-week certification before any live odds acquisition.

No paid OddsAPI call is authorized by this checkpoint.