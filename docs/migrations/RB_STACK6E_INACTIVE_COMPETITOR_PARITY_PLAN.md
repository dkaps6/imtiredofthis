# RB STACK6E — Exact-Inactive Competitor / Train-Eval Parity Repair

Status: FROZEN BEFORE 2025 OUTCOME EVALUATION

## Why this experiment exists

STACK6B and STACK6C showed that generic secondary-back contraction is the wrong fix. The corrected STACK6D source audit also showed that archived injury timestamps do not cover the 2025 P3 population.

A code-level audit of STACK2/P3 identified a narrower implementation/state mismatch:

- STACK2 training begins from the full weekly RB/FB roster (`ACT` + `INA`).
- `add_team_competition()` therefore computes 2024 competition context over the full weekly roster.
- STACK2 2025 evaluation begins from the M94C target table and then merges roster state onto those target rows.
- In the frozen 2025 P3/STACK6 casebook all 1,393 target rows are `ACT`; `roster_inactive == 0` for every target row.
- `add_team_competition()` is then run only over those M94C target rows. Exact inactive competitors are absent from the active player's team-context calculation.
- Existing team-competition features do not encode another player's `roster_inactive` state.

Therefore the historical inactive designation was loaded, but the target player's **competitor availability state** was not represented symmetrically between training and evaluation.

## Historical inactive-state qualification

For this research backtest only, weekly `status == INA` is accepted as reconstructed game-day inactive state because all of the following are true before this experiment is evaluated:

1. nflverse builds modern weekly roster files from the NFL/NGS roster source by season/team/week.
2. nflreadr's roster-status dictionary defines `INA` as inactive / under contract but not active for the game-week roster state.
3. NFL Football Operations states that each club's inactive list is delivered in the mandatory 90-minute pregame officiating meeting.
4. STACK6D delayed semantic validation found 557/557 RB `INA` rows had zero offensive participation.
5. An official NFL Game Book cross-check (2025 GB at DAL) lists RB Jaydon Blue as `Not Active`; the archived weekly NGS roster has Jaydon Blue as `INA` for the same team/week.

This does **not** qualify nflverse weekly rosters as a timestamped live 2026 feed. Any future production use of exact inactive state requires the official game-day inactive release (or another independently timestamped official source) and must fall back to the current P3 path before that state is available.

## Frozen parent

P3 remains champion and parent. No production change is authorized by this experiment.

For scoring/composition:
- Week 1 remains P3 exactly.
- The STACK6 correction domain remains Week 6+, M95F non-risk, pregame depth rank 2+.
- M95F-risk rows and pregame depth-rank 1 rows remain P3 exactly.
- P3 efficiency/context is preserved. Any candidate carry change is converted to yards using the frozen P3 implied efficiency (`parent_yards / parent_att`) when viable.

## Fixed training/evaluation split

- Fit allocation model: 2024 only.
- Evaluate: 2025 only.
- No 2025 outcome is used for feature choice, feature transformation, hyperparameters, thresholds, or arm construction.
- Sportsbook is downstream only and may be loaded only after football disposition is frozen.

## Frozen base allocation learner

Reuse STACK2's learner exactly:
- `HistGradientBoostingRegressor`
- squared-error loss
- learning rate 0.05
- max iterations 160
- max leaf nodes 15
- min samples leaf 30
- L2 regularization 1.0
- random state 17
- same 2024 actual RB carry-share target
- same original `FULL` feature block for the parity arm.

No hyperparameter search.

## Full-roster context construction

For both 2024 training and 2025 feature construction, first build features on the full weekly RB/FB roster (`ACT` + `INA`) before selecting the 2025 M94C/P3 target rows.

Historical player features remain strictly prior-game where they were prior-game in STACK2. Target-game inactive state is an administrative pregame state, not target-game participation/outcome data.

After full-roster team context is constructed, merge the resulting target-player features onto the frozen 2025 P3 target universe. Prediction normalization remains over the frozen M94C/P3 target rows for that team/week so this experiment does not create a new projection universe.

## Arm 1 — FULL_ROSTER_PARITY

Purpose: isolate the train/eval context mismatch without adding new concepts.

- Same original STACK2 `FULL` feature set.
- 2024 training context built on full roster (as already done by STACK2).
- 2025 evaluation context now also built on full roster before selecting target rows.
- Same model and same 50/50 M94C-share + learned-allocation-share composition used by STACK2.

## Arm 2 — INACTIVE_COMPETITOR_STATE

Purpose: add only the exact competitor-availability concepts that the existing active target row cannot express.

Use original STACK2 `FULL` block plus exactly these five fixed features:

1. `inactive_comp_count`
   - number of other same-team RB/FB roster rows with `status == INA`.
2. `inactive_comp_prior3_share`
   - sum of strictly-prior-three RB carry share belonging to inactive same-team competitors.
3. `inactive_comp_prior3_snap`
   - sum of strictly-prior-three offensive snap percentage belonging to inactive same-team competitors.
4. `inactive_above_count`
   - number of inactive same-team competitors with a known pregame depth rank numerically above the target player's rank.
5. `effective_active_depth_rank`
   - `max(1, depth_rank - inactive_above_count)` when depth rank is available; otherwise the original frozen missing-rank treatment.

No other feature may be added.

The same five features are constructed symmetrically in 2024 training and 2025 evaluation.

## Candidate allocation composition

For each arm:

1. Fit the frozen allocation learner on 2024.
2. Predict allocation score for the frozen 2025 target players using full-roster-derived context.
3. Normalize predicted scores within the frozen P3/M94C target team-week rows to obtain the candidate learned allocation share.
4. Preserve STACK2's fixed composition:
   - `candidate_enriched_share = 0.50 * m94c_share + 0.50 * candidate_alloc_share`
   - `candidate_att = candidate_enriched_share * frozen_m94c_team_rb_carry_pool`
5. Apply candidate carries only inside the STACK6 correction domain (Week 6+, M95F non-risk, depth rank 2+).
6. All protected rows remain P3 exactly.
7. Convert eligible candidate carries to yards using frozen P3 implied efficiency/context.

No post-hoc clipping, contraction-only rule, threshold, weight search, or population search is allowed.

## Frozen retention gates

An arm is retainable only if all are true versus P3:

- eligible carry MAE gain >= 0.20 carries
- eligible rushing-yard MAE gain >= 0.15 yards
- eligible Week 13–18 rushing-yard MAE gain > 0
- all-RB Week 6–18 rushing-yard MAE regression <= 0.05 yards
- eligible absolute carry-bias worsening <= 0.25 carries
- max yard change to M95F-risk rows = 0
- max yard change to pregame depth-rank 1 rows = 0

If both arms pass, prefer `FULL_ROSTER_PARITY` when it is within 0.05 eligible-yard MAE gain of the better arm; otherwise select the better eligible-yard gain.

## Downstream market rule

Sportsbook lines are prohibited from fitting, feature engineering, arm selection, thresholds, or retention. Market comparison may run only after the football disposition has been written.

## Possible dispositions

- `STACK6E_RETAIN_FULL_ROSTER_PARITY`
- `STACK6E_RETAIN_INACTIVE_COMPETITOR_STATE`
- `STACK6E_NO_RETAINABLE_INACTIVE_COMPETITOR_REPAIR`

No result promotes production automatically. A retained historical arm would still require a timestamp-safe live 2026 inactive-list implementation and prospective confirmation.
