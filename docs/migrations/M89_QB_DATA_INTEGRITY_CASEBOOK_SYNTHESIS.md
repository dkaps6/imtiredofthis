# Migration 89 — QB Data Integrity, Catastrophic Casebook & Pregame Synthesis

## Status

`PREREGISTERED / FINAL BROAD QB SYNTHESIS`

M89 is the final broad QB research migration before the 2026 production overhaul. It does three things in strict order:

1. **Phase 0 — Data Integrity & Feature Semantics Gate**
2. **Phase 1 — Catastrophic Game Casebook**
3. **Phase 2 — Pregame Synthesis: football-only vs market-assisted**

No Phase 1 postgame field may enter a Phase 2 predictor. No sportsbook field may enter the football-only model.

## Frozen baselines

### 2024-2025 authoritative baseline

Consume the frozen M82 common-cohort artifact:

- 884 stable-primary QB games
- 444 in 2024 / 440 in 2025
- OOS ensemble MAE `56.749517`
- RMSE `72.303902`
- correlation `0.149475`
- 123 100+ yard misses
- artifact ID `9734973786`

### 2023 development baseline

Consume the frozen M88 2023 current-stack artifact:

- 447 stable-primary QB games
- OOS ensemble MAE `58.170780`
- current MC MAE `57.586531`
- artifact ID `9736226603`

2023 is the development/training season for any M89 residual synthesis. 2024-2025 are the locked evaluation seasons. No 2024-2025 result may be used to tune model hyperparameters, feature inclusion, thresholds, or residual caps.

---

# Phase 0 — Data Integrity & Feature Semantics Gate

## 0A. Independent actual-stat reconciliation

For 2023-2025 regular-season QB games, independently reconstruct passer attempts and passing yards from nflverse play-by-play and compare against nflverse weekly player stats.

Required outputs:

- row-level reconciliation by season/week/player ID/team;
- exact-match rates for attempts and passing yards;
- absolute discrepancy distributions;
- unresolved/mismatched identities.

Frozen gate:

- >= `99%` of matched QB-games must agree exactly on official pass attempts;
- >= `99%` must agree exactly on official passing yards;
- any systematic discrepancy pattern must be documented before Phase 2 is interpreted.

This gate tests our normalization/join logic as much as the upstream source.

## 0B. Correct football semantics

Build corrected strictly-prior team-game features from PBP. These are separate from target-game outcomes.

### True situation-adjusted PROE

For eligible offensive plays where nflverse supplies `xpass` or `pass_probability`:

`true_proe = observed_dropback_or_pass_indicator - expected_pass_probability`

Aggregate by completed team-game, then build trailing strictly-prior values for each target week.

Do **not** call observed pass rate minus league pass rate `PROE` in M89.

### Neutral pace

Neutral-state plays require all available conditions:

- regulation quarters 1-3;
- score differential between -7 and +7 when available;
- win probability between 0.20 and 0.80 when available.

Compute seconds between qualifying offensive plays within game. Label any weaker fallback explicitly.

### Pressure semantic correction

The standard public PBP foundation does not reliably provide all hurries/pressures. The existing sack-or-QB-hit construction must be labeled `hit_sack_pressure_proxy`, not full pressure. M89 may use it only under that name unless a source with true pressure is independently qualified.

## 0C. Source availability manifest

Every material input receives one deployment label:

- `LIVE_2026`
- `HISTORICAL_ONLY`
- `PROXY`
- `DEAD_SOURCE`
- `MISSING`

At minimum audit:

- PBP
- weekly player stats
- schedules
- weekly rosters
- injuries
- weather
- participation / man-zone / box
- WR/CB exposure
- game market total/spread

Known issues must be surfaced rather than hidden:

- historical outdoor weather currently lacks archived pregame forecasts;
- nflverse participation is historical-only for recent seasons rather than an in-season live feed;
- the old nflverse injury source is not acceptable as a 2026 live source if it is not updating;
- old `nflreadpy.load_player_stats(..., stat_type="weekly")` usage is incompatible with the maintained API shape and must not be used by M89.

## 0D. Phase-0 corrected benchmark

M89 must report whether the corrected semantic feature construction changes the full-stack QB point projection materially.

If a full-stack rerun is required, run current MC/Bayesian/rules + ML + State at 2,000 MC iterations on 2024-2025 with the corrected deployable feature contract.

The old M82 number remains historically authoritative until a corrected benchmark is explicitly frozen.

---

# Phase 1 — Catastrophic Game Casebook

Phase 1 is forensic only. Target-game box score/PBP may be used only after predictions are frozen.

## Target set

Generate case files for all 123 M82 OOS-ensemble 100+ yard misses, with priority views for:

- the 38 M86 low-event-chaos catastrophic misses;
- the 30 largest absolute ensemble misses overall.

## Required pregame side of each case

Record, when available:

- ensemble / MC / ML / State projection;
- predicted attempts and predicted YPA;
- offense prior plays, true PROE, neutral pace, hit+sack pressure proxy, recent QB attempts/YPA;
- defense prior pass EPA/success/YPA/pass-rate faced/explosive allowance/pressure proxy;
- injuries/availability status;
- venue/weather availability status;
- market total/spread/team implied points as a separately labeled market layer.

Generate a concise mechanical pregame story such as:

`projected_yards ~= predicted_attempts x predicted_ypa` plus the strongest contextual assumptions.

## Required actual-game reconstruction

At minimum:

- pass attempts/yards by quarter;
- first-half vs second-half passing;
- score differential on QB dropbacks/pass attempts;
- trailing/leading/neutral attempt shares;
- drive count and drive-level passing volume;
- opponent scoring by quarter / early scoring pressure;
- longest completion;
- 20+/40+/60+ completion counts;
- YAC totals and max YAC where available;
- receiver passing-yard concentration;
- single-largest-completion contribution;
- passing yards after removing the largest completion;
- sacks, interceptions, scrambles, fourth-down attempts, overtime;
- garbage-time proxy;
- whether the catastrophic miss remains >=100 yards after removing the single largest completion.

## Frozen descriptive failure taxonomy

Assign one primary forensic label using transparent deterministic rules, with `MIXED` fallback:

- `FORCED_PASS_VOLUME`
- `VOLUNTARY_PASS_VOLUME`
- `UNEXPECTED_LOW_VOLUME`
- `SUSTAINED_EFFICIENCY_EXPLOSION`
- `SUSTAINED_EFFICIENCY_COLLAPSE`
- `SINGLE_EXPLOSIVE_PLAY`
- `YAC_DRIVEN_EXPLOSION`
- `RECEIVER_CONCENTRATION`
- `PROTECTION_COLLAPSE`
- `DEFENSIVE_SUPPRESSION`
- `QB_EXECUTION_OUTLIER`
- `TURNOVER_POSSESSION_DISTORTION`
- `GARBAGE_TIME`
- `OVERTIME`
- `PARTICIPATION_INJURY`
- `MIXED`

Taxonomy labels are postgame explanations and are **never eligible Phase 2 features**.

## Casebook outputs

- row-level CSV for all 123 catastrophes;
- quarter/drive summary CSV;
- taxonomy counts by season and direction;
- largest-30 Markdown casebook;
- low-chaos-38 Markdown casebook;
- summary of how many catastrophic misses disappear below 100 yards after removing the single largest completion.

---

# Phase 2 — Pregame Synthesis

Phase 2 tests whether several modest pregame signals together improve the current projection.

## Training/evaluation boundary

- Train/freeze on 2023 only.
- Evaluate once on locked 2024-2025.
- No 2024-2025 hyperparameter tuning.
- Ridge regression only, fixed `alpha=20`.
- Residual target: `actual_pass_yards - base_projection`.
- Residual correction cap fixed before evaluation at `+/- 45 yards`.

## Base projection

For 2023 use the frozen M88 OOS ensemble where available. For 2024-2025 use the frozen/corrected OOS ensemble benchmark established by Phase 0.

## Football-only synthesis feature universe

Only strictly pregame, deployable football information may be used. Freeze this small universe before 2024-2025 scoring:

- base projection;
- MC / ML / State projections and pairwise disagreement summaries;
- predicted attempts;
- predicted YPA;
- offense trailing true PROE;
- offense trailing neutral pace;
- offense trailing pass rate / plays;
- QB trailing attempts / YPA;
- opponent trailing defensive pass EPA / success rate / YPA allowed;
- opponent trailing pass rate faced;
- offense/defense hit+sack pressure proxies under explicit proxy names;
- injury availability indicators only if the historical source is temporally valid;
- indoor/controlled environment and weather only where a legitimate pregame value exists.

Historical-only participation features may not qualify the football-only model as deployable.

## Market-assisted synthesis feature universe

Use the identical football-only feature set plus a separately labeled market layer:

- pregame game total;
- team spread;
- absolute spread;
- team implied points;
- opponent implied points;
- underdog indicator;
- moneyline only where coverage is adequate.

NFL schedule market fields are treated as a **closing/pregame market snapshot**, not as Wednesday information. M89 must state this timing limitation.

## Frozen evaluation metrics

Report on identical 2024-2025 evaluation rows:

- MAE
- RMSE
- bias
- correlation
- median absolute error
- 100+ yard misses
- under-100+ and over-100+ misses
- 2024 and 2025 separately
- bootstrap probability that candidate MAE beats base, 10,000 paired resamples, seed 89

## Promotion gates

### Football-only synthesis

Must satisfy all:

1. combined MAE improves by >= `0.75` yards;
2. MAE non-worse in both 2024 and 2025;
3. RMSE non-worse;
4. correlation improves by >= `0.01`;
5. 100+ misses do not increase;
6. bootstrap P(MAE improvement > 0) >= `0.80`;
7. all used features are `LIVE_2026` or explicitly allowed stable proxies.

### Market-assisted synthesis

This can never replace the independent football model. It may become a separate market-assisted projection only if all are true:

1. combined MAE improves by >= `1.00` yard versus the same base;
2. MAE non-worse in both seasons;
3. RMSE non-worse;
4. 100+ misses do not increase;
5. bootstrap P(MAE improvement > 0) >= `0.80`;
6. market coverage >= `90%` of the locked evaluation cohort.

## Stop rule

M89 is the final broad QB research migration.

- If neither synthesis clears its frozen gate: freeze QB research and move to the 2026 production overhaul / RB / WR / TE work.
- If one candidate clearly clears all gates: allow one M90 confirmation/promotion migration only.
- Do not open another broad QB feature-search loop from M89 failures.

`postgame_casebook_features_used_for_prediction = false`

`sportsbook_features_in_football_model = false`
