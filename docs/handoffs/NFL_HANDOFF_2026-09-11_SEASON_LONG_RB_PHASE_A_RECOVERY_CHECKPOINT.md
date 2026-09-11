# NFL HANDOFF — 2026-09-11 — SEASON-LONG RB PHASE A RECOVERY CHECKPOINT

Status: `PHASE_A_LINEAGE_RECOVERED_LIVE_BRIDGE_FREEZE_NEXT`

Parent handoff: `docs/handoffs/NFL_HANDOFF_2026-09-11_SEASON_LONG_PRODUCTION_READINESS_CURRENT.md`

Branch: `production-season-long-readiness-2026`

## Governing objective

Build one stable 2026 production system that runs every week. Week number may change available pregame state, but must not trigger a new formula/research project. Completed 2026 games are a separate calibration/monitoring stream and may support prospectively frozen future changes only.

## Phase A recovery verdict

The central RB rushing science for a season-long system already exists. The primary unresolved work is productionization of its live weekly state and frozen allocation learner, not invention of a Week-2 model.

### STACK2 authoritative lineage

- branch: `research-rb-stack2-enriched-allocation-integration`
- head: `c07150158c5368c02d618f7504d95efed661ed66`
- run: `33538770934`
- artifact: `9812754276`
- artifact digest: `sha256:793b90494e2ea5a562e53be79311cb59b07d9589d934fb4237ae03e014ea774a`
- fit season: 2024
- held-out evaluation season: 2025
- sportsbook upstream: 0

STACK2 allocation construction:

1. preserve M94C RB/FB opportunity pool;
2. produce `alloc_full_score` from the frozen FULL pregame feature family using a deterministic `HistGradientBoostingRegressor`;
3. normalize those scores within team to `alloc_full_share`;
4. `enriched_share = 0.5 * m94c_share + 0.5 * alloc_full_share`;
5. `enriched_att = enriched_share * M94C team RB/FB opportunity pool`;
6. central architecture combines `enriched_att` with full-stack/STACK1 implied efficiency.

The 50/50 correction was frozen; no blend-weight search was performed in the authoritative disposition.

2025 held-out rushing-yard evidence:

- STACK1 parent all-RB MAE: `20.424163`
- M94C raw all-RB MAE: `21.031150`
- M94C enriched-allocation all-RB MAE: `20.959389`
- `ARCH_ENRICHED_OPP_STACK_EFF` all-RB MAE: **`20.047102`**
- `ARCH_ENRICHED_OPP_STACK_EFF` RMSE: `28.990310`
- `ARCH_ENRICHED_OPP_STACK_EFF` bias: `-2.302435`
- `ARCH_ENRICHED_OPP_STACK_EFF` correlation: `0.626204`

899-row archived-market cohort:

- STACK1 parent MAE: `24.762221`
- `ARCH_ENRICHED_OPP_STACK_EFF` MAE: **`24.450303`**
- Vegas consensus MAE: `23.701891`

Allocation-share evidence on 1,393 2025 RB/FB rows:

- M94C raw share MAE: `0.121555`
- FULL role/usage share MAE: `0.121157`
- frozen 50/50 enriched share MAE: **`0.119405`**

### STACK3 authoritative lineage

- branch: `research-rb-stack3-frozen-state-composition`
- head: `9d7ea5d0173569ac9e4633685da7e91eed5fcd3d`
- run: `33539468967`
- artifact: `9812993290`
- artifact digest: `sha256:3ad1c1a36cc8c1e822814be1393150e5e7a9800c0e37e232bbc26f94b79f17f3`
- disposition: `STACK3_FROZEN_STATE_COMPOSITION_DEVELOPMENT_ONLY`

STACK3 tested frozen M95F/M95I state routers on top of STACK2. They did **not** improve the Weeks 2-18 central rushing architecture enough to replace STACK2.

Weeks 2-18:

- STACK2 parent MAE: **`19.940341`**
- M95F risk override MAE: `20.372934`
- M95I carry/stack-efficiency MAE: `20.014341`
- vacancy enriched route MAE: `20.025772`

Week 1:

- STACK2 parent MAE: `21.689957`
- explicit STACK1/Week-1 override MAE: **`20.090830`**

Therefore the recovered P3 season-long rushing architecture remains:

- **Week 1 initialization:** STACK1 full-stack rushing-yard projection;
- **Weeks 2-18:** enriched STACK2 opportunity/allocation × STACK1 implied efficiency.

This is a single season-long routing contract with a Week-1 initialization regime, not separate weekly models.

## Recovered STACK2 FULL feature contract

Frozen FULL features:

- `depth_present`
- `depth_rank`
- `depth_rank_missing`
- `depth_slot_rb`
- `depth_slot_fb`
- `roster_active`
- `roster_inactive`
- `rookie_flag`
- `drafted_flag`
- `draft_number_log`
- `injury_reported`
- `injury_out_doubtful`
- `injury_questionable`
- `practice_dnp`
- `practice_limited`
- `depth_back_count`
- `injured_comp_count`
- `injured_comp_prior_share`
- `prior_games`
- `prior1_carries`
- `prior3_carries`
- `prior5_carries`
- `prior1_rb_share`
- `prior3_rb_share`
- `prior5_rb_share`
- `prior1_rush_yards`
- `prior3_rush_yards`
- `prior1_snap_pct`
- `prior3_snap_pct`
- `prior1_snap_count`
- `prior3_snap_count`
- `same_team_last_game`
- `max_comp_prior3_share`
- `max_comp_prior3_snap`
- `credible_competitors`
- `prior_backfield_hhi`
- `team_prior1_qb_rush_share`
- `team_prior3_qb_rush_share`

All history functions in the authoritative evaluator use observations with order strictly less than target `season*100 + week`. Target-game carries/snaps/outcomes are not inputs.

## Historical source reconstruction evidence

### ND2A

- branch: `research-rb-nd2a-role-state-source-audit`
- run: `33504022163`
- artifact: `9798861136`
- digest: `sha256:3ac7b8e6ab97793fa9f06cc20db1fb465a94f7c9fa31bf651d227d9d01fed6bb`

ND2A documented that target-game participation is forbidden and only prior/rolling participation is eligible. It exposed the old M94C role-state hole rather than filling it with unsafe current snapshots.

### ND2B

- branch: `research-rb-nd2b-allocation-env-atlas`
- run: `33509092341`
- artifact: `9800848065`
- digest: `sha256:e8f5d3dd60da7464a5000485061087a3b2ef7bdcb66ec9254025e829c1f4be77`

2025 timestamped depth evidence:

- 554,215 total depth records;
- 37,242 RB rows;
- latest snapshot strictly before kickoff selected for every 2025 regular-season team-game;
- 544/544 team-games had pregame depth coverage;
- median snapshot age 10.77 hours;
- p90 age 17.90 hours;
- exact M94C RB/FB depth-rank coverage 94.9749%;
- Week-1 depth-rank coverage 96.4706%.

Lagged snap evidence:

- prior-week offensive-snap coverage all weeks: 78.6073%;
- Weeks 2-18: 83.7156%;
- target-game snaps forbidden.

Conclusion: current-role state is historically reconstructable with strong timestamp integrity and was legitimately used as a signal alongside lagged usage/competition—not as a deterministic carry-share rule.

## Current live-source matrix

| STACK2 information family | Historical research source | Current production/live source status | Season-long disposition |
|---|---|---|---|
| target-week schedule/kickoff | nflreadpy schedules | existing authoritative Full Slate schedule | AVAILABLE / REQUIRED |
| current depth role/index | timestamped nflreadpy depth as-of before kickoff | `scripts/providers/ourlads_depth_status_v1.py` stamps `source`, `source_url`, `source_asof_utc`, preserves status, requires 32 teams | AVAILABLE / REQUIRED |
| current availability | historical injury report | `build_injuries_weekly.py` + official inactives + `build_current_player_availability_v1.py` | AVAILABLE / REQUIRED |
| current eligible RB universe | historical weekly roster/depth | certified `roles_current_production_eligible_v1.csv` / current availability stack | AVAILABLE / REQUIRED |
| prior carries/rush yards/RB share | nflreadpy weekly stats | `nflreadpy.load_player_stats(..., summary_level='week')`; existing PlayerForm already filters active-season evidence to `week < target_week` | AVAILABLE / REQUIRED after Week 1 |
| prior offensive snaps | nflreadpy snap counts | `nflreadpy.load_snap_counts`; existing repo consumers already use source; model historically tolerated incomplete lagged coverage | AVAILABLE WHEN PUBLISHED / MAY BE MISSING, NEVER TARGET-WEEK |
| prior team QB rushing competition | weekly player stats | derivable from same lagged weekly stats | AVAILABLE / REQUIRED after Week 1 |
| roster continuity/team change | weekly roster + prior logs | derivable from current roster identity + lagged logs | AVAILABLE / REQUIRED |
| rookie/draft prior | weekly roster metadata | research source uses `load_rosters_weekly` fields `years_exp`, `rookie_year`, `entry_year`, `draft_number` | SOURCE EXISTS; LIVE 2026 CONTRACT STILL TO VERIFY |
| sportsbook | none | downstream only | PROHIBITED UPSTREAM |

## Important existing production infrastructure recovered

The current repo already has a timestamped Ourlads depth/status sidecar. The legacy `roles_ourlads.csv` strips status/timing, but `roles_ourlads_status_v1.csv` was explicitly introduced to preserve provenance and `source_asof_utc`.

The current availability system already reconciles:

- timestamped Ourlads depth/status;
- weekly injury reports;
- official game-day inactives;
- deterministic role re-ranking after definitive unavailability.

Its promotion lineage passed **35/35** integration gates and is sportsbook-free. Therefore current role/availability plumbing should be reused rather than rebuilt.

PlayerForm v2 already implements the correct season-long evidence boundary:

- prior-season history is an explicit prior;
- current-season evidence is restricted to `week < target_week`;
- same-week leakage is explicitly forbidden;
- a fixed four-game prior-season pseudo-sample allows current-season evidence to take over progressively.

## Newly isolated productionization gap

The authoritative STACK2 evaluator trains `alloc_full_score` with a deterministic `HistGradientBoostingRegressor` using 2024 rows and evaluates on 2025. The repository does not currently expose a persisted production STACK2 allocation model artifact in the recovered search.

This matters because a season-long system must **not refit the allocation learner ad hoc every week**.

The correct productionization path is to materialize/freeze the exact historical allocation learner once, verify its predictions reproduce the archived STACK2 2025 `alloc_full_score` / `alloc_full_share` within numerical tolerance, then use that immutable artifact for every 2026 target week.

This is model-preservation work, not new model science.

## Frozen implementation direction before code changes

No scientific retuning is authorized.

The season-long P3 rushing bridge must preserve all of the following:

1. exact frozen FULL feature names and semantics;
2. exact historical training cohort/learner/hyperparameters unless a separately frozen scientific proposal is later authorized;
3. exact within-team score normalization;
4. exact 50/50 `m94c_share` / `alloc_full_share` blend;
5. exact M94C RB/FB opportunity-pool conservation;
6. exact P3 Weeks 2-18 composition: `enriched_att × STACK1 implied YPC`;
7. Week-1 STACK1 initialization remains unchanged;
8. all current-week state must be timestamped pregame;
9. all usage/snap history must satisfy `week < target_week`;
10. sportsbook inputs remain downstream only;
11. missing optional lagged snap fields remain missing/NaN rather than being filled from target-week outcomes;
12. required-source failure must fail closed with an explicit reason;
13. every weekly pregame state/projection artifact must be preserved for postgame grading.

## Production hard-coding still to remove safely

Current downstream validation/orchestration still contains explicit Week-1 assumptions, including `validate_full_slate_post_pricing_v1.py` requiring eligible RB rush pricing to use `WEEK1_STACK_OVERRIDE`.

Do **not** merely delete that guard. Replace it with a target-week routing invariant:

- week 1 -> `WEEK1_STACK_OVERRIDE`;
- weeks 2-18 -> `WEEKS2_18_ENRICHED_OPP_STACK_EFF`;
- anything else -> fail closed.

## Exact next step

Before implementing the live bridge:

1. verify the 2026 `load_rosters_weekly` metadata contract needed for rookie/draft/status features;
2. materialize the exact frozen 2024-fit STACK2 allocation learner and prove archived 2025 prediction parity;
3. freeze the production artifact hash and feature contract;
4. only then build the generic target-week RB state adapter and replace Week-1-only downstream routing guards.

Do not trigger paid odds acquisition during this work.