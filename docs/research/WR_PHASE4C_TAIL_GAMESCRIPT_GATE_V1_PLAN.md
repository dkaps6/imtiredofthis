# WR Phase 4C — Tail Game-Script Gate V1

**STATUS: FROZEN BEFORE ANY GAME-SCRIPT ASSOCIATION OUTPUT. RESEARCH ONLY. NO MODEL/PRODUCTION/THRESHOLD CHANGE.**

## Authority and motivation

This lane follows the canonical WR Phase 4B V2 attribution:
- run `34987211287`
- artifact `10404525877`
- digest `sha256:0dad0dbc91a4bd6aa3e3f45cbc62e8ed493ca7bf2c95daaa0c4028a54a0fd0d4`
- result record commit `e9d80025f6cc203503fc7c7b169218f4b54cece8`

Phase 4B closed the obvious downstream opportunity-allocation blame:
- preserve M38;
- preserve R15 (`R15_WR2PLUS_ALLOCATION_HEALTHY_OR_IMPROVED`);
- do not build an R15 challenger;
- do not restart the closed R17-R20 target-quality family.

Phase 4B instead showed that opportunity becomes disproportionately important in the large WR-yardage misses. This plan asks one cheap question before authorizing any new script-prediction model:

> Do already-available, leakage-safe pregame market game-script descriptors materially enrich the Phase 4B WR opportunity-tail failures and/or explain the Layer-2 team-target-pool miss?

If not, stop. Do **not** build a realized-script predictor just because game flow is football-plausible.

## Existing game-script source — reuse, do not re-derive

Reuse the reviewed schedule loader/sign convention from PR #558 / `scripts/research/diagnose_market_implied_game_script_v1.py`:
- `market_total`
- `market_team_spread` (positive = this team favored)
- `market_abs_spread`
- `market_team_implied = (market_total + market_team_spread) / 2`

Use the fixed PR #559 spread/total buckets unchanged:
- spread magnitude: `[0-3, 3-7, 7-10, 10-14, 14+]`
- total: `[<38, 38-42, 42-46, 46-50, 50+]`

Important prior evidence is preserved rather than rewritten:
- PR #558 found no general incremental market value over its fitted historical proxy for team plays/pass rate.
- PR #559 showed spread/total are real but noisy descriptors of realized game environment.
- PR #560 showed realized/high-scoring script has a strong WR/TE volume relationship, while pregame Vegas alone is weaker.

Therefore this is a **tail-selection diagnostic**, not a resurrection of PR #558's failed broad injection thesis.

## Gate 0 only — no new script predictor yet

### Canonical Phase 4B inputs

Use only the immutable canonical Phase 4B outputs:
- `phase4b_layer1_yard_decomposition.csv` (4,193 player-games)
- `phase4b_layer2_3_team_game_detail.csv` (1,025 complete anchor-observable team-games)

Join market descriptors by exact `(season, week, team)`.

No player props. No sportsbook spend. No target-game feature engineering beyond the pregame schedule market descriptors above.

### Phase 4B team-pool market-lineage disclosure

`implied_team_target_pool` has **no market-total / spread / sportsbook lineage** in the frozen Phase 4B authority path:
- Phase 4B reconstructs it algebraically as authority `pred_targets / entitlement_tgt_share` on positive-entitlement rows;
- the promoted WR-R15 adapter explicitly forbids sportsbook inputs and its production/audit contract records `sportsbook_inputs_used=False`;
- the R15 conservation artifact consumed by Phase 4B hard-fails if `sportsbook_inputs_used != 0`;
- PR #558 independently documented that the shared historical `project_game_script()` engine did not read live spread/total.

Therefore any Layer-2 association with the market descriptors is interpreted as **new descriptive mechanism evidence about a pregame signal absent from the authority pool**, not as rediscovery/recalibration of a signal already embedded in `implied_team_target_pool`.

### Frozen Layer-1 cohorts

These counts are known from Phase 4B **before any game-script association is inspected**:

1. `ALL`: all 4,193 canonical candidate rows.
2. `ACTUAL_100_PLUS`: 307 rows (`actual_rec_yards >= 100`).
3. `ACTUAL_100_PLUS_OPP_DOM`: 185 rows where `abs(opportunity_yards) > abs(efficiency_yards)`.
4. `ABS_RESIDUAL_30_PLUS`: 1,048 rows (`abs(yard_residual) >= 30`) — retained for continuity with the frozen Phase 4B summary.
5. `ABS_RESIDUAL_30_PLUS_OPP_DOM`: 586 rows.
6. `UNDERPROJECT_30_PLUS`: 809 rows (`yard_residual >= 30`). This is the primary directional large-underprojection cohort.
7. `UNDERPROJECT_30_PLUS_OPP_DOM`: 517 rows. This is the primary opportunity-tail cohort.
8. `OVERPROJECT_30_PLUS`: 239 rows (`yard_residual <= -30`) — disclosure/control only, never pooled into the primary directional conclusion.

The 307 and 1,048 full-tail counts must **not** be described as all opportunity-dominant; only the explicit OPP_DOM subsets are.

### Frozen cluster unit

Every player-row bootstrap/enrichment uncertainty calculation in Section A uses **team-game clusters `(season, week, team)`**, stratified by season for pooled intervals. It must not cluster at `(season, week)` and must not treat individual WR rows as independent.

### Frozen complement/control semantics

For Section A continuous and enrichment comparisons:
- **Primary complement** for each named event cohort = all `ALL` rows not in that cohort, preserving the broad population contrast.
- **Named secondary directional control** = `OVERPROJECT_30_PLUS` for the primary underprojection cohorts (`UNDERPROJECT_30_PLUS` and `UNDERPROJECT_30_PLUS_OPP_DOM`).
- The `OVERPROJECT_30_PLUS` cohort itself is disclosure-only and is not used as its own complement.

Both are reported separately; they must not be conflated.

### Gate-0 reporting

#### A. Player-tail enrichment by existing pregame market descriptors

For every frozen event cohort above except `ALL`, pooled and separately for 2023/2024:
- row count and base prevalence;
- prevalence and enrichment ratio within each fixed PR #559 total bucket;
- prevalence and enrichment ratio within each fixed PR #559 spread-magnitude bucket;
- favorite vs underdog split using the sign of `market_team_spread`;
- mean/median `market_total`, `market_team_implied`, `market_team_spread`, and `market_abs_spread` versus the primary complement cohort;
- for the two underprojection cohorts, repeat the continuous comparison against the named `OVERPROJECT_30_PLUS` directional control.

Use team-game-cluster bootstrap 95% CIs for prevalence/enrichment differences and continuous mean differences so multiple WR rows from one team-game are not treated as independent evidence.

Primary player-tail read = `UNDERPROJECT_30_PLUS_OPP_DOM` (517 rows).
Secondary confirmations = `ACTUAL_100_PLUS_OPP_DOM` (185) and full `UNDERPROJECT_30_PLUS` (809).
The absolute-miss cohorts are continuity/sensitivity only.

#### Comparison-space disclosure

The output must disclose the complete Section-A comparison space, not only significant cells. With seven event cohorts and the frozen market views, the pooled player-tail table contains:
- 5 total-bucket prevalence contrasts,
- 5 spread-bucket prevalence contrasts,
- 2 favorite/underdog prevalence contrasts,
- 4 continuous-descriptor complement contrasts,
for **16 pooled market comparisons per event cohort = 112 pooled cohort/market comparisons**.

The same 112 cells are also reported separately in 2023 and 2024, yielding **336 total pooled+season-specific cohort/market cells** before the named directional-control repeats. The two underprojection cohorts add 4 continuous descriptors × 2 cohorts × 3 scopes (pooled/2023/2024) = **24 named directional-control cells**. These counts are fixed and must be printed in the result metadata.

Condition 1 below is judged on the **primary cohort only**, but all 112/336/24 cells remain visible so the multiple-comparison exposure cannot be hidden.

#### B. Layer-2 team-pool association

On the 1,025 complete Layer-2/3 team-games, pooled and per season, report association of each continuous market descriptor with:
- `team_pool_component` (signed Phase-4B WR-room target contribution from team-pool error),
- `abs(team_pool_component)`,
- direct team-target residual = `actual_team_targets - implied_team_target_pool`.

Report Spearman rho and team-game bootstrap 95% CI.
Also report those same three Layer-2 outcomes by the fixed total/spread buckets above.

This is descriptive mechanism evidence only; no causal claim.

## Prospective stop/go rule before any realized-script model

Do **not** build the new realized-script predictor (Claude's proposed Arm C) unless Gate 0 shows a coherent market-only bridge between pregame script and the Phase 4B opportunity failure.

`ADVANCE_TO_SCRIPT_PREDICTOR_DESIGN` requires all of:
1. **Primary tail enrichment:** `UNDERPROJECT_30_PLUS_OPP_DOM` shows a non-null pregame market association with a team-game-cluster-bootstrap 95% CI excluding zero for at least one of the frozen market descriptor/bucket views.
2. **Season coherence:** the corresponding effect has the same direction in both 2023 and 2024; no pooled-only sign reversal.
3. **Layer-2 coherence:** the same market concept also shows directionally coherent association with either signed/direct team-target-pool error or its magnitude. It cannot be only a player-row artifact with no team-pool connection.
4. **Replication in one secondary tail definition:** either `ACTUAL_100_PLUS_OPP_DOM` or full `UNDERPROJECT_30_PLUS` must point in the same direction.

Otherwise disposition = `NO_ACTIONABLE_TAIL_GAMESCRIPT_BRIDGE`; stop this lane and do not build a script predictor.

No post-hoc threshold/bucket search. Existing PR #559 buckets are fixed. Continuous metrics are reported alongside them to avoid dependence on any one bin edge.

## If and only if Gate 0 advances

Then freeze a separate Phase 4C Stage-1 predictor plan **before fitting it**. Candidate pregame-only inputs may include market spread/total/ML plus strictly-prior team pace/PROE/rest/home-away, with a genuine historical holdout. The decisive later comparison must be incremental:

- A = baseline without market/script predictor
- B = baseline + market descriptors
- C = baseline + market descriptors + our predicted realized-script representation

Only `C > B` out of sample can justify carrying our own script prediction forward. No such model is authorized by this Gate-0 plan itself.

## Boundaries

- No production/model/weight/threshold change.
- No M38/R15 redesign.
- No R17-R20/R21 feature fishing.
- No RB work.
- No paid Full Slate.
- No player-prop odds used upstream.
- No fitting a new realized-script model until Gate 0 passes and a new plan is frozen/reviewed.
