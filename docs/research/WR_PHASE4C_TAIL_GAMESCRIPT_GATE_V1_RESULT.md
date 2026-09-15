# WR Phase 4C — Tail Game-Script Gate V1 Result

**STATUS: CANONICAL GATE-0 RESULT. RESEARCH ONLY. NO PRODUCTION CHANGE.**

## Canonical lineage

- branch: `research-wr-phase4c-tail-gamescript-gate-v1`
- frozen/amended plan: `docs/research/WR_PHASE4C_TAIL_GAMESCRIPT_GATE_V1_PLAN.md`
- plan review-conditions commit: `7baab5f7fc0fd50c5c82977fea8034ce72f42644`
- evaluator mechanics-fix head: `463d56138a48b304ba9671a5501e8a7a1abb0f3c`
- canonical run: `34993703697`
- canonical job: `104464523520`
- artifact: `10407226343` (`wr-phase4c-tail-gamescript-gate-v1`)
- artifact digest: `sha256:fb4ff45d66dac4d68d727ca3786c77f35b213a68b5658d9d721ac650ddd80cc6`
- upstream Phase 4B artifact: `10404525877`
- upstream Phase 4B digest reverified in CI: `sha256:0dad0dbc91a4bd6aa3e3f45cbc62e8ed493ca7bf2c95daaa0c4028a54a0fd0d4`

Two earlier attempts produced no scientific output: run `34993183752` failed during Python-3.12 dependency installation before tests/data; run `34993368410` failed a synthetic gate test before Phase-4B artifact download because pandas `.view` attribute access collided with the `view` column. The latter was fixed mechanically by explicit bracketed column access only; frozen cohorts, statistics, thresholds, and stop/go rules were unchanged.

Canonical run guards all passed: focused tests, exact Phase-4B digest, Gate-0 diagnostic, output-contract guard, artifact upload.

## Frozen cohort counts confirmed

- Layer 1 authority rows: 4,193
- Layer 2/3 complete team-games: 1,025
- `ACTUAL_100_PLUS`: 307
- `ACTUAL_100_PLUS_OPP_DOM`: 185
- `ABS_RESIDUAL_30_PLUS`: 1,048
- `ABS_RESIDUAL_30_PLUS_OPP_DOM`: 586
- `UNDERPROJECT_30_PLUS`: 809
- `UNDERPROJECT_30_PLUS_OPP_DOM`: 517 — primary Gate-0 cohort
- `OVERPROJECT_30_PLUS`: 239 — named directional control only

Bootstrap contract: 1,000 reps; player-row uncertainty clustered on exact `(season, week, team)` and pooled intervals season-stratified.

Comparison-space disclosure was preserved: 112 pooled player-tail cells, 336 pooled+season cells, plus 24 named directional-control cells.

## Frozen disposition

**`ADVANCE_TO_SCRIPT_PREDICTOR_DESIGN`**

Three primary-cohort market views independently satisfied the complete preregistered AND gate (primary CI excludes zero, same player-tail direction in 2023 and 2024, Layer-2 directional coherence, and replication in at least one secondary tail definition):

1. `market_total` bucket `<38`
2. continuous `market_total`
3. continuous `market_team_implied`

No spread-magnitude/favorite-status view advanced.

## Primary tail: `UNDERPROJECT_30_PLUS_OPP_DOM` (n=517)

### Low-total depletion

For posted game totals `<38`:
- pooled base prevalence: `0.123301`
- `<38` prevalence: `0.078947`
- prevalence difference: `-0.044353`
- enrichment ratio: `0.640283`
- team-game-cluster bootstrap 95% CI for prevalence difference: `[-0.070492, -0.019069]`

Season direction was stable:
- 2023: diff `-0.044956` (prevalence `0.075949` vs base `0.120906`)
- 2024: diff `-0.041733` (prevalence `0.083916` vs base `0.125650`)

The frozen rule required same direction by season, not per-season CI exclusion; 2024's individual CI crosses zero and is retained transparently.

### Continuous posted total

Primary cohort vs broad `ALL \ cohort` complement:
- pooled mean difference in `market_total`: `+0.586724`
- 95% cluster CI: `[+0.193427, +0.969713]`
- 2023: `+0.646531`
- 2024: `+0.501195`

Against the separately frozen 239-row `OVERPROJECT_30_PLUS` directional control:
- pooled `market_total` difference: `+0.809474`
- 95% cluster CI: `[+0.140955, +1.514414]`

### Continuous implied team total

Primary cohort vs broad complement:
- pooled mean difference in `market_team_implied`: `+0.354048`
- 95% cluster CI: `[+0.053080, +0.667707]`
- 2023: `+0.496429`
- 2024: `+0.203967`

The 2024 individual CI crosses zero; the preregistered season condition was same direction, which holds.

## Secondary tail replication

The same scoring-environment direction replicated strongly.

### `ACTUAL_100_PLUS_OPP_DOM` (n=185)

- `<38` prevalence difference: `-0.023069`; enrichment ratio `0.477155`; 95% CI `[-0.036573, -0.008941]`
- continuous `market_total` difference: `+1.178325`; 95% CI `[+0.596496, +1.773897]`
- continuous `market_team_implied` difference: `+0.953695`; 95% CI `[+0.457875, +1.471356]`

### Full `UNDERPROJECT_30_PLUS` (n=809)

- `<38` prevalence difference: `-0.050835`; enrichment ratio `0.736523`; 95% CI `[-0.081472, -0.019913]`
- continuous `market_total` difference: `+0.573892`; 95% CI `[+0.282890, +0.883380]`
- continuous `market_team_implied` difference: `+0.594700`; 95% CI `[+0.323680, +0.852235]`

## Layer-2 bridge

The Phase-4B `implied_team_target_pool` has no market-total/spread lineage; WR-R15 explicitly forbids sportsbook inputs and Phase 4B's authority loader hard-fails nonzero sportsbook use. Thus these associations are new descriptive mechanism evidence rather than rediscovery of an embedded market feature.

### Continuous `market_total`

Pooled across 1,025 complete team-games:
- vs signed `team_pool_component`: Spearman rho `+0.066438`, 95% CI `[+0.001081, +0.131335]`
- vs direct team-target residual (`actual_team_targets - implied_team_target_pool`): rho `+0.070911`, 95% CI `[+0.006259, +0.129831]`
- vs absolute team-pool component: rho `+0.009701`, CI crosses zero

Season signs for signed/direct error were positive in both years. The direct residual relationship was stronger in 2023 (`rho +0.094283`, CI excluding zero) and weaker in 2024 (`+0.049115`, CI crossing zero), so this is a small but coherent pooled bridge, not a large deterministic effect.

### Low-total `<38` team-games

Relative to all Layer-2 team-games, `<38` games showed:
- signed team-pool component: `-1.274013` targets pooled (`-0.883235` in 2023; `-1.914209` in 2024)
- direct team-target residual: `-2.342098` targets pooled (`-1.659701` in 2023; `-3.420968` in 2024)

This direction is coherent with the player-tail depletion in low-total environments.

### `market_team_implied`

The primary player-tail association passed the frozen Layer-2 directional requirement, but its Layer-2 evidence is weaker than raw game total: pooled correlations are small and CIs cross zero, with some season sign instability. Preserve it as supporting evidence, not the lead mechanism.

## Interpretation

Gate 0 supports a **scoring-environment / team-volume bridge** to the Phase-4B WR receiving-yard opportunity tail:

- large opportunity-driven WR underprojections are depleted in low-total games;
- they occur in modestly higher posted-total environments;
- the same direction replicates in both the 100+ opportunity-tail and the full large-underprojection cohort;
- posted total also carries a small but coherent association with Phase-4B team target-pool under/over-realization.

This does **not** prove a production feature, causal game script, or profitable betting edge. The effect sizes are modest, the full disclosed comparison space is large, and PR #558 already showed that broad market injection did not generally improve its historical plays/pass-rate proxy. Gate 0 only answers the narrower preregistered question: there is enough coherent market-only bridge evidence to justify designing — not yet fitting — a separate realized-script predictor and testing whether it adds incremental value beyond the market descriptors themselves.

## Required next step

Do not touch production. Do not immediately fit a predictor.

Next, freeze a separate Phase-4C script-predictor design, have it independently reviewed, and preserve the decisive comparison:
- A = baseline without market/script predictor
- B = baseline + market descriptors
- C = baseline + market descriptors + predicted realized-script representation

Only out-of-sample `C > B` can justify carrying the internally predicted script signal forward.

No M38/R15 redesign. No R17-R20/R21 restart. No RB. No paid Full Slate.
