# RB STACK1–STACK5 Continuity — 2026-09-01

This is the authoritative supplement for the latest RB integration sequence. Read it together with `CURRENT_NFL_RESEARCH_HANDOFF.md`. New insight does not erase prior valid capability evidence. Sportsbook remains downstream benchmark only, and 2025 is exposed retrospective development data.

## STACK1 — production-equivalent historical parent

- Run `33535308110`
- Job `99948041928`
- Tested SHA `b3cd9d34d35e93f9dfc32a61a1a5c222a1df81f8`
- Artifact `9811878828`
- SHA256 `b66c3e403f54a8948f63c66524bc5b404f2fe13f73f3b5228ae50f21239b220c`

2025 all 1,393 RB/FB rush-yard games: full-stack ensemble MAE `20.424163`, RMSE `30.069806`, bias `-5.537128`, corr `.616847`; M94C MAE `21.031150`, RMSE `29.861488`, bias `+0.708378`, corr `.601602`.

Exact 899 market-covered games: full stack `24.762221`, M94C `25.515051`, Vegas `23.701891`.

Interpretation: full stack is better in ordinary/Week-1 yard outcomes; M94C keeps carry-center/high-workload strength. Historical full stack still had `role` / `rules_role` coverage 0, so explicit role/allocation enrichment remained necessary.

## STACK2 — missing football state + allocation + integration

Branch `research-rb-stack2-enriched-allocation-integration`.
- Run `33538770934`
- Job `99959516813`
- Tested SHA `c07150158c5368c02d618f7504d95efed661ed66`
- Artifact `9812754276`
- SHA256 `793b90494e2ea5a562e53be79311cb59b07d9589d934fb4237ae03e014ea774a`

Pregame information included timestamp-safe depth state, roster/rookie/draft priors, injuries/practice, lagged carries/share, lagged snaps, competitors, backfield concentration, and QB rushing competition. 2024 fit only; 2025 evaluation; sportsbook upstream 0; blend-weight search 0.

Allocation-share MAE/corr:
- raw M94C `.121555` / `.837965`
- full role+usage `.121157` / `.841682`
- anchored 50/50 M94C + role/usage `.119405` / `.845455` (RMSE `.158479`)

Best architecture: `ARCH_ENRICHED_OPP_STACK_EFF = enriched M94C opportunity/allocation × full-stack implied efficiency/context`.
- all-RB MAE `20.047102`, RMSE `28.990310`, bias `-2.302435`, corr `.626204`
- market-899 `24.450303` vs Vegas `23.701891`

Fixed 75/25, 50/50, 25/75 point blends were inferior to structural integration. Reverse architecture was worse. M94C/enriched opportunity remained materially stronger on actual 20+/25+ workloads.

## STACK3 — frozen M95F/M95I state composition

Branch `research-rb-stack3-frozen-state-composition`.
- Run `33539468967`
- Job `99961832724`
- Tested SHA `9d7ea5d0173569ac9e4633685da7e91eed5fcd3d`
- Artifact `9812993290`
- SHA256 `3ad1c1a36cc8c1e822814be1393150e5e7a9800c0e37e232bbc26f94b79f17f3`

No fit/search/upstream sportsbook.

Retain exact Week-1 full-stack override:
- all-RB `20.047102 -> 19.949524`
- Week-1 `21.689957 -> 20.090830`
- market-899 `24.450303 -> 24.315798`

M95F hard workload-risk point override and M95I carry-tail/vacancy point routing worsened the new parent overall/market despite helping postgame high-workload slices. Retain M95F for workload/tail distribution; retain M95I for transition/tail evidence; do not force either as universal point switches.

Current central development point: `P3 = full stack in Week 1; STACK2 enriched-opportunity × full-stack-efficiency otherwise`.
- all-RB `19.949524`
- market-899 `24.315798`
- Vegas `23.701891`

## STACK4 — frozen M96C efficiency portability

Branch `research-rb-stack4-efficiency-portability`.
- Run `33539814046`
- Job `99962987705`
- Tested SHA `e5f9753790845aedbb566403fe3811f2cf1faf4e`
- Artifact `9813126530`
- SHA256 `8cd946ca22b8aff84df19fdfc14072b5929d952fe0c215b3c0058ccf295b40c7`

No new fit/search. Frozen M96C E/P/D residuals were applied to P3:
- E and P worsened parent.
- D was near neutral.
- D suppressed in frozen M95F-risk games improved all-RB only `19.949524 -> 19.913324`, but worsened market-899 `24.315798 -> 24.337771`.

Do not force E/P/D into the current point. D remains a tiny conditional clue. M95D X remains rejected as isolated tail increment unless a genuinely new distribution/data architecture reopens it.

## STACK5 — remaining Vegas-gap reverse engineering

Branch `research-rb-stack5-market-gap-forensics`.
- Run `33540065380`
- Job `99963812729`
- Tested SHA `c9087faad1bcb4788dc3ea4281bf637ce3b3af99`
- Artifact `9813220944`
- SHA256 `e53ef9afa4f103eb9c8368fcc9b7165eeb4fc9abe49952e38f42f9fa85872188`
- no fit; sportsbook downstream only.

P3 vs Vegas exact 899:
- P3 MAE `24.315798`
- Vegas `23.701891`
- gap `+0.613907`
- P3 closer `48.1646%`

All-899 oracle decomposition remains joint: opportunity recovery `7.8845`, efficiency recovery `7.9392`, opportunity dominant `50.95%`.

### The remaining market gap is localized

Frozen M95F workload-risk rows (428): P3 `28.385169` vs Vegas `28.514019`; **P3 better by ~0.129** and closer 52.10%.

M95F non-risk rows (471): P3 `20.617941` vs Vegas `19.329087`; **Vegas better by ~1.289**, P3 closer 44.59%.

Depth hierarchy:
- rank1: P3 `27.202466`, Vegas `27.039130` (gap +`.163`)
- rank2: P3 `21.316647`, Vegas `20.365782` (gap +`.951`)
- rank3+: P3 `21.204248`, Vegas `19.660000` (gap +`1.544`)

Large model-above-Vegas 10+ rows (135): P3 `21.167141`, Vegas `18.811111`, gap `+2.356`; projected carries `11.28`, actual `9.63`, carry error `+1.65`; opportunity-oracle recovery `8.5445` vs efficiency `5.7451`; opportunity dominant `57.78%`.

This is the clearest remaining systematic failure: **large false-high projections are primarily opportunity/allocation overestimates.**

Model-below-Vegas 10+ rows (158): P3 `28.093802`, Vegas `27.310127`, gap +`.784`; predicted carries `12.16`, actual `14.54`, carry error `-2.38`; opportunity/efficiency mechanism 50/50. False-low misses include workload shocks and giant efficiency/explosive outcomes, but the market edge is less systematic.

Week 1 is no longer dominant: 53 listed W1 rows P3 `23.709666`, Vegas `23.198113`, gap `.512`. Rookies are nearly tied: P3 `22.820668`, Vegas `22.807927`. Injury-report rows still show ~`1.110` Vegas advantage and deserve exact availability-role audit.

Representative false-high opportunity cases include Chuba Hubbard, Rico Dowdle, Miles Sanders, Alvin Kamara, D'Andre Swift, Tank Bigsby, Jacory Croskey-Merritt, Tyjae Spears, Breece Hall and others where predicted carry allocation remained materially above actual despite aggregate depth/snap enrichment.

## Retained capability map

- `P3` central point: current best retrospective development parent.
- M94C/enriched opportunity: retain central/team opportunity and high-workload strength.
- Full stack: retain efficiency/context and Week-1 initialization strength.
- Depth/snap/role allocation: retain, but aggregate snap/depth state is not yet detailed enough for secondary-back substitution roles.
- M95F: retain workload distribution/high-workload state; its risk subset now matches/beats Vegas. Do not force mean router.
- M95I: retain vacancy/transition/tail evidence; no current mean-point role.
- M95C/ENV1: environment real but weakly deterministic; conditional context only.
- M96C E/P: redundant/non-portable against P3.
- M96C D: tiny conditional clue only, not current point.
- M95D X: rejected isolated increment.

## NEXT — RB-STACK6 / ND3 Secondary-Back Role & Substitution State

STACK5 justifies a new football-data family focused on ordinary/non-risk RB2/RB3 and false-high opportunity overallocation.

Audit/build timestamp-safe pregame features for:
1. prior-game play-level participation/personnel/formation identity;
2. early-down vs third-down vs two-minute role;
3. short-yardage, red-zone, inside-10 and inside-5 rushing roles;
4. single-RB vs two-RB personnel and series/drive rotation where reconstructable;
5. exact active/inactive competitor identity and game-day availability if source integrity qualifies;
6. competitor return/injury state and current usable RB count;
7. coach/team backfield-rotation tendencies from prior games;
8. OL availability only with a demonstrably timestamp-safe historical source.

Primary target remains player share of a constrained team RB opportunity pool, with special evaluation on M95F-nonrisk depth-rank2/3 backs. Do not disturb the M95F-risk/lead-back strength without evidence.

2025 remains exposed development. Freeze any promising architecture for prospective 2026 confirmation. Do not reopen generic M96 router threshold tuning or global efficiency searches.
