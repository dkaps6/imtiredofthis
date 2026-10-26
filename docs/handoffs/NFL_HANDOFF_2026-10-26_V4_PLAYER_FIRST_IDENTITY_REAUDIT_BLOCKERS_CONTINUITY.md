# NFL HANDOFF — 2026-10-26 — V4 — PLAYER-FIRST IDENTITY RE-AUDIT BLOCKERS / ENVIRONMENT CONTINUITY

Repository: `dkaps6/imtiredofthis`

This is the canonical deep handoff for the next ChatGPT session after the 2026-10-26 player-first Identity Card re-audit. It is designed so a cold session can resume without the user re-explaining the project.

## 1. PERMANENT ASSIGNMENT

PRISM is an **INDIVIDUAL-PLAYER PROJECTION MODEL**.

> THE PLAYER SHOULD HAVE THE RULES. THE POSITION SHOULD NOT HAVE THE PLAYER.

> ANOMALIES ARE QUESTIONS, NOT TRENDS.

Position MAE/RMSE is downstream QA only. Do not optimize the project around aggregate positional error. Improve named-player projections and let aggregate metrics reflect that improvement.

Identity/state/opponent work is additive. Do not scrap/rebuild validated PRISM science merely because this next player-first layer is incomplete.

Always distinguish:
- SOURCE READY
- SOURCE BLOCKED
- PREDICTIVE CLOSED
- QUALIFIED
- DESCRIPTIVE ONLY
- PROMOTED/PRODUCTION

## 2. CONTROL ROOM / OWNERSHIP

- Claude = executor / research lead.
- GPT-5.6 = independent auditor / project control.
- Shared control lane = GitHub Issue #535.
- GitHub is canonical.
- Audit newer Claude work before duplicating it.
- Do not recursively read old handoffs.

Before acting in a new chat:
1. read `AGENTS.md`;
2. read ONLY the newest TOP checkpoint in `CURRENT_NFL_RESEARCH_HANDOFF.md`;
3. read this handoff;
4. read newest Issue #535 comments, especially the governing comments listed below;
5. reconcile live `main`, open PRs, exact-head Actions/artifacts before asserting status.

## 3. LIVE STATE AT HANDOFF CREATION

Pre-handoff `main` was:
`dc6add89347cd6c177dfac655b755425e393e909`

The handoff commit is a direct child of that SHA. Re-fetch `main` before acting.

Active research branch:
`research-prism-player-first-identity-environment-2026-10-26`

Research branch head at handoff:
`aa70663cae1b65b32687b2809c57326d66093949`

Important: that research branch is behind current main by the docs-only continuity commits and must be rebased/ported onto current main before any merge. Preserve the already-accepted Walker / JSN-Kupp-Barner / zero-history audit lineage while doing so.

Current exact-head main Actions before this handoff:
- Full Slate run `40392535113` — SUCCESS
- artifact `12685836060`
- digest `sha256:b607ad03cec823b75f080a6fc53852b462e63c41ff3793b688f01122de359da4`
- Full Slate job `121816968596` — SUCCESS
- Repo CI run `40392535296` — FAILURE
- CI job `121816970972` — only the known stale workflow exact-string assertion:
  - `tests/test_full_slate_workflow_contract.py::test_full_slate_main_push_defaults_to_no_credit_mode`
  - 660 passed / 1 skipped / 1 failed
  - **DO NOT weaken the live-odds guard to make this pass.**
- Archive Market Track Record run `40393072600` — SUCCESS.

No paid OddsAPI pull is authorized.

Open PRs remain mostly historical/parallel. In particular:
- PR #691 is stale/superseded continuity; do not merge.
- PR #690 belongs to the source-blocked RB-rushing Stage-1 lineage; do not treat it as production science.
- PR #672 is a large research lineage; do not wholesale merge it to solve this player-first lane.
- Re-fetch all open PRs before acting because this list can change.

## 4. GOVERNING ISSUE #535 COMMENTS

Key lineage:
- `6420504608` — GPT audit blocked original Identity Card V1 on five player-first correctness defects.
- `6420724646` — Walker rushing environment-transmission decomposition.
- `6420875189` — JSN / Kupp / Barner receiving environment-transmission decomposition.
- `6421015183` — slate-wide zero-history position-prior entitlement defect.
- `6421086265` — Stage-2 Week-7 source-readiness fail-closed by chronology.
- `6422827149` — Claude corrected card build/source audit.
- `6422904132` — Claude correctly admitted the card still violated the anomaly gate.
- `6422987723` — Claude corrected their own bridge-recovery prediction.
- `6423234263` — anomaly gate enforced, including correction of bye-week false positives.
- `6425282770` — GPT accepted the Identity Card as a **descriptive substrate only**, no predictive clearance, and set priorities 1-5.
- `6426050333` — Claude completed priorities 1-4 and presented revised fixed target `aa70663c`.
- `6426307861` — GPT re-audit: team-transition share bug + authority materiality decomposition required.
- `6426366490` — latest governing GPT audit at handoff: **two blocking player-first correctness defects plus materiality decomposition; revised target NOT cleared.**

The next session must fetch newer #535 comments in case Claude has responded after `6426366490`.

## 5. IDENTITY CARD V1 — WHAT IS ACCEPTED

The original five audit blockers were materially fixed enough for the Identity Card to be accepted as a **descriptive substrate**, not predictive authority.

Accepted fixed descriptive substrate:
- commit `d93e4ccab6416b1d8ca256d505fa8f3cc6a8594e`
- green run `40371208533`
- 35,196 cards
- 3,053 players
- 52 columns at that fixed point
- `sportsbook_inputs_used = 0`
- scientific boundary: `DESCRIPTIVE_STATE_ONLY_NO_FIT_NO_PROMOTION`

Original A-E fixes that stand:
A. snap counts are consumed as a real participation denominator;
B. active zero-opportunity weeks survive as zeros; DNP/no-participation weeks do not;
C. target-week team comes from weekly roster authority, not historical PBP mode;
D. sparse/no-history roster players receive explicit cards rather than disappearing;
E. PBP/snaps use a stable `gsis_id <-> pfr_id` bridge, not a name join.

Durable-card universe decision:
- keep complete roster universe for identity/provenance;
- model-facing opportunity universe must explicitly use `can_receive_opportunity == True`;
- current opportunity-eligible positions: QB/RB/FB/HB/WR/TE;
- non-opportunity roster rows may not silently receive carry/target entitlement.

Identity bridge:
- player-level bridge is correct architecture;
- 2,520 players mapped;
- no ambiguous PFR IDs;
- partial-bridge players = 0;
- `untrusted_denominator_cards = 0`;
- 381 players never bridge because nflverse weekly roster data lacks `pfr_id` for them in all rows;
- Claude initially predicted the player-level bridge would recover most missing weeks; actual recovery was only 6 player-weeks and Claude corrected the claim;
- relevant coverage: 2,939 / 2,945 = **99.80%** on opportunity-eligible cards with current-season opportunity;
- Cody White remains the notable signal-bearing no-participation-rate exception; rates correctly fail closed.

## 6. REVISED TARGET aa70663c — GOOD WORK THAT STANDS

Claude's revised target:
`aa70663cae1b65b32687b2809c57326d66093949`

Green runs:
- card `40391557775`
- threshold sweep `40391557902`
- 68 tests against CI-pinned pandas 1.5.3 / numpy 1.26.4.

Committed artifacts:
- `data/research/player_identity/player_identity_card_v1_2026.csv`
- `data/research/player_identity/player_identity_card_v1_2026_model_facing.csv`
- `data/research/player_identity/player_identity_card_v1_2026_status.json`
- `data/research/player_identity/player_state_threshold_sweep_v1.csv`
- `data/research/player_identity/player_state_threshold_sweep_v1_churn.csv`
- `data/research/player_identity/player_state_threshold_sweep_v1_examples.csv`
- `data/research/player_identity/player_state_threshold_sweep_v1_status.json`

### Player-state resolver

`load_injuries` is available for 2026 Weeks 1-7 and keyed on `gsis_id`.

Claude added source-qualified labels:
- `RETURN_FROM_INJURY = 213`
- `INJURY_LIMITED = 67`
- `UNRESOLVED_CONTEXT` reduced 1,358 -> 1,078.

The withheld set intentionally does **not** shrink just because a cause is explained. An injury-explained abnormal observation is explained, not automatically representative. This is correct.

`PARTIAL_GAME` remains **SOURCE BLOCKED** for 2026:
- `load_participation` reports season support only through 2025.
- PBP is not a valid substitute because it only observes ball-touchers.
- Do not invent a PARTIAL_GAME label from last-touch timing.

Gate version:
`PLAYER_STATE_ANOMALY_GATE_V2_INJURY_RESOLVED`

### Threshold sensitivity

Authorized descriptive sweep:
`0.30, 0.40, 0.50, 0.60, 0.70`

Results:

| ratio | withheld weeks | unresolved | eligible cards with usable evidence |
|---|---:|---:|---:|
| 0.30 | 907 | 654 | 2,651 |
| 0.40 | 1,099 | 833 | 2,573 |
| 0.50 | 1,358 | 1,078 | 2,517 |
| 0.60 | 1,643 | 1,347 | 2,417 |
| 0.70 | 2,030 | 1,711 | 2,289 |

Adjacent label churn: 192 / 259 / 285 / 387 player-weeks.

Spread:
- 362 of 2,651 opportunity-eligible cards = **13.66%**
- disposition: `MODERATELY_THRESHOLD_SENSITIVE`.

This is **uncertainty**, not a selection exercise.
- no threshold winner;
- `threshold_recommended = null`;
- `no_winner_selected = true`;
- outcomes not inspected.
Do not promote 0.50 merely because it is current.

### Evidence authority

The revised card explicitly surfaces:
- `position_prior_only`
- `prior_only`
- `prior+current`
- `current_only`
- participation evidence / observed-zero distinction
- `opportunity_from_population_fallback`.

Raw card-wide counts at `aa70663c`:
- position_prior_only: 29,515 all / 5,422 opportunity-eligible
- prior_only: 2,714 / 2,593
- prior+current: 2,409 / 2,407
- current_only: 558 / 538

The 5,422 / 10,960 = **49.5%** figure is useful as card-level authority coverage across Weeks 1-7.

**DO NOT call 49.5% the current live-production severity rate.**
It mixes:
- Weeks 1-7;
- both carry and target cards;
- position-kind combinations with zero/tiny priors;
- structurally sparse cross-kind rows.

The materiality must be decomposed before using it as production severity.

## 7. CURRENT BLOCKERS — THIS IS THE EXACT STOPPING POINT

Governing disposition:

`PLAYER_IDENTITY_CARD_V1_REVISED_TARGET_BLOCKED_ON_NO_BASELINE_CONTAMINATION_AND_TRADE_SHARE_SEMANTICS`

The previously accepted descriptive substrate remains accepted. The revised `aa70663c` target is **not** cleared as the next canonical predictive/scoring authority.

### BLOCKER A — NO_BASELINE can bypass known injury contamination

Current ordering allows:
1. insufficient baseline -> `NO_BASELINE`;
2. injury resolver revisits only `UNRESOLVED_CONTEXT`;
3. `HEALTHY_FULL` and `NO_BASELINE` become `USABLE`.

Therefore a sparse-history player can have a **known same-week limiting injury designation** in his first/second participation week, remain `NO_BASELINE`, and enter trajectory/efficiency as usable.

Required correction:
- insufficient baseline may prevent snap-collapse inference;
- it may NOT override independent source-qualified contamination;
- same-week limiting injury must force WITHHELD even with baseline < minimum;
- source-qualified return-from-injury must force WITHHELD even with baseline < minimum;
- `NO_BASELINE` means insufficient baseline **and no independent contamination signal**.

Required tests:
1. first/second participation week + limiting injury -> WITHHELD;
2. return from injury-explained absence with insufficient baseline -> WITHHELD;
3. sparse player without injury/absence contamination -> remains NO_BASELINE.

### BLOCKER B — current-team share is not trade/team-transition safe

At `aa70663c`:
- `team_now` comes from target-week roster;
- current-season player opportunity history can include old-team weeks;
- numerator filters by accepted week but not necessarily target-team tenure;
- denominator uses target/current team.

That can mix an old-team numerator with a new-team denominator.

Required semantic:
**target-team-tenure share**.

For a target-week current-team share:
- build strictly-prior `(season, week, gsis_id) -> team` from weekly roster authority;
- only use player-weeks where historical roster team equals `team_now`;
- numerator = player's target-team opportunities in those same accepted weeks;
- denominator = target team's opportunities in those exact same weeks;
- if no prior accepted week with target team exists -> NaN / explicit no-current-team-evidence;
- same rule for red-zone and inside-5 shares;
- same semantic for any `*_all_weeks` current-team comparison field.
- old-team historical role may be retained separately as identity history, but it must not be called `team_share_current`.

Required test:
- synthetic in-season trade with large old-team workload and sparse/new target-team tenure;
- prove no old-team opportunity enters the target-team share.

### REQUIRED MATERIALITY DECOMPOSITION — outcome-blind

Before interpreting the 49.5% fallback number as architecture severity, report:
1. evidence authority by target_week;
2. by opportunity_kind;
3. by roster_position × opportunity_kind;
4. Week-7 specifically;
5. position_prior_only split into:
   - no_history
   - participation_only_zero_opportunity;
6. identify which position-kind rows actually enter current production allocation versus merely exist descriptively;
7. include production prior value and whether that fallback can move finite allocation;
8. preferably quantify implied fallback opportunity mass entering the finite room, analogous to the prior Week-7 zero-history audit.

No outcomes. No fitting. No subgroup rescue.

## 8. WEEK-7 ZERO-HISTORY POSITION-PRIOR ENTITLEMENT DEFECT — ALREADY CONFIRMED

Durable audit:
- `docs/research/PRISM_WEEK7_ZERO_HISTORY_POSITION_PRIOR_ALLOCATION_AUDIT_V1.md`
- commit `aef06922ae3039c73080749226a2074ac0c3dc69`
- #535 `6421015183`.

Paid W7 football universe:
- 396 skill-player rows
- 28 teams
- 34 zero-history players (prior_games=0 AND current_games=0), across 21/28 teams
- 69 players with current_games=0, across 25/28 teams.

Position-prior values observed:
- RB/FB rush share 0.359568
- RB/FB target 0.079174
- QB rush 0.111763
- WR target 0.165842
- TE target 0.129391.

Rushing materiality:
- 13 zero-history rows entered a team top-five rushing allocator
- 11/28 teams
- combined final rush allocation probability 1.824446
- analytic expected carries 45.829
- 6.87% of modeled top-five carry expectation.

Zero-current-season:
- 30 top-five rush rows
- 21/28 teams
- analytic expected carries 101.156
- 15.17% of modeled carry expectation.

Target-room materiality:
- finite modeled player target mass = 26.6 (=28*0.95)
- zero-history final target entitlement 1.264686 = 4.75%
- zero-current final target entitlement 2.134477 = 8.02%.

Do NOT "fix" this by deleting priors or zeroing all sparse players. Position priors are valid priors. The architecture defect is population fallback becoming named-player finite entitlement without enough target-week player participation/role authority.

## 9. WALKER-FIRST OPPONENT / ENVIRONMENT TRANSMISSION — STILL OPEN

Durable audit:
- `docs/research/PRISM_WEEK7_WALKER_ENVIRONMENT_TRANSMISSION_DECOMPOSITION_V1.md`
- commit `cca416bd2cdad8f142a5a063923149344c7e5c5d`
- #535 `6420724646`.

Kenneth Walker III exact W7 path:
- Bayes rush share 0.5249473141
- Bayes YPC 4.9596105995
- rules plays 56.9321209523
- generic pass rate 0.57
- rules rush share 0.5249473141
- rules YPC 4.9596105995
- rush_eff_mult 1.0
- volatility 1.0
- final rush-yards mean 66.045893
- SD 28.097688
- main line 84.5
- canonical direction UNDER.

SEA pregame contained meaningful rush-defense context:
- def_rush_epa -0.202932
- explosive allowed 0.073034
- light box .497
- heavy box .164
- success_rate_def .367978
- RB YPT allowed 5.1
- YBC 0.7
- stuff 24.5% / 19.1%.

Generic RB rushing consumed none of those except coarse box thresholds, and SEA crossed neither.

Exact conclusion for Walker rushing:
- opponent-caused rush opportunity movement = 0
- efficiency movement = 0
- final mean movement = 0
- sigma movement = 0
versus a neutral opponent that also trips no box threshold.

Walker receiving was different:
- SEA zone .892 triggers RB target-share x1.20;
- receiving opportunity therefore moved while rushing did not.

This is an architecture finding, not a fitted claim that SEA "should have" moved Walker by some coefficient.

## 10. WR/TE ENVIRONMENT TRANSMISSION — STILL OPEN

Durable audit:
- `docs/research/PRISM_WEEK7_SEA_RECEIVERS_VS_KC_ENVIRONMENT_TRANSMISSION_V1.md`
- commit `87dcfd99e33f9a03bc3f5086efbbaacf7bee9204`
- #535 `6420875189`.

JSN:
- Bayes target = rules target 0.316891
- Bayes YPT = rules YPT 10.174822
- direct KC target movement = 0
- direct KC efficiency movement = 0.

Cooper Kupp:
- upstream alignment SWR
- depth role WR3
- generic rules role blank
- Bayes target = rules target 0.151325
- YPT 7.940591 unchanged
- direct KC movement = 0.
Do NOT revive the old SLOT carry candidate; it prospectively failed Week-3 targeted SWR metrics.

AJ Barner:
- Bayes target 0.136722
- KC zone .789 => x1.15
- KC middle-open 50.7% => x1.10
- rules target 0.172954
- direct +26.5% target movement
- YPT unchanged.

Finite target-room coupling:
- SEA raw Bayes target sum 1.468634
- post-rule sum 1.601149
- as-run scale .593324
- neutral scale .646860
- JSN/Kupp indirectly compressed ~8.28%
- Barner final entitlement lifted ~16%.

Conclusion:
- receiving is not fully opponent-blind;
- transmission is coarse threshold opportunity + finite-room coupling;
- rich opponent efficiency context is still mostly stranded;
- pass split remains hardcoded 0.57;
- alignment can be lost;
- specialist WR-R15/TE-R5P are player/room specialists, not opponent-specific coefficients.

Opponent/environment transmission remains independently OPEN regardless of Identity Card status.

## 11. STAGE-2 READINESS — FAIL CLOSED BY CHRONOLOGY

Existing frozen readiness workflow:
`.github/workflows/research-prism-week7-stage2-translation-readiness-v1.yml`

Attempt rerun:
- run `39093471632`
- attempt 2
- exact head `b89d1f5942ea31c7f067e7044e1feb5d6c7dc149`
- job `121693160661` SUCCESS
- artifact `12666876129`
- digest `sha256:3e9295f8271112261eb64e47a9b8bfe7b345e11df6a3037a7f4e2865569fd32f`.

Disposition:
`PRISM_FUTURE_STAGE2_TRANSLATION_NOT_READY__FAIL_CLOSED`

Blocker:
`RuntimeError: target/future stat rows already present; refuse audit: weeks=[7]`

This is a correct chronology guard.
Do NOT:
- weaken/filter the guard;
- call Week 7 prospectively source-ready;
- reconstruct a Week-7 Stage-2 shadow postgame;
- use Week-7 outcomes to rescue it.

Next Stage-2 readiness target must be genuinely future and frozen before target-week data exists.

Historical Stage-2 qualification remains unchanged. Stage-2 is not production-wired.

## 12. WEEK-7 BETTING CONSTRUCTION POSTMORTEM — KEEP SEPARATE FROM SCIENCE

Paid Week-7 board authority:
- run `39849371821`
- artifact `12456298985`
- digest `sha256:d1a1fb48e5bb920c5291f35b7457b5870801b689751deb2d742f98c555603a85`
- exact artifact head `4dc178b59da8708dbea978b8e54e753b1e2f75c5`.

Ticket evidence is durable on main from PR #695:
`docs/research/evidence/2026-10-25_week7_ticket_ledger/`

Contains Sunday/SNF regular tickets, RB-integrated tickets, ATD tickets, 68 prices, mappings/hashes, and one-row-per-leg evidence.

User result:
- only ONE Sunday ticket hit;
- no SNF regular ticket hit.

Accepted construction diagnosis:
- "safe" parlays were only ~41-47% naive model-implied hit probability;
- 25/57 unique yardage legs were bought-down OVERs while canonical main-line model direction was UNDER;
- model SD / exact P(clear) were not used correctly;
- heavy repeated-player concentration;
- ATD was uncertified but consumed risk budget.

Boundary:
**Week-7 parlay failure is primarily a construction/threshold/exposure problem. It does NOT close the separate opponent/environment-transmission architecture problem.**

Until construction V2 is evidence-backed:
- alt direction conflicting with canonical main-line direction => default PASS;
- raw cushion != safety;
- show model mean, SD/distribution authority, main line/direction, exact alt threshold, P(clear), exact price, exposure;
- do not call ~41% a "Safe" parlay;
- surface same-game and repeated-player concentration;
- no universal anchor on every tier;
- ATD remains uncertified/fun and separate.

Betting is CLOSED during this research handoff.

## 13. VERIFIED MODEL ARCHITECTURE FACTS

- `scripts/modeling/ml_v2.py`: ML features are lagged player history + position indicators; no opponent/team-defense authority.
- `scripts/modeling/state_v2.py`: player outcome-state transitions + position; no opponent authority.
- `scripts/modeling/bayesian_v2.py`: population prior + prior/current player evidence; no opponent; FB maps to RB group; zero player evidence becomes `position_prior_only`.
- `scripts/modeling/rules_v2.py` / `simulation_rules.py`: generic game-script pass share hardcoded 0.57; lead/trail diagnostics do not alter the generic split; matchup thresholds coarse.
- `scripts/simulation_v2.py`: team rushing picks top five rush shares and normalizes/caps modeled sum to .95.
- `scripts/modeling/target_entitlement_v1.py`: target shares feed finite team cap .95, residual .05; TE-R5P and WR-R15 conserve their room pools afterward.
- `scripts/run_pricing_with_full_roster_universe_v3_core.py`: finite entitlement -> TE-R5P -> WR-R15 -> joint simulation -> QB C2.
- V4/V5 wrappers apply Week-1-only RB refinements; outside Week 1 they no-op to prior result.
- Stage-2 conditional allocation is research-qualified but NOT production-wired.
- Current availability reranking maps FB into RB namespace; this was part of the zero-history entitlement problem.

## 14. EXISTING SCIENCE BOUNDARIES — DO NOT REOPEN CASUALLY

- QB M89/M90 + C2 active.
- WR-R15 active.
- TE-R5P active.
- RB P3/R22 Week-1-only; not a Week-7 authority.
- Stage-2 qualified research but not production-wired.
- Direct WR/CB source unavailable/gated.
- ATD model uncertified.
- RB rushing Stage-1 source reset SOURCE BLOCKED.
- RB/TE team-context reset candidates PREDICTIVE CLOSED.
- Do not refit anything to Week 7.
- Week 7 is forensic/case-finding only.
- No sportsbook variable upstream of the independent football projection.
- No new paid OddsAPI pull without explicit user authorization.

## 15. EXACT NEXT ACTIONS

If Claude has not posted newer work:
1. fix BLOCKER A: NO_BASELINE cannot override source-qualified injury/return contamination;
2. fix BLOCKER B: make target-team share strictly target-team-tenure / trade-safe;
3. add the required tests for both;
4. produce the outcome-blind evidence-authority/materiality decomposition:
   - target_week;
   - opportunity_kind;
   - roster_position × opportunity_kind;
   - explicit Week-7 slice;
   - no_history vs participation_only_zero_opportunity;
   - production prior value / whether the fallback can move allocation;
   - implied fallback finite-room opportunity mass where possible;
5. rebase/port the research lane onto current main without losing Walker, WR/TE, zero-history and Identity Card evidence;
6. post a new fixed head + exact green runs in #535;
7. GPT independently audits the exact diff/artifacts/code path before any predictive scoring.

If those clear:
8. freeze the FIRST predictive use of Identity Card state prospectively before scoring; no post-hoc Week-7 rescue;
9. continue player-first opportunity-authority tracing and opponent/environment transmission as separate architecture work;
10. betting-layer Construction V2 remains separate and evidence-backed;
11. eventually port stranded orphan-branch graders/tooling file-by-file and grade Week 7, but do not let that interrupt current player-state correctness work.

## 16. USER EXPECTATION

The user explicitly does not want to teach this project again every time a chat fills. The next assistant should behave as the same control-room auditor continuing the same project, not as a fresh consultant.

Do not respond to a simple "continue" with a giant recap. Reconcile GitHub, inspect the newest Claude work, and continue from the exact unresolved blocker.

No paid OddsAPI. No Week-7 fit. No aggregate-MAE optimization. No post-hoc subgroup rescue. No reopening closed science without new causal evidence.
