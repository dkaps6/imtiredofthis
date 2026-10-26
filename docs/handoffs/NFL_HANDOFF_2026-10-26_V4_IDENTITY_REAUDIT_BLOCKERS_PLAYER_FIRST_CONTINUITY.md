# NFL PRISM DEEP HANDOFF — 2026-10-26 — V4

## PURPOSE

This is the canonical deep continuity handoff for the next ChatGPT session.

Repository: dkaps6/imtiredofthis

The goal is seamless continuation. The next assistant must not ask the user to reconstruct the project, must not restart closed science, and must not confuse descriptive player-state work with predictive clearance.

GitHub is canonical.

## REQUIRED READ ORDER FOR THE NEXT CHAT

1. AGENTS.md
2. ONLY the single active TOP checkpoint in CURRENT_NFL_RESEARCH_HANDOFF.md
3. This file
4. Newest Issue #535 comments, especially:
   - 6425282770
   - 6426050333
   - 6426307861
   - 6426366490
5. Reconcile live main, the active research branch, open PRs, and exact-head Actions before asserting status.

Do not recursively read old handoffs.

## PERMANENT SCIENTIFIC ASSIGNMENT

PRISM is an INDIVIDUAL-PLAYER PROJECTION MODEL.

The governing principles remain:

> THE PLAYER SHOULD HAVE THE RULES. THE POSITION SHOULD NOT HAVE THE PLAYER.

> ANOMALIES ARE QUESTIONS, NOT TRENDS.

The required reasoning order is:
1. player identity;
2. current role/state;
3. opportunity allocation;
4. opponent/environment;
5. market-specific outcome;
6. reliability.

Position-level MAE/RMSE is downstream QA. It is not the objective. Do not optimize the project around lowering aggregate positional error while losing named-player correctness.

Identity/state/opponent work is additive to the validated stack. Do not scrap PRISM or reopen frozen science without new causal evidence.

## CONTROL ROOM

Claude = executor / research lead.
GPT-5.6 = independent auditor / project control.
Shared asynchronous control lane = GitHub Issue #535.

The assistant should audit newer Claude work before duplicating it. Claude should consume durable GPT audits rather than rediscovering them.

## HARD BOUNDARIES

- No Week-7 outcome fitting.
- No post-hoc subgroup rescue.
- No new paid OddsAPI pull without explicit user authorization.
- Betting is currently CLOSED while research continuity is being repaired.
- Sportsbook information stays downstream of football projection unless explicitly labeled market-assisted.
- Stage-2 Week-7 prospective readiness is permanently fail-closed by chronology.
- Do not weaken the live-odds authorization guard merely to make Repo CI green.
- Do not wholesale merge stale/orphan research branches.
- Do not promote the Identity Card into predictive/scoring use until the current V4 blockers are cleared and independently re-audited.

## LIVE GITHUB STATE AT HANDOFF

Pre-handoff main:
a6f296234ecf9db50d30b82e43587e2c5e9b8482

This V4 continuity commit is docs-only and advances main from that parent. The next chat must re-fetch the live SHA.

Active research branch:
research-prism-player-first-identity-environment-2026-10-26

Research head at handoff:
aa70663cae1b65b32687b2809c57326d66093949

Pre-handoff comparison against main:
- status: diverged
- ahead by 30 commits
- behind by 2 commits
- merge base: c4de3d4044e828e39d80aa7db19af0de2caf652a

Because this V4 handoff adds a docs-only main commit, the research branch will be one additional docs commit behind unless Claude rebases/ports it.

The latest Issue #535 comment observed before this handoff is 6426366490.

There is NO Claude response after 6426366490 as of this handoff.

## EXACT-HEAD ACTIONS ON PRE-HANDOFF MAIN

For main a6f296234ecf9db50d30b82e43587e2c5e9b8482:

- Full Slate run 40392535113 — SUCCESS
- Archive Market Track Record run 40393072600 — SUCCESS
- Repo CI run 40392535296 — FAILURE
  - compile and static repo audit succeeded;
  - unit-test step failed;
  - continuity still treats this as the known stale no-credit exact-string FETCH_LIVE_ODDS assertion unless a later exact-head inspection proves otherwise;
  - do NOT weaken the live-odds authorization guard to make CI pass.

## OPEN PR CONTINUITY

Relevant open PRs observed at handoff:

- #691 — Docs: collapse continuity to one active checkpoint, archive the other 71. Stale/superseded by current continuity. Do not merge opportunistically.
- #690 — RB-rushing Stage-1 source-cert ordering repair. Research remains source-blocked; do not treat this PR as predictive clearance.
- #688 — RB NGS efficiency source audit rerun 2, draft.
- #677 — Week-5 priced-board continuity, older ops lineage.
- #672 — Individual opportunity roadmap / historical trajectory, large draft lineage. Do not wholesale merge.
- #669 — GSIS RB successor-lineup research.
- #668 — market-offer residual probability research.
- #667 — market-relative bet selector research.
- #665 — free historical WR-CB archive research.
- #663 — market snapshot history research.
- #662 — GSIS point-in-time archive contract.
- Numerous older research PRs remain open. Treat them as lineage, not implicit permission to merge.

## CANONICAL WEEK-7 PAID BOARD / FORENSIC EVIDENCE

One paid Week-7 board is canonical:

- Full Slate run 39849371821 — SUCCESS
- artifact 12456298985
- digest sha256:d1a1fb48e5bb920c5291f35b7457b5870801b689751deb2d742f98c555603a85
- artifact exact head 4dc178b59da8708dbea978b8e54e753b1e2f75c5

Week-7 ticket evidence is durable on main from PR #695 under:
data/research/evidence/2026-10-25_week7_ticket_ledger/

It contains Sunday/SNF regular tickets, RB-integrated tickets, ATD tickets, 68 exact prices, mappings/hashes, and one-row-per-leg evidence.

The user reported only ONE Sunday ticket hit and none of the SNF regular tickets.

Accepted independent construction diagnosis:
- so-called safe parlays were only about 41–47% naive model-implied hit probability;
- 25 of 57 unique yardage legs were bought-down OVERs while the canonical main-line model direction was UNDER;
- model SD / exact P(clear) was not properly used;
- repeated-player and same-game concentration was too high;
- ATD remained uncertified but consumed ticket share.

Important causal separation:
- Week-7 betting failure was primarily a CONSTRUCTION / THRESHOLD / EXPOSURE problem.
- Opponent/environment transmission is a separate OPEN architecture problem.
- Do not blame all losing tickets on the projection model.
- Do not use the construction failure to close the environment problem.

Betting corrections that remain in force until an evidence-backed Construction V2 exists:
- alt direction conflicting with canonical main-line direction => default PASS;
- cushion is not safety;
- show model mean, model SD/distribution authority, main line/direction, exact alt threshold, P(clear), exact price, and exposure;
- do not label roughly 41% probability Safe;
- surface repeated-player and same-game concentration;
- no universal anchor across all ticket tiers;
- ATD stays uncertified/fun and separate;
- do not invent exposure caps from Week 7 outcomes.

## STAGE-2 READINESS — PERMANENT WEEK-7 CHRONOLOGY RESULT

Frozen readiness workflow:
.github/workflows/research-prism-week7-stage2-translation-readiness-v1.yml

Existing authority:
- original first/reprobe run 39060798892
- run 39093471632
- attempt 2 job 121693160661 — SUCCESS
- exact head b89d1f5942ea31c7f067e7044e1feb5d6c7dc149
- artifact 12666876129
- digest sha256:3e9295f8271112261eb64e47a9b8bfe7b345e11df6a3037a7f4e2865569fd32f

Disposition:
PRISM_FUTURE_STAGE2_TRANSLATION_NOT_READY__FAIL_CLOSED

Current blocker:
RuntimeError: target/future stat rows already present; refuse audit: weeks=[7]

This is NOT the old DEN/SEA source-maturation blocker. Week-7 target rows now exist, so chronology correctly refuses a retrospective reconstruction.

Do not:
- weaken/filter around the guard;
- call Week 7 prospectively Stage-2 ready;
- reconstruct a Week-7 Stage-2 shadow postgame;
- reuse Week 7 as a retrospective substitute.

Historical Stage-2 qualification remains unchanged.
The existing Week-6 prospective shadow remains unchanged.
The next readiness target must be genuinely future and frozen before target-week stat/snap rows can exist.

## WALKER-FIRST OPPONENT / ENVIRONMENT TRANSMISSION AUDIT

Durable document:
docs/research/PRISM_WEEK7_WALKER_ENVIRONMENT_TRANSMISSION_DECOMPOSITION_V1.md

Commit:
cca416bd2cdad8f142a5a063923149344c7e5c5d

Issue #535 audit:
6420724646

Player:
Kenneth Walker III, KC RB vs SEA in the Week-7 paid artifact.

Strictly pregame PlayerForm:
- prior games 17
- current games 5
- prior rush share 0.435897
- current rush share 0.731034
- blended PF rush share 0.599862
- prior YPC 4.647059
- current YPC 6.273585
- blended PF YPC 5.550684
- prior target share 0.079121
- current target share 0.149660
- blended PF target share 0.118309

Empirical-Bayes live:
- rush share 0.5249473141
- YPC 4.9596105995
- target share 0.1043247769
- YPT about 6.243173

SEA pregame context in the artifact included:
- def_rush_epa -0.202932
- explosive_play_rate_allowed 0.073034
- light box 0.497
- heavy box 0.164
- success_rate_def 0.367978
- zone 0.892
- man 0.056
- pressure_generated 0.209069
- def_pass_epa_allowed -0.205481
- def_pass_success_allowed 0.375842
- def_ypa_allowed 4.753773
- RB YPT allowed 5.1
- WR YPT allowed 6.2
- TE YPT allowed 4.4
- outside YPT 7.7
- slot YPT 4.2
- yards-before-contact per RB rush 0.7 / 0.7
- rush stuff 24.5% / 19.1%

Live generic rush path:
- rules plays 56.9321209523
- pass rate hardcoded 0.57
- rush share 0.5249473141
- YPC 4.9596105995
- rush efficiency multiplier 1.0
- volatility 1.0

Week-7 rush-yards output:
- MC 40.568542
- ML 98.073554
- State 58.685803, but frozen State ensemble weight = 0
- MC weight 0.5569542426070742
- ML weight 0.4430457573929258
- final mean 66.045893
- model SD 28.097688
- main line 84.5
- canonical direction UNDER

Critical result:
For a neutral generic opponent that activates no light/heavy-box threshold, the current production architecture produces the same Walker rushing path.

Opponent-caused movement in this exact Walker rushing case:
- rush opportunity = 0
- rush efficiency = 0
- final mean = 0
- sigma = 0

Receiving asymmetry:
SEA zone 0.892 triggered the generic RB target x1.20 rule.
- Walker Bayes target share 0.1043247769
- rules target share 0.1251897322
- actual final entitlement 0.0753093458
- neutral no-zone entitlement about 0.0635980562
- about +18.41% final entitlement
- generic pass mean 32.451309
- expected targets actual about 2.443887 vs neutral about 2.063840

Architecture conclusion:
environment transmission exists unevenly. Rich rush-defense context is largely stranded in the generic Walker rushing path, while a coarse coverage threshold changes receiving opportunity.

No predictive coefficient was authorized from this audit.

## REPRESENTATIVE WR / TE TRANSMISSION AUDIT

Durable document:
docs/research/PRISM_WEEK7_SEA_RECEIVERS_VS_KC_ENVIRONMENT_TRANSMISSION_V1.md

Commit:
87dcfd99e33f9a03bc3f5086efbbaacf7bee9204

Issue #535:
6420875189

Players:
JSN, Cooper Kupp, AJ Barner versus KC.

Direct generic-path findings:
- JSN Bayes target share = rules target share = 0.316891; YPT unchanged at 10.174822; direct KC opportunity movement 0; direct efficiency movement 0.
- Cooper Kupp Bayes target share = rules target share = 0.151325; YPT unchanged at 7.940591; direct KC opportunity movement 0; direct efficiency movement 0.
- AJ Barner Bayes target share 0.136722; KC zone and middle-open gates combine to x1.265; rules target share 0.172954; direct target-share movement +26.5%; YPT unchanged at 6.902557.

Finite target-room coupling means KC can indirectly compress unchanged WRs:
- SEA Bayes raw target-share sum 1.468634
- post-rule sum 1.601149
- as-run scale 0.593324
- neutral scale 0.646860
- JSN final about -8.28% versus neutral
- Kupp final about -8.28%
- Barner final about +16.0% versus neutral

Approx analytic mean materiality from the prior audit:
- JSN about -4.95 receiving yards
- Kupp about -1.60
- Barner about +2.87

Boundary:
These are deterministic/analytic architecture decompositions, not new 25k-draw prospective promotions.

Do not revive the old SLOT candidate. It failed the frozen Week-3 targeted SWR prospective gate.

## ZERO-HISTORY POSITION-PRIOR ENTITLEMENT AUDIT

Durable document:
docs/research/PRISM_WEEK7_ZERO_HISTORY_POSITION_PRIOR_ALLOCATION_AUDIT_V1.md

Commit:
aef06922ae3039c73080749226a2074ac0c3dc69

Issue #535:
6421015183

Full paid Week-7 football universe:
- 396 skill-player rows
- 28 playing teams
- 34 players with prior_games=0 and current_games=0
- 21/28 teams contain at least one such player
- 69 players have current_games=0

Zero-history position-prior values observed:
- RB/FB rush share 0.359568
- QB rush share 0.111763
- WR target share 0.165842
- TE target share 0.129391

Rushing materiality:
- 13 zero-history rows entered top-five rushing allocators
- across 11/28 teams
- combined final rush allocation probability 1.824446
- analytic expected carries 45.829
- 6.87% of modeled top-five carry expectation

Zero-current players:
- 30 top-five rush rows
- 21/28 teams
- analytic expected carries 101.156
- 15.17% of modeled carries

Target-room materiality:
- finite target mass across 28 teams = 26.6
- zero-history final target entitlement 1.264686
- 4.75% of modeled player target mass
- zero-current final target entitlement 2.134477
- 8.02% of modeled target mass

Representative issue:
current availability reranking maps FB into the RB namespace, so a zero-history FB can inherit an RB-family position prior and compete materially inside the rushing room.

Do NOT fix this by setting all zero-history players to zero or deleting position priors. Position priors are valid priors. The defect is allowing a population fallback to become named-player finite entitlement without enough target-week participation/role authority.

## PLAYER IDENTITY CARD — EVOLUTION AND CURRENT STATUS

### Original V1 blockers from GPT audit 6420504608

A. snap counts were loaded but not consumed;
B. active zero-opportunity weeks disappeared from denominators;
C. current team was inferred from historical PBP mode;
D. universe contained only opportunity-observed players;
E. snap/PBP identity lacked a stable bridge.

Those five blockers were materially closed for the descriptive substrate.

### Descriptive substrate acceptance

GPT comment 6425282770:
PLAYER_IDENTITY_CARD_V1_DESCRIPTIVE_SUBSTRATE_ACCEPTED__NO_PREDICTIVE_CLEARANCE

Important meaning:
- the card is useful descriptive infrastructure;
- it is NOT cleared as predictive/scoring authority.

The accepted architecture decision:
- durable card keeps the complete roster universe;
- model-facing opportunity research uses an explicit can_receive_opportunity == True subset;
- downstream outputs must say which universe they use;
- non-eligible roster rows may not silently receive carry/target entitlement.

### Identity bridge

The bridge was correctly moved to player-level pfr_id <-> gsis_id with ambiguity fail-close.

Claude initially predicted this would recover most unmatched offensive player-weeks; that prediction was wrong.

Measured result:
- bridge match rate 0.8135 is not the important statistic;
- offensive player-weeks unmatched 4,900;
- player-level change recovered only six player-weeks;
- never-bridged players 381;
- partial-bridge players 0;
- ambiguous pfr ids 0.

The relevant signal-bearing coverage:
2,939 / 2,945 = 99.80% of opportunity-eligible cards with current-season opportunity have a participation rate.

Cody White is the notable opportunity-bearing unresolved denominator case and correctly fails closed.

### Anomaly gate progression

The first corrected card still violated the PLAYER-STATE ANOMALY INVESTIGATION GATE. Claude caught that and fixed it.

A subsequent gate defect falsely treated team bye gaps as player absences:
- unresolved initially 1,743
- after bye correction 1,358
- 385 player-weeks were false positives
- 22% of the initial unresolved flags were schedule artifacts

The participation-only gate correctly withholds unresolved abnormal weeks from trajectory/stability/efficiency while preserving healthy zero-opportunity weeks as real zeros.

### Injury resolver on revised target aa70663c

Claude then added source-qualified injury context.

Reported revised state:
- UNRESOLVED_CONTEXT 1,358 -> 1,078
- RETURN_FROM_INJURY 213
- INJURY_LIMITED 67
- 280 of 1,358 previously unresolved weeks now carry a cause
- withheld set intentionally remains unchanged: explained abnormal weeks remain withheld rather than becoming representative
- PARTIAL_GAME remains SOURCE BLOCKED for 2026 because nflverse load_participation supports only through 2025
- PBP is not an acceptable substitute for partial-game authority

This direction is accepted as conservative descriptive state resolution.

### Threshold sensitivity sweep

Authorized descriptive sweep:
0.30, 0.40, 0.50, 0.60, 0.70

No outcomes.
No threshold winner.
No tuning.

Reported:
- 0.30: 907 withheld, 654 unresolved, 2,651 eligible cards with usable evidence
- 0.40: 1,099 withheld, 833 unresolved, 2,573 usable
- 0.50: 1,358 withheld, 1,078 unresolved, 2,517 usable
- 0.60: 1,643 withheld, 1,347 unresolved, 2,417 usable
- 0.70: 2,030 withheld, 1,711 unresolved, 2,289 usable

Adjacent churn:
192 / 259 / 285 / 387

Spread:
362 cards = 13.66%

Disposition:
MODERATELY_THRESHOLD_SENSITIVE

This uncertainty is intentionally unresolved. Do not select 0.50 merely because it is current.

Named high-churn examples reported:
Tim Patrick, Nick Chubb, DJ Moore, Adam Thielen, Tyler Higbee, Jonnu Smith.

### Evidence authority / model-facing universe on revised target aa70663c

Claude added evidence_authority and opportunity_from_population_fallback.

Reported counts:
- position_prior_only: 29,515 all cards; 5,422 opportunity-eligible
- prior_only: 2,714 all; 2,593 eligible
- prior+current: 2,409 all; 2,407 eligible
- current_only: 558 all; 538 eligible

Complete durable roster artifact:
35,196 rows.

Model-facing opportunity artifact:
10,960 rows.

Important:
The raw 5,422 / 10,960 = 49.5% figure is NOT yet accepted as the live production severity rate. It pools target weeks and includes position-kind combinations whose position prior may be zero or immaterial.

## REVISED FIXED TARGET PRESENTED BY CLAUDE

Branch:
research-prism-player-first-identity-environment-2026-10-26

Head:
aa70663cae1b65b32687b2809c57326d66093949

Green runs Claude reported:
- card 40391557775
- threshold sweep 40391557902
- 68 tests under pandas 1.5.3 / numpy 1.26.4

Committed artifacts include:
- data/research/player_identity/player_identity_card_v1_2026.csv
- data/research/player_identity/player_identity_card_v1_2026_model_facing.csv
- data/research/player_identity/player_identity_card_v1_2026_status.json
- data/research/player_identity/player_state_threshold_sweep_v1.csv
- data/research/player_identity/player_state_threshold_sweep_v1_churn.csv
- data/research/player_identity/player_state_threshold_sweep_v1_examples.csv
- data/research/player_identity/player_state_threshold_sweep_v1_status.json

Claude's summary is Issue #535 comment 6426050333.

## CURRENT GOVERNING RE-AUDIT — aa70663c IS NOT CLEARED

Two GPT re-audit comments govern the current state:
- 6426307861
- 6426366490

The latter is the more complete blocker statement.

Current disposition:
PLAYER_IDENTITY_CARD_V1_REVISED_TARGET_BLOCKED_ON_NO_BASELINE_CONTAMINATION_AND_TRADE_SHARE_SEMANTICS

The previously accepted descriptive substrate remains accepted.
This blocks the revised target from becoming the next canonical descriptive fixed point and absolutely blocks predictive/scoring promotion.

### BLOCKER A — NO_BASELINE can bypass known injury contamination

Current logic order on aa70663c:
- insufficient baseline -> NO_BASELINE;
- otherwise abnormal/return -> UNRESOLVED_CONTEXT;
- injury resolver revisits only UNRESOLVED_CONTEXT;
- HEALTHY_FULL or NO_BASELINE -> USABLE.

Therefore an early-career / sparse-history player can have a same-week known limiting injury designation in one of his first two participation weeks, remain NO_BASELINE, and enter trajectory/efficiency as usable.

Required correction:
- baseline insufficiency can prevent a snap-collapse judgment;
- it cannot override an independent source-qualified contamination signal;
- limiting injury or source-qualified return-from-injury must be able to force WITHHELD even when baseline_weeks < MIN_BASELINE_WEEKS;
- NO_BASELINE must mean insufficient baseline AND no independent contamination signal.

Required regression tests:
1. first/second participation week + limiting injury context => withheld;
2. return from an injury-explained absence with insufficient baseline => withheld;
3. sparse player with no injury/absence contamination => remains NO_BASELINE, not invented as unhealthy.

### BLOCKER B — target-team share is not trade/team-transition safe

Current revised logic improved week-matching but still uses target-week team_now against a player current-season opportunity frame containing all historical teams.

For an in-season trade:
- old-team opportunities can enter the numerator;
- denominator can be the target-week/new team's opportunity total for those calendar weeks.

That is football-semantically wrong even if the ratio is numerically reconstructible and <= 1.

Required preferred semantic:
TARGET-TEAM-TENURE SHARE.

For a target-week current-team share:
- build strictly-prior player-week team authority from weekly rosters;
- include only player-weeks where the player's roster team equals target-week team_now;
- numerator uses only the player's opportunities for team_now in those accepted weeks;
- denominator uses team_now opportunities in exactly those same weeks;
- newly acquired player with no accepted prior week for team_now => NaN / explicit no-current-team-evidence;
- apply the same semantic to red-zone and inside-5 shares;
- apply it to the all-weeks current-team comparison fields too;
- add a synthetic in-season-trade regression test proving old-team workload cannot leak into new-team share.

A multi-team historical identity feature is allowed separately, but it must use the player's actual team each week and must not be called team_share_current.

### MATERIALITY / REPORTING CORRECTION — 49.5% is not yet production severity

The 49.5% is useful card-level authority coverage across Weeks 1-7, but it is not yet an accepted current-production fallback burden.

Required outcome-blind decomposition:
1. evidence-authority counts/fractions by target_week;
2. by opportunity_kind;
3. by roster_position x opportunity_kind;
4. Week-7 slice explicitly;
5. split position_prior_only into no_history versus participation_only_zero_opportunity;
6. identify which position-kind rows actually enter the current production allocation path versus only existing descriptively;
7. include the production prior value / whether the fallback can actually move finite allocation;
8. preferably calculate implied fallback opportunity mass entering the finite room, using the same materiality framing as the earlier Week-7 zero-history audit.

Do not delete rare-use rows. This is reporting and authority tracing, not permission to hard-code unusual players away.

### REBASE / PORT REQUIREMENT

Before any merge, the active research branch must be rebased/ported onto current main.

At handoff it is ahead 30 / behind 2 before the V4 docs-only commit and has merge base c4de3d40.

Do not lose the already-accepted durable audit lineage:
- Walker decomposition
- JSN/Kupp/Barner decomposition
- zero-history position-prior allocation audit
- Identity Card and threshold-sweep artifacts

## WHAT CLAUDE SHOULD DO NEXT

There is no Claude response after 6426366490 yet.

When Claude resumes:
1. fix NO_BASELINE injury-contamination precedence;
2. add the required three regression tests;
3. fix current-team share to target-team-tenure semantics;
4. add the in-season trade regression test;
5. produce the evidence-authority/materiality decomposition;
6. rebase/port the research branch onto live main without losing audit lineage;
7. post a NEW fixed head plus exact green runs in #535;
8. stop and wait for GPT re-audit before predictive/scoring use.

GPT should then independently inspect exact code/tests/artifacts, not merely accept Claude's summary.

## AFTER THE CARD CLEARS

Do not immediately fit or score it.

The first predictive use must be frozen/preregistered before outcomes are consulted and must obey the anti-retest contract already recorded in Issue #535.

Player-state evidence and opponent/environment transmission are separate layers. Clearing the card does not close environment transmission.

Likely next science sequence after identity clearance:
1. freeze a prospective player-state use contract;
2. continue player-first opportunity-authority trace;
3. continue opponent/environment transmission architecture work, starting from the already-documented Walker / JSN / Kupp / Barner findings;
4. future-window Stage-2 readiness must use a genuinely future week;
5. keep betting construction V2 downstream and separate.

## PRODUCTION ARCHITECTURE FACTS ALREADY VERIFIED

- scripts/modeling/ml_v2.py uses lagged player history + position indicators; no opponent/team-defense features.
- scripts/modeling/state_v2.py uses player outcome-state transitions + position; no opponent.
- scripts/modeling/bayesian_v2.py uses population prior + prior/current player evidence; no opponent.
- FB maps into the RB position group in Bayesian logic.
- scripts/modeling/rules_v2.py / simulation_rules.py carry the coarse generic matchup rules.
- generic game-script pass share is hardcoded at 0.57.
- diagnostic lead/trail does not change the generic pass/rush split.
- scripts/simulation_v2.py takes team top-five rushing shares and normalizes modeled rushing allocation to 0.95.
- scripts/modeling/target_entitlement_v1.py caps modeled player target mass at 0.95 and leaves residual 0.05.
- TE-R5P and WR-R15 operate after the finite target room and conserve their room pools.
- Stage-2 conditional allocation research is not production-wired.
- current availability reranking maps FB into the RB namespace.

## EXISTING SCIENCE BOUNDARIES TO PRESERVE

- QB M89/M90 + C2: active/frozen production lineage.
- WR-R15: active.
- TE-R5P: active.
- RB P3/R22: Week-1-only promotion; not a Week-7 authority.
- Stage-2: qualified research, not production-wired.
- direct WR/CB source lane: unavailable/gated.
- ATD model: uncertified.
- RB rushing Stage-1 source reset: SOURCE BLOCKED.
- RB/TE team-context reset candidates: PREDICTIVE CLOSED.
- failed SLOT carry candidate: do not revive from architecture anecdotes.
- receiver-room targets-per-play discovery failed closed prospectively for 2024-2025.
- no re-opening closed science without new causal evidence.

## USER INTENT / WORKING STYLE THAT MATTERS TO THIS PROJECT

The user wants this to feel like one continuous research program, not a series of fresh starts.

Do not:
- ask the user to re-explain the model;
- blindly agree with a claim without checking the repository/evidence;
- treat aggregate MAE as the goal;
- go rogue into broad new research while an assigned blocker is unresolved;
- silently weaken fail-closed guards;
- overwrite or discard prior validated work;
- call descriptive infrastructure predictive;
- call position priors bad merely because they are priors.

Do:
- verify named-player logic;
- distinguish source readiness from predictive qualification;
- investigate anomalies before learning trends;
- preserve evidence durability in git;
- report mistakes explicitly when found;
- keep Claude and GPT roles separate but coordinated through #535.

## EXACT RESUME STATE

The next chat should begin by checking for a Claude response after Issue #535 comment 6426366490.

If there is a newer Claude comment:
- audit that exact work against Blocker A, Blocker B, the materiality decomposition, and branch/main reconciliation.

If there is no newer Claude comment:
- do not redo the already-completed audits;
- wait/continue project-control around the required fixes;
- do not start predictive scoring.

That is the exact point where this conversation ends.
