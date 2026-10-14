# NFL HANDOFF — 2026-10-14 — PLAYER-FIRST INDIVIDUALIZATION / TARGETED-ROUTE MATCHUP SOURCE READY

**This is the newest authoritative continuity document for the active NFL research lane.**

GitHub is canonical. Chat memory is secondary.

Repository:
- `dkaps6/imtiredofthis`

Production main at handoff creation:
- `812ac2cde79fe3846174613a2f2777e57b34b739`
- message: `Docs: preserve model-backed alt-line parlay construction method`

Active research branch:
- `research-individual-opportunity-roadmap-2026-10-08`
- branch head immediately before this handoff write: `08ef29f1bff0c6f93336990586869938feea7d8a`
- draft PR: **#672**
- continuity issue: **#535**

No production model code was changed by the work in this handoff.
No paid OddsAPI call was used.

---

# 1. START RULE FOR THE NEXT CHAT

Read in this exact order:

1. `AGENTS.md`
2. ONLY the newest TOP checkpoint in `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. THIS FILE
4. `docs/research/PLAYER_TARGETED_ROUTE_MATCHUP_V1_FREEZE.md`
5. `docs/research/PLAYER_TARGETED_ROUTE_MATCHUP_V1_SOURCE_RESULT.md`
6. `docs/research/WR_PLAYER_OPPORTUNITY_UNCERTAINTY_V1_RESULT.md`
7. `docs/research/TE_PLAYER_OPPORTUNITY_UNCERTAINTY_V1_RESULT.md`
8. `docs/research/PLAYER_HIGH_RESOLUTION_ROLE_STATE_SOURCE_AUDIT_2026-10-14.md` if needed by exact path lookup
9. newest comments on Issue #535
10. live branch / PR #672 / exact-head GitHub Actions

Do **not** recursively read older handoffs.

Do **not** restart completed W1-W5 replay/postmortem work, player landscape work, output decomposition, WR-R3, player-x-environment, broad personnel, generic matchup, efficiency-tail, Q90 tail, or receiver-conversion lanes merely to reacquire context.

If a newer result has landed after this handoff, that newer frozen/result document beats this narrative.

---

# 2. USER'S MODEL STANDARD — NON-NEGOTIABLE

The user explicitly corrected the research standard in this conversation:

> This is an **individual-player projection system**, not a position-average model.

The intended projection is conceptually:

`named player × current role × own history × teammates × team offense × opponent × exact matchup × game environment -> individual projection/distribution`

Important consequences:

- A position-wide MAE/RMSE result is a **guardrail**, not the scientific hypothesis grain.
- Do not equate "this position-level candidate failed" with "individual-player research is exhausted."
- The first question must be whether the **named player's own state/history** contains information about that player's next game.
- Player-level reporting must include qualifying named players, median per-player behavior, sign consistency, and player-clustered inference where appropriate.
- Position-wide aggregates can stop a harmful promotion, but they cannot by themselves prove that individual-player signal is absent.
- Shared positional shrinkage is allowed as a fallback when individual history is insufficient, but it is not the model vision.
- Generic defense-v-position multipliers are not acceptable substitutes for football/player knowledge.

Canonical status remains:

`PLAYER_INDIVIDUALIZATION_PARTIAL`

The model has a real individual core, but shared positional transforms/shrinkage and shared environment responses remain material.

---

# 3. CRITICAL CORRECTION FROM THIS CHAT

We explicitly re-audited prior player-specific work before deciding what to do next.

## WR-R3 was NOT evidence that individual signal was exhausted

The exact WR-R3 combined candidate did fail its frozen promotion gates, so do not reopen or promote it.

But the underlying player-first evidence was real:
- same-player signed bias persistence;
- same-player difficulty persistence;
- same-player extreme-miss persistence.

The later combined candidate improved MAE in **all six seasons 2020-2025** and improved the median qualifying player's MAE, plus WR1/WR2/WR3 slices. It still failed because:
- MAE gain missed the predeclared >=1% bar;
- 20+ yard miss rate worsened;
- exact parent/cohort parity was slightly off.

Interpretation:
- **exact implementation failed**;
- **individual-player phenomenon did not disappear**.

Do not use WR-R3 as proof that individual WR modeling is closed.

## RB same-player persistence is also real

Historical RB-PD2:
`RB_PLAYER_ERROR_PERSISTENCE_DETECTED`

Same-player carry/yard difficulty persisted. Some later mean-correction implementations failed their own gates, but that does not invalidate the underlying player-specific reliability signal.

Historical distribution qualification:
`RB_YARD_DIFFICULTY_MC_WIDTH_QUALIFIED`

This remains valid and separate.

## QB is different

Dedicated QB same-player error-persistence work was genuinely null. QB also already has stronger individualized mean machinery from M89/M90.

Do not reopen generic QB bias persistence/mean correction without materially new information.

---

# 4. PLAYER-FIRST UNCERTAINTY — NEW CONFIRMED EVIDENCE

This conversation produced/reconciled two important direct player-vs-position-pool tests.

## TE Player Opportunity Uncertainty V1

Canonical result:
`docs/research/TE_PLAYER_OPPORTUNITY_UNCERTAINTY_V1_RESULT.md`

Run:
- `38594365393` SUCCESS
- job `115957146392`
- artifact `11909006619`
- digest `sha256:76308043d022ebf57aa7728c330d8562051d31bc1ae5b65dddc33ebf225e82a3`
- exact scored head `32f11c67b3028d49b6d915085b3ec7c0d972181f`

Disposition:
`TE_PLAYER_OPPORTUNITY_UNCERTAINTY_CONFIRMED`

Question:
Does this named TE's own strictly-prior opportunity difficulty forecast his uncertainty better than pooled TE uncertainty?

Result:
- 2,005 player-games
- 121 TEs
- 2024 CRPS: 10.36052 -> 9.95972
- 2025 CRPS: 9.02549 -> 8.78711
- pooled CRPS: 9.67669 -> 9.35909 (~3.28% relative improvement)
- player-cluster bootstrap 95% CI for gain: [+0.19727, +0.46175]
- P(gain>0)=1.000
- pooled 80% coverage: 77.46% -> 80.00%
- pooled 90% coverage: 83.44% -> 86.23%
- high-difficulty Q4 CRPS: 15.05654 -> 14.64296
- Q4 80% coverage: 59.16% -> 84.66%
- Q4 90% coverage: 68.92% -> 90.24%

No fitted coefficient. No mean change. No sportsbook/OddsAPI. No 2026 outcomes. No production change.

## WR Player Opportunity Uncertainty V1

Canonical result:
`docs/research/WR_PLAYER_OPPORTUNITY_UNCERTAINTY_V1_RESULT.md`

Run:
- `38595088817` SUCCESS
- job `115959621036`
- artifact `11908832554`
- digest `sha256:32fc0e08fc06cdf388151773338f25ce1614cc6038044d054b12cba4c5367bf5`
- exact scored head `3f672b6e4ca675a36525a5efb83db8d0a20055c8`

Disposition:
`WR_PLAYER_OPPORTUNITY_UNCERTAINTY_CONFIRMED`

Result:
- 3,224 player-games
- 211 WRs
- 2023 CRPS: 12.50878 -> 12.35416
- 2024 CRPS: 13.21114 -> 12.99474
- pooled CRPS: 12.91551 -> 12.72511 (~1.47% relative improvement)
- player-cluster bootstrap 95% CI: [+0.08180, +0.30846]
- P(gain>0)=0.9996
- pooled 80% coverage gap: 2.705pp -> 1.588pp
- pooled 90% coverage gap: 4.702pp -> 4.547pp
- high-difficulty Q4 CRPS: 17.87079 -> 17.61484
- Q4 80% coverage: 63.03% -> 88.46%
- Q4 90% coverage: 74.57% -> 93.18%

Again:
- named-player own history beat pooled positional uncertainty;
- this is distribution/reliability evidence, not a mean correction;
- no production change.

Cross-position interpretation:

The repo now has repeated player-specific distribution evidence across:
- WR
- TE
- RB

This is strong evidence for a **player-specific reliability/distribution layer**.

Do not collapse this into generic position width.

---

# 5. PLAYER X ENVIRONMENT RESPONSE V1 — CLOSED, SUPPORT-LIMITED

Frozen diagnostic asked:

> Given the existing full-stack projection and shared environment multiplier, does THIS named player's own prior response to that environment predict his next residual?

Canonical run:
- `38463830141`
- artifact `11847639388`

Disposition:
`PLAYER_X_ENVIRONMENT_RESPONSE_NULL`

But interpret carefully:
- 9,253 source rows had environment multipliers;
- only 165 rows / 37 players survived the frozen same-player history + distinct-environment-state support requirements;
- WR: 54 rows/12 players in 2024, 63/16 in 2025;
- TE: 25/6 in 2024, 23/6 in 2025;
- RB: zero qualifying rows.

Reason:
shared production environment states are highly discrete (roughly 0.98/1.00/1.01 receiving, 0.94/1.00/1.07 RB rush).

This closes the exact V1. Do not lower support/history gates or pool players/seasons to rescue it.

It is not evidence that named-player response is universally absent.

---

# 6. BROAD PLAYER PERSONNEL-CONDITIONED OPPORTUNITY V1 — CLOSED

The branch then tested a clean player-conditioned 11/12/OTHER personnel construction.

Run:
- `38627780518`
- exact scored head `5afaefb2dd250cd6c0c5a9ea5cac473c5dd8439c`
- artifact `11924472635`
- digest `sha256:3522e4fb14d6a0fd31c9b5733fe719887bbabeada55c037303e9285282500219`

Formal result:
`docs/research/PLAYER_PERSONNEL_CONDITIONED_OPPORTUNITY_V1_RESULT.md`

Disposition:
`NO_ACTIONABLE_PLAYER_PERSONNEL_CONDITIONED_OPPORTUNITY`

Key facts:
- 4,220 eligible rows
- 100% parent identity mapping
- zero chronology violations
- WR pooled rho +0.02356; player-cluster CI crosses zero
- TE negative in both 2024/2025; pooled rho -0.03950

Exact 11/12/OTHER player-target-propensity × team personnel-mix construction is closed.

Do not retune 11/12/OTHER or invent slightly different broad personnel buckets.

This null does **not** negate player-specific uncertainty or the targeted-route lane below.

---

# 7. HIGH-RESOLUTION ROLE-STATE SOURCE AUDIT

Latest branch work before the targeted-route source test recorded a source/anti-retest audit at commit:

`932380be97bdfc1384d16ce75a0936f23577bb5f`
message: `Research: record high-resolution player role source audit`

Important conclusions:

## Historical total routes / route participation
Blocked for source parity.
- nflverse participation `route` is NOT total routes run.
- do not count route labels as all routes.

## Player alignment
Current/live slot/wide fields exist, but clean historical parity remains unproven.
- potentially useful for prospective locks;
- not authorized as a historical fitted feature yet.
- TE inline/slot/flex historical source still absent.

## Broad participation formation/personnel
Available, but the exact broad 11/12/OTHER hypothesis is now closed.

## FTN motion
Play-level motion flag exists, but it is not player-attributed for this purpose.
Do not call it Player A motion rate.

## NGS
Exact prior NGS target-model implementation failed.
Do not repackage it.

## Free WR-CB archive
Projected alignment/matchups exist as a separate WR lane, but projected pairing is not observed route responsibility and does not solve TE role state.

## Still legitimately open
- true player-attributed alignment history if source parity is found;
- actual all-route participation/family history if source parity is found;
- player-attributed motion if source is found;
- genuinely new named-player hierarchy/role state;
- targeted-route mix, under strict semantics.

---

# 8. NEW TARGETED-ROUTE × OPPONENT ROUTE-FAMILY LANE — SOURCE GATE PASSED

This is the most important newest development.

The chat intentionally moved away from:
- defense-v-position;
- generic matchup multipliers;
- broad personnel;
- team-level tactical averages.

Instead it froze a truly player-specific matchup question:

> For THIS named WR/TE, does his own strictly-prior mix of route families on plays where he was targeted interact with THIS opponent's strictly-prior results allowed on those same route families?

Two teammates can therefore receive different matchup states against the same defense because their own targeted-route mixes differ.

Frozen plan:
`docs/research/PLAYER_TARGETED_ROUTE_MATCHUP_V1_FREEZE.md`

Source auditor:
`scripts/research/audit_player_targeted_route_matchup_source_v1.py`

Tests:
`tests/test_player_targeted_route_matchup_source_v1.py`

Workflow:
`.github/workflows/research-player-targeted-route-matchup-source-v1.yml`

Formal source result:
`docs/research/PLAYER_TARGETED_ROUTE_MATCHUP_V1_SOURCE_RESULT.md`

## Mechanical history

Initial run:
- `38629644127`
- failed mechanically on earlier head
- do NOT interpret scientifically

Mechanical import-path fixes followed.
No science/gates changed.

Canonical replacement run:
- `38637901016` — SUCCESS
- job `116105672352`
- exact scored head `159b8213a734e3f4ee16fc86fbb51950099f3323`
- artifact `11928258474`
- digest `sha256:b957ff0b37f0b2d8f45232bcd44657928cecc75f5f5e6d1c292ee8e35729228c`

Disposition:
`PLAYER_TARGETED_ROUTE_MATCHUP_SOURCE_READY`

Every frozen source/support gate passed.

## Source facts

Historical source seasons:
- 2022-2025

Scored seasons:
- 2023-2025

Raw route-labeled participation rows:
- 75,773

Joined route/PBP rows:
- 72,206

Join rate:
- 100%

Valid targeted-route WR/TE rows:
- 46,666

Distinct valid WR/TE players:
- 554

Stable identity coverage on taxonomy-eligible route rows:
- 93.1827%

Eligible named players with enough history:
- 2023: 257
- 2024: 246
- 2025: 253

Defense support by Week 8:
- 32/32 teams in 2023
- 32/32 in 2024
- 32/32 in 2025

Valid targeted-route rows:
- 2023: 14,028
- 2024: 13,923
- 2025: 13,567

Ten well-supported route families >=1,000 rows:
- HITCH/CURL 7,609
- QUICK OUT 6,986
- GO 5,607
- IN/DIG 4,543
- SCREEN 4,228
- SLANT 4,205
- DEEP OUT 3,538
- POST 3,214
- SHALLOW CROSS/DRAG 3,114
- CORNER 2,611

Lower-support:
- SWING 521
- WHEEL 410
- TEXAS/ANGLE 80

Do not regroup route families after outcome exposure.

## Semantic boundary

CRITICAL:
`route` is only the route label for the **primary/targeted receiver on the play**.

It is NOT:
- total routes;
- route participation;
- TPRR;
- YPRR;
- non-target route state.

Do not violate this in Stage B.

---

# 9. EXACT NEXT SCIENTIFIC ACTION — STAGE B

Stage A/source is complete.

Next chat should immediately build and run **Stage B** exactly as frozen.

For target player-game g:

## Player targeted-route mix
Use:
- same named player only;
- completed games strictly before g;
- trailing 8 eligible player-games maximum;
- minimum 4 prior player-games;
- only valid targeted-route rows;
- normalize counts to route-family targeted-route weights.

## Opponent route-family susceptibility
For each route family:
- target opponent only;
- completed defensive plays strictly before g;
- trailing 8 defensive games maximum;
- minimum frozen support;
- compute:
  - completion rate allowed;
  - yards per target allowed;
  - receiving yards per target allowed.

No target game may enter history.
No same-week contamination.

## Neutral control
League strictly-prior route-family performance.

## Player-specific matchup expectation

For each metric m:

`matchup_m = sum_r(player_targeted_route_mix_r * opponent_prior_allowed_m_r)`

`neutral_m = sum_r(player_targeted_route_mix_r * league_prior_allowed_m_r)`

`matchup_delta_m = matchup_m - neutral_m`

This is the pregame player-specific matchup state.

## Primary outcomes

WR and TE separately:
- next-game receiving yards per target;
- next-game catch rate;
- next-game receiving yards.

Target count is diagnostic only because the source has no all-route denominator.

## Validation split

Frozen:
- 2023 development
- 2024 independent confirmation
- 2025 independent confirmation

Do not tune family grouping/window/support after 2023 output.

Required to advance:
- sign-consistent relationship in both 2024 and 2025 for at least one predeclared metric;
- player-clustered 95% CI excludes zero in same direction pooled 2024-2025;
- game-clustered 95% CI excludes zero;
- signal is not just a renamed target-depth/air-yard signal;
- WR and TE separate;
- player-level reporting required;
- position-wide MAE cannot qualify the hypothesis alone.

A Stage-B PASS authorizes only a separately frozen full-stack integration candidate beyond M38/WR-R15 or TE-R5P.

Do not jump directly to production from the diagnostic.

---

# 10. MEAN VS DISTRIBUTION — KEEP SEPARATE

The current player-first program has two parallel tracks.

## Distribution / uncertainty track — positive

Confirmed:
- WR named-player opportunity uncertainty
- TE named-player opportunity uncertainty
- RB yard-difficulty width historical qualification

These are not mean corrections.

They support player-specific reliability/uncertainty distributions and should proceed toward separately frozen integration/shadow work.

## Mean / matchup track — unresolved but not exhausted

Closed exact implementations:
- WR-R3 combined calibration
- broad 11/12/OTHER personnel opportunity
- player-x-environment V1
- receiver conversion mechanism V1
- efficiency-tail volatility V1
- Q90 upper-tail hinge
- generic matchup transmission candidates
- team man/zone coverage ablation
- target-depth symmetric widening
- NGS target model
- generic WR-CB historical assignment due source limitation

Open newest mean-adjacent lane:
- targeted-route × opponent route-family Stage B

Also potentially open only if source parity can be established:
- player alignment history;
- all-route participation/history;
- player-attributed motion.

Do not confuse a distribution success with solved mean accuracy.

---

# 11. PRIOR WEEK-5 / SEASON CONTINUITY

Week-5 canonical preserved-odds Full Slate:
- run `37860592615` SUCCESS
- artifact `11585973062`
- digest `sha256:bf52aa4bc30bf6e8cc6029a0a86c0f33cdfaa09f587c38a138f23cef6b7436f4`
- production authority `24704e5f1ef872ec86c36446d2c73f6d377c824c`
- pricing status `PRESERVED_REPLAY`
- 1,472 priced offers
- 754 player-markets
- zero unresolved positions
- DAL-TB retained by acquisition-time pre-T75 lock
- ZERO second OddsAPI acquisition

Week-5 selected board:
- 199-151
- 56.86%
- +26.17u
- +7.48% ROI

Through Week 5 selected board:
- 1,048-969
- 2,017 decided
- 51.96%
- -17.93u
- approx -0.89% simple cumulative ROI

Week-5 raw model:
- model MAE 18.9288
- median preserved-line MAE 17.8431
- model closer than line 50.26%
- direction hit rate 55.32%

Interpretation:
selected-board profitability improved, but raw player mean accuracy remains unresolved.

Top-40 Week-5 miss decomposition:
- 19 efficiency-dominant
- 14 opportunity-dominant
- 7 mixed

Do not claim mean layer solved.

---

# 12. WEEK-6 READINESS / FUTURE LOCK

Latest known readiness from Oct 13:
`PLAYER_STATE_FUTURE_WEEK_NOT_READY`

Reason:
- exact Week-6 weekly roster authority missing;
- roster source week remained 5;
- target-week injury rows unavailable;
- Week-5 stats/snaps complete;
- chronology clean.

Do not substitute stale Week-5 roster identity for an exact Week-6 lock.

If next chat checks and exact Week-6 roster authority is now available, it may resume the frozen Week-6 opportunity/joint-allocation prospective lock path. Keep that separate from Stage B historical science.

---

# 13. CLOSED SCIENCE / ANTI-RETEST

Do not reopen unchanged:

- raw-edge sorting as confidence;
- generic position averages as the individual projection solution;
- generic defense-v-position multipliers;
- universal symmetric target-depth widening;
- receiver conversion V1;
- individual efficiency-tail volatility V1;
- Q90 upper-tail hinge V1;
- broad 11/12/OTHER personnel-conditioned opportunity;
- player-x-environment V1;
- generic team man/zone coverage as WR efficiency;
- historical WR-CB assignment from nflverse;
- WR-R3 exact combined candidate;
- NGS target model R9-R11;
- RB historical receiving efficiency transforms YPR/YPT/YAC/xYAC/YACOE;
- RB PD3/PD4/PD5 exact mean-calibration formulas;
- generic QB player-error persistence;
- generic QB internal disagreement reliability;
- generic Vegas/game-script;
- previously failed simple Football Matchup Transmission candidates;
- M95A/M95B generic RB role × defense;
- M96 stop remains binding.

A materially different information contract may be tested only after anti-retest proof and a frozen plan.

---

# 14. USER'S WORKING EXPECTATIONS

The user wants:
- execution, not repeated roadmap speeches;
- concise check-ins after meaningful milestones;
- actual GitHub work;
- no fabrication;
- no "position average" framing when the model is meant to predict individuals;
- autonomous continuation;
- real Actions runs;
- mechanical failures fixed without changing frozen science;
- result docs committed;
- production and research separated;
- no paid OddsAPI without explicit authorization;
- no asking them to re-explain prior work.

When they say "keep going" or "checking in":
continue real GitHub work.

---

# 15. EXACT RESUME COMMAND FOR THE NEXT CHAT

After reading the files above:

1. reconcile current research branch head, PR #672, Issue #535 latest comments, and exact-head Actions;
2. verify that `PLAYER_TARGETED_ROUTE_MATCHUP_SOURCE_READY` remains the newest source authority;
3. build Stage B evaluator/tests/workflow from the already-frozen contract;
4. do not change route families, windows, support gates, or outcomes after seeing 2023 results;
5. run 2023 development then frozen 2024/2025 confirmation;
6. report player-level + player/game clustered results separately for WR and TE;
7. write formal result doc immediately after run;
8. if Stage B passes, freeze a separate full-stack integration candidate;
9. if Stage B fails, close this exact interaction with no rescue;
10. continue player-specific WR/TE/RB distribution integration in parallel only under separately frozen contracts;
11. no production mutation and no paid OddsAPI.

This is the immediate frontier.
