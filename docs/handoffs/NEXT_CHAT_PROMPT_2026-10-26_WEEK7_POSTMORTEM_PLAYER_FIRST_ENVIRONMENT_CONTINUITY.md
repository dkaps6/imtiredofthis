Pick up my NFL PRISM project seamlessly from the newest canonical GitHub checkpoint.

Repo: `dkaps6/imtiredofthis`.

Do NOT ask me to re-explain anything. GitHub is canonical. I need this to feel like the exact same assistant continuing this exact conversation.

READ IN THIS EXACT ORDER:

1. `AGENTS.md`
2. ONLY the newest TOP checkpoint in `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. `docs/handoffs/NFL_HANDOFF_2026-10-26_WEEK7_POSTMORTEM_PLAYER_FIRST_ENVIRONMENT_CONTINUITY.md`
4. Newest Issue #535 comments — especially:
   - 6417917738
   - 6418739459
   - 6419138959
   - 6419872553
   - 6420003475
   - 6420198257
5. Reconcile live `main`, open PRs, exact-head Actions/artifacts before asserting status.

Do NOT recursively read old handoffs. Do NOT restart closed science.

PERMANENT ASSIGNMENT:

PRISM is an INDIVIDUAL-PLAYER PROJECTION MODEL.

> THE PLAYER SHOULD HAVE THE RULES. THE POSITION SHOULD NOT HAVE THE PLAYER.

> ANOMALIES ARE QUESTIONS, NOT TRENDS.

Position MAE/RMSE is downstream QA only, never the objective.

Always distinguish SOURCE READY / SOURCE BLOCKED / PREDICTIVE CLOSED / QUALIFIED.

CONTROL ROOM:
- Claude = executor / research lead
- GPT-5.6 = independent auditor / project control
- shared lane = Issue #535
- if Claude has posted anything newer, audit that first rather than duplicating work.

CURRENT CRITICAL STATE:

- Week-7 Sunday/SNF ticket evidence is durable on main from PR #695 under:
  `docs/research/evidence/2026-10-25_week7_ticket_ledger/`
- It includes all Sunday non-RB drafts/finals, RB-integrated tickets, Sunday ATDs, SNF ATDs, SNF regular Safe/Moderate/Medium/High/Lottery, one-row-per-leg CSV, 68 transcribed FanDuel prices, screenshot mappings and hashes.
- User reported only ONE ticket hit across Sunday and none of the SNF regular cards hit.
- Claude's independent forensic concluded the ticket result itself is primarily a BETTING-CONSTRUCTION failure, not proof of model failure:
  - “Safe” parlays were only ~41–47% naive model-implied hit probability.
  - 25/57 unique yardage legs were bought-down OVERs while PRISM's main-line edge direction was UNDER.
  - model_sd / exact P(clear) was not properly used.
  - portfolio was highly concentrated in repeated players.
- GPT ACCEPTED that construction diagnosis.

CRITICAL BOUNDARY:

**Do NOT close opponent/environment transmission.**

The correct disposition is:

- Week-7 parlay failure = primarily construction / threshold / exposure problem.
- Opponent/environment transmission = still an independent OPEN architecture/research problem.

Prior code-path audit already established:
- def_rush_epa reaches TeamContext but is not consumed by generic RB matchup rules;
- explosive-play suppression available but not consumed generically;
- WR/TE/RB YPT and outside/slot YPT are dropped before generic player projection;
- YBC/stuff run-defense fields are dropped;
- generic project_game_script hardcodes pass_share=0.57;
- lead/trail does not change generic pass/rush split;
- Stage-2 qualified allocation research does not currently reach production.

Walker is the preferred FIRST generic-skill transmission decomposition case, not Mahomes, because QB M89/M90 is already much richer.

For Kenneth Walker III, audit read-only:
1. neutral/current-player baseline;
2. every legal pregame opponent input available;
3. what production actually consumed;
4. what was collected but dropped;
5. opportunity movement;
6. efficiency movement;
7. mean movement;
8. sigma/uncertainty movement;
9. rushing-vs-receiving defensive asymmetry.

Then representative WR/TE cases such as JSN vs Kupp and Barner.

Do not fit anything to Week 7. Week 7 is forensic/case-finding only.

PLAYER IDENTITY:

Claude corrected an earlier overstatement. Historical player identity is NOT blocked.

A strictly-prior Player Identity Card can be built from football history using nflverse PBP/player stats/snap counts/schedule and other source-valid history.

Player-conditional promotion gate direction:
- preregister football-semantic partitions before scoring;
- same-direction replication across independent seasons/windows;
- pooled population must not materially harm;
- declared partition must improve consistently;
- minimum support;
- no top-N rescue;
- no post-hoc named-player carveout;
- report per-player heterogeneity even when candidate closes.

BETTING-LAYER CORRECTIONS:

Until an evidence-backed construction V2 is frozen:
- if proposed alt direction conflicts with canonical main-line model edge direction => default PASS;
- do not equate raw yards of cushion with safety;
- show model mean, model SD/distribution authority, main line, main-line direction, exact alt threshold, estimated P(clear), exact alt price, and portfolio exposure;
- do not call a ~41% parlay “Safe”;
- surface same-game and repeated-player concentration;
- no universal anchor across every tier;
- ATD remains uncertified/fun and separate from yardage science.

CANONICAL PAID WEEK-7 BOARD:
- run 39849371821 SUCCESS
- artifact 12456298985
- digest sha256:d1a1fb48e5bb920c5291f35b7457b5870801b689751deb2d742f98c555603a85
- durable files under `data/market_track_record/`
- NO NEW PAID ODDSAPI PULL without my explicit authorization.

STAGE-2:
Qualified conditional allocation exists in research but is not wired to production.
A fresh source-readiness re-probe under the already-frozen contract was the exact assigned next step unless newer #535 evidence shows it already happened.

OPERATIONS:
- pre-handoff main was 97ecae88f391a692cf694971c2bf2f34c04cf593; the continuity merge will advance it, so re-fetch.
- Repo CI after #695 was red only on the known stale no-credit exact-string assertion: 660 passed / 1 skipped / 1 failed.
- Do NOT weaken the live-odds guard to satisfy that stale test.
- #691 stale/superseded; do not merge.
- #690 previously audit-cleared, science remains SOURCE_BLOCKED.
- #672 large draft research lineage; do not wholesale merge.

EXACT NEXT BEHAVIOR:

1. Reconcile live state and newest Claude message.
2. If Claude has advanced the Player Identity Card / Stage-2 readiness / Walker decomposition, audit that exact work.
3. Otherwise continue the active player-first identity + environment-transmission work; do not restart.
4. Keep betting construction as a separate downstream lane from predictive science.
5. Do not spend an OddsAPI credit.
6. Do not stop at a status summary if there is actionable research/audit work.

This must feel like the same chat continuing.