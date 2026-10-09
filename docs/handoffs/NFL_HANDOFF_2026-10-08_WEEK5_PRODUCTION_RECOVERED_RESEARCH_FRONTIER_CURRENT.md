# NFL HANDOFF — 2026-10-08 — WEEK 5 PRODUCTION RECOVERED / RESEARCH FRONTIER CURRENT

**Repository:** `dkaps6/imtiredofthis`  
**GitHub is canonical. Chat memory is secondary.**  
**Purpose:** exact cross-chat continuity after Week-5 production execution interrupted the active individual-opportunity research lane.

This handoff must be treated as the immediate resume authority together with the newest top checkpoint in `CURRENT_NFL_RESEARCH_HANDOFF.md`.

---

## 1. USER / PROJECT OPERATING CONTRACT — PRESERVE

The user does **not** want this treated as a generic statistical prop model. The model is intended to price real football:

`GAME STATE -> TEAM OPPORTUNITY -> ROOM / POSITION OPPORTUNITY -> PLAYER ENTITLEMENT -> PLAYER EFFICIENCY -> MATCHUP TRANSMISSION -> JOINT SIMULATION -> DISTRIBUTIONS -> SPORTSBOOK COMPARISON`

Important user framing that was active immediately before game-day execution:

- football projections should come from football structure, roles, team tendencies and matchups, not just better averages;
- individual players are different, teams are different, roles are different;
- some defenses are genuinely better/worse against certain position/usage families, and matchup value should transmit **conditional on the player's role and opportunity**, not as a generic position multiplier;
- sportsbook is downstream only and must never select the football starter, role or projection;
- user prefers fewer, higher-confidence plays rather than treating every large model-vs-book gap as equally actionable;
- never fabricate missing data, never weaken truth gates merely to make a run green;
- never claim a GitHub Action is running unless live Actions shows queued/in_progress;
- no paid OddsAPI pull without explicit authorization;
- if a paid snapshot already exists, reuse it rather than spending a second credit.

Do not confuse the user's football-layer concern with permission to reopen generic matchup multipliers. The historical Football Matchup Transmission work showed meaningful football context is sometimes weakly transmitted, but all three simple predeclared integration formulas failed. Matchup work remains secondary/conditional until opportunity allocation is resolved.

---

## 2. CANONICAL PRODUCTION STATE — WEEK 5 RECOVERED

### Current production authority before this docs-only handoff branch

- `main`: `24704e5f1ef872ec86c36446d2c73f6d377c824c`
- canonical orchestration: `.github/workflows/full-slate.yml`
- canonical Week-5 preserved-odds Full Slate:
  - run **37860592615**
  - conclusion **SUCCESS**
  - event `workflow_dispatch`
  - exact head `24704e5f1ef872ec86c36446d2c73f6d377c824c`
  - artifact **11585973062**
  - artifact name `run_37860592615`
  - artifact digest `sha256:bf52aa4bc30bf6e8cc6029a0a86c0f33cdfaa09f587c38a138f23cef6b7436f4`
  - artifact size ~8.0 MB
  - `FETCH_LIVE_ODDS=false`
  - **zero new OddsAPI acquisition**

### Preserved paid sportsbook source

Original paid Week-5 run:
- run **37852811339**
- paid source artifact **11582178322**
- digest `sha256:4948e08003fcab0f519b35bd4b23d8c895554e4faced993238575acdc5d2e765`
- the paid pull occurred successfully before the original run failed on a downstream quality gate;
- do **not** refetch this slate unless the user explicitly reauthorizes a new paid pull.

Canonical replay now supports pinned preserved sportsbook artifacts directly inside `full-slate.yml`.

The replay is explicitly labeled:
- `pricing_status=PRESERVED_REPLAY`
- prices are **not** asserted current;
- `Bettable Now` is suppressed for replayed prices;
- sportsbook remains downstream-only.

### Final Week-5 board produced by canonical run 37860592615

- workbook: `outputs/NFL_BETTING_MODEL_MASTER.xlsx`
- priced offer rows: **1,472**
- player-market rows: **754**
- unresolved position rows: **0**
- DAL-TB priced rows: **254**
- sportsbook downstream only: **true**

Do not quote these as current live odds after the fact. They are the exact preserved paid snapshot repriced through the current football model.

---

## 3. PRODUCTION BLOCKERS FOUND AND CLOSED TODAY

Game-day execution exposed real operational assumptions. These were fixed without changing model science.

### A. Week-5 bye-week QB C2 scope

Week 5 has **30 active teams** because two teams are on bye. The old QB C2 source-context logic assumed 32 active slate teams and failed.

Closed via production hotfix chain:
- main `66c22026880178e60d8011ea2400127249be83ac`: schedule/role-certified 30-team C2 active context;
- later main hotfixes preserved full-schedule integrity while permitting the certified active complete-game subset;
- no fake bye opponents;
- no model parameter changes;
- original protected availability seams preserved.

### B. Data-quality coverage scope

Raw Ourlads legitimately contains all 32 NFL teams while the active Week-5 schedule contains 30. The quality classifier was corrected to distinguish:
- 32-team season/provider roster universe;
- 30-team active schedule;
- certified complete-game current eligible subset.

Do not restore hard-coded 32 active-slate assumptions.

### C. Current NFL.com injury scope

NFL.com injury reporting had to be made bye-week-aware:
- active Week-5 scope = 30 scheduled teams;
- teams with rows + explicit `No Injuries Reported` sections can jointly certify full scope;
- priced Full Slate refreshes current football-only injury scope before availability;
- sportsbook data is not used to set availability.

### D. Current identity alias

Verified identity bridge added:
- current `Mitch Tinsley`
- historical `Mitchell Tinsley`
- CIN WR
- GSIS `00-0038839`

Identity metadata only; no football usage invented.

### E. Uncertified backup-QB sportsbook market

Sportsbooks posted Tyler Huntley passing-yards offers while the football-only authority still had Huntley as BAL QB2.

Correct action:
- **do not let the book choose the starter**;
- Huntley Week-5 props are quarantined at the final-board stage only;
- Baltimore's other markets and internal football simulation remain intact.

This follows the same fail-closed production pattern used previously for Tyson Bagent / Marcus Mariota.

### F. Preserved snapshot vs refreshed roster

A preserved non-core anytime-TD row for Camden Brown no longer mapped to refreshed current PlayerForm.

Canonical replay behavior:
- strict yardage/reception market player mismatches remain fatal;
- stale **non-core** replay rows may be quarantined deterministically;
- paid source artifact itself remains immutable evidence;
- no lines/odds are altered.

### G. T-75 acquisition-time game lock

The first canonical preserved replay on `d36daa...` later crossed the T-75 official-inactives boundary for TB-DAL and current availability correctly withheld the game because official sections were missing.

Current main `24704e5f...` added a replay-only pregame lock seam:

For a preserved paid snapshot, an entire game can restore its exact acquisition-time football eligibility only when:
- the source game was production-eligible;
- source snapshot was >=75 minutes before kickoff;
- current game is still pre-kickoff;
- event identity/matchup/kickoff are unchanged;
- current withholding is only `REQUIRED_MISSING_FAIL_CLOSED` with both official sections missing.

For TB-DAL:
- kickoff `2026-10-09T00:15:00Z`
- source as-of `2026-10-08T22:22:28.077633Z`
- source minutes to kickoff **112.532**
- source state `NOT_YET_REQUIRED`
- restored games = 1
- restored teams = DAL, TB
- current definitive-unavailable facts preserved
- resurrected players = 0
- sportsbook inputs used for eligibility = 0

The restored state is explicitly stamped `PRESERVED_PAID_ACQUISITION_LOCK`.

### H. PR #679 status

PR **#679** (`P0: preserve tonight's pre-T75 football lock in canonical odds replay`) is now **stale/superseded**:
- it is based on old main `d36daa...`;
- its branch is dirty against current main;
- the equivalent/preferred fix is already on current main as `24704e5f...`;
- run **37860592615** proves the current-main implementation works.

**Do not merge #679 blindly.** Treat it as superseded historical lineage unless a targeted diff audit finds unique unmerged content.

---

## 4. LIVE DAL-TB GAME-DAY STATE — PRESERVE BUT DO NOT CONFUSE WITH RESEARCH

A pre-kickoff core SGP was frozen on Issue #535 before kickoff, using the canonical preserved board and football process.

Preferred core:
1. Jalon Daniels 175+ passing yards
2. Cade Otton 25+ receiving yards
3. Jake Ferguson 20+ receiving yards

Backup if Ferguson 20+ unavailable:
- Emeka Egbuka 25+ receiving yards

The ticket was deliberately built as:
- promoted QB passing leg;
- same-team TE receiving leg;
- opposite-team TE receiving leg;
- no ATD;
- no forced RB rushing/rush+rec while RB opportunity science remains unresolved;
- alt thresholds below main lines for survival.

Do **not** retroactively alter this ticket from outcomes. Do not grade it unless the user asks to grade the game.

This ticket is not a validated downstream selector and is not evidence for model changes.

---

## 5. RESEARCH STATE BEFORE GAME-DAY EXECUTION INTERRUPTED US

This is the critical continuity section.

### Headline scientific conclusion remains

`OPPORTUNITY_DOMINANT_ACROSS_ALL_PRIMARY_MARKETS__INDIVIDUAL_ROLE_SHARE_ALLOCATION_NEXT`

Exact decomposition authority:
- run **37777319154** SUCCESS
- scientific head `cb1e511adcb58e40fc41d0949306083a304e83dd`
- artifact **11550446909**
- digest `sha256:7acfc453258a49f3aadf0d0ea386b13827eb715f1dfb055b96505f733320d204`
- result doc `docs/research/PLAYER_OUTPUT_COMPONENT_DECOMPOSITION_V1_RESULT.md` on branch `research-player-output-component-decomposition-v1`

Opportunity oracle vs efficiency oracle — MAE removed:
- QB pass yards: **23.9% vs 12.7%**
- RB rush yards: **45.1% vs 7.2%**
- RB rec yards: **45.1% vs 27.5%**
- RB receptions: **57.2% vs -2.4%**
- WR rec yards: **42.8% vs 28.2%**
- WR receptions: **54.4% vs 11.6%**
- TE rec yards: **55.0% vs 23.5%**
- TE receptions: **68.3% vs 8.1%**

Meaning:
- the model is genuinely player-specific;
- the dominant remaining cross-position error is allocating the correct pregame opportunity to the named player;
- do **not** pivot back to generic efficiency or generic matchup multipliers as the primary repair.

The opportunity roadmap explicitly separates:
1. participation / active-role probability;
2. team opportunity volume;
3. room share / named-player allocation.

### Completed all-player / player-landscape work

Do not rerun:
- Player Individualization Audit V1;
- all-player/all-position W1-W4 replay;
- Player Landscape Transmission Audit V1;
- Player Output Component Decomposition V1.

Key landscape disposition:
`INDIVIDUAL_PLAYER_CORE_CONFIRMED__LANDSCAPE_TRANSMISSION_INCOMPLETE`

---

## 6. RB RESEARCH — CURRENT STATUS

Preserve:
- RB receiving-room share retrospective improvement is real:
  - targets MAE improvement **7.60%**
  - receptions **3.83%**
  - rec yards **2.89%**
  - rush+rec **1.66%**
- RB receiving-room Week-5 prospective lock is frozen.
- RB carry/snap Week-5 prospective lock is frozen.
- M96 retrospective stop remains authoritative.
- do **not** back-apply the Week-5 carry/snap shadow to W1-W4.
- do **not** reopen failed generic RB mean/width families or M95A/M95B role x defense.
- RB opportunity is still a more important unresolved layer than generic efficiency.

The Week-5 locks remained outcome-blind when research paused. Do not grade them retrospectively unless their prospective protocol says the evaluation window has matured.

---

## 7. WR / TE RESEARCH — HISTORICAL OOS TRAJECTORY SUPPORT COMPLETED

Current research branch:
- `research-individual-opportunity-roadmap-2026-10-08`
- draft PR **#672**
- current branch head/result-doc commit `af913a8bab0b7cbd68d835eb0e22709b46d14e67`

The original final-fit source overlap problem was correctly caught:
- deployed WR-R15 / TE-R5P final assets include overlapping training seasons;
- those final-fit vectors cannot be called independent OOS historical validation.

Exact fold authorities were then recovered before scoring.

Parent recovery:
- run **37782189538** SUCCESS
- artifact **11552447166**
- digest `sha256:94a4bee7011877b1d7a24708d5ba3e86ccc5d29b427bea650af574fc1f09fa24`
- WR original OOS fold bundle rearchived;
- TE-R5P exact original fold authority mechanically rehydrated from pinned TE-R3/R4 artifacts;
- original published metrics reproduced exactly;
- no provider refit;
- no sportsbook.

Frozen historical OOS-fold integration:
- run **37783801061** SUCCESS
- scientific head `4566fb9d1c6f2523487873d16b9ff76470b07b50`
- artifact **11552284671**
- digest `sha256:6d4e9a2a87fc2b364a847e6328d201f42fa4bf63b10f87416001c4fec8de03d2`
- strict repo audit PASS
- result doc `docs/research/PLAYER_TARGET_SHARE_TRAJECTORY_HISTORICAL_OOS_FOLD_INTEGRATION_V1_RESULT.md`

Disposition:
`HISTORICAL_OOS_FOLD_TRAJECTORY_INTEGRATION_SUPPORTED`

Primary 2024 W5+ common OOS-fold population:
- 2,463 WR+TE rows
- 328 players
- target MAE 1.95101097 -> 1.94604884 (**+0.254%**)
- rec-yard MAE 20.56999067 -> 20.51221057 (**+0.281%**)
- room-share MAE 0.06479171 -> 0.06446356 (**+0.506%**)
- WR target +0.329%
- WR rec yards +0.349%
- TE target +0.088%
- TE rec yards +0.105%
- bootstrap P(target AE improvement > 0) = **0.9466**
- 95% percentile interval still crosses zero
- all predeclared disposition gates PASS

Secondary TE 2025:
- target +0.390%
- rec yards +0.014%
- room share +1.152%

Interpretation:
- real but small support;
- no coefficient/window/cap retuning;
- no automatic promotion;
- exact four-prior-same-season-game eligibility stays frozen;
- no W1-W4 eligibility rescue;
- Week-5+ prospective lock remains the primary forward authority.

---

## 8. WR / TE / MATCHUP CLOSED OR SOURCE-BLOCKED LANES

Do not reopen without genuinely new information:

- universal symmetric target-depth distribution transform: failed W1-W4 CRPS; unpromoted;
- route participation / YPRR historical testing: source-parity blocked;
- WR-R3 combined calibration: `NO_ACTIONABLE_WR_R3_COMBINED_CALIBRATION`;
- Coverage-v2 team man/zone: near-null;
- WR-CB historical assignment: source blocked; user cannot justify a paid WR/CB source;
- generic game-script / Vegas state: no actionable pregame signal;
- all three simple Football Matchup Transmission V1 integration formulas: failed closed;
- generic “bad defense vs position => boost player” formulas: do not reintroduce without role-conditional replicated evidence.

Important nuance:
The user's football intuition that defenses and player roles matter is **not rejected**. What failed were simple generic transmission formulas. Any future matchup work must be conditional on actual player usage/opportunity and must demonstrate incremental replicated predictive value.

---

## 9. QB RESEARCH — EXACT FRONTIER WHEN GAME-DAY INTERRUPTED US

This is where research was moving immediately before Week-5 production urgency took over.

Do not reopen generic QB mean/YPA. M89/M90 mean remains frozen/promoted. QB C2 distribution remains promoted.

Existing diagnostic chain already localizes residual QB opportunity error upstream:

`TEAM PASS OPPORTUNITY -> PASS-OPPORTUNITY RATE -> WITHIN-STATE PASS PROPENSITY -> FIRST-DOWN PLAY SELECTION`

Previously established:
- team pass opportunity/dropback volume is the primary QB opportunity bottleneck;
- fixed production pass-opportunity rate / play-selection uncertainty is more important than generic efficiency;
- shared QB/receiver residual linkage is real;
- first-down within-state play choice is the surviving mechanism;
- field position did not explain it;
- score-state occupancy/reference-level did not explain it;
- simple pass-vs-run EPA/success economics failed;
- schedule/rest D1 improved some raw opportunity quantities but worsened downstream pass-yard behavior / failed confirmation;
- PBP D2 penalty/fourth-down candidate had no independent survivor;
- generic same-data attempt-volume repackaging is closed.

The anti-retest inventory had just begun when game-day execution interrupted us. The safe preserved conclusion is:

**Do not build another generic QB attempt feature from already-used box/PBP state. Any new QB opportunity candidate must contain genuinely new pregame information or attack calibration/uncertainty rather than recycling same-data mean features.**

Important prior QB documents:
- `docs/research/overnight/QB_GAP_FINDINGS.md`
- `docs/handoffs/NFL_HANDOFF_2026-09-11_QB_WR_SHARED_OPPORTUNITY_CURRENT.md`

Known open/unfinished possibilities from that chain:
1. finish the already-frozen QB PD3 internal-disagreement/reliability lane if still scientifically relevant;
2. treat unresolved first-down choice as variance/calibration uncertainty rather than another mean-feature hunt;
3. only use coordinator/team choice persistence if it is genuinely distinct from previously closed state decompositions;
4. public pregame intent language was the only genuinely new mean-adjacent information family left, but manual collection was explicitly paused as operationally unacceptable.

Public-intent rule:
- do not resume a 500+ team-week manual crawl;
- if revisited, freeze an automation-first V1B source-validation protocol;
- hard time cap roughly one working day / ~6–10 serious research hours;
- no odds, no outcomes, no target residuals in source collection;
- close it if automation/source quality does not qualify.

### Exact research resume move

Before opening any new QB mechanism:
1. reconcile the full completed QB opportunity chain and stop-rules;
2. identify whether PD3/internal disagreement was actually completed elsewhere;
3. verify whether any genuinely new pregame team pass-intent source exists;
4. if not, prefer a bounded uncertainty/calibration study over another mean-feature search;
5. freeze contract/cohort/gates before scoring;
6. keep sportsbook completely downstream.

Do not restart from “how can we predict QB attempts better?” as if prior work did not exist.

---

## 10. POSITION READINESS SUMMARY

### QB
- promoted mean/distribution stack exists;
- generic mean hunt closed;
- remaining research = team pass-opportunity / intent / uncertainty seam;
- this is the **next unresolved opportunity frontier** after WR/TE trajectory support.

### RB
- receiving-room retrospective support exists;
- Week-5 receiving + carry/snap locks frozen;
- M96 stop preserved;
- further player allocation work must respect those locks and avoid generic role/defense retests.

### WR
- M38 + WR-R15 production entitlement exists;
- target-share trajectory supported historically OOS-fold;
- Week-5+ prospective trajectory lock frozen;
- generic efficiency/matchup rescue closed/source-blocked.

### TE
- TE-R5P production entitlement exists;
- target-share trajectory supported historically OOS-fold;
- Week-5+ prospective trajectory lock frozen;
- same generic matchup/source-block rules as WR.

---

## 11. CURRENT OPEN / STALE GITHUB STATE

### PR #672
- draft research PR
- branch `research-individual-opportunity-roadmap-2026-10-08`
- current preserved head `af913a8bab0b7cbd68d835eb0e22709b46d14e67`
- contains the individual-opportunity roadmap and supported WR/TE OOS-fold trajectory integration
- **do not merge/promote automatically**

### PR #679
- stale/superseded pre-T75 production PR
- current main already contains the intended fix and has green canonical proof
- do not merge blindly

### Issue #673
The Week-5 30-team bye / QB C2 / preserved replay blocker is effectively closed by current main and run 37860592615. Preserve it as lineage, not an active science lane.

### Issue #535
Newest continuity comments include:
- WR/TE OOS-fold trajectory support;
- canonical Week-5 preserved-odds production recovery;
- pre-kickoff DAL-TB core ticket.

Read newest comments only; do not recursively reread the issue unless needed.

---

## 12. WHAT NOT TO DO IN THE NEXT CHAT

Do not:
- ask the user to re-explain the project;
- restart research from scratch;
- rerun the all-player W1-W4 replay;
- rerun Player Landscape Transmission Audit;
- rerun Player Output Component Decomposition;
- rerun the completed WR/TE OOS-fold trajectory integration;
- weaken the four-prior-game target-share trajectory rule;
- grade Week-5 prospective locks prematurely;
- reopen generic QB mean/YPA;
- reopen schedule/rest D1 or PBP D2 unchanged;
- reopen M96, M95A/M95B, WR-R3, Coverage-v2, WR-CB, route/YPRR parity, generic Vegas/game-script, failed FMT formulas;
- use sportsbook lines to select starters or construct upstream football features;
- spend another OddsAPI credit without explicit authorization;
- call replayed odds “current”;
- merge PR #679 blindly;
- claim an Action is active without live verification.

---

## 13. EXACT NEXT-CHAT START SEQUENCE

Read in this order:

1. `AGENTS.md`
2. **ONLY the newest TOP checkpoint** in `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. this file:
   `docs/handoffs/NFL_HANDOFF_2026-10-08_WEEK5_PRODUCTION_RECOVERED_RESEARCH_FRONTIER_CURRENT.md`
4. newest Issue #535 continuity comments
5. live `main`, open PRs, relevant branches and Actions
6. PR #672 current head/result only if resuming research

Then:

### Immediate production check
- verify current main has not moved materially from the documented production authority;
- verify canonical Week-5 run 37860592615 remains SUCCESS and artifact available;
- do **not** repull OddsAPI.

### Immediate research continuation
Resume the **individual opportunity allocation** program exactly where it paused:
- WR/TE trajectory historical OOS-fold support is completed;
- next unresolved seam is QB team pass-opportunity / intent / uncertainty;
- first perform anti-retest reconciliation;
- choose only a genuinely new pregame information source or a bounded calibration/uncertainty test;
- freeze before scoring.

If the user instead asks about the DAL-TB game, board or ticket, treat the canonical preserved replay artifact as the Week-5 authority and keep research conclusions separate from live betting discussion.

---

## 14. ONE-SENTENCE CONTINUITY STATE

**Production is recovered and canonical; Week-5 odds are preserved without a second paid pull; the research program remains opportunity-dominant, WR/TE target-share trajectory has small but real OOS-fold support, and the next scientific frontier is the unresolved QB team-pass-opportunity / pregame-intent / uncertainty seam without reopening closed same-data mean research.**
