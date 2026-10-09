# NFL HANDOFF — 2026-10-08 — WEEK 5 CANONICAL REPLAY + RESEARCH FRONTIER CURRENT

This is the authoritative deep handoff for the next ChatGPT session.

Repository: `dkaps6/imtiredofthis`

User requirement: **GitHub is canonical. Do not ask the user to re-explain anything. Do not restart completed research. Do not recursively reread old handoffs. Do not spend another OddsAPI credit without explicit authorization.**

Read order for the next session:
1. `AGENTS.md`
2. ONLY the newest top checkpoint in `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. this file
4. targeted live GitHub state: `main`, open PRs, Issue #535 newest comment, Issue #673 newest comment, exact-head Actions
5. only then the specific PR #672 research docs named below if resuming research

---

## 1. CURRENT CANONICAL PRODUCTION STATE — WEEK 5 RECOVERY IS GREEN

### Canonical main

At this handoff:

- `main = 24704e5f1ef872ec86c36446d2c73f6d377c824c`
- commit message:
  `P0: preserve pinned pre-T75 game state during paid replay`

The only production entrypoint remains:

`.github/workflows/full-slate.yml`

Do not create or use a sidecar production pipeline.

### Final canonical preserved-odds Full Slate

The Week 5 canonical replay is **COMPLETE and SUCCESSFUL**:

- workflow: `Full Slate`
- run: `37860592615`
- event: `workflow_dispatch`
- head: `24704e5f1ef872ec86c36446d2c73f6d377c824c`
- conclusion: **SUCCESS**
- output artifact: `11585973062`
- artifact name: `run_37860592615`
- digest:
  `sha256:bf52aa4bc30bf6e8cc6029a0a86c0f33cdfaa09f587c38a138f23cef6b7436f4`

The canonical run completed:
- current-output availability seams;
- pinned sportsbook restore;
- sportsbook availability resolution;
- live-boundary compaction;
- opponent map;
- player identity audit;
- Full Slate data-quality classification;
- exact bookmaker-offer assembly;
- deterministic metrics;
- certified football-first pricing;
- final-board verified quarantines;
- immutable preserved-sportsbook proof;
- strict repository audits;
- master workbook build;
- artifact upload.

No failed step.

### ZERO additional OddsAPI acquisition

The user explicitly did **not** want another paid OddsAPI pull after the original Week 5 run had already spent the credit.

That requirement was satisfied.

Final replay:
- `FETCH_LIVE_ODDS=false`
- no new sportsbook acquisition;
- no second OddsAPI charge;
- sportsbook data came only from the exact preserved paid source.

Original paid source:
- run: `37852811339`
- source artifact: `11582178322`
- source artifact name: `run_37852811339`
- digest:
  `sha256:4948e08003fcab0f519b35bd4b23d8c895554e4faced993238575acdc5d2e765`

Do **not** fetch Week 5 odds again unless the user explicitly authorizes a new paid call.

### Final workbook / board provenance

Final workbook:
- `outputs/NFL_BETTING_MODEL_MASTER.xlsx`
- pricing status: **PRESERVED_REPLAY**
- priced offer rows: **1,472**
- player-market rows: **754**
- unresolved-position rows: **0**
- sportsbook downstream-only: **true**

Important semantic rule:
- preserved replay prices are **not labeled CURRENT**;
- `Bettable Now` is suppressed for replayed prices;
- the workbook explicitly distinguishes the preserved sportsbook snapshot from a fresh same-run fetch.

Final `outputs/props_priced_clean.csv` still contains **254 DAL-TB priced rows**.

The user may still inspect the board, but never describe the replayed prices as fresh/current after the original acquisition time.

---

## 2. WHY WEEK 5 FAILED INITIALLY, AND EXACTLY WHAT WAS FIXED

### A. Bye-week QB C2 32-team blocker

Week 5 has 30 active scheduled teams because two teams are on bye.

Initial Full Slate production failure:
- run `37778827288`
- QB C2 state builder expected 32 active team rows after schedule filtering;
- Week 5 legitimately had 30.

This was an operations contract problem, not a football-model-science failure.

P0 remediation:
- Issue #673
- hotfix PR #674
- merged main at:
  `66c22026880178e60d8011ea2400127249be83ac`

The fix:
- made QB C2 active schedule context schedule-aware;
- preserved protected availability seams;
- did not create fake bye opponents;
- did not weaken the football-only source contract;
- accepted complete active matchup sets rather than hard-coded league count.

Focused hotfix run:
- `37852225057` SUCCESS

Canonical no-odds validation:
- `37852305029` SUCCESS

### B. One paid Week 5 acquisition was then launched

The original paid run:
- `37852811339`

It successfully reached the paid sportsbook acquisition and preserved the exact odds snapshot, but the downstream production run failed later.

Because the paid artifact already existed, every recovery attempt after that was required to reuse the preserved snapshot.

### C. Canonical preserved-paid-odds replay was added to Full Slate

PR #678 moved replay capability into the actual production workflow instead of leaving it in a sidecar recovery workflow.

Merged production commit before the later T-75 repair:
- `d36daa45db38f667bdb6566ed13e835df391227f`

Key permanent behavior:
- `fetch_live_odds=true` and preserved replay are mutually exclusive;
- replay requires exact source run ID, artifact ID and digest;
- artifact metadata is bound to the exact workflow run;
- immutable sportsbook boundary is verified;
- no OddsAPI code path is used in replay mode;
- football stack is rebuilt normally;
- replay workbook gets explicit non-current provenance.

### D. Week 5 data-quality / identity / publication repairs

The production/replay work also exposed and resolved these operational seams:

1. **Mitch Tinsley / Mitchell Tinsley**
   - current Ourlads identity vs historical nflverse identity;
   - stable historical GSIS alias anchored;
   - identity metadata only, no invented football usage.

2. **NFL.com injury scope**
   - bye-week-aware active-team injury scope;
   - official NFL.com context used for priced Full Slate current injury certification;
   - 30 scheduled teams certified;
   - no generic assumption that “no row = healthy”.

3. **Tyler Huntley BAL Week 5**
   - sportsbook posted pass-yard offers while football-only depth authority still had him as QB2;
   - sportsbook cannot choose or override the football starter;
   - Huntley Week-5 props are quarantined at the **final-board publication stage**;
   - BAL's other markets and football simulation remain intact.

4. **Preserved non-core stale prop reconciliation**
   - a preserved anytime-TD entity (Camden Brown/DAL) was no longer in refreshed current PlayerForm;
   - only non-core replay rows may be quarantined;
   - strict yardage/reception markets remain fail-closed if a player is absent from current football authority;
   - lines/odds themselves are not rewritten.

### E. T-75 timing problem and final P0 repair

A later canonical replay attempt:
- run `37858437447`

failed because time had moved forward after the paid acquisition.

Observed DAL-TB case:
- original paid snapshot acquired **112.532 minutes before kickoff**;
- replay later ran inside T-75;
- current official inactive sections were then required but unavailable;
- current availability correctly withheld DAL and TB;
- preserved sportsbook rows still contained DAL-TB;
- replay therefore failed closed because PlayerForm no longer contained those teams.

The correct fix was **not to delete DAL-TB**.

Final main P0:
- `24704e5f1ef872ec86c36446d2c73f6d377c824c`

Canonical replay can preserve the exact acquisition-time game eligibility only when:
- source game was production-eligible;
- source was at least T-75;
- current game is still pre-kickoff;
- event identity and kickoff are unchanged;
- current withholding is only the timing-driven required-missing-inactives condition;
- current player-level definitive-unavailable facts remain authoritative.

Final successful replay evidence for DAL-TB:
- source as-of:
  `2026-10-08T22:22:28.077633Z`
- source minutes to kickoff: **112.532**
- source certification state: `NOT_YET_REQUIRED`
- restored games: **1**
- restored teams: `[DAL, TB]`
- replay state:
  `PRESERVED_PAID_ACQUISITION_LOCK`
- definitive-unavailable players resurrected: **0**
- sportsbook inputs used for game eligibility: **0**

This is now the canonical solution to the Week-5 bye/T-75/preserved-paid-odds blocker.

### F. Current stale/open PR warning

At handoff, GitHub still showed:

- PR #679:
  `P0: preserve tonight's pre-T75 football lock in canonical odds replay`
- PR #677:
  `Ops: preserve successful Week 5 priced artifact and scoped Huntley quarantine`

Do **not** blindly merge either.

Main already contains the final P0 replay-lock implementation and the canonical replay has succeeded on `24704e5...`.

PR #679's open head is stale relative to main and has review history that was superseded by the final main implementation.

PR #677 is continuity from an earlier incomplete stage and explicitly recorded data-quality limitations that were later repaired. Reconcile/close as obsolete rather than reintroducing old state.

Always re-query before acting because another chat may have already cleaned them up.

---

## 3. PRODUCTION TIMELINE — DO NOT REPEAT THESE RUNS

Useful exact lineage:

1. `37778827288` — Full Slate failed at QB C2 hard-coded 32-team Week-5 context.
2. PR #674 / `66c220...` — bye-week C2 repair.
3. `37852305029` — canonical no-odds Full Slate SUCCESS.
4. `37852811339` — one explicitly authorized paid Week-5 Full Slate; sportsbook acquired; later downstream failure; preserved paid artifact `11582178322`.
5. `37854558239` — successful offline recovery using preserved odds; important evidence, but not the final canonical entrypoint.
6. PR #678 / `d36daa45...` — canonical replay mode added to `full-slate.yml`.
7. `37858050907` — canonical no-odds Full Slate SUCCESS on the #678 production merge.
8. `37858437447` — canonical preserved replay failed at the later T-75 timing/availability seam.
9. `24704e5...` — acquisition-time pre-T75 lock preservation merged to main.
10. `37860224986` — automatic main push Full Slate SUCCESS.
11. `37860592615` — final canonical preserved-paid-odds Full Slate SUCCESS, zero new OddsAPI, artifact `11585973062`.

Do not rerun these merely to “confirm” them.

---

# 4. RESEARCH STATE BEFORE WEEK-5 PRODUCTION TOOK OVER

The user explicitly wants the next chat to remember the model-development work that was in progress **before we paused to run Week 5**.

The scientific headline remains:

`OPPORTUNITY_DOMINANT_ACROSS_ALL_PRIMARY_MARKETS__INDIVIDUAL_ROLE_SHARE_ALLOCATION_NEXT`

Research draft PR:
- PR #672
- branch:
  `research-individual-opportunity-roadmap-2026-10-08`
- current research result commit:
  `af913a8bab0b7cbd68d835eb0e22709b46d14e67`
- research-only; no automatic promotion.

## 4.1 Completed all-player landscape work

Do not rerun:
- 2026 W1-W4 all-player/all-position replay;
- Player Landscape Transmission Audit;
- Player Output Component Decomposition;
- historical availability parity audit;
- share coverage/residual audit.

The model is genuinely player-specific; it is not just positional averages with player names attached.

But the largest remaining cross-position miss is getting the player's **pregame opportunity** right.

### Player Output Component Decomposition V1

Authority:
- run `37777319154` SUCCESS
- head:
  `cb1e511adcb58e40fc41d0949306083a304e83dd`
- artifact:
  `11550446909`
- digest:
  `sha256:7acfc453258a49f3aadf0d0ea386b13827eb715f1dfb055b96505f733320d204`
- result:
  `docs/research/PLAYER_OUTPUT_COMPONENT_DECOMPOSITION_V1_RESULT.md`

Opportunity oracle removed more MAE than efficiency oracle in all eight cells:

- QB pass yards: **23.9% vs 12.7%**
- RB rush yards: **45.1% vs 7.2%**
- RB rec yards: **45.1% vs 27.5%**
- RB receptions: **57.2% vs -2.4%**
- WR rec yards: **42.8% vs 28.2%**
- WR receptions: **54.4% vs 11.6%**
- TE rec yards: **55.0% vs 23.5%**
- TE receptions: **68.3% vs 8.1%**

Do not pivot back to a generic “improve YPA/YPC/YPT averages” program.

## 4.2 Corrected opportunity allocation diagnosis

Corrected ACT-only volume-vs-share authority:
- run `37694474836`

The key decomposition:

- **QB:** team opportunity volume dominates; actual team-volume oracle removes about **60.3%** of attempt MAE.
- **RB carries:** player share dominates, about **64.5%** oracle MAE removed.
- **RB targets:** player share dominates, about **76.5%**.
- **WR targets:** player share dominates, about **68.6%**.
- **TE targets:** player share dominates, about **77.1%**.

This is why the next work was framed as:
1. participation / active-role probability;
2. team opportunity volume;
3. room share / player allocation.

Do not collapse those layers.

---

# 5. POSITION-BY-POSITION RESEARCH STATE

## QB — unresolved team-pass opportunity seam, but generic searches are closed

Broad QB mean research remains frozen after M89/M90.

The next QB question is **not** generic YPA or “better average passing yards”.

Existing opportunity-chain evidence says upstream `TEAM_PASS_OPPORTUNITY` is the dominant QB chain error:
- about **68.29%** pooled absolute chain mass;
- about **76.24%** dominant-row rate;
- even stronger in the largest misses.

Anti-retest work already established:
- generic QB mean/YPA reopening is not authorized;
- same-data team pass-volume repackaging is exhausted;
- generic Vegas/game-script is closed;
- schedule/rest D1 improved raw pass opportunity / attempts but **worsened passing yards** once synthesis interacted with it, consistent with double-counting; no confirmation;
- PBP D2 penalty/fourth-down state produced no independent survivor.

Therefore the legal QB path is a **source gate**:
- only reopen the team-pass-volume lane if a materially new, pregame-valid game-plan / intent / starter / pass-volume observable with as-of provenance exists;
- otherwise record the seam as source-gated and stop;
- protect M89/M90 and QB C2.

Do not “just build another attempts model” from the same inputs.

## RB receiving — real retrospective improvement, prospective lock frozen

Completed W1-W4 within-RB-room receiving allocation improvement:
- targets MAE: **7.60% better**
- receptions MAE: **3.83% better**
- receiving-yards MAE: **2.89% better**
- rush+rec MAE: **1.66% better**
- rushing unchanged.

Authority:
- RB Receiving Room Share impact run:
  `37703522415`
- Week-5 receiving lock:
  `37705464974`

Preserve:
- total RB+FB receiving target mass;
- R26 vacancy ordering;
- R22 tail logic;
- team pass volume;
- fixed per-target efficiency.

Do not invent a second RB receiving proxy.

## RB carries — prospective carry/snap lock frozen

Week-5 carry/snap allocation lock:
- run `37560824479`
- frozen 50/50 recent carry-share + snap-fraction allocation.

Do not:
- back-apply it retrospectively to W1-W4;
- reopen M96;
- choose post-hoc subgroups after outcomes.

M96 retrospective stop remains authoritative.

## WR / TE — Target Share Trajectory V1 supported, but still prospective

Base signal:
- run `37638269235`

Week-5 immutable prospective lock:
- run `37654382316`
- artifact `11497153776`
- row digest:
  `sha256:afbfd7f360c50fcd4850c0967be40f9a333da1bd5f835cdc676c2e88d777c1f3`

Exact eligibility cannot be weakened:
- at least four prior same-season same-team target-team games;
- two most recent vs all earlier completed games;
- therefore **zero eligible 2026 W1-W4 rows**.

Do not rewrite this rule after seeing outcomes.

### Historical exact OOS-fold integration is now complete

Disposition:

`HISTORICAL_OOS_FOLD_TRAJECTORY_INTEGRATION_SUPPORTED`

Final authority:
- run `37783801061` SUCCESS
- scientific head:
  `4566fb9d1c6f2523487873d16b9ff76470b07b50`
- artifact:
  `11552284671`
- digest:
  `sha256:6d4e9a2a87fc2b364a847e6328d201f42fa4bf63b10f87416001c4fec8de03d2`
- strict repo audit PASS
- result:
  `docs/research/PLAYER_TARGET_SHARE_TRAJECTORY_HISTORICAL_OOS_FOLD_INTEGRATION_V1_RESULT.md`

Recovered fold authority:
- run `37782189538` SUCCESS
- artifact `11552447166`
- digest:
  `sha256:94a4bee7011877b1d7a24708d5ba3e86ccc5d29b427bea650af574fc1f09fa24`
- disposition:
  `WR_TE_FOLD_AUTHORITIES_RECOVERED_EXACT`

Primary 2024 W5+ common OOS fold:
- 2,463 WR+TE rows
- 328 players

Pooled:
- target MAE:
  **1.95101097 -> 1.94604884**
  (**+0.254%**)
- receiving-yards MAE:
  **20.56999067 -> 20.51221057**
  (**+0.281%**)
- room-share MAE:
  **0.06479171 -> 0.06446356**
  (**+0.506%**)

WR:
- target MAE **+0.329%**
- receiving yards **+0.349%**

TE:
- target MAE **+0.088%**
- receiving yards **+0.105%**

Bootstrap:
- P(target AE improvement > 0) = **0.9466**
- 95% percentile interval crosses zero.

Secondary TE 2025:
- target MAE **+0.390%**
- receiving-yards MAE **+0.014%**
- room-share MAE **+1.152%**

Interpretation:
- real but small support;
- no coefficient/window/cap tuning;
- no automatic production promotion;
- prospective confirmation remains necessary.

---

# 6. CLOSED / SOURCE-BLOCKED / DO-NOT-RETEST LANES

Preserve all of these stopping rules.

### Target-depth universal transform
The universal symmetric player target-depth distribution transform did not improve W1-W4 CRPS.

Unpromoted. Do not rescue with post-hoc tuning.

### Route participation / YPRR
Historical source parity is blocked.

Do not fabricate route participation from nflverse participation labels.

### WR-R3 combined calibration
Already built/run.

Disposition:
`NO_ACTIONABLE_WR_R3_COMBINED_CALIBRATION`

Closed.

### Coverage / WR-CB
- team Coverage-v2 man/zone information is near-null;
- direct historical WR-CB assignment remains source blocked;
- user cannot rely on a paid WR-CB source;
- do not hallucinate assignment data.

### Generic game-script / Vegas
No actionable pregame state confirmed.

Closed unless genuinely new independent state exists.

### Football Matchup Transmission simple integrations
The user's football concern was valid: Phase A showed meaningful football matchup state can be weakly transmitted or dropped by the generic stack.

However the simple integration candidates tested afterward failed closed.

Do not respond by adding a generic “bad vs position -> boost player” multiplier.

The science currently says:
- player/team/role differences matter;
- matchup context still matters conceptually;
- but **opportunity allocation is the dominant measurable error**;
- efficiency/matchup should remain secondary until the relevant opportunity seam is resolved.

### M95A / M95B
Generic RB role x defense families remain closed.

### M96
Retrospective RB width/mean reopening remains stopped.

---

# 7. PROSPECTIVE WEEK-5 LOCKS — DO NOT GRADE PREMATURELY

As of the research checkpoint before game-day production took over, the Week-5 prospective mechanisms were still outcome-blind.

Preserve that state unless the user explicitly moves into a postgame grading/postmortem phase.

Do not use partial Thursday outcomes to retune:
- RB receiving-room lock;
- RB carry/snap lock;
- WR/TE target-share trajectory lock.

The original confirmation contracts remain intact.

---

# 8. EXACT NEXT RESEARCH ACTION AFTER PRODUCTION URGENCY

When the user is ready to return from Week-5 execution to model development:

1. Start from PR #672, not a new research branch unless a new source/mechanism earns one.
2. Do **not** rerun the all-player W1-W4 replay, landscape audit, opportunity decomposition, historical availability parity, or WR/TE OOS-fold trajectory integration.
3. Finish the bounded QB opportunity anti-retest/source gate:
   - determine whether a genuinely new pregame team-pass-intent / pass-volume input exists;
   - if not, formally mark the QB opportunity seam source-gated and stop it;
   - do not fit another same-data attempts model.
4. Keep RB receiving and carry/snap mechanisms under their existing prospective locks.
5. Keep WR/TE trajectory under the existing prospective lock; historical support is already established.
6. Only open a new efficiency/matchup lane if it is incremental to role/opportunity and survives the existing anti-retest map.
7. Do not use sportsbook lines as upstream football predictors.
8. Freeze any exact new mechanism before scoring it.

Research reference files on PR #672:
- `docs/research/INDIVIDUAL_OPPORTUNITY_ALLOCATION_ROADMAP_V1.md`
- `docs/research/PLAYER_TARGET_SHARE_TRAJECTORY_HISTORICAL_INTEGRATION_V1_GATE.md`
- `docs/research/PLAYER_TARGET_SHARE_TRAJECTORY_HISTORICAL_OOS_FOLD_INTEGRATION_V1_RESULT.md`
- `docs/research/PLAYER_TARGET_SHARE_TRAJECTORY_HISTORICAL_FOLD_PARENT_INVENTORY_V1.md`

---

# 9. NEXT-CHAT EXECUTION RULES

The next session should feel like the same operator continuing.

Do not ask the user:
- what repo;
- what happened this week;
- whether live odds were already pulled;
- what failed;
- what the research priority was;
- whether WR/TE trajectory was tested;
- whether RB allocation work already exists.

Those answers are in this handoff.

At the start of the next chat:
1. verify live `main`;
2. verify open PRs;
3. verify Issue #535 newest continuity comment;
4. verify Issue #673 newest operational comment;
5. verify run `37860592615` and its artifact still exist if relevant;
6. do not claim a run active unless Actions says queued/in_progress;
7. do not spend OddsAPI without explicit authorization.

If the user asks for the Week-5 board:
- use canonical artifact `11585973062`;
- remember its sportsbook prices are preserved/replayed, not current;
- do not label it Bettable Now from stale prices;
- no new OddsAPI pull unless authorized.

If the user says “keep going” on research:
- resume the QB opportunity/source-gate frontier from PR #672;
- preserve all anti-retest rules above.

---

# 10. ONE-SCREEN STATE

Production:
- Week-5 canonical preserved-odds replay: **GREEN**
- main: `24704e5...`
- final run: `37860592615`
- final artifact: `11585973062`
- zero second OddsAPI call
- 1,472 priced offers / 754 player-markets
- DAL-TB retained via pre-T75 acquisition lock
- replay pricing explicitly non-current.

Research:
- headline:
  `OPPORTUNITY_DOMINANT_ACROSS_ALL_PRIMARY_MARKETS__INDIVIDUAL_ROLE_SHARE_ALLOCATION_NEXT`
- QB: team opportunity is the main unresolved seam, but recycled same-data attempts/means are closed; source gate only.
- RB receiving: retrospective improvement confirmed + Week-5 lock.
- RB carries: carry/snap Week-5 lock; M96 stop.
- WR/TE: Target Share Trajectory V1 has small real OOS-fold support + prospective Week-5 lock.
- generic target-depth transform failed.
- routes/YPRR source blocked.
- WR-R3 closed.
- coverage near-null; WR-CB assignment blocked.
- generic Vegas/game-script closed.
- simple matchup-transmission integrations closed.
- no automatic promotion from historical trajectory support.
- no premature Week-5 grading.

That is the exact state to continue from.
