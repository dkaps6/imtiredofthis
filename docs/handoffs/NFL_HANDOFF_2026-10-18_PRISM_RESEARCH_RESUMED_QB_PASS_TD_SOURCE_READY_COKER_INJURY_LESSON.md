# NFL HANDOFF — 2026-10-18 — RESEARCH RESUMED / QB PASS-TD SOURCE READY / COKER INJURY-STATE LESSON

**Status:** CURRENT CROSS-CHAT CONTINUITY AUTHORITY  
**Repository:** `dkaps6/imtiredofthis`  
**Model:** **PRISM — Player Role & Individual Situation Model**  
**Production branch:** `main`  
**Long-running research branch / PR:** `research-individual-opportunity-roadmap-2026-10-08` / PR #672  
**Primary issue:** #535

This handoff supersedes the earlier same-day locked-bets handoff wherever research has advanced beyond it. The user’s Sunday bets are locked. The next chat must return immediately to PRISM research and continue from the newest completed science below.

---

## 1. IMMEDIATE MODE

**Betting work is CLOSED unless the user explicitly reopens it.**

Do not:
- rebuild today’s slips;
- reprice already-entered tickets;
- spend a paid OddsAPI credit;
- use live outcomes to tune research.

The user explicitly wants the next chat to continue the **research lane**.

---

## 2. NON-NEGOTIABLE PRISM OBJECT

PRISM is an **individual-player projection model**.

Binding phrase:

> **The player should have the rules. The position should not have the player.**

Canonical reasoning order:

`PLAYER IDENTITY/HISTORY -> CURRENT ROLE/STATE/TRAJECTORY -> TEAMMATE COMPETITION -> FINITE OPPORTUNITY ALLOCATION -> OPPORTUNITY TYPE/QUALITY -> TEAM/GAME ENVIRONMENT -> PLAYER-CONDITIONED OPPONENT MATCHUP -> INDIVIDUAL EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`.

Position-average MAE/RMSE is downstream QA only.

Carries, targets, receptions, receiving yards, rushing yards, passing yards, passing TDs, rushing TDs and anytime-TD questions are distinct market-specific football objects.

The Dobbins principle remains binding:
- football case for the named player first/alongside PRISM;
- exact current role and workload;
- teammate competition;
- opportunity type/quality;
- scoring-area ownership where relevant;
- team/game environment;
- player-specific conversion;
- then the exact market.

---

## 3. BETTING/TICKET LESSONS — PRESERVE, BUT DO NOT RESUME

Canonical construction doc:
`docs/research/MODEL_BACKED_ALT_LINE_PARLAY_CONSTRUCTION_V1.md`.

Binding order:
`FOOTBALL PROJECTION -> RELIABILITY/TRUST -> MARKET COMPARISON -> APPROVED PLAY -> ALT-LINE/CORRELATION/TICKET CONSTRUCTION`.

User clarified:
- positive-side parlay legs should use actual sportsbook milestone props with a `+` sign when available, e.g. `175+ passing`, `20+ receiving`, `1+ reception`;
- approved UNDER opinions use the actual normal sportsbook under line;
- do not invent alternate-under products that are not actually offered;
- the exact line is the bet;
- one weak named-player read must not be allowed to kill every ticket;
- Watson concentration lesson remains binding.

Recent RB live re-evaluation remains preserved:
- Weeks 3-5 RB bets: **237-194, 55.0%, +19.20u**;
- RB receptions were the strongest recent RB lane;
- RB rushing yards were profitable over the recent sample;
- raw RB receiving yards remained weak;
- underlying RB point projections still retain material low-side bias, so profitable betting results do not mean the RB projection engine is solved.

---

## 4. IMPORTANT GAME-DAY DATA-QUALITY CORRECTION — JALEN COKER

The prior chat initially misread Jalen Coker’s 2026 game log as a role/production decline:
`138 -> 66 -> 18 -> 0`.

That interpretation is **invalid**.

Correct player-state interpretation:
- Week 3: Coker was already dealing with an **ankle injury**; he was limited Wednesday, did not practice Thursday, was limited Friday and entered the game questionable. His 18 receiving yards therefore cannot be treated as an ordinary healthy-role observation.
- Week 4: the apparent `0` is not a zero-yard performance; he **did not play**.
- He then did not produce a normal intervening healthy game before returning to the Week-6 practice state.
- Week 6: he returned to full participation and carried no final game designation.

Binding PRISM lesson:

> **Injury-affected games, partial games, and DNPs must not be treated as ordinary player trajectory evidence.**

Future player-state / trajectory work must distinguish at minimum:
- healthy full-participation observation;
- injury-limited / questionable observation;
- early-exit / partial-game observation where source-valid;
- DNP / inactive / did-not-play;
- return-from-injury state.

Do not infer declining role from a sequence that is actually driven by availability.

This is a direct example of why PRISM’s scientific unit must be the named player in his real current state rather than a blind recent-game average.

This lesson is important for future trajectory/source audits, but it does **not** authorize a post-hoc change to already-locked bets or an unfrozen model coefficient.

---

## 4A. GENERALIZED ANOMALY INVESTIGATION GATE — BINDING ACROSS FUTURE CHATS

The Coker error is not a one-player exception. It exposes a general PRISM reasoning requirement now also written into `AGENTS.md`:

> **Anomalies are questions, not trends.**

If a player's recent stat/usage sequence contains an abrupt discontinuity, zero/near-zero, missed game, partial game, injury-limited game, return from injury, or sudden opportunity change, the next operator must investigate the cause **before** calling it role growth/decline, trajectory, efficiency change or reliability evidence.

Reconcile source-valid injury/practice status, active/DNP state, snaps/routes/carries/targets, early exits, QB/starter changes, teammate availability and other plausible context. State-label abnormal observations before using them.

If the cause is unresolved, fail closed as `UNRESOLVED_CONTEXT`; do not use the observation as affirmative evidence for a bet or model adjustment.

This is binding for game-day football checks, postmortems, forensic replays, player-state research and future production feature design.

---

## 5. SIGNED-OPPORTUNITY FRONTIER — PRESERVED

### Stage-2 player-specific mechanism learning
- source run `39098772787`: SOURCE READY ALL;
- hierarchical future utility run `39099164620`:
  - RB rushing QUALIFIED;
  - RB receiving QUALIFIED;
  - TE receiving CLOSED;
- player-specific authority bridge run `39099474655`:
  - **RB receiving QUALIFIED**;
  - RB rushing CLOSED.

Do not rescue RB rushing.

### Stage-1 personalization
- source run `39099745136`: SOURCE READY ALL;
- hierarchical utility run `39099930673`:
  - RB QUALIFIED;
  - TE QUALIFIED;
  - QB CLOSED;
  - WR CLOSED;
- direct Stage-1 authority bridge run `39100106463`:
  - RB CLOSED;
  - TE CLOSED.

Do not rescue closed Stage-1 bridges.

### WR/TE signed component bias
Source audit:
- run `39100442923`;
- artifact `12126685270`;
- disposition `PRISM_WR_TE_PLAYER_SIGNED_COMPONENT_BIAS_SOURCE_READY_ALL`.

Signal:
- run `39109423266`;
- artifact `12128983909`;
- disposition `PRISM_WR_TE_PLAYER_SIGNED_COMPONENT_BIAS_SIGNAL_QUALIFIED_PARTIAL`.

Results:
- **WR opportunity signed bias QUALIFIED**;
- WR efficiency signed bias CLOSED;
- **TE opportunity signed bias QUALIFIED**;
- TE efficiency signed bias CLOSED.

Do not reopen closed efficiency lanes.

---

## 6. WR EXACT AUTHORITY RECOVERY — FAIL-CLOSED

WR-R15 exact authority recovery:
- run `39110742338`;
- job `117626156590`;
- disposition: **SOURCE RECOVERY BLOCKED / NO BRIDGE SCORE**.

Frozen expected prediction SHA:
`91d108ee7981935d78179257b204a0204e9a4f8061e6eba4d6810845ed057d05`.

Rebuilt SHA:
`7dce6b0e194d37ec3b814caf79942eea3b83fcda520b481788ec6b2b3f0778c3`.

Stop rules:
- do not replace expected digest;
- do not weaken byte identity;
- do not score WR topology/bridge from rebuilt authority;
- do not use this to retune the frozen bridge action;
- no efficiency rescue.

The upstream WR opportunity-bias science remains valid.

---

## 7. RESEARCH ADVANCED AFTER THE EARLIER LOCKED-BETS HANDOFF

The earlier handoff said to determine whether TE could execute independently. That question is now answered and must **not** be repeated.

### 7A. TE signed-opportunity bridge independent-path decision — BLOCKED

Research commit:
`56e8d9cb109f854a3bc4f0e33bdafe32646cfde7`.

Result:
`docs/research/PRISM_TE_SIGNED_OPPORTUNITY_BRIDGE_INDEPENDENT_PATH_DECISION_2026_10_18.md`.

Disposition:
`PRISM_TE_SIGNED_OPPORTUNITY_CONSERVED_BRIDGE_EXECUTION_BLOCKED_BY_FROZEN_COMBINED_SOURCE_GATE`.

Important nuance:
- TE-R5P source authority is independently preserved in run `36332720457` / artifact `10935999653`;
- TE opportunity-bias signal is live from run `39109423266`;
- however the **already-frozen shared action contract** requires combined WR/TE recovery disposition `PRISM_WR_TE_OPPORTUNITY_BIAS_CONSERVATION_BRIDGE_SOURCE_RECOVERED_READY_ALL`;
- failed WR byte recovery prevents that combined state;
- no separately frozen TE-only bridge contract exists.

Therefore:
- no TE bridge score is authorized;
- do not create a TE-only exception after seeing the WR failure;
- do not rewrite the frozen shared precondition.

### 7B. WR/TE opportunity-uncertainty historical integration — BLOCKED / PROSPECTIVE ONLY

Research commit:
`96d877c982cb96670d68ba033b4b01ab4f47c5bf`.

Result:
`docs/research/PRISM_WR_TE_PLAYER_OPPORTUNITY_UNCERTAINTY_INTEGRATION_SOURCE_DECISION_2026_10_18.md`.

Disposition:
`PRISM_WR_TE_PLAYER_OPPORTUNITY_UNCERTAINTY_HISTORICAL_INTEGRATION_SOURCE_BLOCKED__PROSPECTIVE_SHADOW_ONLY`.

Preserve:
- WR player opportunity uncertainty remains CONFIRMED;
- TE player opportunity uncertainty remains CONFIRMED.

But:
- exact historical production-order specialist distribution parent from run `34722725629` is no longer recoverable;
- later opportunity authority preserves means/state, not the exact specialist distribution arrays/SDs;
- generic right-tail artifact is a different parent and may not be substituted;
- deterministic reconstruction already failed frozen-parent parity in run `35944066618` with max `mc_proj` drift `14.020649281259615`.

Therefore:
- no historical full-stack uncertainty integration score is authorized;
- future prospective shadow remains legal only if frozen before the target slate;
- do not back-create a Week-7 lock after games began.

---

## 8. QB PASSING-TD GAP MOVED FROM BACKLOG TO ACTIVE SOURCE FRONTIER

The fantasy question in the prior chat exposed a real PRISM market gap:
- PRISM has strong QB passing-yards science;
- it does **not** yet have a dedicated certified QB passing-TD projection layer.

This gap has now advanced beyond a mere note.

Frozen source/anti-retest plan:
`docs/research/PRISM_QB_PASS_TD_MARKET_SPECIFIC_V1_SOURCE_PLAN.md`.

Freeze commit:
`af709a937f61c39240b73a1d6960ab47e60eb152`.

Source-audit implementation fix/current exact research head observed:
`f150d6078e72ddce27a298f049c9aaa18c0150f7`.

Canonical source run:
`39152687318`.

Job:
`117749381414` — **SUCCESS**.

Disposition:
`PRISM_QB_PASS_TD_MARKET_SPECIFIC_SOURCE_READY`.

Source audit facts:
- 2022: 19,455 pass attempts / 114 passers / 32 teams;
- 2023: 19,807 / 119 / 32;
- 2024: 19,224 / 110 / 32;
- 2025: 18,822 / 102 / 32;
- stable passer-ID coverage 100%;
- pass-TD coverage 100%;
- yardline coverage 100%;
- 2024 strict-prior feasibility: 604 target rows / 72 QBs with >=3 prior QB-games;
- 2025 strict-prior feasibility: 596 target rows / 77 QBs with >=3 prior QB-games.

No QB passing-TD predictive model was scored by this audit.
No 2026 / Week-6 / Week-7 outcomes were used.
No sportsbook/OddsAPI inputs were used.
No production mutation occurred.

---

## 9. EXACT ACTIVE NEXT ACTION

**Do not rerun the QB passing-TD source audit. It is SOURCE READY.**

The next scientific action is to freeze a **separate predictive mechanism contract before scoring any TD relationship**.

The architecture already predeclared by the source plan is:

`QB PASS OPPORTUNITY -> TEAM PASS-TD CREATION / SCORING ACCESS -> NAMED QB OWNERSHIP -> QB-SPECIFIC PASS-TD CONVERSION -> PASS-TD COUNT DISTRIBUTION`.

The next chat should:

1. anti-retest existing QB/TD/scoring work so the candidate is genuinely distinct;
2. freeze the predictive question, chronology, player/team decomposition, development/confirmation boundary, baselines, candidate formula(s), shrinkage, named-player/equal-player guards, distribution scoring and pass/fail gates **before outcome scoring**;
3. keep current starter/QB authority pregame;
4. keep named-QB history strictly prior and shrink it toward a broader prior rather than using raw recent TD rate;
5. keep scoring-area pass opportunity separate from pass-TD conversion;
6. keep team creation separate from QB-specific conversion;
7. keep passing TDs separate from passing yards, rushing TDs and generic ATD;
8. only after the predictive plan is frozen, execute the source-ready historical test exactly once under the predeclared gates.

Do **not**:
- infer TDs from passing yards with a league-average multiplier;
- use sportsbook pass-TD prices/lines upstream;
- use generic ATD as a proxy;
- reopen M89/M90 passing-yards mean science;
- reopen closed QB Stage-1 or Stage-2 allocation;
- reopen closed player/team high-value families under a new label;
- use 2026 outcomes;
- fit after looking at confirmation results.

The purpose is a **player-first QB passing-TD model**, not a generic positional TD average.

---

## 10. LIVE REPO / ACTIONS STATE AT THIS HANDOFF

Main continuity head observed before this update:
`d7c8d17a31b7d275ab82872e526c11d13342a030`
(`docs: promote locked-bets research-return checkpoint`).

PR #672 remains the long-running draft research PR:
- title: `Research: individual opportunity roadmap and historical trajectory OOS source gate`;
- state observed via live search: open/draft lineage remains active;
- branch: `research-individual-opportunity-roadmap-2026-10-08`.

The latest QB pass-TD source work is represented by research commits `af709a...` -> `f150d60...` and workflow run `39152687318`; do not assume those commits are already on `main` or PR #672 without reconciling live refs first.

Exact run `39152687318`:
- job `117749381414`;
- status completed;
- conclusion **success**;
- source audit, certification, strict repo audit and immutable artifact upload all passed.

For exact commit `f150d607...`, the connector returned no pull-request-triggered workflow runs / combined statuses. Do not interpret that as evidence that no workflow exists; the authoritative source audit run above is successful.

Next chat must re-query:
- live `main`;
- PR #672 state/head;
- the branch/ref containing the QB pass-TD source commits;
- newest Issue #535 comments;
- exact-head Actions before any write.

---

## 11. HARD ANTI-RETEST

Do not:
- ask the user to re-explain PRISM;
- restart betting work;
- treat Coker’s injury game or DNP as healthy trajectory evidence;
- rescue WR exact authority with reconstructed bytes;
- create a TE-only bridge exception after the shared gate failed;
- substitute a generic distribution parent for expired specialist arrays;
- back-create prospective locks after target games began;
- reopen closed efficiency or generic matchup families;
- use sportsbook data upstream;
- spend paid OddsAPI;
- claim a workflow is running without live Actions evidence;
- optimize position averages instead of named-player mechanisms.

---

## 12. USER OPERATING EXPECTATIONS

The user wants:
- seamless cross-chat continuity;
- GitHub canonical over chat memory;
- actual execution, not planning-only discussion;
- player-specific football reasoning;
- failures closed cleanly;
- frozen hypotheses before scoring;
- no hindsight rescue;
- concise concrete progress updates;
- no paid OddsAPI without explicit permission.

When the user says **“keep going”**, continue with the QB passing-TD predictive freeze from the source-ready state unless live GitHub shows a newer canonical checkpoint.

---

## 13. ONE-SCREEN STATE

**Bets:** LOCKED. Do not rebuild.

**Coker correction:** Week-3 18 yards was injury-affected; Week-4 apparent zero was DNP. Do not read this as role decline. Injury/DNP state must be explicit in player trajectory evidence.

**Signed opportunity:** WR opportunity QUALIFIED; TE opportunity QUALIFIED; WR/TE efficiency CLOSED.

**WR bridge:** exact authority recovery FAIL-CLOSED.

**TE bridge:** independently preserved TE source exists, but frozen shared WR/TE READY_ALL gate blocks execution. CLOSED under existing contract.

**WR/TE uncertainty integration:** historical full-stack source blocked; prospective shadow only if frozen before a future target slate.

**QB passing TD:** source audit **READY** — run `39152687318` SUCCESS at research head `f150d607...`.

**NEXT:** freeze the first player-first QB passing-TD predictive mechanism before scoring:
`PASS OPPORTUNITY -> TEAM PASS-TD CREATION -> QB OWNERSHIP -> QB CONVERSION -> TD COUNT DISTRIBUTION`.

No 2026 outcomes. No sportsbook upstream. No production mutation.
