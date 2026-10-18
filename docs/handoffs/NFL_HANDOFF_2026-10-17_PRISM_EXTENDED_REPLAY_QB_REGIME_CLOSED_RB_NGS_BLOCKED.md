# NFL HANDOFF — 2026-10-17 — PRISM EXTENDED PLAYER-FIRST REPLAY / QB-REGIME CLOSED / RB NGS SOURCE BLOCKED

**Status:** CURRENT / ELITE CONTINUITY HANDOFF  
**Repository:** `dkaps6/imtiredofthis`  
**Model:** **PRISM — Player Role & Individual Situation Model**  
**Production branch:** `main`  
**Research branch:** `research-individual-opportunity-roadmap-2026-10-08`  
**Research PR:** #672 — OPEN / DRAFT  
**Primary research issue:** #535

This handoff supersedes the earlier Week-4/5 replay-active continuity for purposes of resuming this exact conversation. Do **not** recursively read older handoffs.

---

## 1. NON-NEGOTIABLE MODEL PHILOSOPHY

PRISM is an **INDIVIDUAL PLAYER PROJECTION MODEL**.

The named player in his actual football situation is the scientific object.

Think in this exact order:

`PLAYER IDENTITY/HISTORY -> CURRENT ROLE/STATE/TRAJECTORY -> TEAMMATE COMPETITION -> FINITE OPPORTUNITY ALLOCATION -> OPPORTUNITY TYPE/QUALITY -> TEAM/GAME ENVIRONMENT -> PLAYER-CONDITIONED OPPONENT MATCHUP -> INDIVIDUAL EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`.

Position-average MAE/RMSE is downstream QA only. Never turn this project back into “how do we lower RB/WR average error?”

The prior **Construct NFL Tickets** discussion beginning with the user asking why J.K. Dobbins was selected to score a TD remains binding intuition for the whole model:

- understand this exact player's history;
- current workload / trajectory;
- teammate competition;
- exact opportunity type;
- scoring-area / high-value role;
- team context;
- matchup path;
- individual efficiency;
- then answer the **exact market**.

Carries, targets, receptions, yards and TDs are different football questions. A player can be a poor yards candidate and a good TD candidate.

Ticket order remains downstream only:

`FOOTBALL PROJECTION -> RELIABILITY/TRUST -> MARKET COMPARISON -> APPROVED PLAY -> ALT-LINE/CORRELATION/TICKET CONSTRUCTION`.

Sportsbook is downstream. Do not use sportsbook to build the football projection.

---

## 2. LIVE REPO STATE AT HANDOFF

Live reconciliation immediately before this handoff:

### main
Current main before this continuity write:

`dbb8d4d037a4b769a11afb0a7429a14c5ff9793c`

Current main top checkpoint before this handoff was:

`PRISM WEEK-4/5 REPLAY COMPLETE / FINITE ALLOCATION COHERENCE GAP CONFIRMED`.

### active research branch
`research-individual-opportunity-roadmap-2026-10-08`

Current head:

`9279975fd4ef9c207aced7c72dc31f7175311b90`

Latest branch message:

`Research: record RB NGS efficiency source block`.

### PR #672
- OPEN
- DRAFT
- head = `research-individual-opportunity-roadmap-2026-10-08`
- head SHA = `9279975fd4ef9c207aced7c72dc31f7175311b90`

Do not blindly rebase/merge. Reconcile live state first.

---

## 3. WEEK-4 / WEEK-5 CHRONOLOGY — IMPORTANT

The replay begins at Week 4 because Week 4 can legally use:

- prior-season / historical named-player identity where source-valid;
- **Weeks 1-3 of 2026** as current-season state.

Week 5 can use:

- prior-season / historical named-player identity;
- **Weeks 1-4 of 2026** as current-season state.

Weeks 1-3 are therefore **not ignored**. They are upstream evidence for Week 4.

The exact WR dual-reliability rule requiring >=4 prior current-season games is first eligible in Week 5. Do not manufacture the exact score for Week 4.

Do not wait for Week 6 merely to study already-completed Week-4/5 projections.

---

## 4. ORIGINAL CANONICAL MAIN REPLAY — STILL IMPORTANT

Main's earlier canonical replay remains:

- run `39043393975` — SUCCESS;
- exact head `c77a31900c8f99692201453114685cd15796e568`;
- artifact `12105086305`;
- digest `sha256:653e58cd37511dc2ddae499c0b123a564a5317b9912d556a2c427cb05148a363`;
- disposition `PRISM_WEEK4_5_ALL_ROW_PLAYER_FIRST_REPLAY_COMPLETE__DIAGNOSTIC_ONLY__NO_PROMOTION`.

Universe:
- 836 named player-game-market rows;
- W4 440 / W5 396;
- 819 settled;
- 236 unique players;
- 100 informative cases;
- 100 matched correct controls;
- 20 repeat same-player+market rows / 10 identities;
- 405 exact/preserved opportunity-efficiency component rows.

100-case attribution:
- opportunity quantity misread: 40;
- source blocked/unavailable: 28;
- efficiency misread: 20;
- mixed: 12.

This run produced the player-first architecture conclusion now on main.

---

## 5. EXTENDED WEEK-4/5 PLAYER-FIRST REPLAY COMPLETED IN THIS CHAT

A broader diagnostic replay was built and certified on the active research branch.

Canonical successful execution:

- run `39060246734` — SUCCESS;
- exact head `4cd7b4483be1e93e8107a8686381b14379a6d569`;
- artifact `12111007952`;
- digest `sha256:0f120e2002452af548fbea20f0ff1355ec04eb926e53b5845ca663ea2e98a5be`;
- disposition `PRISM_WEEK4_5_PLAYER_FIRST_FORENSIC_REPLAY_EXECUTED__NO_PROMOTION`.

The extended replay fixed an important scope issue rather than pretending the older 440-row Week-4 postmortem board was the entire Week-4 authority.

Preserved Week-4 authority:
- **873 player-market rows total**;
- **483 supported PRISM player markets** replayed;
- **390 anytime-TD rows preserved but intentionally not attributed**, because PRISM's TD pathway is not certified.

Week 5:
- **396 all-row records** retained;
- 4 DNP/inactive rows preserved rather than silently dropped.

Combined extended replay:
- **879 player-market rows**;
- exact final-error component identities:
  - Week 4 = **375**
  - Week 5 = **40**
- detailed named-player casebook = **80**
- matched correct controls = **80**
- 2025 same-player identity support coverage = **83.732%**
- prioritized cases with pregame role state already above projection/rules = **33**
- max final-error identity gap = `1.631e-11`.

Integrity:
- Week-6 outcomes read = 0;
- OddsAPI calls = 0;
- sportsbook refetch/upstream predictor = 0;
- fitted coefficients = 0;
- threshold searches = 0;
- production changes = false;
- exact Week-4 WR dual score manufactured = false.

Issue #535 checkpoint:
comment `6239034454`.

### Important control lesson

The replay/control comparison reinforced a central PRISM rule:

> Correct final output does not mean the component reasoning was correct.

Among matched correct controls with a pregame role-above-projection/rules signal, most still realized more opportunity than projected, but many did **not** beat the final output because efficiency offset the opportunity miss.

Therefore:
- “more opportunity” is not automatically “yards over”;
- opportunity and efficiency can cancel;
- target/carry/reception/yards/TD questions must remain separated.

Do not use final-number correctness as proof that the internal player state was correct.

---

## 6. COMPONENT-CANCELLATION CONTROL AUDIT — IMPORTANT

Later exact audit:

- run `39061107622`;
- exact head `b0b9fd64d1ad83b140f3cae8169bd5a0b4d3048a`;
- disposition `PRISM_COMPONENT_CANCELLATION_CONTROL_AUDIT_COMPLETE__DIAGNOSTIC_ONLY`.

Findings:
- exact component rows = 3,991;
- bottom final-error decile median final error = 0.7818;
- same rows median gross component magnitude = 3.9901;
- median cancellation fraction = 0.8597;
- p75 cancellation fraction = 0.9383;
- replay matched controls with exact components = 15;
- matched-control median cancellation fraction = 0.8086.

Interpretation:

A “correct” projection can hide **large opposing opportunity/efficiency errors**. This means matched correct controls are useful but not automatically structurally correct controls.

Do not create a reliability selector from this forensic result.

---

## 7. FINITE-ALLOCATION ARCHITECTURE GAP — CONFIRMED

Main's critical architecture audit:

- run `39044043517`;
- exact head `189c23a063c58c74473de27f9033b7a4b411a9cc`;
- artifact `12105037700`;
- digest `sha256:fcdbe6732df9aa1803aa235680ee151dd23c3067221ebfb5dd74bc8a78801da8`;
- disposition `PRISM_WEEK5_FINITE_ALLOCATION_TRANSMISSION_AUDIT_COMPLETE__ARCHITECTURE_GAP_CONFIRMED__NO_MODEL_CHANGE`.

Week-5 preserved player-state:
- 30/30 target rooms exceeded modeled-player mass 0.95 before conservation;
- raw target mass mean = 1.5052;
- mean target scale factor = 0.6352;
- 30/30 top-five rushing rooms exceeded 0.95;
- top-five rush mass mean = 1.2682;
- mean rush scale factor = 0.7614.

Brock Bowers exact frozen example:
- upstream rules target share = 0.277666;
- final target probability = 0.206265;
- clean scale factor ~0.742851;
- attenuation ~25.7%.

Interpretation:

Independent named-player claims can collectively overbook finite team opportunity. The valid conservation layer then must attenuate even a strong upstream role estimate.

**Do not remove conservation.**

The scientific target is:
**jointly coherent named-player entitlement / allocation before conservation**.

This is a player + teammate competition + finite opportunity problem, not a position-average problem.

---

## 8. STAGE-2 ACTIVE-ROLE / CONDITIONAL-ALLOCATION SCIENCE

Existing qualified architecture remains:

`PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED_PARTIAL`.

Qualified:
- RB rushing;
- RB receiving;
- TE receiving.

Closed / not scored:
- QB exact Stage 2 CLOSED;
- WR Stage 2 source-blocked / not scored.

Important later work:

### Week-5 frozen Stage-1/Stage-2 retrospective replay
Run `39057189966`, exact head `25b95767f3aeb1a52c28e0cbb07e6375f36a44ee`.

Disposition:
`WEEK5_STAGE12_CURRENT_SEASON_REPLAY_DIRECTIONALLY_SUPPORTIVE`.

Directionally supportive:
- RB rushing improved the most;
- RB receiving and TE receiving improved modestly;
- candidate was frozen before outcomes.

This is not production promotion.

### Independent 2023 translation source audit
Run `39057913591`.

Disposition:
`PRISM_STAGE2_2023_TRANSLATION_SOURCE_BLOCKED`.

Hard blocker:
the frozen Stage-1 architecture requires exact official game-day inactive identity; canonical authority contains 2024-2025 only, with no exact 2023 equivalent.

Do not proxy 2023.

### Future Stage-2 market-translation mechanics
Run `39058313282`.

Status:
`READY` mechanics only.

Eligible cells:
- RB rushing;
- RB receiving;
- TE receiving.

Exact translation:
coherent carries/targets -> existing fixed efficiency authorities.

No scientific target lock was created merely by mechanics readiness.

### Week-7 readiness probe
Run `39060798892`.

Disposition:
`PRISM_FUTURE_STAGE2_TRANSLATION_NOT_READY__FAIL_CLOSED`.

Reason:
Week-6 prior-week state was incomplete at probe time; only DEN/SEA had the required Week-6 snap/stat coverage.

This is a **timing/data readiness block**, not a scientific failure. Do not use incomplete Week-6 data or grade Week-6 early.

---

## 9. CURRENT-ROLE / ROOM STATE CLOSED OR BLOCKED FAMILIES

Protect all of these:

### Raw recent-share replacement
CLOSED.

Do not reopen with:
- RB-only;
- WR-only;
- TE-only;
- alternate Bayes weights;
- different caps;
- “use recent share when gap is large.”

### Opportunity-state disagreement as uncertainty
CLOSED.

### Active-role participation
QUALIFIED ALL.

### WR/TE target-share trajectory
SUPPORTED / protected.
Do not rename or retune.

### RB carry/snap state shadow
Protected prospective science.
Do not back-apply / retune.

### Teammate-availability role elasticity V1
SOURCE BLOCKED.
Do not lower gates or convert into vacancy transfer.

### Room Competition State V1
SOURCE BLOCKED:
- run `39039540742`;
- head `73542d0b0eed3fd212352e0d51c7d550decd04e2`;
- artifact `12103282541`;
- disposition `PRISM_ROOM_COMPETITION_STATE_SOURCE_BLOCKED`.

### RB/TE explicit depth-role transition
SOURCE BLOCKED:
- run `39043968618`;
- head `a57b6160a7fea6c31f88a9ea4be3eb198d0349d8`;
- artifact `12105236525`.

Do not infer target-game role from outcomes/snaps.

---

## 10. PLAYER MECHANISM-INSTABILITY TRUST V1 — CLOSED

Source audit first passed for WR/TE:

- run `39061606501`;
- disposition `PRISM_PLAYER_MECHANISM_INSTABILITY_TRUST_V1_SOURCE_READY_WR_TE`.

But frozen historical scoring failed.

Canonical scoring:
- run `39061903343`;
- exact head `3bf408334ee0867bd867417beae466748c8847bc`;
- freeze `95f1e831cf12a3c5b24234089169e0b049ec252f`;
- disposition `PRISM_PLAYER_MECHANISM_INSTABILITY_TRUST_V1_CLOSED`.

WR:
- 2023 development partial rho +0.02714;
- 2024 confirmation +0.01527;
- confirmation player-cluster CI crossed zero;
- confirmation team-week CI crossed zero.

TE:
- 2024 development partial rho -0.03887;
- 2025 confirmation +0.02229;
- CIs crossed zero;
- development primary direction failed.

Do not reopen as:
- another trust threshold;
- alternate instability window;
- position-specific rescue;
- market selector.

---

## 11. PASS-CATCHER × QB-REGIME ALLOCATION — SOURCE GREAT, SIGNAL CLOSED

A new player-first object was frozen:

> named WR/TE × pregame-selected QB identity × strictly-prior target allocation.

Source audit plan:
`docs/research/PRISM_PASS_CATCHER_QB_REGIME_ALLOCATION_STATE_V1_SOURCE_AUDIT.md`.

### Source audit — PASSED ALL

Canonical:
- run `39067415356` — SUCCESS;
- exact head `d189d2305b745d0c7f104e6f4fd8d0d305d1da16`;
- artifact `12113054267`;
- digest `sha256:61b701091bdfcdad32e86d0c6ba2e6a471b773656c95b4fa6179adbbb54214a2`;
- disposition `PRISM_PASS_CATCHER_QB_REGIME_ALLOCATION_STATE_V1_SOURCE_READY_ALL`.

All four cells passed every frozen source gate:

WR 2024:
- QB-regime coverage 98.897%;
- >=2 same-QB 76.355%;
- >=3 same-QB 65.977%;
- support 2,141 rows / 219 players.

TE 2024:
- QB-regime coverage 98.897%;
- >=2 same-QB 78.174%;
- >=3 same-QB 67.993%;
- support 1,336 / 127.

WR 2025:
- QB-regime coverage 98.346%;
- >=2 same-QB 75.596%;
- >=3 same-QB 65.018%;
- support 2,094 / 221.

TE 2025:
- QB-regime coverage 98.346%;
- >=2 same-QB 76.914%;
- >=3 same-QB 66.647%;
- support 1,296 / 129.

Chronology violations = 0.

### Predictive signal — FAILED CLOSED

Frozen candidate:
- baseline = ordinary strictly-prior recent target-share state;
- signal = same-QB prior mean target share minus overall prior mean target share;
- apply only >=2 same-QB prior games;
- no fitted coefficient;
- finite WR/TE room mass conserved;
- 2024 development / 2025 untouched confirmation.

Canonical result:
- run `39067914161` — SUCCESS;
- exact head `87e57f4341c1ca6ae79d585a9b52bbc80a9a29b2`;
- artifact `12113942923`;
- digest `sha256:8019c23d69db0a9fb2b8c68b180277bcbdd17bdd99c3e9f393ce51286c329fbf`;
- result doc `docs/research/PRISM_PASS_CATCHER_QB_REGIME_ALLOCATION_SIGNAL_V1_RESULT.md`;
- disposition `PRISM_PASS_CATCHER_QB_REGIME_ALLOCATION_SIGNAL_V1_CLOSED`.

WR:
- 2024 target-share MAE 0.055351 -> 0.056281 — worse;
- 2025 0.058520 -> 0.058893 — worse;
- 2025 player-cluster CI crossed zero;
- team-week CI crossed zero.

TE:
- 2024 0.042077 -> 0.042593 — worse;
- 2025 0.042547 -> 0.042857 — worse;
- both 2025 CIs entirely above zero, meaning statistically worse.

Close this exact family.

Do not rescue with:
- >=3-only;
- alternate recent window;
- fitted shrink coefficient;
- QB-change subgroup;
- WR1/TE1 carveout;
- family pooling;
- receiving-yards translation.

Important:
this does **not** close QB effects on opportunity type / target quality. It closes direct same-QB target-allocation delta -> current target-share adjustment.

---

## 12. RB NGS RUSHING EFFICIENCY IDENTITY V1 — SOURCE BLOCKED

Result:
`docs/research/PRISM_RB_NGS_RUSHING_EFFICIENCY_IDENTITY_V1_SOURCE_RESULT.md`.

Canonical:
- run `39068580334` — SUCCESS;
- exact child-runner head `3b39bfce0cdff0f3ecef09518b8b9dd04795cca8`;
- artifact `12113799476`;
- digest `sha256:9511828f360ad01aceb0fae6e9bff4d2b70981ed07d54bca84740bed41d65f00`;
- runner PR #688;
- disposition `PRISM_RB_NGS_RUSHING_EFFICIENCY_IDENTITY_SOURCE_BLOCKED`.

NGS provider quality itself was excellent:
- 2023 / 2024 / 2025 stable ID, team, RB position and required process fields = 100%;
- 32 teams;
- 80+ distinct RB source-season identities;
- zero duplicate player-week identities.

But the frozen >=3 strictly-prior same-player support gate failed.

2024:
- 1,845 target pregame RB rows;
- history-ready = 476;
- distinct history-ready RBs = 52;
- required >=900 rows / >=80 RBs.

2025:
- 1,866 target rows;
- history-ready = 511;
- distinct history-ready RBs = 52.

Do not:
- lower >=3 game rule;
- lower support thresholds;
- pool seasons after seeing support;
- use target-game NGS;
- fit a YPC rescue;
- use 2026 outcomes.

This is a source-support block for exact V1, not evidence player-specific rushing process is worthless.

---

## 13. OPPONENT / MATCHUP BOUNDARY

Generic defense-vs-position is closed.

Do not use:
- generic defense-vs-position;
- generic recent allowance;
- usage × generic matchup weakness;
- failed FMT formulas.

A future opponent mechanism must be player-conditioned on the named player's actual role/opportunity subtype with exact historical/live source parity.

Opponent should alter the **path** for the named player, not replace player identity / role as the primary object.

---

## 14. WEEK-6 / WEEK-7 PROTECTED PROSPECTIVE SCIENCE

Do not touch immutable Week-6 locks.

Protected:
- Week-6 Sunday/Monday Stage-2 shadow;
- Week-6 WR opportunity trust lock;
- Week-6 WR efficiency trust lock;
- Week-6 TE trust lock;
- no Week-6 WR dual backfill;
- no early Week-6 grading;
- Weeks 7-10 WR dual-reliability prospective protocol.

Week-7 Stage-2 readiness failed only because Week-6 source data were incomplete at the time of the probe.

Do not fabricate missing Week-6 source state.

---

## 15. COST / SPORTSBOOK RULES

No paid OddsAPI pull is authorized from this handoff.

Do not spend OddsAPI without explicit user authorization.

Sportsbook is downstream only.

Do not use market lines to choose the football starter, role, usage, allocation, efficiency or projection.

---

## 16. WHAT THIS CHAT ESTABLISHED SCIENTIFICALLY

The broad player-first direction was not a new pivot. It was already the intended PRISM architecture, and this chat restored it after some research became too narrowly centered on WR reliability.

The important current picture is:

1. Individual-player opportunity allocation remains a major failure seam.
2. Strong upstream role knowledge can be attenuated by incoherent independent room claims before valid conservation.
3. Correct final numbers can hide badly wrong components due to opportunity/efficiency cancellation.
4. Raw recent-share replacement is closed.
5. Same-QB target-allocation history is source-rich but predictively null/worse under the frozen direct formulation.
6. Generic mechanism-instability trust is closed.
7. RB NGS individual efficiency V1 has excellent source quality but insufficient strictly-prior same-player support.
8. Stage-2 active-set / conditional allocation remains one of the most promising already-qualified architecture pieces for RB rushing, RB receiving and TE receiving.
9. Opportunity and efficiency must remain market-specific; yards conclusions cannot be reused for TDs or vice versa.

---

## 17. EXACT NEXT ACTION FOR THE NEXT CHAT

Do **not** rerun the completed Week-4/5 replay.

Do **not** reopen:
- raw recent share;
- same-QB target-allocation delta;
- mechanism-instability trust;
- RB NGS V1 with weakened gates;
- Room Competition State V1;
- RB/TE depth-role transition V1;
- generic DvP;
- older generic conservation rescues.

First:

1. Read this handoff and the root checkpoint only.
2. Reconcile live:
   - `main`;
   - branch `research-individual-opportunity-roadmap-2026-10-08`;
   - PR #672;
   - newest Issue #535 comments;
   - exact-head Actions after branch head `9279975...`.
3. Confirm no newer branch result supersedes the RB NGS source block.

Then continue autonomously from the **post-RB-NGS player-first frontier**.

The next new candidate must be genuinely distinct under anti-retest.

Preferred research posture:

- keep the named player as the object;
- target an unresolved **role regime / teammate competition / opportunity subtype or quality / individual efficiency state** seam;
- require strictly-prior source parity before scoring;
- freeze one exact mechanism before outcomes;
- preserve finite room conservation;
- separate opportunity from efficiency;
- translate to market outcomes only in a separate frozen bridge;
- fail closed if source parity/support is not strong enough.

A particularly important open distinction after the QB-regime closure is:

> QB identity may still alter **opportunity type / target quality** for a named pass catcher even though direct same-QB target-share adjustment failed.

But do not automatically score that idea. Anti-retest the already-closed opportunity-quality/QB-pair families first and prove it is genuinely distinct with source parity.

In parallel, once complete Week-6 prior-state data legally exist, the future Stage-2 Week-7 readiness mechanics may be rechecked under their existing frozen contract. Do not backfill outcomes or change Week-6 locks.

Do not stop at a plan if a source-valid next action is available. Keep pushing.

---

## 18. NEXT-CHAT READ ORDER

Read in exactly this order:

1. `AGENTS.md`
2. ONLY the newest TOP checkpoint in `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. this handoff:
   `docs/handoffs/NFL_HANDOFF_2026-10-17_PRISM_EXTENDED_REPLAY_QB_REGIME_CLOSED_RB_NGS_BLOCKED.md`
4. `docs/research/PRISM_PASS_CATCHER_QB_REGIME_ALLOCATION_SIGNAL_V1_RESULT.md`
5. `docs/research/PRISM_RB_NGS_RUSHING_EFFICIENCY_IDENTITY_V1_SOURCE_RESULT.md`
6. newest relevant Issue #535 comments, especially from `6239034454` onward
7. live reconcile main / research branch / PR #672 / Actions before any write.

Do not recursively read older handoffs unless this handoff explicitly points to one for a specific scientific fact.

GitHub is canonical over chat memory.
