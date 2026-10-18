# NFL HANDOFF — 2026-10-18 — BETS LOCKED / RETURN TO PRISM RESEARCH / SIGNED OPPORTUNITY FRONTIER

**Status:** CURRENT CROSS-CHAT CONTINUITY AUTHORITY  
**Repository:** `dkaps6/imtiredofthis`  
**Model:** **PRISM — Player Role & Individual Situation Model**  
**Production branch:** `main`  
**Research branch:** `research-individual-opportunity-roadmap-2026-10-08`  
**Research PR:** #672 — OPEN / DRAFT  
**Primary issue:** #535

This handoff exists because the prior chat reached its conversation limit immediately after the user said the Sunday bets were locked. The next chat must feel like the same operator continuing. Do not ask the user to repeat the ticket work or model philosophy.

---

## 1. IMMEDIATE MODE SWITCH

**Game-day betting work is CLOSED unless the user explicitly reopens it.**

The user said:
> bets are locked

Do not spend the next chat reconstructing today's slips, re-checking prices, or trying to improve already-entered bets.

The explicit next direction is:
> **go back to the research lane and continue from the exact canonical PRISM frontier.**

No paid OddsAPI call is authorized.

---

## 2. NON-NEGOTIABLE PRISM OBJECT

PRISM is an **individual-player projection model**.

Binding phrase:
> **The player should have the rules. The position should not have the player.**

Reasoning order:
`PLAYER IDENTITY/HISTORY -> CURRENT ROLE/STATE/TRAJECTORY -> TEAMMATE COMPETITION -> FINITE OPPORTUNITY ALLOCATION -> OPPORTUNITY TYPE/QUALITY -> TEAM/GAME ENVIRONMENT -> PLAYER-CONDITIONED OPPONENT MATCHUP -> INDIVIDUAL EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`.

Position-average MAE/RMSE is downstream QA only.

Carries, targets, receptions, receiving yards, rushing yards, passing yards and touchdowns are different football questions about the same named player.

The prior Dobbins discussion remains binding model intuition:
- lead-back / current workload;
- recent workload trajectory;
- teammate competition;
- exact opportunity type;
- scoring-area ownership;
- team/game environment;
- player-specific conversion context;
- then market/price only after the football case exists.

A player can be a weak yards candidate and a strong touchdown candidate without contradiction.

---

## 3. BET-CONSTRUCTION LESSONS PRESERVED FOR FUTURE USE

Although today's bets are closed, preserve the construction contract because it is important downstream research context.

Canonical doc:
`docs/research/MODEL_BACKED_ALT_LINE_PARLAY_CONSTRUCTION_V1.md`

Binding order:
`FOOTBALL PROJECTION -> RELIABILITY/TRUST -> MARKET COMPARISON -> APPROVED PLAY -> ALT-LINE/CORRELATION/TICKET CONSTRUCTION`.

The user clarified the sportsbook product semantics very explicitly:
- positive-side parlay legs should use **FanDuel milestone props with a "+" sign** when available, e.g. `175+ passing`, `20+ receiving`, `1+ reception`;
- do **not** invent alternate-under ladders that FanDuel does not offer;
- if an UNDER is approved, use the actual normal sportsbook under line;
- the exact playable line is the bet, not merely the player name;
- HOU-JAX became unavailable after kickoff and was excluded from all later construction;
- no new paid odds pull was used.

### Watson lesson

The user specifically referenced the prior slate where several otherwise-good slips lost because the repeated Deshaun Watson leg failed.

Preserved lesson:
- do not use one player as a universal anchor;
- one wrong named-player projection must not kill the entire portfolio;
- use the same **approved player/market pool**, but diversify tickets;
- a useful working constraint remains no player on more than two tickets unless there is an explicit reason.

### Dobbins two-layer gate

The user reaffirmed that **model support alone is insufficient**.

Every future leg should pass:
1. independent player-specific football logic;
2. PRISM agreement on the exact market;
3. current availability/starter reconciliation;
4. exact playable line/milestone;
5. ticket-tier construction.

A large model edge that conflicts with current football context should be removed or downgraded.

---

## 4. RB LIVE PERFORMANCE RE-EVALUATION — IMPORTANT CORRECTION

The prior chat challenged the stale blanket idea that RBs should still be broadly distrusted.

The settled Weeks 3-5 postmortem artifacts were re-read and the RB betting rows recomputed.

### RB by week

| Week | Record | Hit rate | Units |
|---|---:|---:|---:|
| Week 3 | **84-70** | **54.5%** | **+5.67u** |
| Week 4 | **78-75** | **51.0%** | **-3.74u** |
| Week 5 | **75-49** | **60.5%** | **+17.27u** |
| Weeks 3-5 | **237-194** | **55.0%** | **+19.20u** |

431 decided RB bets.

### RB by market, Weeks 3-5

| Market | Record | Hit rate | Units |
|---|---:|---:|---:|
| Receptions | **57-34** | **62.6%** | **+19.08u** |
| Rush + Rec Yards | **53-41** | **56.4%** | **+5.89u** |
| Rushing Yards | **81-68** | **54.4%** | **+4.27u** |
| Receiving Yards | **46-51** | **47.4%** | **-10.03u** |

Important interpretation:
- do **not** say "RBs are bad" as a blanket statement;
- RB receptions have been the strongest recent RB betting lane;
- RB rushing yards were profitable in each of Weeks 3, 4 and 5;
- rush+receiving improved sharply in Week 5;
- raw RB receiving yards remained weak.

But the point-projection problem is **not solved**:
- Weeks 3-5 RB model MAE ~18.43 versus market-line MAE ~16.52;
- overall RB signed bias ~-6.35;
- rush-yards bias ~-9.47;
- rush+receiving bias ~-12.42.

Therefore:
> recent RB bet performance has improved materially, but the underlying player projection still has structural low-side opportunity/volume issues in important RB markets.

Do not confuse profitable recent selection results with a solved RB projection engine.

---

## 5. NEW GAP EXPOSED TODAY — QB PASSING TOUCHDOWNS / FANTASY OUTPUT

The user asked a fantasy question about Brock Purdy vs Drake Maye and then correctly noticed that PRISM does **not** currently have a dedicated certified passing-touchdown projection layer.

Current QB production is strong on:
- attempts/team opportunity;
- passing efficiency;
- passing yards;
- some rushing context.

But a complete fantasy QB evaluation requires:
- passing yards;
- passing TD expectation;
- interception risk;
- rushing yards;
- rushing TD expectation.

Repository search in the prior chat did not find a certified QB passing-TD projection object in the current PRISM stack.

This is now a legitimate future research observation:
> **PRISM can answer passing-yard questions much better than full fantasy-QB scoring because passing TD expectation is missing as a dedicated market-specific player output.**

Do **not** immediately open an unfrozen TD model just because this was noticed on game day. Treat it as a new research backlog item requiring:
- anti-retest inventory;
- source audit;
- player/team scoring-opportunity decomposition;
- market-specific touchdown mechanism;
- frozen historical/prospective validation.

This is conceptually aligned with the player-first architecture, but it is **secondary to the currently active signed-opportunity frontier** unless the canonical research queue changes.

---

## 6. CURRENT CANONICAL RESEARCH STATE — PRESERVE EXACTLY

### Stage-2 player-specific mechanism learning

Source:
- run `39098772787` — SOURCE READY ALL.

Hierarchical future mechanism-utility:
- run `39099164620`;
- RB rushing QUALIFIED;
- RB receiving QUALIFIED;
- TE receiving CLOSED on the frozen tail-harm guard.

Player-specific authority bridge:
- run `39099474655`;
- **RB receiving QUALIFIED**;
- RB rushing CLOSED.

RB receiving 2025:
- CRPS gain +0.008520;
- player CI low +0.005441;
- game CI low +0.006117;
- equal-player CI low +0.005582;
- 78.51% named players improved;
- finite opportunity exactly conserved.

Do not rescue RB rushing.

### Stage-1 personalization

Source:
- run `39099745136` — SOURCE READY ALL.

Hierarchical future mechanism utility:
- run `39099930673`;
- RB QUALIFIED;
- TE QUALIFIED;
- QB CLOSED;
- WR CLOSED.

Direct Stage-1 authority bridge:
- run `39100106463`;
- RB CLOSED;
- TE CLOSED.

Do not rescue Stage-1 bridge with soft blends, threshold changes, support changes, recency or subsets.

Architecture lesson:
> named-player mechanism utility can predict whether a mechanism helps without identifying the correct projection intervention.

Utility prediction and projection action remain separate scientific objects.

---

## 7. WR / TE SIGNED COMPONENT-BIAS FRONTIER — LATEST MATERIAL SCIENCE

Source audit:
- run `39100442923`;
- artifact `12126685270`;
- digest `sha256:d29de06947a0b3aead1e20f05b66d5cdfacfa393d8b1c535808533ed3b0a0d7a`;
- disposition `PRISM_WR_TE_PLAYER_SIGNED_COMPONENT_BIAS_SOURCE_READY_ALL`.

Signal freeze:
- `docs/research/PRISM_WR_TE_PLAYER_SIGNED_COMPONENT_BIAS_V1_SIGNAL_FREEZE.md`;
- frozen-before-scoring commit `dcf70236709d0f586c72bcaa4e3550dbefe31efb`.

Predictive signal run:
- `39109423266` SUCCESS;
- exact head `ac31b82907c416f76ffb078e9b2af8b6100a9ea6`;
- artifact `12128983909`;
- digest `sha256:bb14d6474778d0afd53ca74bb60237b381aa4e973195bab3d09d54ed87ef2490`;
- Issue #535 comment `6247355342`.

Disposition:
`PRISM_WR_TE_PLAYER_SIGNED_COMPONENT_BIAS_SIGNAL_QUALIFIED_PARTIAL`.

Results:
- **WR opportunity QUALIFIED**;
- WR efficiency CLOSED;
- **TE opportunity QUALIFIED**;
- TE efficiency CLOSED.

Sign convention:
`opportunity error = (predicted_targets - actual_targets) * predicted_YPT`.

Positive = overprojected opportunity.  
Negative = underprojected opportunity.

Any future correction would subtract a learned positive opportunity bias; do not reverse the sign.

---

## 8. FROZEN DOWNSTREAM BRIDGES AND THE CURRENT BLOCK

Two downstream contracts were frozen before outcome scoring.

WR topology deployment eligibility freeze:
- commit `9b90018f0906f64371daa7e350fc9375bb7996d4`;
- ANCHOR and SECONDARY score independently;
- no pooled WR rescue;
- source recovery must be READY before scoring.

WR/TE conserved bridge action freeze:
- commit `618b8861a072eb06be442ade966be98f101a5a68`;
- fixed modifier:
  `exp(-hierarchical_signed_bias / parent_predicted_receiving_yards)`;
- WR must respect R15 anchor/secondary hierarchy and exact room conservation;
- TE must act inside exact R5P team-TE-pool softmax;
- no clipping/tuning/support retuning/efficiency changes/sportsbook/2026 outcomes/production mutation.

### WR exact authority recovery FAILED CLOSED

Recovery run:
- `39110742338`;
- job `117626156590`;
- disposition: **SOURCE RECOVERY BLOCKED / NO BRIDGE SCORE**.

Expected exact WR-R15 predictions SHA:
`91d108ee7981935d78179257b204a0204e9a4f8061e6eba4d6810845ed057d05`.

Rebuilt SHA:
`7dce6b0e194d37ec3b814caf79942eea3b83fcda520b481788ec6b2b3f0778c3`.

The exact byte-identity gate correctly failed.

Stop rules:
- do not update the expected digest;
- do not weaken byte identity;
- do not score WR topology or WR conserved bridge from the rebuilt file;
- do not tune the frozen action;
- no efficiency rescue;
- no Week-6/2026 outcome contamination;
- no sportsbook/OddsAPI;
- no production mutation.

The upstream WR/TE signed-opportunity signal remains valid. This is a source/materialization failure, not a reversal of the science.

---

## 9. LIVE GITHUB STATE OBSERVED AT HANDOFF

Reconciled live on 2026-10-18 before writing this handoff.

PR #672:
- state OPEN;
- DRAFT = true;
- merged = false;
- mergeable = false at observation;
- base `main`;
- base SHA observed via PR metadata:
  `dbb8d4d037a4b769a11afb0a7429a14c5ff9793c`;
- head branch:
  `research-individual-opportunity-roadmap-2026-10-08`;
- head SHA:
  `e1ce9880ba8fd2ccd71cd26fe609a1a96cf66c90`.

Issue #535 latest material continuity comments observed:
- `6246076330` — WR/TE signed component-bias source READY;
- `6247355342` — WR/TE signed component-bias predictive signal; WR/TE opportunity qualified, efficiency closed;
- `6247561852` — downstream bridge contracts frozen pre-result;
- `6249913115` — WR-R15 exact authority recovery blocked / no bridge score.

The connector returned no pull-request-triggered workflow runs for exact PR head `e1ce9880...` at this observation. Do not infer that no other Actions exist; re-query live before claiming anything is running or complete.

---

## 10. EXACT RESTART ACTION FOR THE NEXT CHAT

The next chat should **return immediately to research**, not betting.

Read in order:
1. `AGENTS.md`;
2. ONLY the newest TOP checkpoint in `CURRENT_NFL_RESEARCH_HANDOFF.md`;
3. this handoff;
4. newest material Issue #535 comments;
5. reconcile live `main`, research branch, PR #672 and exact-head Actions/status.

Then:

### Primary task

Resume from the **signed-opportunity frontier after the WR authority fail-close**.

Do **not** rescue the blocked WR bridge.

First determine, from the already-frozen contracts and source authorities, whether there is an independently valid **TE conserved bridge execution path** that does not depend on the failed WR-R15 exact-authority recovery.

- If yes, execute only that already-frozen TE path under its existing gates.
- If no, document the exact source dependency that blocks it and move to the next genuinely distinct player-specific mechanism rather than weakening source integrity.

For WR:
- source recovery remains blocked until exact original authority or a separately frozen independently certified equivalent source exists;
- no digest rewrite;
- no approximate bridge scoring.

### Secondary backlog item

Record QB passing-TD / fantasy-output modeling as a **new player-first market-specific research gap**, but do not let it displace the current authorized frontier unless the canonical queue is deliberately changed.

### Protected prospective state

Preserve:
- Week-6 WR/TE locks immutable;
- Week-7 source-readiness fail-close;
- Weeks 7-10 WR dual prospective protocol;
- finite opportunity conservation;
- sportsbook downstream only;
- no paid OddsAPI without explicit authorization;
- no production mutation from research-only forensic work.

---

## 11. HARD ANTI-RETEST / DO-NOT-DO

Do not:
- ask the user to re-explain PRISM;
- restart ticket construction unless explicitly asked;
- reopen closed RB rushing Stage-2 authority;
- rescue closed Stage-1 RB/TE authority bridges;
- reopen WR/TE efficiency signed-bias after closure;
- weaken exact WR authority byte identity;
- replace the expected WR digest with the rebuilt digest;
- score the frozen WR bridge from unauthorized reconstructed authority;
- use sportsbook lines to construct football projections;
- use Week-6 outcomes prematurely;
- create a generic defense-vs-position multiplier;
- chase positional averages instead of named-player mechanics;
- spend a paid OddsAPI credit.

---

## 12. USER OPERATING EXPECTATIONS

The user wants:
- seamless continuity;
- GitHub canonical over chat memory;
- actual execution rather than planning-only responses;
- player-specific football reasoning;
- closed failures to stay closed;
- frozen hypotheses before scoring;
- no post-hoc rescue;
- concise concrete progress updates;
- live Actions verification before saying something is running;
- no paid OddsAPI spend without explicit permission.

When the user says **"keep going"**, perform the next authorized research action.

When the user asks **"where are we?"**, report exact live GitHub state and the current scientific question, not a generic project summary.

---

## 13. ONE-SCREEN STATE

**Betting:** Sunday bets LOCKED; game-day lane closed unless explicitly reopened.

**Ticket method preserved:** football reasoning + PRISM + exact market + real `+` milestone/normal under line; diversify player exposure; Watson lesson preserved.

**RB live evidence:** Weeks 3-5 = 237-194, 55.0%, +19.20u; receptions strongest; underlying RB point-projection low bias still unresolved.

**New gap noted:** no dedicated certified QB passing-TD layer; relevant to fantasy/full QB output, secondary backlog.

**Research:**
- RB receiving Stage-2 player-specific authority QUALIFIED;
- Stage-1 direct RB/TE authority CLOSED;
- WR opportunity signed bias QUALIFIED;
- TE opportunity signed bias QUALIFIED;
- WR efficiency signed bias CLOSED;
- TE efficiency signed bias CLOSED;
- WR-R15 exact authority recovery FAIL-CLOSED;
- no WR bridge score authorized.

**Next:** resume signed-opportunity research from the fail-close; test only an independently source-valid already-frozen TE bridge path if such a path exists, otherwise fail closed and advance to a genuinely distinct player-specific frontier.
