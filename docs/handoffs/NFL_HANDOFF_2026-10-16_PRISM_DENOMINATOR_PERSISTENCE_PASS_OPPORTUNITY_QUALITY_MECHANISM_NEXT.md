# NFL HANDOFF — 2026-10-16 — PRISM DENOMINATOR PERSISTENCE PASS / OPPORTUNITY-QUALITY MECHANISM NEXT

**Status:** CURRENT  
**Repository:** `dkaps6/imtiredofthis`  
**Model:** **PRISM — Player Role & Individual Situation Model**  
**Production branch:** `main`  
**Active research branch:** `research-individual-opportunity-roadmap-2026-10-08`  
**Research PR:** #672 — OPEN / DRAFT  
**Issue:** #535 — live research ledger  
**Canonical research state:** `PLAYER_INDIVIDUALIZATION_PARTIAL`

This handoff supersedes the earlier 2026-10-16 handoffs for the active frontier. Do **not** recursively read old handoffs unless this file explicitly points to one.

---

## 1. USER INTENT / NON-NEGOTIABLE MODEL PHILOSOPHY

PRISM is an **individual-player projection model**.

The user has been explicit and repeated about this: do not optimize abstract position-average MAE as though QB/RB/WR/TE are the modeling objects. Position MAE/RMSE is downstream QA. The scientific object is the named player in his actual football situation.

Canonical reasoning order:

`PLAYER IDENTITY -> CURRENT ROLE/STATE -> TEAMMATE COMPETITION -> OPPORTUNITY ALLOCATION -> OPPORTUNITY QUALITY -> OPPONENT/ENVIRONMENT -> EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`.

For every player, PRISM should ultimately reason about:
- who this player is historically;
- what his role is **right now**;
- whether he is actually likely to participate;
- which teammates compete for the same finite work;
- how much opportunity belongs to him;
- what *kind* of opportunities those are;
- what opponent/environment he must move through;
- what he individually does with those chances;
- how reliable/volatile his projection is;
- which prop market best matches the football thesis.

Do not collapse this into:
- generic position means;
- generic recent-share formulas;
- defense-vs-position shortcuts;
- sportsbook-derived football inputs;
- "lower average positional MAE" as the primary objective.

The user's core principle remains:

> Fix the individual player and his situation; aggregate error improves as a consequence.

---

## 2. READ ORDER FOR THE NEXT CHAT

Read only this sequence before doing new work:

1. `AGENTS.md`
2. **ONLY the newest TOP checkpoint** in `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. **this handoff**
4. `docs/research/PRISM_OPPORTUNITY_QUALITY_PARENT_DENOMINATOR_PERSISTENCE_V1_RESULT.md`
5. newest relevant Issue #535 comments
6. live PR #672 / research branch / exact-head Actions

Do not recursively ingest older handoffs unless needed to resolve a specific referenced authority.

GitHub is canonical. Reconcile live state before any write.

---

## 3. MODEL NAME

The model is now canonically named:

**PRISM — Player Role & Individual Situation Model**

Use PRISM in future docs and conversation.

---

## 4. STAGE 1 — NAMED-PLAYER ACTIVE-ROLE STATE — QUALIFIED

Canonical result:

`PLAYER_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED_ALL`

Authority:
- run `38907459996` — SUCCESS;
- job `116994537005`;
- exact head `1c2c92ca498e8dce1eee19d39b2f276dfd37d8f7`;
- artifact `12049120604`;
- digest `sha256:41b1acd3f1b8d77abb53980db160ece32f8fc40351939d4827248b9caad0c64b`.

Exact qualified Stage-1 semantics:

Pregame universe:
- weekly roster ACT;
- stable identity;
- exact official inactives removed;
- valid scheduled team-week;
- valid snap source.

Target:
- 1 = any offensive snap;
- 0 = zero offensive snaps.

Position-family baseline:
`p_baseline = (K_pos + 0.5) / (N_pos + 1)`
with `N_pos >= 100`.

Named-player state:
- same stable player;
- same current team;
- most recent up to 4 eligible strictly-prior games;
- minimum 3.

Candidate:
`p_candidate = (k_player + 2 * p_baseline) / (n_player + 2)`.

2025 independent-confirmation Brier improvements:
- QB +0.092720;
- RB +0.038602;
- WR +0.015566;
- TE +0.006296.

Every position passed player-cluster, game-cluster and equal-player frozen confirmation gates.

Stage 1 is a separate upstream state:
`P(any offensive snap | this named player's strictly-prior same-team participation history)`.

Do not convert it into a direct share multiplier outside the qualified bridge.

---

## 5. STAGE 2 — CONDITIONAL FINITE OPPORTUNITY ALLOCATION — QUALIFIED PARTIAL

Source result:
`PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_SOURCE_READY_PARTIAL`.

Historical bridge result:
`PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED_PARTIAL`.

Canonical scored authority:
- run `38957575645` — SUCCESS;
- job `117164201650`;
- exact head `eba5787c0b06aa5f975008965df7d023b3525ba7`;
- artifact `12070774378`;
- digest `sha256:e07edbad1bf97a4917ba95c58156c8941448e7b2668bd0eae2e9bcad5d4945b0`.

### Qualified cells

**RB rushing**
- 2025 CRPS +0.385893;
- expected-count AE +0.267115;
- room allocation +0.008439;
- player-cluster CI low +0.297996;
- game-cluster CI low +0.333263;
- equal-player CI low +0.277780;
- 86.78% of evaluable players improved.

**RB receiving**
- 2025 CRPS +0.034723;
- expected-count AE +0.050076;
- room allocation +0.000975;
- player-cluster CI low +0.019115;
- game-cluster CI low +0.027288;
- equal-player CI low +0.015982.

Preserve the player heterogeneity: only 47.11% of evaluable RBs improved individually, but the frozen equal-player mean/bootstrap still passed. Do not call this universal benefit.

**TE receiving**
- 2025 CRPS +0.018198;
- expected-count AE +0.018672;
- room allocation +0.000395;
- player-cluster CI low +0.009962;
- game-cluster CI low +0.013674;
- equal-player CI low +0.008875;
- 57.89% of evaluable players improved.

### Closed / blocked cells

**QB passing V1 — CLOSED**
- CRPS improved, but expected attempts and room allocation worsened in both seasons;
- 2025 player/game/equal-player intervals failed;
- do not rescue with thresholds, starter carveouts, active-state correlation, backup reassignment or a new share coefficient.

**WR receiving — NOT SCORED**
- 2024 OOS authority exists;
- 2025 WR-R15 independent OOS authority is unavailable/forbidden because the final fit overlaps training;
- do not relabel it OOS or score 2024 only.

---

## 6. WEEK-6 PROSPECTIVE PRISM SHADOW — LOCKED

The interim Cleveland identity blocker is **closed**. Do not hand the next chat the stale blocker.

Final guarded prospective lock:

`PRISM_WEEK6_SUNMON_STAGE2_SHADOW_LOCKED`

Authority:
- run `38991032116` — SUCCESS;
- exact head `3c0190cc8f1cd6271bfd6912cf84cff82008ff80`;
- artifact `12084848700`;
- digest `sha256:ec1376dfa90fff3b5fc31e199588f7fe5f570017946adfd0b85be847af901e6e`;
- result `docs/research/PRISM_WEEK6_SUNMON_STAGE2_SHADOW_V1_LOCK_RESULT.md`.

Population:
- 13 Sunday/Monday Week-6 games;
- SEA-DEN excluded because it was already played before the Stage-2 freeze;
- 248 locked player-cell rows;
- RB rushing / RB receiving / TE receiving only.

Source gates all passed:
- prior roster team-game coverage 100%;
- official-inactive team-game coverage 100%;
- prior snap team-game coverage 100%;
- stable roster identity 100%;
- skill inactive -> roster bridge 97.5535%;
- skill positive-snap -> roster bridge 99.6386%;
- Stage-1 `p_active` attach 100%;
- canonical q finite 100%;
- maximum candidate mass gap 2.22e-16.

Named-player Stage-1 probability was used on 93.1452% of locked player-cell rows.

Important identity repair that is already DONE:
- Audric Estim -> Audric Estimé -> GSIS `00-0039373`;
- Drew Ogletree -> Andrew Ogletree -> GSIS `00-0037292`.

Do not reopen those as active blockers.

At lock:
- Week-6 target outcomes read = 0;
- SEA-DEN outcome read by shadow = 0;
- sportsbook upstream = 0;
- OddsAPI calls = 0;
- production mutation = false.

**Do not grade this shadow until all 13 included games are final.**
Continue research while waiting.

---

## 7. OPPORTUNITY QUALITY / TOPOLOGY — SOURCE READY AND RECONCILED

This is the current active architectural layer.

Critical anti-retest distinction:

The old question is CLOSED:
> Does a named player's recent high-value scoring-area share itself persist better than ordinary usage?

That exact V1:
`PLAYER_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_CLOSED`.

Do not rescue it with:
- alternate bands;
- alternate windows;
- lead-back/WR1/TE1 subsets;
- TD outcomes;
- threshold search.

The **new** question is materially different:

> Given PRISM's existing finite opportunity quantity and current named-player entitlement, how much high-value opportunity does the TEAM create?

Only after team high-value opportunity quantity is established may it be mapped through the current player allocation/state.

### Stage-2 compatibility source authority

Run:
- `38991429245` — SUCCESS;
- head `d14ba07f7f0e79f2b3f16a0b018a50c30744be91`;
- artifact `12085262613`;
- digest `sha256:77739799f34237538d7ffcba9715939c59d6a5b50f811f06f2f5836ff9deb31d`.

Disposition:
`PRISM_OPPORTUNITY_QUALITY_SOURCE_READY_ALL`.

Source-ready player-compatible quality cells:
- RB rushing: inside-10 rushing opportunity;
- RB receiving: red-zone targets;
- TE receiving: red-zone targets.

### Raw full finite team-pool source

Canonical hardened run:
- `38991605252` — SUCCESS;
- exact head `e590ce4754da2c7f7aa01beab2ee148aa5b613a0`;
- artifact `12085367606`;
- digest `sha256:68b680c745ae8b8479b3dbaf8f41dcc4f6cb728a0f39b0ac10b77dc3ac8e4b90`.

Disposition:
`PRISM_OPPORTUNITY_QUALITY_TEAM_VOLUME_SOURCE_READY`.

2022-2025 all pass:
- scheduled team-game coverage 100%;
- rusher identity 100%;
- target identity 100%;
- strictly-prior offense state attach 100%;
- strictly-prior opponent state attach 100%.

Drive topology is source-ready.
Optional end-zone fields are source-ready.
**Source readiness is not predictive permission.**

### Exact source reconciliation

Run:
- `38991737241` — SUCCESS;
- exact head `c1abb0bacd0c78dd0b38861214bd28c04cc9f376`;
- artifact `12085278127`;
- digest `sha256:0be3d571129876ce0a8df755762215e147faec2ff1ac1822e05cbd84508a12bb`.

Disposition:
`PRISM_OPPORTUNITY_QUALITY_SOURCE_RECONCILED_READY_ALL`.

Exact 2024/2025 Weeks 5-18 parity:
- 832 / 832 team-games;
- 0 left-only;
- 0 right-only;
- 0 mismatch rows;
- team rushes drift 0;
- inside-10 rushes drift 0;
- team targets drift 0;
- red-zone targets drift 0.

Interpretation:
- raw PBP owns the finite team high-value-pool truth;
- Stage-2 source owns player entitlement compatibility.

No predictive opportunity-quality candidate had been scored at this point.

---

## 8. NEWEST MILESTONE — PARENT DENOMINATOR PERSISTENCE PARITY PASSED

This is the newest fact and the one that must replace the older "denominator persistence active" checkpoint.

Frozen addendum:
`docs/research/PRISM_OPPORTUNITY_QUALITY_PARENT_DENOMINATOR_PERSISTENCE_V1_ADDENDUM.md`.

Freeze commit:
`3a980c9ba821c5e0a3b41231314db5c21bb0adef`.

Result:
`docs/research/PRISM_OPPORTUNITY_QUALITY_PARENT_DENOMINATOR_PERSISTENCE_V1_RESULT.md`.

Canonical execution:
- run `38991915503` — **SUCCESS**;
- job `117275946409`;
- exact head `02e96ad9faabee62c1ecb31303b48746e4945a8c`;
- artifact `12085194208`;
- digest `sha256:c2be00311f9cf2149ebef36d419851214add08cae83bbc3b947990ce60a5e569`.

Disposition:

`PRISM_PARENT_DENOMINATOR_PERSISTENCE_PARITY_PASS`.

Exact parity:
- **14,284 rows**;
- key sets exact = true;
- denominator finite flags exact = true;
- ready cells exact = true;
- scientific source disposition exact = true;
- maximum absolute canonical entitlement-q drift = **2.0816681711721685e-16**.

The persisted numeric parent field is:

`authority_denominator_mean`.

Semantics were **not** changed:

RB rushing:
`rules_plays_est * (1 - rules_pass_rate)`.

Receiving:
`rules_plays_est * rules_pass_rate`.

This was an output-lineage repair only:
- no denominator formula change;
- no entitlement change;
- no new share model;
- no predictive score;
- no sportsbook;
- no OddsAPI;
- no 2026 outcomes;
- no production mutation.

This closes the final source-lineage seam that was blocking a clean opportunity-quality mechanism.

---

## 9. EXACT NEXT SCIENTIFIC ACTION

The next chat should **not** search for another source first.

The source work for the first opportunity-quality mechanism is ready.

The next action is:

### Freeze ONE team high-value-opportunity mechanism BEFORE scoring.

Question:

> Conditional on PRISM's already-existing canonical pregame parent opportunity quantity, does a team's own strictly-prior high-value opportunity creation state improve the distribution/expectation of high-value opportunity count in the next game?

Keep this first mechanism intentionally narrow.

Allowed quality pools:
- RB rushing: **team inside-10 rushing opportunities**;
- RB receiving: **team red-zone targets**;
- TE receiving: **team red-zone targets**.

The mechanism must be built **on top of**:
- the persisted canonical parent denominator;
- existing canonical player entitlement;
- qualified Stage-1 active-role state;
- qualified Stage-2 conditional allocation.

The player-side allocation remains the current canonical entitlement / qualified Stage-2 logic.

Do **not** use recent player scoring-area share as the predictor.

### Chronology

Use:
- 2024 = development;
- 2025 = independent confirmation.

No 2026 outcomes.

### First-mechanism boundary

Do **not** add opponent/environment transmission yet.

This first experiment isolates **team creation of high-value opportunity**.

Opponent/environment belongs to the next architectural layer only after this team opportunity-quality mechanism is independently qualified.

### Required freeze philosophy

Freeze before target outcomes are scored:
- exact strictly-prior team history window/state;
- baseline;
- candidate formula;
- finite count/distribution law;
- support floors;
- player/team dependence-aware inference;
- mean/count protection;
- anti-rescue rules.

Do not tune after seeing 2025.

If this exact mechanism fails, close it and move forward.

---

## 10. PROTECTED / TERMINAL LANES

Do not reopen without genuinely new information:

- `PLAYER_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_CLOSED`;
- player role elasticity V1 source block;
- Availability -> Opportunity rule-order gap;
- M78/M79 generic inactive-QB identity;
- RB Vacancy;
- RB carry/snap allocator research already frozen/closed where applicable;
- RB receiving-room share duplicate proxies;
- WR/TE target-share trajectory duplicate research;
- WR excess-share / Dirichlet closed variants;
- TE allocation difficulty / joint Dirichlet closed variants;
- generic QB mean/YPA/attempt-volume retunes after M90;
- generic Vegas/game-script;
- generic defense-vs-position shortcuts;
- WR-CB paid-source dependency / unsupported historical assignment;
- generic ATD certification claims.

QB Stage-2 V1 remains closed.
WR Stage-2 remains source-blocked.
Generic ATD remains uncertified.

Preserve WR/TE player difficulty trust/selectivity qualifications and prospective locks.

---

## 11. ODDS / PRODUCTION GUARDRAILS

Exactly **one** paid Week-6 OddsAPI acquisition has already been used:
- run `38791377571`;
- artifact `11999923880`;
- digest `sha256:0916cd1cbec93f9a2529190b3df7c39172eee011a31eb9df429a35cdb9b41f61`.

**Do not spend another OddsAPI credit without explicit user authorization.**

Sportsbook data remains downstream comparison only.

Current research lane:
- sportsbook inputs 0;
- OddsAPI calls 0;
- 2026 outcomes 0;
- production mutation false.

Do not production-promote research merely because historical gates pass.

---

## 12. REPO / PR STATE AT HANDOFF CREATION

Immediately before writing the final handoff docs:
- PR #672 = OPEN / DRAFT;
- research branch = `research-individual-opportunity-roadmap-2026-10-08`;
- research branch head = `fd5572cbbeb96edd3e26889bf0b63e8b5edda279`;
- that commit publishes the denominator-persistence result;
- main had already advanced through the opportunity-quality source-reconciled checkpoint.

The handoff/checkpoint commits themselves will advance these SHAs.

Therefore the next chat must reconcile:
- live `main`;
- live research branch;
- PR #672;
- Issue #535 newest comments;
- exact-head Actions

before doing any new write.

---

## 13. OPERATING STYLE FOR THE NEXT CHAT

Do not ask the user to re-explain PRISM.

Do not narrate for long periods instead of working.

Do real GitHub work, give short progress updates, and continue through mechanical failures where the frozen science is unchanged.

Fail closed rather than weaken a gate.

Do not call a run successful until GitHub says SUCCESS.

Do not treat source readiness as predictive proof.

Do not treat historical qualification as production promotion.

Do not grade the Week-6 prospective shadow before all included games are final.

The immediate continuation is:
**freeze the first team high-value-opportunity creation mechanism, then run it under the frozen 2024-development / 2025-confirmation contract.**
