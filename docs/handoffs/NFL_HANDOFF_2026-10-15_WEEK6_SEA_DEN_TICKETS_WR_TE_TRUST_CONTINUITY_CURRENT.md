# NFL HANDOFF — 2026-10-15 — WEEK 6 SEA-DEN TICKETS + WR/TE TRUST LOCKS + RESEARCH CONTINUITY

GitHub is canonical. This handoff exists because the prior chat hit its conversation limit while switching between **tonight's SEA-DEN betting tickets** and the **individual-player research lane**.

The next chat must feel like the exact same conversation continuing. Do not ask the user to re-explain anything.

## 0. Read order

Read in this exact order:

1. `AGENTS.md`
2. ONLY the newest TOP checkpoint in `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. this file:
   `docs/handoffs/NFL_HANDOFF_2026-10-15_WEEK6_SEA_DEN_TICKETS_WR_TE_TRUST_CONTINUITY_CURRENT.md`

Then reconcile live:
- `main`
- research branch `research-individual-opportunity-roadmap-2026-10-08`
- draft PR #672
- Issue #535 newest comments/state
- exact-head Actions relevant to any resumed research work.

Do not recursively read old handoffs.

---

## 1. User's model philosophy — non-negotiable

This is an **individual-player projection system**, not a position-average system.

The user explicitly rejects "we improved positional MAE" as sufficient proof that a player-level mechanism is solved.

Research and betting decisions must remain player-first:
- player usage / role;
- team offensive environment;
- player history and talent;
- opponent/matchup state only where source-qualified / validated;
- individual reliability/fragility;
- current starter and availability context.

Position aggregates are diagnostics/guardrails only.

The user wants fewer, better bets, not a board where everything becomes a giant "edge."

---

## 2. Production Week 6 — complete and protected

Canonical production main:
`9b70b12e2fb4bc39f08556f244370068aad9c47b`

### Football-only validation
- run `38790425676` SUCCESS
- artifact `11999618082`
- digest `sha256:d7473705d4534c3172b58843272e7d95f5fbbd9aa910238d48916a8567cde1de`
- 14/14 games eligible
- 0 withheld
- no live odds.

### One and only authorized paid Week-6 OddsAPI acquisition
Paid source run:
- run `38791377571`
- artifact `11999923880`
- digest `sha256:0916cd1cbec93f9a2529190b3df7c39172eee011a31eb9df429a35cdb9b41f61`

That acquisition succeeded. A downstream lineage issue was repaired without refetching.

**DO NOT SPEND ANOTHER WEEK-6 ODDSAPI CREDIT.**

### Final green preserved replay
- run `38792341891` SUCCESS
- artifact `12000113891`
- digest `sha256:dd54f2f505d7ec96caa98ba2f660454d25d2efcacb91bfeb0ba2f5d7fdd187c5`
- pricing provenance: `PRESERVED_REPLAY`
- `odds_api_refetched=false`.

The Jalon Daniels starter-style rows were quarantined after current role authority still had Baker Mayfield as QB1.

Final priced board:
- 2,528 side rows
- 315 players
- Jalon Daniels rows = 0.

---

## 3. Tonight's game — SEA @ DEN, Thursday 2026-10-15

Correct game:
**Seattle Seahawks at Denver Broncos**, 8:15 PM ET.

Do not accidentally call tonight PIT-TB. PIT-TB is Sunday.

The user is currently focused on creating tickets for this game.

### Frozen model-backed SEA-DEN candidate pool already recovered

Core individual board:

| Player | Market | Main/working line | Model mean | Direction |
|---|---|---:|---:|---|
| Sam Darnold | passing yards | 223.5 | 212.6 | UNDER |
| Cooper Kupp | receiving yards | 21.5 | 28.1 | OVER |
| Evan Engram | receiving yards | 20.5 | 25.8 | OVER |
| Courtland Sutton | receptions | 3.5 | 2.56 | UNDER |
| Troy Franklin | receiving yards | 18.5 | 22.2 | OVER |
| Horton | receiving yards | 3.5 | 16.1 | OVER |

Do not infer new current odds from these numbers. Reconcile a user's sportsbook screenshot or exact current book offer if needed.

### Ticket ladder that had already been built before the user asked to return to the repo construction model

Recovered verbatim working tickets:

**Core / safer**
- Darnold U249.5 passing
- Kupp 20+ receiving
- Engram 15+ receiving

**Core + juice / moderate**
- Darnold U239.5 passing
- Kupp 20+ receiving
- Engram 20+ receiving
- Sutton U4.5 receptions

**Aggressive**
- Darnold U229.5 passing
- Kupp O21.5 receiving
- Engram O20.5 receiving
- Sutton U3.5 receptions
- Franklin 15+ receiving

The exact final lottery ticket had NOT been faithfully recovered in chat. Do not fabricate one from memory.

The user then correctly said: use the actual GitHub ticket-creation process rather than hand-reconstructing it.

---

## 4. Exact ticket-construction contract to use now

Canonical process:
`docs/research/MODEL_BACKED_ALT_LINE_PARLAY_CONSTRUCTION_V1.md`

Status:
`DOCUMENTED WORKING PROCESS — NOT A VALIDATED DOWNSTREAM BET SELECTOR`

The process says:

1. start from the canonical football model, not largest raw sportsbook edge;
2. require a coherent football reason for every candidate;
3. prefer clean disagreement over gigantic disagreement;
4. reconcile current availability/starter state and actual entry line;
5. use alt lines to preserve the football thesis while reducing ordinary game variance;
6. keep known distorted/uncertified lanes out;
7. diversify fragile assumptions / players across tickets;
8. record main line, chosen line, model mean, lineage, availability, main-vs-alt rationale, caveats and ticket assignment.

Official tiers in the repo:
- **Core / "lock-style"**: 2-3 legs, strongest lineage, alt lines favored when useful.
- **Core + juice**: 4-5 legs, same quality standard, more independent football hypotheses.
- **Aggressive**: 5-7+ legs, more accumulated variance, every leg still model/football backed.
- **High-ceiling / lottery**: intentionally tiny ticket probability through leg count; **individual leg quality does not fall**.

The lottery ticket must NOT be created by simply adding random longshots.

### Immediate unfinished task

Resume exactly here:

> Use `MODEL_BACKED_ALT_LINE_PARLAY_CONSTRUCTION_V1.md` plus the frozen Week-6 SEA-DEN board to construct tonight's final Core / Moderate / Aggressive / Lottery tickets. Reconcile current lines if the user supplies a sportsbook screenshot. Do not merely sort raw edge.

The user specifically wants the lottery ticket produced from this contract, not a hand-waved reconstruction.

---

## 5. ATD permission for tonight — important exception with boundaries

Dedicated ATD science is still **NOT certified**:
`atd_dedicated_science_certified=false`.

Normally the construction V1 excludes uncertified ATD probabilities from core bets.

However, for tonight the user explicitly authorized cautious ATD suggestions as a **separate football-judgment layer**.

Allowed tonight:
- reason from role, red-zone/goal-line usage, receiving/rushing role, expected game environment, opponent, and the offered ATD price;
- compare price versus plausible football likelihood;
- use as an optional high-ceiling/lottery ingredient if the football case is coherent.

Required wording:
- ATD is judgment-only / non-certified by the current model;
- do not present a model probability that the model does not actually produce;
- do not let an ATD become the foundation of the "safe/core" ticket;
- price matters; if current odds have changed materially, reassess.

Previously discussed rough ATD ideas included:
- JSN around +107
- J.K. Dobbins around +180 to +190
- Cooper Kupp around +410
- Evan Engram around +550

Those prices were discussion-time values, NOT guaranteed current odds. Verify current offered price before using them.

---

## 6. Research lane — do not lose this while handling tonight's game

Research branch:
`research-individual-opportunity-roadmap-2026-10-08`

Draft PR:
#672

Live branch head immediately before this handoff work:
`638037c82b81e3e0f4d01ce37b1db24e5cc8144b`

Canonical overall status:
`PLAYER_INDIVIDUALIZATION_PARTIAL`

### TE trust/selectivity — qualified

Historical:
- disposition `TE_PLAYER_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED`
- run `38821176218` SUCCESS
- artifact `12013570473`
- digest `sha256:66c7b39f08b956dc1a96bd068272d1b97ddb5ddba22ef928a7d6e02c7f463530`

Meaning:
same-player strictly-prior TE allocation difficulty predicts reliability/fragility of the **unchanged TE-R5P projection**.

It does NOT authorize another TE width/mean/joint law.

Prospective Week-6 TE shadow:
- status `WEEK6_TE_TRUST_PREGAME_LOCK_FROZEN`
- run `38822346088` SUCCESS
- artifact `12013018536`
- digest `sha256:98667675753bfb410c6ac6259dc097476a87e49ce29f5cfe71be6fe14a74e1f1`
- row digest `sha256:fb5e2f83822a76adf3f09a721738237158bfddd9f95a3e63a97d3e4334d8ffa5`
- cohort 105
- supported 103
- HIGH 25 / MIDDLE 52 / LOW 26 / unsupported 2
- Week-6 outcomes read 0.

### WR trust/selectivity — NEWLY QUALIFIED immediately before timeout

Canonical historical result:
- result `docs/research/WR_PLAYER_OPPORTUNITY_DIFFICULTY_TRUST_SELECTIVITY_V1_RESULT.md`
- disposition `WR_PLAYER_OPPORTUNITY_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED`
- run `38823063793` SUCCESS
- job `116723999254`
- exact scored head `35a10d09ba92fd1218d37e812768fda5d2f4023e`
- artifact `12013153664`
- digest `sha256:12fd0de1dc3ebceee4d356669d59f30d96776d8c0800d1fb3ac6a293af457b7c`

Exact lineage parity proves this is reliability of the **final WR-R15 receiving-yard point projection**, not merely a proxy:
- 3,224 matched rows
- 211 WRs
- max signed/absolute parity gap ~2.84e-14.

Historical result:
- 2023 association +0.29403
- 2024 independent association +0.28453
- pooled +0.28853
- pooled HIGH_TRUST absolute error 14.61 yd
- overall 23.25 yd
- LOW_TRUST 30.47 yd
- pooled player-cluster 95% CI [+0.24177,+0.33078]
- pooled game-cluster 95% CI [+0.25525,+0.32106]
- 2024 player-cluster 95% CI [+0.23014,+0.33658]
- 2024 game-cluster 95% CI [+0.24337,+0.32665]
- all frozen gates passed.

Interpretation:
same-player strictly-prior WR opportunity difficulty strongly separates reliable from fragile **unchanged WR-R15 receiving-yard projections**.

This does NOT authorize:
- mean correction;
- width law;
- WR1-specific rescue;
- automatic over/under;
- automatic bet/fade.

### Week-6 WR trust shadow — NEWLY FROZEN successfully

Canonical:
- result `docs/research/WR_PLAYER_OPPORTUNITY_DIFFICULTY_TRUST_WEEK6_PREGAME_LOCK_V1_RESULT.md`
- status `WEEK6_WR_TRUST_PREGAME_LOCK_FROZEN`
- run `38823520954` SUCCESS
- job `116725474887`
- exact head `638037c82b81e3e0f4d01ce37b1db24e5cc8144b`
- artifact `12012884262`
- digest `sha256:e5bc113e507ded651ecc29080c7fdbe4fd55182d1d039d0dd14057ada4b7bb38`
- row digest `sha256:81d9e950b96d7ac9d5699eba8490fc741a5aab2c938ef29287dc261534346b8f`

Week-6 WR cohort:
- 152 current WRs
- 141 supported
- 11 unsupported
- HIGH_TRUST 35
- MIDDLE 70
- LOW_TRUST 36
- reconstructed W1-W5 history rows 717
- week-pool MAE 16.1793567183 yd
- Week-6 outcomes read 0
- sportsbook 0
- OddsAPI 0
- models fit 0
- projection/distribution changes 0
- production unchanged.

Do not alter these labels with Week-6 outcomes. Score only after Week 6 settles.

---

## 7. Research anti-retest / closed lanes

Do not reopen completed/closed work merely because the next chat is new.

Key protected closures:
- broad generic QB mean/YPA hunts closed;
- target-depth full symmetric distribution transform failed and is closed;
- WR targeted-route mechanism confirmed for YPT/catch but exact full-stack mean integration closed;
- WR allocation-only width bridge closed;
- TE global receiving-yard width closed;
- TE player allocation marginal width exact mapping closed;
- TE Normal-width -> rho -> Dirichlet/Beta-Binomial joint transmission closed;
- TE targeted-route families closed;
- situational target earning closed;
- generic game script / market confirmation closed as upstream football input;
- simple football matchup transmission candidates closed;
- defender-injury source-parity blocked;
- WR-CB historical source unavailable without paid source.

Existing confirmed named-player uncertainty:
- WR `WR_PLAYER_OPPORTUNITY_UNCERTAINTY_CONFIRMED`
- TE `TE_PLAYER_OPPORTUNITY_UNCERTAINTY_CONFIRMED`
- RB historical `RB_YARD_DIFFICULTY_MC_WIDTH_QUALIFIED`.

The new WR/TE trust results are reliability/selectivity layers, not production mutations.

---

## 8. Research frontier after tonight's ticket work

Before opening anything new, reconcile Issue #535 and the newest result docs.

The most important scientific lesson now is that both WR and TE have strong **same-player reliability state** around unchanged final receiving-yard projections.

Legal next directions must be genuinely distinct and player-first.

Potential frontier from the prior TE handoff:
- named-player **efficiency reliability/regime state**, distinct from already-closed generic efficiency-tail volatility;
- preserve existing means on first diagnostic;
- separate opportunity from efficiency;
- require season replication and player/game clustered support;
- anti-retest before freezing.

Do NOT jump into a new lane until the newest WR/TE trust locks are recorded and the user is no longer actively handling tonight's game.

---

## 9. Communication behavior

The user expects:
- concise but substantive progress updates during long GitHub work;
- no fake/local-only runs presented as model runs;
- real GitHub Actions authority;
- GitHub as canonical;
- no paid OddsAPI without explicit authorization;
- no asking them to repeat context already recorded;
- player-level football reasoning, not positional averages.

Right now prioritize tonight's ticket request first if the user asks about bets. Research is saved and should resume afterward.

---

## 10. Immediate next response in new chat

After reading/reconciling, say plainly that continuity is loaded.

If the user wants tonight's ticket:
1. use `MODEL_BACKED_ALT_LINE_PARLAY_CONSTRUCTION_V1.md`;
2. use the frozen SEA-DEN candidate pool;
3. reconcile any current sportsbook lines they provide;
4. produce Core / Moderate / Aggressive / Lottery;
5. use ATD only as explicitly labeled judgment-only optional high-ceiling material;
6. do not burn another Week-6 OddsAPI credit.

If the user says "go back to research":
resume from the WR + TE trust-lock state above and anti-retest Issue #535 before opening a distinct player-level frontier.
