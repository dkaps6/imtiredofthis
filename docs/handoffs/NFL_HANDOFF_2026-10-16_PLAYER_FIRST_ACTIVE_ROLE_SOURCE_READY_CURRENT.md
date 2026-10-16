# NFL HANDOFF — 2026-10-16 — PLAYER-FIRST ACTIVE-ROLE SOURCE READY / HIGH-VALUE PERSISTENCE CLOSED

**Status:** CURRENT  
**Repository:** `dkaps6/imtiredofthis`  
**Production branch:** `main`  
**Active research branch:** `research-individual-opportunity-roadmap-2026-10-08`  
**Research PR:** #672 — OPEN / DRAFT  
**Canonical research status:** `PLAYER_INDIVIDUALIZATION_PARTIAL`

This handoff is intentionally self-contained enough for a new chat to resume as if the conversation had not ended.

> **FINAL CONTINUITY SEAL — 2026-10-16 08:06 ET**
>
> Live GitHub immediately before this docs-only seal:
> - production `main`: `29213acc71e342c09537c198626c8a5c79a33353` — `Docs: advance canonical active-role source checkpoint`;
> - research branch / PR #672 head: `b77faa7f0f85a122b391683a7079c96a071d1036` — `Docs: advance top checkpoint to active-role source ready`;
> - PR #672: OPEN / DRAFT;
> - Issue #535 newest continuity comment: `6215000334`;
> - newest scientific authority remains active-role source run `38895205962` — SUCCESS, exact scientific head `28ebf8d7159fdd1fec8c2e4a0fe1c9e2c691537c`, artifact `12044155086`, digest `sha256:a42322579aee682497223c7f53bce0b14a96dc6f466558e811770e681a035dad`.
>
> **No active-role mechanism result exists yet.** The exact next action remains to freeze that mechanism before scoring. Do not skip ahead, do not use 2026 outcomes, and do not spend another Week-6 OddsAPI credit.
>

---

## 1. NON-NEGOTIABLE MODEL OBJECTIVE

This is an **individual-player projection model**.

The primary scientific unit is the named player in his current football situation.

Do not optimize around position-average MAE/RMSE as the main objective. Those are downstream diagnostics. Individual players create those aggregate errors.

The model must reason in this order:

`PLAYER IDENTITY -> CURRENT ROLE/STATE -> OPPORTUNITY ALLOCATION -> OPPORTUNITY QUALITY -> OPPONENT/ENVIRONMENT -> EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`.

The user explicitly wants the model to know:

- who the player is;
- what his role is now;
- whether his role is growing/shrinking/changing;
- who he shares opportunity with;
- what situations he is used in;
- what kind of opportunity he receives;
- what obstacle/matchup/environment he faces this week;
- how that player historically behaves in comparable role/state contexts;
- whether the current projection is reliable for this exact player.

Position aggregates remain valid as priors, shrinkage, sparse-player fallback, calibration and QA. They do **not** define the player.

Do not kill validated existing science. This player-first architecture is additive.

Canonical files:
- `AGENTS.md`
- `docs/research/PLAYER_IDENTITY_ROLE_STATE_ARCHITECTURE_V1.md`

---

## 2. THE FOOTBALL MOMENT THAT DROVE THE CURRENT RESEARCH

SEA @ DEN on 2026-10-15 crystallized the user's long-running argument.

The model-backed ticket legs mostly failed. The user reported:
- Sam Darnold passing under destroyed the slips after ~100 first-half passing yards and a much larger second half;
- Evan Engram receiving did not hit;
- Cooper Kupp receiving yards hit;
- J.K. Dobbins anytime TD hit.

The Dobbins ATD was **not** selected by certified ATD science. It was a cautious football-judgment call based on the player's current role, recent/current usage, likely scoring-area opportunity and the expectation that if Denver reached a scoring situation, Dobbins had a strong chance to receive the valuable carry.

The user then made the key point:

> That is exactly what the projection model should be able to do for every named player.

This is not a request to abandon historical modeling. Historical data should help define **this player's identity and role playbook**, while current-week state determines what is true today.

Examples of the intended reasoning:
- Jahmyr Gibbs history can show usage trajectory, red-zone role, third-down role and teammate sharing patterns;
- that history does not mean the same thing automatically happens today;
- the opponent/environment is an obstacle the player faces, not the player's identity;
- teammate competition determines how much of a finite opportunity pool belongs to this player;
- different markets ask different football questions;
- a player can simultaneously be a weak rushing-yards candidate and a strong TD-opportunity candidate.

Do not reduce this lesson to a single red-zone coefficient.

---

## 3. WEEK-6 PRODUCTION / ODDS GUARDRAIL

Week-6 production is already complete and protected.

Football-only run:
- run `38790425676` — SUCCESS
- artifact `11999618082`
- digest `sha256:d7473705d4534c3172b58843272e7d95f5fbbd9aa910238d48916a8567cde1de`
- 14/14 games eligible
- no live odds.

Exactly **ONE** authorized paid Week-6 OddsAPI acquisition has already been used:
- run `38791377571`
- artifact `11999923880`
- digest `sha256:0916cd1cbec93f9a2529190b3df7c39172eee011a31eb9df429a35cdb9b41f61`.

Final green preserved replay:
- run `38792341891` — SUCCESS
- artifact `12000113891`
- pricing provenance `PRESERVED_REPLAY`
- `odds_api_refetched=false`.

**Do not spend another Week-6 OddsAPI credit without explicit user authorization.**

Research work remains sportsbook-independent upstream.

---

## 4. TICKET CONSTRUCTION CONTINUITY

Canonical ticket-construction process:
`docs/research/MODEL_BACKED_ALT_LINE_PARLAY_CONSTRUCTION_V1.md`.

Core rules:
- football model first, not raw edge;
- coherent player/market football reason required;
- prefer clean disagreement over giant-looking disagreement;
- reconcile current availability and exact entry line;
- alt lines preserve the thesis and reduce ordinary variance;
- overs buy down, unders buy up;
- same candidate quality across all ticket tiers;
- payout comes from accumulated leg count, not garbage unsupported legs;
- avoid universal player anchors when possible;
- no largest-edge/top-N automation;
- no stale projections after availability changes;
- sportsbook remains downstream.

ATD remains **judgment-only / non-certified** unless a future dedicated ATD model qualifies.

Do not promote Dobbins' one successful ATD selection into model proof.

---

## 5. QUALIFIED WR / TE RELIABILITY STATE THAT MUST REMAIN PRESERVED

### WR opportunity-difficulty trust

Historical disposition:
`WR_PLAYER_OPPORTUNITY_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED`

Authority:
- run `38823063793` — SUCCESS
- job `116723999254`
- exact scored head `35a10d09ba92fd1218d37e812768fda5d2f4023e`
- artifact `12013153664`
- digest `sha256:12fd0de1dc3ebceee4d356669d59f30d96776d8c0800d1fb3ac6a293af457b7c`
- result `docs/research/WR_PLAYER_OPPORTUNITY_DIFFICULTY_TRUST_SELECTIVITY_V1_RESULT.md`.

Key:
- 3,224 rows / 211 WRs;
- pooled rank association +0.28853;
- HIGH_TRUST abs error 14.61 yd;
- overall 23.25 yd;
- LOW_TRUST 30.47 yd;
- dependence-aware support passed.

Meaning:
same-player strictly-prior opportunity difficulty is a qualified reliability selector around the **unchanged WR-R15 projection**.

It is NOT:
- over/under direction;
- bet/fade;
- mean mutation;
- width mutation.

Week-6 prospective WR trust lock:
- `WEEK6_WR_TRUST_PREGAME_LOCK_FROZEN`
- run `38823520954`
- artifact `12012884262`
- 152 current WR / 141 supported
- HIGH 35 / MIDDLE 70 / LOW 36 / unsupported 11
- Week-6 outcomes read 0 at lock time.

### TE trust

Historical:
`TE_PLAYER_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED`
- run `38821176218`
- artifact `12013570473`
- digest `sha256:66c7b39f08b956dc1a96bd068272d1b97ddb5ddba22ef928a7d6e02c7f463530`.

Week-6 lock:
`WEEK6_TE_TRUST_PREGAME_LOCK_FROZEN`
- run `38822346088`
- exact head `1edecb1b3f60089e95bfc4c100522baf60d560a9`
- artifact `12013018536`
- 105 TE-R5P players / 103 supported
- HIGH 25 / MIDDLE 52 / LOW 26 / unsupported 2
- Week-6 outcomes read 0 at lock time.

Do not alter those frozen labels using target-week results.

---

## 6. RESEARCH PROGRESSION SINCE THE DOBSINS DISCUSSION

### A. Named-player role elasticity under teammate availability — SOURCE BLOCKED

Question:
when a named player's competitive room changes before kickoff, does this exact player absorb/lose opportunity in a repeatable way?

Frozen source result:
`PLAYER_ROLE_ELASTICITY_SOURCE_BLOCKED`

Authority:
- run `38847501580` — SUCCESS
- job `116801249742`
- exact head `fb516aea9e8af21563af71a6f1506edd154d29af`
- artifact `12024335417`
- digest `sha256:1a22efad89da4593124d366d21176d8c70d0d5f2cc0e7498a6216fab57353186`
- result `docs/research/PLAYER_ROLE_ELASTICITY_TEAMMATE_AVAILABILITY_V1_SOURCE_RESULT.md`.

Important:
identity/snap/event/multi-room-state support was strong.

Only blocker:
frozen >=90% strictly-prior same-team opportunity-history coverage failed.

Observed:
- RB: 88.68% / 88.36% in 2024/2025;
- WR: 88.67% / 89.26%;
- TE: 85.59% / 84.88%.

Do not lower the gate.
Do not infer missing prior opportunity from target-game outcomes.
Do not rescue the old vacancy formula.

This is a source-readiness block, not evidence that role elasticity is false.

---

### B. High-value scoring-opportunity identity — SOURCE READY

Current generic ATD trace was audited.

Existing generic logic:
- historical player offensive TD/game rate;
- position fallback if player history unavailable;
- blended `rz_tgt_share`;
- blended `rz_carry_share`;
- `rz_share = max(rz_tgt_share, rz_carry_share)`;
- generic `0.75 + rz_share` modifier;
- small team-win-probability script modifier;
- Bernoulli TD draw.

Canonical validator still says:
`DEDICATED_ATD_WALK_FORWARD_CERTIFICATION_REQUIRED`.

Its required conceptual frontier:
`football-only team TD opportunity -> red-zone/end-zone pool -> player entitlement -> probability calibration`.

Source audit:
`PLAYER_HIGH_VALUE_SCORING_OPPORTUNITY_SOURCE_READY`

Authority:
- run `38848080453` — SUCCESS
- exact head `1448fcce54f803d9737a7b3f7d9ff451b5189add`
- artifact `12023852453`
- digest `sha256:1d8a22bf6240b4d4443ab7432d3224b63eaef97439639e7b47f536aed174adda`
- result `docs/research/PLAYER_HIGH_VALUE_SCORING_OPPORTUNITY_IDENTITY_V1_SOURCE_RESULT.md`.

The repo can cleanly measure from 2022-2025:
- named RB inside-20 / inside-10 / inside-5 rushing opportunity;
- named RB/WR/TE red-zone target opportunity;
- stable player identity;
- optional end-zone-target source also passed.

No sportsbook / OddsAPI / 2026 outcomes were used.

---

### C. Simple high-value opportunity persistence V1 — CLOSED

Frozen question:
does the player's recent high-value share predict next-game high-value share better than ordinary overall usage share?

Result:
`PLAYER_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_CLOSED`

Authority:
- run `38848538570` — SUCCESS
- exact scored head `89b92b56d9b9786a3cc8802a3d2b645caeca9ba5`
- artifact `12024037594`
- digest `sha256:2b02f5a6c7a8a05962517f72a9d62ba287d06ffe5ced37abfd70b623a5120afe`
- result `docs/research/PLAYER_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_V1_RESULT.md`.

RB rush:
- 2022 worsened;
- 2023/24 tiny positive;
- 2025 +0.00437 share-MAE gain, but both 2025 clustered CIs crossed zero;
- pooled gain only +0.00046 and pooled clustered CIs crossed zero.
CLOSED.

RB receiving:
- favorable all four seasons;
- pooled gain +0.00220;
- pooled player CI [+0.00011,+0.00419];
- pooled game CI [+0.00090,+0.00357];
- but 2025 player CI [-0.00175,+0.00611];
- 2025 game CI [-0.00044,+0.00449].
CLOSED under frozen independent-confirmation gates.

WR receiving:
- worse all four seasons;
- pooled gain -0.00263 with clustered CIs strictly negative.
CLOSED.

TE receiving:
- worse all four seasons;
- pooled gain -0.00571 with clustered CIs strictly negative.
CLOSED.

Interpretation:
the Dobbins reasoning is **not** captured by one recent red-zone-share coefficient.

Do not rescue with:
- different yardline bands;
- different recency windows;
- lead-back / WR1 / TE1 subsets;
- pooled positions;
- TD outcomes.

---

## 7. CURRENT FRONTIER — ACTIVE-ROLE / ZERO-STATE PARTICIPATION

This is the most current research state and where the next chat should resume.

The W1-W4 full-roster replay exposed a distinct zero-state problem:
- 1,837 player-weeks;
- 610 / 1,837 = 33.21% had no weekly-stat row;
- QB 7.81%;
- RB 30.15%;
- WR 31.55%;
- TE 45.31%.

The correct response is **not** to delete players using postgame evidence.

The player-first model is now explicitly two-stage:

1. **Active-role participation:** will this named player have a real offensive role at all?
2. **Conditional opportunity allocation:** if he participates, how much team/room opportunity belongs to him?

Do not combine the two before Stage 1 is scientifically qualified.

Frozen source contract:
`docs/research/PLAYER_ACTIVE_ROLE_PARTICIPATION_V1_SOURCE_AUDIT.md`.

Source result:
`docs/research/PLAYER_ACTIVE_ROLE_PARTICIPATION_V1_SOURCE_RESULT.md`.

### Canonical corrected source result — READY ALL

Disposition:
`PLAYER_ACTIVE_ROLE_PARTICIPATION_SOURCE_READY_ALL`

Canonical corrected run:
- run `38895205962` — SUCCESS
- job `116953914726`
- exact head `28ebf8d7159fdd1fec8c2e4a0fe1c9e2c691537c`
- artifact `12044155086`
- digest `sha256:a42322579aee682497223c7f53bce0b14a96dc6f466558e811770e681a035dad`.

The corrected head specifically excludes bye roster weeks before adjudication.

Per-position dispositions:
- `QB_ACTIVE_ROLE_PARTICIPATION_SOURCE_READY`
- `RB_ACTIVE_ROLE_PARTICIPATION_SOURCE_READY`
- `WR_ACTIVE_ROLE_PARTICIPATION_SOURCE_READY`
- `TE_ACTIVE_ROLE_PARTICIPATION_SOURCE_READY`.

Every frozen source gate passed in both 2024 and 2025:
- roster team-week coverage 100%;
- snap team-week coverage 100%;
- stable roster identity 100%;
- positive-snap -> roster bridge 100%;
- chronology violations 0;
- inactive bridge above frozen 95%;
- position-specific support floors passed.

### Source support

QB:
- 2024: 1,097 pregame player-games / 90 players / 416 valid zero-snap rows / 62.08% participation;
- 2025: 1,107 / 95 / 420 zero-snap / 62.06%.

RB:
- 2024: 1,845 / 166 / 233 zero-snap / 87.37%;
- 2025: 1,866 / 170 / 235 zero-snap / 87.41%.

WR:
- 2024: 2,834 / 253 / 183 zero-snap / 93.54%;
- 2025: 2,817 / 253 / 190 zero-snap / 93.26%.

TE:
- 2024: 1,730 / 143 / 66 zero-snap / 96.18%;
- 2025: 1,715 / 146 / 69 zero-snap / 95.98%.

This is a real, source-supported player-state problem.

### IMPORTANT: no mechanism result exists yet

The source result authorizes the next step. It does **not** prove a participation predictor.

No active-role predictive/mechanism test has been frozen/scored yet.

The next scientific question is:

> Does the named player's own strictly-prior participation history forecast whether he records any offensive snap in the target game better than a strictly-prior position-level participation baseline?

Target:
- binary any-offensive-snap in a valid target team-week.

Suggested source-defined state from the already-frozen audit:
- same-team last-4 eligible games;
- minimum 3;
- prior participation occurrence only;
- no target-game information.

But the exact baseline/candidate/probability mapping/inference/support gates must be **frozen before scoring**.

This is the exact next action.

---

## 8. ACTIVE-ROLE ANTI-RETEST BOUNDARY

Do NOT relabel/reopen:

- `AVAILABILITY_OPPORTUNITY_RULE_ORDER_GAP_CONFIRMED`;
- M78/M79 generic official-inactive QB identity;
- RB Vacancy;
- player role-elasticity teammate-availability V1;
- RB carry/snap allocation;
- RB receiving-room share;
- WR/TE target-share trajectory;
- WR-R15 / TE-R5P snap entitlement;
- route participation/YPRR historical source work;
- postgame DNP/zero settlement rules.

This active-role probability is a separate upstream zero-state layer.

Existing WR/TE snap state used in mean-entitlement models does not equal a calibrated cross-position participation probability.

Do not define target-game participation using:
- whether weekly stats exist;
- target-game carries/targets;
- target-game snap percentage as a predictor;
- postgame DNP labels.

Pregame roster universe is fixed before target-game snap outcome.

---

## 9. OTHER PERMANENT / IMPORTANT STOP RULES

Do not restart these completed or terminal lanes unless a genuinely new source/mechanism is first demonstrated:

- generic QB mean/YPA hunt after M90;
- target-depth symmetric distribution transform;
- WR route full-stack mean integration;
- WR allocation-only width bridge;
- TE global/marginal width;
- TE Normal-width -> rho -> Dirichlet/Beta-Binomial mappings;
- TE targeted-route families;
- situational third-down/red-zone/two-minute target earning as direct opportunity-residual explanation;
- generic game-script/market-confirmation upstream features;
- simple football matchup transmission;
- defender-injury source-parity lane;
- WR-CB historical source without a free qualifying source;
- generic individual efficiency-tail volatility V1;
- direct WR/TE depth-YPT;
- RB direct run-location-YPC;
- RB-PD2 retrospective rescue.

RB-PD2 Weeks 4/5/6 are permanent prospective non-observations.

Do not use sportsbook data to train football state.

---

## 10. LIVE GITHUB STATE AT THIS HANDOFF

At handoff construction:

Production main:
- `6053c6862544ab4c35e772207b61f861484c01ef`
- message: `Docs: advance canonical player-first research checkpoint`.

Research PR #672:
- OPEN
- DRAFT
- branch `research-individual-opportunity-roadmap-2026-10-08`.

Current research branch head before this handoff write:
- source-ready correction head `28ebf8d7159fdd1fec8c2e4a0fe1c9e2c691537c`
- plus source-result documentation commit `13d4cc5910451965dafb9b26e4992d0c2846b61d`.

Issue #535 remains the anti-retest/control-room ledger.

Newest relevant Action:
- `38895205962` — `Research Player Active Role Participation Source Audit V1` — SUCCESS.

No paid OddsAPI was used in any of this research.

---

## 11. EXACT NEXT ACTION FOR THE NEXT CHAT

Do this in order:

1. Read `AGENTS.md`.
2. Read **ONLY the newest TOP checkpoint** in `CURRENT_NFL_RESEARCH_HANDOFF.md`.
3. Read this handoff.
4. Reconcile live:
   - `main`;
   - branch `research-individual-opportunity-roadmap-2026-10-08`;
   - draft PR #672;
   - Issue #535 newest comments;
   - exact-head GitHub Actions.
5. Confirm the corrected active-role source result still equals:
   `PLAYER_ACTIVE_ROLE_PARTICIPATION_SOURCE_READY_ALL`,
   run `38895205962`,
   artifact `12044155086`.
6. **Freeze the active-role mechanism contract before any outcome scoring.**
7. Keep QB/RB/WR/TE separate.
8. Use named-player same-team strictly-prior participation state.
9. Compare against a strictly-prior position-level participation baseline.
10. Add player-level heterogeneity and player/game clustered inference.
11. Do not mutate production from source readiness alone.
12. Do not use sportsbook/OddsAPI/2026 outcomes for this historical mechanism.
13. If the mechanism qualifies, only then design Stage 2:
    conditional opportunity allocation given participation.

Do not ask the user to re-explain any of this.

---

## 12. COMMUNICATION / OPERATING STYLE

The user expects:
- real GitHub work, not fabricated/local-only claims;
- exact Actions/run/artifact authority;
- concise check-ins while work is running;
- no abandoned partial experiments;
- fail closed rather than weaken gates;
- no repeated explanation of already-canonical context;
- player-level football reasoning;
- clear distinction between what is source-ready, scientifically qualified, prospectively frozen, production-promoted, or judgment-only.

When a lane fails:
record it, protect the anti-retest boundary, and move forward.

When a lane passes:
do not over-promote it. Freeze the next bridge separately.

The objective is not to make a metric look better.

The objective is to project **this player, in this role, in this game, against this environment**, as accurately and coherently as possible.
