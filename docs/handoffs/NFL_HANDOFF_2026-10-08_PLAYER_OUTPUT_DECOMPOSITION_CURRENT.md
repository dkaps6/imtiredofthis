# NFL HANDOFF — 2026-10-08 — PLAYER OUTPUT DECOMPOSITION / INDIVIDUAL OPPORTUNITY NEXT

**Repository:** `dkaps6/imtiredofthis`  
**GitHub is canonical. Chat memory is secondary.**  
**This document is the current detailed continuity authority for the next chat.**

---

## 0. Why this handoff exists

The current conversation reached the end of its usable context after a long player-centric research sequence.

The user explicitly wants the next chat to feel like the same assistant continuing with no reset, no re-explanation, no duplicated research, and no reopening of closed science.

The exact philosophical requirement from the user is:

> The model must project the **individual player**, not merely a position row with a name attached. For a named player it should understand his role, expected usage, room competition, offense, opponent, defensive matchup, injuries/vacancies, efficiency, and uncertainty, and explain why his specific output should be higher or lower.

That requirement has now been audited directly.

The key new scientific conclusion is even more specific:

> The current model **is genuinely individualized**, but the dominant remaining error is **individual opportunity/workload allocation**, not generic efficiency or matchup conversion.

The next research work must therefore stay inside **pregame role/share/opportunity allocation** until that bottleneck is resolved.

---

# 1. Operating contract — preserve exactly

Read root `AGENTS.md` first.

Hard rules:

- canonical repo: `dkaps6/imtiredofthis`
- production authority: `.github/workflows/full-slate.yml`
- current production season: 2026
- no paid OddsAPI pull without explicit user authorization
- sportsbook/player props may not construct the independent football projection
- do not silently production-promote research
- do not weaken integrity tests to make a run green
- do not reopen closed science under a new name
- do not recursively read old handoffs unless a targeted anti-retest question requires it
- after substantive research milestones, update `CURRENT_NFL_RESEARCH_HANDOFF.md` on `main`
- never say an Action is running unless live Actions confirms queued/in_progress
- important results should be checkpointed to GitHub because the user’s UI frequently times out

User communication preference:

- execute, do not just theorize
- concise but meaningful progress updates while work is running
- if a run fails, say exactly where and why
- distinguish plumbing/integrity failures from scientific failures
- do not ask the user to repeat project context already available in GitHub
- the user strongly prefers seamless continuity over “starting fresh”

---

# 2. Canonical repository state at handoff preparation

Verified before this handoff was written:

- `main` before handoff docs commit:
  `7336994c8c5c5b232648ff8d899bca806628cff8`
- player landscape branch:
  `research-player-landscape-transmission-audit-v1@1c214b61205a272956720624158b31d9dff707f4`
- player output decomposition branch:
  `research-player-output-component-decomposition-v1@35fb319cad55c4588124cdd5b8e56522d6838054`
- RB receiving Week-5 branch:
  `research-rb-receiving-room-share-week5-lock-v1@56d6871b80a946047179bba9c5f7a346f5644794`

Always re-query live refs and Actions before editing.

---

# 3. Current project-wide state

## 3.1 Live betting / board context

Week 4 is fully settled.

- Week 4: **220-207, 51.52%, -3.38u**
- Weeks 1-4 cumulative: **849-818, 50.93%, -44.10u**

The board has repeatedly shown that raw model edge / fair-probability magnitude is overconfident.

No simple position/market/side/edge slice survived the clustered multiple-testing gate.

Do not mistake player-projection improvement for a validated betting selector.

The current player-centric research is about improving the football projection first.

---

# 4. Completed player-centric sequence

The following sequence is complete. Do not restart it.

## 4.1 Target-depth distribution Week-5 shadow

Branch:
`research-player-target-depth-distribution-shadow-v1`

Frozen Week-5 lock:
- run `37677366697` SUCCESS
- artifact `11507970080`
- artifact digest:
  `sha256:ecd7c8683d65bf9701353f0dbccb3ace59c992157bdf75b2b87191b08a8f314f`
- 277 WR/TE players
- feature available 210 / 277
- exact receiving-yard means preserved
- no sportsbook inputs
- no Week-5 outcomes
- no production change

Exact scale:
`depth_scale = sqrt(prior8_target_depth_sd / position_anchor)`

Frozen anchors:
- WR: `10.02786868561449`
- TE: `6.735519692444827`

This candidate was later evaluated retrospectively in W1-4 and **did not improve pooled CRPS**.

Do not promote it universally.

---

## 4.2 RB player-level scope resolution

Disposition:
`RB_PLAYER_LEVEL_SCOPE_BUTTONED_UP_BASELINE_PLUS_PROSPECTIVE_ALLOCATION`

Important boundaries:

- M96 retrospective stop remains preserved
- Week-5 RB carry/snap player-state allocation is prospective only
- do not back-apply that Week-5 shadow to Weeks 1-4
- do not reopen closed RB generic mean/width families

---

## 4.3 All-player / all-position W1-4 replay

Branch:
`research-all-player-all-position-replay-v1`

Canonical successful run:
- run `37683439543` SUCCESS
- exact successful head:
  `5e6d15cdebbb47d88217eb1f3437c2a506d56051`
- artifact `11509468659`
- artifact digest:
  `sha256:abb5bb7ba85d13306c0f64f83dd7c1e9840847c3ad308b18d859724121b8a711`

Canonical result:
`docs/research/ALL_PLAYER_ALL_POSITION_REPLAY_V1_RESULT.md`

Coverage:

- **4,488** point rows
- **1,837** unique player-weeks
  - QB 128
  - RB 471
  - WR 748
  - TE 490

Market row counts:

- QB pass_yards: 128
- RB rush_yards: 471
- RB rec_yards: 471
- RB receptions: 471
- RB rush_rec_yards: 471
- WR rec_yards: 748
- WR receptions: 748
- TE rec_yards: 490
- TE receptions: 490

Point MAE:

- QB pass_yards: **73.99**
- RB rush_yards: **19.11**
- RB rec_yards: **10.46**
- RB receptions: **1.182**
- RB rush_rec_yards: **24.37**
- WR rec_yards: **22.30**
- WR receptions: **1.514**
- TE rec_yards: **15.62**
- TE receptions: **1.427**

The replay exposed the critical cross-position pattern:

> **individual workload is compressed toward the middle.**

Low-workload players are overprojected.
High-workload focal players are underprojected.

Spearman correlation between projection error and realized opportunity was negative in every tested market:

- QB pass_yards: -0.586
- RB rec_yards: -0.611
- RB receptions: -0.809
- RB rush_rec_yards: -0.683
- RB rush_yards: -0.663
- TE rec_yards: -0.550
- TE receptions: -0.664
- WR rec_yards: -0.592
- WR receptions: -0.698

Examples:

- WR rec_yards, 1-2 realized targets: bias +10.88y
- WR rec_yards, 9+ targets: bias -45.16y
- TE rec_yards, 1-2 targets: +8.06y
- TE rec_yards, 9+ targets: -40.34y
- RB rush_yards, 1-3 carries: +13.19y
- RB rush_yards, 15+ carries: -35.81y
- QB pass_yards, <=20 attempts: +110.00y
- QB pass_yards, 41+ attempts: -83.30y

This is one of the most important current project findings.

---

## 4.4 Full-roster participation / zero-state finding

The replay also showed that a player simply appearing in the pregame football universe does not guarantee meaningful realized workload.

Of 1,837 player-weeks, **610 (33.21%)** had no weekly stat row:

- QB: 10 / 128 = 7.81%
- RB: 142 / 471 = 30.15%
- TE: 222 / 490 = 45.31%
- WR: 236 / 748 = 31.55%

Yet the model still gave meaningful average projections to many of those rows.

Interpretation:

> there is a real pregame **participation / role / opportunity-state** problem in the full player universe.

Do not “fix” this by deleting zero-stat players after outcomes.
Any solution must be pregame.

---

# 5. Target-depth W1-4 retrospective result

WR/TE distribution rows: 1,238.

Feature available: 859 / 1,238 = 69.39%.

Mean invariance:
max gap `4.263e-14` — passed.

Pooled empirical CRPS:

- baseline: **13.43749**
- target-depth shadow: **13.50947**
- improvement: **-0.07198** (worse)

Feature-available rows only:
- baseline 15.11523
- shadow 15.21897

Feature-available + weekly-stat rows:
- baseline 16.37459
- shadow 16.40012

Disposition:

`TARGET_DEPTH_FULL_SYMMETRIC_DISTRIBUTION_TRANSFORM_NOT_CONFIRMED_W1_W4`

Descriptive quartiles suggested high-dispersion widening may help while low-dispersion narrowing hurts, but that was post-outcome descriptive evidence only.

Do not rescue with a threshold selected from the same outcomes.

---

# 6. RB receiving-room share — major confirmed player-level improvement

This became the cleanest example that individual-player opportunity allocation materially improves final projections.

## 6.1 State coverage audit

Branch:
`research-rb-receiving-share-state-coverage-v1`

Canonical run:
- `37702111845` SUCCESS

Current within-RB-room target-share MAE:
**0.22336**

Strict-prior raw `prior_rb_room_share` MAE:
**0.18807**

Improvement:
**~15.8%**

The current model and strict-prior state were similar at identifying the room leader.

The gain came from **share concentration / magnitude**, not just picking a different leader.

Interpretation:

> the model often knows *who* the receiving back is, but spreads the room too evenly.

---

## 6.2 W1-4 retrospective final-projection impact

Branch:
`research-rb-receiving-room-share-impact-v1`

Run:
- `37703522415` SUCCESS
- artifact `11518826839`

Frozen no-fit mechanism:
- state = strict-prior `prior_rb_room_share`
- preserve exact RB/FB total receiving entitlement
- redistribute only inside RB/FB room
- no fitted coefficient
- no threshold
- no sportsbook
- rush arrays hard-locked to baseline

W1-4 results, 436 RB/FB player-games:

- targets MAE: **1.3863 -> 1.2810** = **7.60% better**
- receptions MAE: **1.1906 -> 1.1451** = **3.83% better**
- receiving-yards MAE: **10.7223 -> 10.4121** = **2.89% better**
- rush+receiving-yards MAE: **24.0169 -> 23.6176** = **1.66% better**
- rush-yards: exactly unchanged

Targets and receiving yards improved in **all four weeks**.

Player-by-player:
- target candidate closer: 242 vs baseline 193
- rec-yards candidate closer: 251 vs baseline 185
- receptions candidate closer: 247 vs baseline 189

For 6+ realized targets:
- target MAE 10.52% better
- rec-yards MAE 5.12% better
- receptions MAE 6.06% better

This is real individual-player projection improvement.

---

## 6.3 Week-5 RB receiving prospective lock

Branch:
`research-rb-receiving-room-share-week5-lock-v1`

Successful run:
- `37705464974` SUCCESS
- artifact `11518604971`

Frozen population:
- 98 RB identities
- 30 team RB rooms
- strict-prior receiving history: **98 / 98**
- exact GSIS identity bridge
- parameters fit: 0
- sportsbook inputs: 0
- Week-5 outcomes read: 0
- production changed: false

This exact rule is prospectively frozen.
Do not alter it after outcomes.

---

# 7. Player Landscape Transmission Audit — user’s “are we truly individual?” question

Branch:
`research-player-landscape-transmission-audit-v1`

Canonical result:
`docs/research/PLAYER_LANDSCAPE_TRANSMISSION_AUDIT_V1_RESULT.md`

Latest result evidence recorded in that document:
- successful audit run `37717016519`
- artifact `11524296325`
- result disposition:
  `INDIVIDUAL_PLAYER_CORE_CONFIRMED__LANDSCAPE_TRANSMISSION_INCOMPLETE`

The audit explicitly traced:

`identity -> availability -> player state -> room state -> team environment ->
opponent environment -> opportunity -> efficiency -> final mean -> distribution`

Result:

**Yes, the model is genuinely individualized.**

All 9 required player markets have:
- individualized usage
- individualized efficiency
- materially consumed opponent context

But the full landscape does not transmit perfectly.

Static inventory:
- 51 concrete feature/mechanism families
- 227 position-market transmission rows
- 143 actively consumed
- 84 gap / blocked / prospective / closed rows

Important interpretation:

> It is not “a position model with names attached.”

It is currently better described as:

> **individual-player core + material positional shrinkage + promoted player specialists + partial team/opponent transmission**

The main generic RB/WR/TE gaps include richer context that exists upstream but is not fully consumed.

However, that **does not authorize dumping all unused features into production**.

---

# 8. Player landscape anti-retest / source-blocked conclusions

These were reconciled before opening another lane.

## 8.1 Route participation / YPRR

Historical predictive route-rate/YPRR work is **SOURCE PARITY BLOCKED**.

Canonical historical PlayerForm only populates routes when a real routes/routes_run field exists.

nflverse weekly stats do not provide a clean historical total-routes-run field with live semantic parity.

Do not relabel:
- targets
- primary receiver flags
- participation route labels

as total routes run.

RB prior disposition:
`LIVE_RB_ROUTE_VOLUME_CONFIRMED_HISTORICAL_WEEKLY_PARITY_NOT_CLEARED`

The same historical parity issue blocks a clean WR/TE route-rate backtest.

---

## 8.2 WR-R3 combined player calibration

An older handoff incorrectly suggested this was never built.

It was later built and run.

Branch:
`research-wr-r3-combined-calibration`

Run:
`34726509088`

Final disposition:
`NO_ACTIONABLE_WR_R3_COMBINED_CALIBRATION`

It descriptively improved several MAEs but failed frozen promotion gates, including the >=1% 2025 MAE requirement and miss-rate guard.

Do not rebuild or retune it.

---

## 8.3 Coverage-v2 team man/zone

Already ablation-tested and effectively near-null.

Do not reopen the same feature family.

Player-level WR-CB assignment remains source-blocked without a free reproducible historical assignment source.

Do not restore retired static coverage penalty.

---

## 8.4 Generic game script / market confirmation

Already researched.

Key conclusion:
game-level market variables did not create an actionable incremental pregame confirmation state.

No player-prop or betting-market input should be moved upstream into the football model.

---

## 8.5 Football Matchup Transmission V1

Architecture audit correctly found dropped/weakly-transmitted football context.

But the three frozen simple integration candidates all failed closed:

- RB opponent pass-rate-faced
- WR true PROE
- TE opponent pass-success allowed

Do not resurrect or retune those exact formulas.

M95A/M95B generic RB role × run-defense family remains closed.

Opponent matchup is not irrelevant; those exact simple transmission mechanisms failed.

---

## 8.6 Defender injuries

Opponent-defender injury context remains source-parity blocked.

Blank injury reports are not equivalent to healthy.

Do not force a defender-injury feature without a qualified pregame source.

---

# 9. PLAYER OUTPUT COMPONENT DECOMPOSITION V1 — newest decisive result

This is the newest canonical scientific milestone and the immediate authority for next steps.

Branch:
`research-player-output-component-decomposition-v1`

Canonical result:
`docs/research/PLAYER_OUTPUT_COMPONENT_DECOMPOSITION_V1_RESULT.md`

Latest exact-head successful workflow:
- run: **`37777319154`**
- head: **`cb1e511adcb58e40fc41d0949306083a304e83dd`**
- conclusion: SUCCESS
- tests: SUCCESS
- decomposition: SUCCESS
- certification: SUCCESS
- strict repo audit: SUCCESS

Artifact:
- id: **`11550446909`**
- name: `player-output-component-decomposition-v1-37777319154`
- digest:
  `sha256:7acfc453258a49f3aadf0d0ea386b13827eb715f1dfb055b96505f733320d204`
- retention expiry: 2027-01-06

Branch head is later because RESULT / continuity docs were committed after the scientific run:
`35fb319cad55c4588124cdd5b8e56522d6838054`

Frozen parents:
- all-player W1-4 point replay run `37683439543`
- replay-matched baseline opportunity artifact from run `37687979574`

No sportsbook/player-prop inputs upstream.
No paid OddsAPI.
No fitted parameter.
No automatic promotion.

---

## 9.1 Integrity / grading source

Primary paired rows:
**4,017**

Oracle-scoreable rows:
**3,991**

Rows with unavailable model-effective efficiency:
**21**

Explicit grading-source exclusions:
- **5 rows**
- **2 unique player-weeks**
- reason:
  `GRADING_SOURCE_CONFLICT_UNRESOLVED_PBP_TARGET`

These rows remain visible in row-level evidence but are excluded from every oracle.

No target count was fabricated.

Target grading:
- completed-game PBP target rows: **2,056**
- verified zero-receiving-use fallback rows: **1,360**
- resolved PBP-vs-frozen target-count discrepancies: **0**

The target-grade identity resolver ultimately uses:

**PBP receiver GSIS ID -> weekly roster GSIS aliases -> canonical player key**

with narrow PBP-name fallback only when necessary.

This was required because weekly-stat identity handling missed/ambiguously represented some players, including dual-role / suffix / team-history edge cases.

Do not replace this with name-only matching.

Algebraic integrity:
- max baseline reconstruction gap:
  `2.842170943040401e-14`
- max full-actual identity gap:
  `5.684341886080802e-14`

Both pass the frozen `1e-10` tolerance.

---

## 9.2 Primary output decomposition result

Frozen disposition:

`OPPORTUNITY_DOMINANT_ACROSS_ALL_PRIMARY_MARKETS__INDIVIDUAL_ROLE_SHARE_ALLOCATION_NEXT`

The opportunity oracle removed more final-output MAE than the efficiency oracle in **every single predeclared market**.

| Position | Market | Baseline MAE | Opportunity-oracle MAE | Opportunity MAE removed | Efficiency-subset baseline | Efficiency-oracle MAE | Efficiency MAE removed |
|---|---|---:|---:|---:|---:|---:|---:|
| QB | pass_yards | 73.995 | 56.328 | **23.9%** | 61.274 | 53.469 | 12.7% |
| RB | rush_yards | 19.846 | 10.893 | **45.1%** | 21.273 | 19.732 | 7.2% |
| RB | rec_yards | 10.466 | 5.747 | **45.1%** | 11.529 | 8.357 | 27.5% |
| RB | receptions | 1.180 | 0.505 | **57.2%** | 1.175 | 1.203 | **-2.4%** |
| WR | rec_yards | 22.275 | 12.739 | **42.8%** | 23.404 | 16.800 | 28.2% |
| WR | receptions | 1.511 | 0.689 | **54.4%** | 1.479 | 1.308 | 11.6% |
| TE | rec_yards | 15.615 | 7.026 | **55.0%** | 16.238 | 12.423 | 23.5% |
| TE | receptions | 1.427 | 0.453 | **68.3%** | 1.347 | 1.238 | 8.1% |

This is unusually consistent.

---

## 9.3 Error transmission

Correlation of opportunity error with final output error is larger than efficiency error in all eight markets:

| Position | Market | Corr(opportunity error, output error) | Corr(efficiency error, output error) |
|---|---|---:|---:|
| QB | pass_yards | **0.814** | 0.491 |
| RB | rush_yards | **0.828** | 0.401 |
| RB | rec_yards | **0.789** | 0.619 |
| RB | receptions | **0.923** | 0.330 |
| WR | rec_yards | **0.749** | 0.553 |
| WR | receptions | **0.871** | 0.383 |
| TE | rec_yards | **0.788** | 0.515 |
| TE | receptions | **0.907** | 0.336 |

The opportunity-error relationship is larger in every cell.

---

# 10. Scientific interpretation — this controls the next lane

The current individual-player stack is **not primarily failing because it needs one more generic YPT/YPC/YPA/catch-rate or defense multiplier**.

The dominant remaining problem is:

> **How much opportunity the named player receives.**

By position:

- QB: pass attempts / active-QB workload state
- RB: carries + targets
- WR: targets
- TE: targets

This aligns with the original full-player replay:
- low-workload players overprojected
- focal high-workload players underprojected

And it aligns with the RB receiving-room success:
correcting individual room share improved final player projections.

Therefore:

### DO NOT pivot back to generic efficiency/matchup tuning now.

Opponent, defense, and conversion efficiency still matter.

But the newest controlled decomposition says they are **not the dominant bottleneck today**.

---

# 11. Current per-position opportunity state

## QB

What is good:
- M89/M90 mean path is protected and strongly individualized
- richer QB specialist environment than generic skill positions

What remains:
- decomposition still says opportunity dominates pass-yards error
- player/workload state means pass-attempt / active-QB opportunity needs to remain the focus, not another generic QB mean hunt

Do not reopen broad QB mean science.

Any new QB work must be specifically about pregame active-QB / attempt opportunity state and must clear anti-retest against prior QB volume families.

---

## RB rushing

What exists:
- Week-5 RB carry/snap player-state shadow is prospectively frozen
- do not back-apply it to W1-4
- M96 retrospective stop remains protected

What decomposition says:
- correcting carries has far more theoretical leverage than perfecting YPC

Next work should remain:
- pregame player role
- carry share
- room concentration
- starter/committee state
- teammate availability redistribution

Do not open another generic YPC or run-defense multiplier.

---

## RB receiving

This is the strongest completed player-share success.

- strict-prior RB room receiving share improves W1-4 final projections
- Week-5 exact rule is prospectively frozen

Do not retune it.

Future action:
grade exact frozen Week-5+ rule only after outcomes settle.

---

## WR

Current promoted opportunity stack:
- M38
- WR-R15

Prospective:
- target-share trajectory V1 frozen Week-5+

Blocked/closed:
- historical route-rate/YPRR parity not cleared
- WR-R3 combined calibration closed
- team Coverage-v2 near-null
- exact WR-CB historical assignment source unavailable
- simple true-PROE integration failed

Decomposition says:
WR rec-yards and receptions remain opportunity-dominant.

Do not open a new WR efficiency or matchup multiplier before exhausting player target-allocation state.

---

## TE

Current promoted opportunity stack:
- TE-R5P

Prospective:
- target-share trajectory V1 frozen Week-5+

Blocked/closed:
- route-rate/YPRR historical parity not cleared
- simple opponent pass-success integration failed
- universal target-depth transform unpromoted

Decomposition says:
TE rec-yards and receptions are strongly opportunity-dominant.

TE receptions are the strongest result:
**68.3% of baseline MAE removed by the perfect-opportunity oracle**.

---

# 12. Frozen prospective shadows to protect

Do not change these formulas after seeing Week-5 outcomes.

## RB carry/snap allocation shadow
- run `37560824479` SUCCESS
- 98 RB/HB/FB identities
- 30 teams
- prospective only
- no Week-5 outcomes at lock time

## WR/TE Target Share Trajectory Week-5 shadow
- run `37654382316` SUCCESS
- artifact `11497153776`
- 277 WR/TE rows
- 247 changed
- exact eligibility requires 4 prior same-season team games
- M38 WR1 preserved
- WR2+/TE pool and team target mass conserved

Do not weaken the eligibility rule.

## RB receiving-room Week-5 shadow
- run `37705464974` SUCCESS
- 98 / 98 strict-prior receiving history
- exact GSIS identity authority
- no Week-5 outcomes
- no production change

## Target-depth Week-5 distribution shadow
Still frozen prospectively, but W1-4 universal transform was not supported.

Do not promote universally.

---

# 13. Exact next research action

The next lane must be **individual pregame role/share/opportunity allocation**.

Do not begin by adding defense features.

Do not begin by fitting a post-hoc correction to output.

Do not begin by reopening WR-R3 / FMT / Coverage-v2 / generic RB defense families.

### Immediate sequence for the next chat

1. **Read only the memory-efficient set listed below.**
2. Re-query:
   - live `main`
   - `research-player-output-component-decomposition-v1`
   - relevant opportunity-shadow branches
   - latest Actions
   - Issue #535 latest comments
3. Verify run `37777319154` and the canonical RESULT doc.
4. Build a concise **opportunity-authority map** for every primary market:
   - QB pass attempts
   - RB carries
   - RB targets
   - WR targets
   - TE targets
5. For each, explicitly mark:
   - production authority today
   - prospective frozen shadow already available
   - retrospective evidence available
   - closed families
   - source-blocked families
   - genuinely unresolved pregame role/share state
6. Do **not** create a new model until the map proves the gap is novel.
7. Then open the smallest legal lane that attacks the unresolved individual workload compression.

The most likely unresolved cross-position mechanism is:

> **pregame participation / role-state / room-concentration authority**

because the full player replay showed many rostered players receiving no realized workload while focal players were underallocated.

But do not pre-decide the exact feature formula.

First reconcile existing authorities and anti-retest boundaries.

Potential questions the next candidate must answer:

- Is this player actually expected to participate meaningfully?
- Is he a starter, rotational player, or inactive-risk row?
- Has his strictly-prior role changed?
- How concentrated is his room this week?
- Which teammate absences redistribute work to him?
- Does his current room state support a focal workload versus committee workload?
- Can that state be reconstructed historically with source parity?
- Can it be tested without target-game leakage?

A valid candidate should alter **opportunity/share only**, not efficiency, unless a separate efficiency lane is explicitly opened later.

---

# 14. Strong anti-retest rules

Do not reopen or rescue:

- generic QB mean hunt after M89/M90
- RB M96 retrospective router/threshold research
- RB R23-R27D closed receiving families
- WR-R3 combined calibration
- raw QB-receiver pair YPT
- team-level Coverage-v2 man/zone
- retired static coverage penalty
- exact WR-CB assignment without a real historical source
- generic game-script / market-confirmation lane
- FMT RB pass-rate-faced candidate
- FMT WR true-PROE candidate
- FMT TE pass-success candidate
- M95A/M95B generic RB role × defense family
- opponent-defender injury until source parity clears
- route-rate/YPRR historical candidate until total routes-run source parity clears
- universal symmetric target-depth transform based on W1-4
- post-hoc target-depth thresholds chosen from the same outcomes

Do not use sportsbook/player-prop information upstream.

---

# 15. Week-5 / prospective grading rule

Week-5 prospective locks are confirmation evidence.

Do not change their formulas after outcomes.

When Week-5 results are actually settled and authoritative:
- grade exact frozen locks
- compare baseline vs frozen candidate
- do not re-select thresholds
- preserve immutable row digests / identity sets
- report promotion evidence separately from retrospective discovery evidence

---

# 16. Important distinction: “individual” versus “complete individual landscape”

The user asked this repeatedly and it matters for communication.

Correct answer:

- yes, the model has genuine player-specific rows, usage, efficiency, room hierarchy, and opponent context
- no, not every available player/team/opponent signal transmits completely
- however, the newest decomposition proves that **opportunity allocation is the dominant current bottleneck**
- therefore the model should first get the named player’s workload right, then revisit residual efficiency/matchup conversion if necessary

The user is not asking for random feature accumulation.

They want a coherent football statement like:

> “This specific player is expected to get this amount of work, for these pregame football reasons, in this room/offense/opponent context, and therefore projects to this output distribution.”

That is the end state.

---

# 17. Memory-efficient next-chat read set

Read only:

1. `AGENTS.md`
2. root `CURRENT_NFL_RESEARCH_HANDOFF.md` newest top checkpoint
3. this file
4. `docs/research/PLAYER_OUTPUT_COMPONENT_DECOMPOSITION_V1_RESULT.md`
5. `docs/research/PLAYER_LANDSCAPE_TRANSMISSION_AUDIT_V1_RESULT.md`
6. `docs/research/ALL_PLAYER_ALL_POSITION_REPLAY_V1_RESULT.md`
7. `docs/research/RB_RECEIVING_ROOM_SHARE_IMPACT_V1_RESULT.md`
8. `docs/research/RB_RECEIVING_ROOM_SHARE_SHADOW_V1_WEEK5_LOCK.md`
9. `docs/research/RB_PLAYER_STATE_ALLOCATION_SHADOW_V1_WEEK5_LOCK.md`
10. `docs/research/PLAYER_TARGET_SHARE_TRAJECTORY_SHADOW_V1_WEEK5_LOCK.md`
11. Issue #535 latest relevant comments
12. live main / relevant branches / Actions

Do not recursively read older handoffs unless needed for a specific anti-retest check.

---

# 18. Exact resume sentence

When the next chat starts, the working state should be understood as:

> **Player individualization audit is complete; all-player W1-4 replay is complete; RB receiving-share retrospective impact is confirmed and Week-5 locked; player landscape transmission audit is complete; player output component decomposition is complete and proves opportunity dominates efficiency across all eight primary markets. Next work is bounded individual pregame role/share/opportunity allocation, with anti-retest and source-parity gates enforced.**

That is the current project frontier.
