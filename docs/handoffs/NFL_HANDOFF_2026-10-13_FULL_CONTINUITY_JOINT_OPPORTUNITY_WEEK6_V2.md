# NFL HANDOFF — 2026-10-13 — FULL CONTINUITY / JOINT OPPORTUNITY UNCERTAINTY / WEEK-6 EXECUTION FRONTIER V2

**This is the newest authoritative cross-chat continuity document.**

GitHub is canonical. Chat memory is secondary.

Repository:
- `dkaps6/imtiredofthis`

Active research container:
- branch: `research-individual-opportunity-roadmap-2026-10-08`
- branch head immediately before this handoff write: `3134992c2b48aee705499992c3f5235cbf8c4491`
- draft PR: **#672**
- continuity issue: **#535**

Production main:
- `812ac2cde79fe3846174613a2f2777e57b34b739`
- `Docs: preserve model-backed alt-line parlay construction method`

Do not merge research conclusions into production merely because a research
workflow is green.

---

# 1. START RULE FOR THE NEXT CHAT

Read in this exact order:

1. `AGENTS.md`
2. ONLY the newest TOP checkpoint in `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. THIS FILE
4. `docs/research/WEEKS1_5_CONTINUITY_WEEK5_POSTMORTEM_SYNTHESIS_V1.md`
5. `docs/research/PLAYER_OPPORTUNITY_JOINT_ALLOCATION_DIRICHLET_V1_HISTORICAL_RESULT.md`
6. `docs/research/PLAYER_OPPORTUNITY_JOINT_ALLOCATION_DIRICHLET_V1_PROSPECTIVE_SHADOW_MECHANICS_RESULT.md`
7. `docs/research/WEEK6_INDIVIDUAL_OPPORTUNITY_LOCK_READINESS_RESULT.md`
8. newest comments on Issue #535 and PR #672
9. live branch / main / exact-head GitHub Actions

Do **not** recursively read older handoffs.

Do **not** rerun completed W1-W4 replay, Player Landscape, output decomposition,
Week-5 postmortem, share-volatility discovery, Beta-Binomial historical tests,
joint Dirichlet historical tests, or failed matchup tests merely to reacquire
context.

If GitHub has moved after this handoff, newer frozen plans/results on the active
research branch beat this narrative.

---

# 2. USER'S MODEL VISION — NON-NEGOTIABLE

The user is building an individual-player football model, not a position-average
model.

The intended end state is:

> Project every player as an individual football player in the exact game he is
> about to play, using current role, usage, team environment, offensive context,
> opponent, individual history, efficiency, matchup interaction, injuries /
> personnel, and uncertainty.

Conceptually:

`player projection = active-role state × team opportunity × individual share × individual efficiency × player-specific context/matchup + individual uncertainty`

The user repeatedly emphasized:
- individual players are different;
- teams are different;
- roles are different;
- defenses can be good or bad against specific mechanisms;
- football knowledge should inform projections;
- generic positional averages are not enough;
- do not call a generic defense-v-position multiplier “football knowledge”;
- every position eventually needs player-specific treatment;
- fewer, higher-confidence wagers are preferred over a giant raw-edge board;
- own-line thinking is eventually desired;
- alt lines should be used intelligently;
- no fabricated certainty;
- no repeated failed research under new names.

Current model-development priority remains **individual opportunity allocation
and uncertainty**, with **individual efficiency-tail uncertainty** now the next
independent historical root-cause lane.

---

# 3. USER WORK STYLE / EXECUTION EXPECTATIONS

The user wants execution, not repeated roadmap speeches.

Do:
- take the wheel;
- continue autonomously;
- give concise status after meaningful milestones;
- fix mechanical failures and rerun until resolved;
- preserve all meaningful results in GitHub;
- distinguish scientific failure from CI/runtime mechanics;
- do not ask the user to re-explain prior project history;
- do not restart completed science;
- keep production, research, betting selection, and ticket construction separate;
- preserve why an experiment was tried and why it passed/failed.

The user is frustrated by:
- UI timeouts;
- apparent restarts;
- research that does not translate into player-level projections;
- Week 6 arriving before the model feels finished;
- parlays losing by one leg;
- wrong player theses being repeated across multiple tickets.

The user explicitly said after discussing tonight's game:
> go back to research please

Unless they ask again, stay in research.

---

# 4. PRODUCTION / WEEK-5 LIVE AUTHORITY

Canonical Week-5 preserved-odds Full Slate:
- run: `37860592615` SUCCESS
- artifact: `11585973062`
- digest:
  `sha256:bf52aa4bc30bf6e8cc6029a0a86c0f33cdfaa09f587c38a138f23cef6b7436f4`
- production behavior authority:
  `24704e5f1ef872ec86c36446d2c73f6d377c824c`
- pricing status: `PRESERVED_REPLAY`
- 1,472 priced offers
- 754 player-markets
- 0 unresolved positions
- DAL-TB retained via acquisition-time pre-T75 replay lock

Original paid source:
- run: `37852811339`
- artifact: `11582178322`
- digest:
  `sha256:4948e08003fcab0f519b35bd4b23d8c895554e4faced993238575acdc5d2e765`

Critical:
- ZERO second OddsAPI acquisition was used;
- do not spend another OddsAPI credit without explicit user authorization;
- preserved Week-5 prices are stale replay authority, never “Bettable Now.”

---

# 5. WEEK 1-5 FACTUAL PERFORMANCE STATE

Canonical synthesis:
`docs/research/WEEKS1_5_CONTINUITY_WEEK5_POSTMORTEM_SYNTHESIS_V1.md`

## Weeks 1-4 selected board

- W1: 204-205, -20.92u
- W2: 192-198, -22.06u
- W3: 233-208, +2.26u
- W4: 220-207, -3.38u

Cumulative W1-W4:
- 849-818
- 50.93%
- -44.10u
- -2.65% ROI
- model MAE 17.89
- selected-line MAE 16.86
- model closer than line 46.07%

Raw edge / stated probability was badly overconfident:
- 70-100% stated-prob rows: n=653
- mean stated probability 80.52%
- realized win rate 53.60%
- 57 clustered eligible slices
- 0 BH-FDR survivors

Do not resurrect raw-edge sorting as confidence.

## Week 5 selected board

- 350 decided bets
- 199-151
- 56.86%
- +26.17u
- +7.48% ROI

By position:
- QB 25-28, -5.73u
- RB 75-49, +17.27u
- WR 73-50, +14.51u
- TE 26-24, +0.12u

By market:
- rush+rec 20-6, +11.64u
- receptions 66-45, +14.11u
- rec yards 61-51, +3.05u
- pass yards 14-12, +0.47u
- rush yards 38-37, -3.10u

Through Week 5:
- 1,048-969
- 2,017 decided bets
- 51.96%
- -17.93u
- approximate simple cumulative ROI -0.89%

## Raw model vs market

Week-5 full football board:
- model MAE: 18.9288
- preserved median-line MAE: 17.8431
- model closer than line: 50.26%
- model direction hit rate vs median line: 55.32%

Interpretation:
Week-5 selected-board profitability came primarily from downstream selection /
price handling. The raw player mean layer is **not solved**.

---

# 6. WEEK-5 FAILURE DECOMPOSITION

Top 40 absolute final-model player-market misses:
- efficiency-dominant: **19**
- opportunity-dominant: **14**
- mixed opportunity/efficiency: **7**

Therefore two distinct root-cause lanes remain:

1. player opportunity / role allocation uncertainty;
2. player efficiency-tail uncertainty.

Do not collapse these into one generic “projection error” problem.

---

# 7. USER-FACING TICKET / PORTFOLIO POSTMORTEM

Canonical:
`docs/research/WEEK5_USER_FACING_TICKET_CONCENTRATION_POSTMORTEM_V1.md`

Keep this downstream from football science.

## Deshaun Watson

User reported 155 passing yards.

Canonical model:
- model mean ~238.23
- actual 155
- predicted attempts 24.77
- actual attempts 33
- predicted YPA 7.28
- actual YPA 4.70
- label: mixed opportunity/efficiency miss

Interpretation:
- genuine projection failure;
- not primarily an alt-line construction failure;
- repeated exposure across tickets amplified one bad thesis.

Process defect:
`CROSS_TICKET_PLAYER_CONCENTRATION`

## BUF-LAR core ticket example

Suggested:
- Josh Allen 200+ pass -> 210 HIT
- Puka Nacua 50+ rec -> 138 HIT
- Kyren Williams 50+ rush -> 49 MISS by 1

Interpretation:
near-threshold variance, not equivalent to Watson.

Higher-payout expansion also had:
- Keon Coleman 40+ -> 7 MISS
- Dalton Kincaid 40+ -> 7 MISS

Stafford under also failed badly.

Required downstream controls:
- cap repeated non-lock player exposure across paid tickets;
- model mean > alt line is NOT calibrated survival probability;
- tag failure type before reusing a player thesis;
- track ticket/portfolio survival;
- separate lottery entertainment from core tickets.

The user said all prior tickets were one leg away and felt “we did something
right.” Preserve that as a ticket-construction observation, not model-science
evidence.

---

# 8. CORE SCIENTIFIC FOUNDATION — COMPLETE / DO NOT REDO

Completed:
- W1-W4 all-player replay;
- Player Individualization / Landscape work;
- Player Landscape Transmission Audit;
- Player Output Component Decomposition;
- availability parity / opportunity audits;
- WR/TE trajectory historical work;
- Week-5 full-model postmortem;
- player failure casebook;
- Weeks 1-5 continuity synthesis.

Key decomposition:
opportunity oracle beats efficiency oracle in all 8 tested primary cells.

Approximate corrected ACT-only opportunity-oracle mass:
- QB team pass opportunity ~60.3%
- RB carry share ~64.5%
- RB target share ~76.5%
- WR target share ~68.6%
- TE target share ~77.1%

This is why the roadmap stayed on individual opportunity allocation first.

---

# 9. PRIOR OPPORTUNITY MEAN / ALLOCATION STATE

## RB receiving-room share

Historical W1-W4 impact was positive:
- targets +7.60% MAE improvement
- receptions +3.83%
- rec yards +2.89%
- rush+rec +1.66%

Preserve this as a confirmed mean-allocation improvement.

## RB carry/snap Week-5 shadow

Frozen 50/50 recent carry-share + snap-fraction mechanism.
Week-5 prospective result was worse than recent-carry control.
Do not promote from Week 5.

## WR/TE target-share trajectory

Historically real signal.
Week-5:
- WR essentially flat/slightly worse
- TE small first-lock positive

Do not weaken eligibility:
- same-season
- same-team
- >=4 prior games
- >=2 earlier games plus RECENT2 logic
- M38 WR1 frozen
- WR2+ / TE room mass conserved

The future-week mechanics remain useful for prospective locks, but mean
promotion is not authorized from Week 5 alone.

---

# 10. MAJOR PLAYER-SPECIFIC SIGNAL — SHARE VOLATILITY

## Corrected 2024-2025 V1

Canonical:
- run `38292658912` SUCCESS
- exact head `1d224c8cc5804eec110f9c9750d60d0f506bc7e4`
- artifact `11767472231`
- digest
  `sha256:021a9fc79f28b4568b8663be6e57d194686b785ccbfd4c6d0a088d7e44a4f5a3`
- result:
  `docs/research/PLAYER_OPPORTUNITY_SHARE_STABILITY_V1_RESULT.md`

Important repair:
an earlier all-zero result was INVALID because roster IDs used raw GSIS
(`00-...`) while PlayerForm used canonical identity keys
(`gsis:00-...`).

Final run canonicalized identity and added a non-degenerate-share gate.

Signal replicated in BOTH 2024 and 2025 for:
- RB carry share
- RB target share
- WR target share
- TE target share

Representative Spearman:
- RB carry: .5766 / .6477
- RB target: .5989 / .6662
- WR target: .5181 / .5063
- TE target: .6268 / .5634

All relevant game/player cluster CI lower bounds >0.

## Untouched 2023 holdout

Canonical:
- run `38294106319` SUCCESS
- exact head `cad154e0f2fd0bdd2d1d5f502eda77c13ae2f636`
- artifact `11767104207`
- digest
  `sha256:e5a042d84ba575fedd31148f9c5e9a5bebc9bd8f1ae36e2decd9df173280e8ce`
- result:
  `docs/research/PLAYER_OPPORTUNITY_SHARE_STABILITY_2023_HOLDOUT_RESULT.md`

2023 confirms all four:
- RB carry rho +.6409
- RB target +.6116
- WR target +.4686
- TE target +.5699

Disposition:
`PLAYER_OPPORTUNITY_SHARE_STABILITY_THREE_SEASON_CONFIRMED`

This is one of the strongest player-specific findings in the project.

Interpretation:
a player's own strictly-prior usage-share volatility strongly predicts how
unstable his next-game opportunity allocation will be.

This is initially an uncertainty/distribution finding, not a mean change.

---

# 11. V2 — EXCESS ROLE VOLATILITY BEYOND FINITE-COUNT NOISE

Canonical:
- run `38295515693` SUCCESS
- exact head `81cd2beeb6f8da9fd5e485d70daee546285e6905`
- artifact `11768432238`
- digest
  `sha256:7b31dd29e4b94674d6948cf84af24fae7181d1a4f502a07beaa43919d4d7f3a6`
- result:
  `docs/research/PLAYER_OPPORTUNITY_SHARE_VOLATILITY_MECHANISM_V2_RESULT.md`

Disposition:
`PLAYER_OPPORTUNITY_EXCESS_VOLATILITY_EXISTS`

V2 removes expected finite-count sampling variance and partials prior mean share
+ expected sampling SD.

Three-season confirmed:
- RB carry share
- RB target share
- WR target share

TE target:
- fails frozen three-season mechanism gate;
- 2023 player-cluster CI low -0.0134;
- do not rescue TE by pooling/threshold changes.

Representative excess-SD rho:
- RB carry .4989 / .4609 / .5024
- RB target .3293 / .3053 / .3049
- WR target .2502 / .2927 / .2730

Meaning:
there is genuine player-specific role instability beyond ordinary multinomial
count noise.

---

# 12. MARGINAL BETA-BINOMIAL UNCERTAINTY

Canonical:
- run `38296726460` SUCCESS
- exact head `b94e2188b6fd0c20b41263b6dc9c8161b151d5da`
- artifact `11768483266`
- digest
  `sha256:403360d3245e78507cfe7646022291a676535fe6bf90d607ba85f89f900eae4c`
- result:
  `docs/research/PLAYER_OPPORTUNITY_ALLOCATION_UNCERTAINTY_BETABINOMIAL_V1_RESULT.md`

Candidate preserves expected opportunity exactly:
`E[count] = N × prior_share_mean`

Historical PASS all 2023/24/25:
- **RB carry share**
  - CRPS +.1814 / +.1437 / +.1033
  - clustered CI lower bounds >0 all seasons
  - paired NLL improves all seasons
- **WR target share**
  - CRPS +.0184 / +.0253 / +.0277
  - clustered CI lower bounds >0 all seasons
  - paired NLL improves all seasons

FAIL CLOSED:
- RB target share:
  - 2025 CRPS point +.0020
  - game CI low -0.00132
  - player CI low -0.00159
- TE target never authorized because V2 failed.

Do not reopen RB-target or TE-target under this exact uncertainty family.

---

# 13. MAJOR CURRENT PASS — JOINT TEAM-CONSERVING DIRICHLET

This is the strongest current opportunity-uncertainty result.

Canonical historical result:
- run `38401069806` SUCCESS
- exact head `5f35f96f0a43b9963691a7768a330b32b4854a5b`
- artifact `11815490905`
- digest
  `sha256:2b336114dc5ef0145c0a40706266da850a0ff1d25c8935958481b996333fce29`
- result:
  `docs/research/PLAYER_OPPORTUNITY_JOINT_ALLOCATION_DIRICHLET_V1_HISTORICAL_RESULT.md`

Passing cells:
- **RB_CARRY_SHARE**
- **WR_TARGET_SHARE**

Disposition for both:
`JOINT_ALLOCATION_DIRICHLET_HISTORICAL_PASS_FREEZE_FORWARD_SHADOW`

Properties:
- exact canonical player means preserved;
- exact finite team opportunity totals conserved;
- player-specific latent role uncertainty from strictly-prior excess share variance;
- no fitted coefficient;
- no threshold search;
- no sportsbook;
- no 2026 outcomes;
- no production change.

Integrity:
- 2,354 joint team-game rooms
- 6,279 eligible player marginals
- canonical-q match rate 100%
- max room expected-count gap 0
- max marginal expected-count gap 0

## RB carry

2023:
- joint log-score improvement +3.42013
- game CI [2.94566, 3.90988]
- team CI [2.69551, 4.19594]
- marginal CRPS +.45697

2024:
- +3.08394
- game CI [2.72594, 3.46330]
- team CI [2.45067, 3.74149]
- marginal CRPS +.36223

2025:
- +2.88473
- game CI [2.49467, 3.27033]
- team CI [2.38669, 3.40813]
- marginal CRPS +.32082

## WR target

2023:
- joint log-score +.78218
- game CI [.63155, .93797]
- team CI [.57994, .98635]
- marginal CRPS +.05301

2024:
- +1.04293
- game CI [.83940, 1.26814]
- team CI [.72820, 1.37987]
- marginal CRPS +.07221

2025:
- +1.04630
- game CI [.85871, 1.24904]
- team CI [.78508, 1.31141]
- marginal CRPS +.06805

Scientific interpretation:
the current fixed-share multinomial allocation is too narrow for:
- individual RB carry allocation;
- individual WR target allocation.

This is a **distribution/uncertainty improvement**, NOT a new mean allocator.

---

# 14. PROSPECTIVE JOINT SHADOW — MECHANICS CERTIFIED

Canonical:
- run `38402018307` SUCCESS
- exact head `fb4136b639ac4b2281a57a6d3a383c97b5c79fe6`
- result:
  `docs/research/PLAYER_OPPORTUNITY_JOINT_ALLOCATION_DIRICHLET_V1_PROSPECTIVE_SHADOW_MECHANICS_RESULT.md`

Future manual runner:
`.github/workflows/research-future-week-joint-opportunity-shadow-v1.yml`

Certified:
- exact target-week player-state authority required;
- stale prior-week roster rejected;
- only RB carry / WR target accepted;
- canonical mean q preserved exactly;
- alpha/alpha0 equals q within 1e-12;
- room probability mass conserved;
- strictly-prior source week required;
- zero-excess / unauthorized state stays ineligible;
- no production simulator import/reference;
- no target-week outcomes;
- research-only.

Do not mutate production pricing from this shadow.

---

# 15. FUTURE-WEEK PLAYER OPPORTUNITY INFRASTRUCTURE — READY

Mechanically certified:
- future-week player-state auditor;
- exact target-week roster lock boundary;
- RB carry/snap future-week lock;
- RB receiving-room future-week lock;
- WR/TE trajectory future-week lock;
- share-volatility lock mechanics;
- unified future-week opportunity lock pack;
- joint RB-carry / WR-target Dirichlet prospective shadow.

The lock pack is intentionally fail-closed.

Do not rebuild these mechanisms unless a genuine bug is found.

---

# 16. WEEK-6 LOCK — LAST VERIFIED HARD BLOCK

Canonical readiness:
- run `38399980031`
- exact head `70090477c416541ce2475b218ea502f3d9833091`
- artifact `11812829801`
- result:
  `docs/research/WEEK6_INDIVIDUAL_OPPORTUNITY_LOCK_READINESS_RESULT.md`

Disposition:
`PLAYER_STATE_FUTURE_WEEK_NOT_READY`

Ready:
- Week-5 stats complete
- Week-5 snaps complete
- schedule context available
- current usage coverage among played ~99.458%
- RB/WR/TE snap coverage ~99.379%
- stable ID coverage ~99.848%
- chronology violations 0
- 28 scheduled Week-6 teams

Hard blocker at last check:
- target week = 6
- roster source week = 5
- `target_week_roster_source_ready = false`
- target-week injury rows available = 0

No legal Week-6 immutable lock was created:
- no share-volatility lock;
- no RB carry/snap lock;
- no RB receiving lock;
- no WR/TE trajectory lock;
- no unified Week-6 opportunity pack;
- no Week-6 joint Dirichlet shadow.

**Never substitute Week-5 roster identity.**

FIRST operational move next chat:
check whether exact Week-6 weekly roster authority is now available.

If YES:
1. dispatch existing frozen Week-6 opportunity lock pack;
2. require exact Week-6 identity + complete Week-5 stats/snaps;
3. freeze raw share-volatility state;
4. freeze RB carry/snap;
5. freeze RB receiving;
6. freeze WR/TE trajectory;
7. freeze unified opportunity lock pack;
8. run/freeze joint Dirichlet shadow for **RB carry + WR target only**;
9. preserve run/artifact/digest before Week-6 outcomes;
10. remain research-only.

If NO:
- stay fail-closed;
- do NOT change the source contract;
- immediately continue the non-time-gated **individual efficiency-tail uncertainty** lane.

---

# 17. NEXT HISTORICAL ROOT-CAUSE LANE — INDIVIDUAL EFFICIENCY TAILS

This is the next non-time-gated football lane if Week-6 roster authority is still
unavailable.

Reason:
Week-5 top-40 misses were:
- 19 efficiency-dominant
- 14 opportunity-dominant
- 7 mixed

Opportunity uncertainty finally has a major historical PASS. Efficiency tails
remain unresolved.

The next historical question should be frozen BEFORE scoring:

> Does strictly-prior individual efficiency volatility / regime state identify
> next-game efficiency-tail risk beyond the current mean model?

Potential cells:
- QB YPA / pass-efficiency tail
- WR receiving YPT / catch-efficiency tail
- TE receiving YPT / catch-efficiency tail
- RB rushing YPC tail
- RB receiving YPT / catch-efficiency tail if justified

Important:
- this is primarily an uncertainty/tail-risk question first;
- do not tune to Watson, Stafford, Coleman, Kincaid, or any Week-5 miss;
- no Week-5 outcome-derived thresholds;
- reconcile all prior efficiency research / target-depth work before freezing;
- no generic matchup multiplier;
- no sportsbook;
- freeze exact feature/window/statistic/gates before scoring;
- require replicated seasons and clustered inference;
- keep mean-changing work separate from uncertainty work.

A logical first move:
1. anti-retest audit of completed efficiency research;
2. inventory which strictly-prior player-level efficiency dispersion/regime
   features already exist;
3. freeze one bounded historical source/diagnostic plan;
4. score only after freeze.

---

# 18. QB TEAM PASS-INTENT SIDE LANE — PASSIVE

This is one possible upstream QB team-pass-opportunity input. It is NOT the main
roadmap.

Corrected archive transport:
- V1I-R1 run `38186579290` SUCCESS
- artifact `11719552673`
- digest
  `sha256:c371cde47ac800fb9665dcdd511d33c6626e59a4f47ba3f45e10fe17f7f063bd`
- corrected combined transport 141/158 = 89.24%
- frozen 80% transport gate PASS

Historical semantic automation:
- V1J independent NLI FAILED safety
- V1K blinded precision:
  - 4/32 genuinely eligible explicit target-game intent
  - 25/32 generic/descriptive
  - 3/32 insufficient
- blanket historical automation CLOSED

Do not:
- manually adjudicate all 378;
- retune V1F;
- cascade more NLI models;
- promote V1G wholesale.

V1L prospective:
- initial run `38187787580`, artifact `11719983641`: 0 retained
- Monday attempt 2 artifact `11752326387`,
  digest `sha256:415a04ebfe254f2b18665d641943ed9dbb6996dff6cca613856e92bd39736d0b`
- 28/28 attempted
- 0 retained again
- source/time-gated

Let this collect passively when genuine first-party coach/OC media appears.
Never loosen the source family to manufacture coverage.

Protected QB science:
- M89/M90
- QB C2

Closed:
- generic QB mean/YPA
- same-data attempts repackaging
- schedule/rest D1
- PBP D2

---

# 19. PLAYER-SPECIFIC MATCHUP WORK — CURRENT CLOSURES

Do not restart generic matchup work.

## WR/TE individual target-depth × opponent depth-defense

Source qualified strongly:
- 7,184 source-ready player-weeks
- WR 4,666
- TE 2,518

Predictive direct YPT multiplier:
- WR worsened in both 2024 and 2025
- TE improved/supportive in 2025 but not 2024
- cross-season gate failed
- scorer also violated untreated-output isolation

Result:
`docs/research/PLAYER_ROLE_MATCHUP_DEPTH_INTERACTION_PREDICTIVE_V1_RESULT.md`

Closed. Do not rescue via:
- depth-band retune
- shrinkage
- slot/deep/WR1 subgroup mining
- post-hoc completion-rate switch

## RB individual run-location × opponent location-defense

Source qualified:
- 2,654 source-ready RB player-weeks

Predictive authority:
- run `38258940285` SUCCESS
- exact head `20230872a13729f8c5269b653c46ffec030c1c78`
- artifact `11752168196`
- digest
  `sha256:476000f0f6814af60f956f71bf96708442057f6a4b55a37e975bdc0bfac77ed2`

Disposition:
`RB_RUN_LOCATION_MATCHUP_HISTORICAL_FAIL_CLOSED`

2024:
- n=1,063
- MAE 23.7181 -> 23.6520
- +0.0661 improvement
- game/player CI cross zero

2025:
- n=1,018
- MAE 23.2285 -> 23.2506
- slight worsening

Closed:
- no location-band retune
- no success-rate blend
- no bellcow/inside/outside subgroup rescue
- no shrinkage
- no M95 recombination

---

# 20. OTHER CLOSED / PROTECTED SCIENCE

Protect:
- QB M89/M90
- QB C2
- M38 WR1 anchor
- WR-R15
- TE-R5P
- RB receiving-room historical improvement
- exact WR/TE trajectory eligibility
- Week-5 production/preserved board authority
- joint opportunity uncertainty historical PASS

Closed:
- generic QB mean/YPA
- same-data attempts repackaging
- schedule/rest D1
- PBP D2
- historical QB-intent blanket automation
- WR-R3 combined calibration
- coverage-v2
- direct historical WR-CB assignment unless a genuinely new free source exists
- generic FMT direct matchup transforms
- M95A/B
- M96 retrospective RB program
- universal target-depth symmetric distribution transform
- player-depth direct YPT V1
- RB run-location direct YPC V1
- RB target Beta-Binomial V1
- TE target Beta-Binomial/joint family under current mechanism
- Opportunity State Conflict uncertainty

A new attempt must use genuinely new information or a genuinely different
football mechanism.

---

# 21. CURRENT GITHUB / CI STATE AT HANDOFF WRITE

Production main:
- `812ac2cde79fe3846174613a2f2777e57b34b739`

Research branch immediately before handoff write:
- `3134992c2b48aee705499992c3f5235cbf8c4491`
- message: `docs: set Oct13 NFL frontier as current top checkpoint`

PR:
- #672 draft research container

Issue:
- #535 continuity ledger

Exact-head PR checks at `3134992...`:
- Repo CI: SUCCESS
- Backtest Historical Input Validation: SUCCESS
- Historical Benchmark Clean Rebuild V1: was still IN_PROGRESS at last check
  - earlier exact heads on this lane completed SUCCESS
- Backtest Weeks 1-2 Full Board: FAILURE
- Replay Paid Full Slate Artifact Once: FAILURE

The two red workflows are known mechanical environment failures:
- Polars / pyarrow / NumPy compatibility
- not a football-science failure
- not a new OddsAPI pull
- not a Week-5 authority failure

Do not weaken science gates to make them green.

Mechanically repair those workflows when useful:
- use Polars `to_dicts()` / direct conversion to avoid pyarrow dependency,
  or
- pin a pyarrow version compatible with NumPy 1.26.4.

Do not change football formulas while repairing them.

---

# 22. NO-ODDS / SPORTSBOOK RULE

No new paid OddsAPI acquisition without explicit user approval.

Sportsbook remains downstream comparison/selection information unless a model is
explicitly labeled market-assisted.

Research in this branch must remain football-only unless the frozen experiment
explicitly says otherwise.

---

# 23. EXACT NEXT ACTIONS — DO THESE IN ORDER

## Step 1 — reconcile live state

Re-fetch:
- production main
- active research branch
- PR #672
- Issue #535 latest comments
- exact-head Actions

Do not assume any run is active.

## Step 2 — check Week-6 exact roster authority

This is the FIRST operational question.

If exact Week-6 weekly roster authority is now present:
- dispatch/freeze the existing Week-6 opportunity lock pack;
- require exact Week-6 identity;
- require complete Week-5 stats/snaps;
- preserve chronology;
- freeze raw volatility;
- freeze RB carry/snap;
- freeze RB receiving;
- freeze WR/TE trajectory;
- freeze unified pack;
- run/freeze joint Dirichlet shadow for RB carry + WR target ONLY;
- preserve run/artifact/digest;
- no Week-6 outcome use;
- no production promotion.

If still unavailable:
- remain fail-closed;
- do not substitute Week-5 roster;
- immediately proceed to Step 3.

## Step 3 — open individual efficiency-tail uncertainty lane

Do an anti-retest inventory first.

Then freeze a bounded historical diagnostic/source contract before scoring.

Goal:
identify player-specific efficiency-tail risk beyond current means using only
strictly-prior football information.

Do not tune to Week-5 misses.

## Step 4 — repair PR-side Arrow/NumPy workflows when convenient

Mechanical only.
No football-science changes.

## Step 5 — QB V1L passive refresh only

Refresh when genuine first-party media availability makes sense.
Never loosen the source rules.

## Step 6 — maintain research/ticket separation

Do not use user-facing bet outcomes to retune football science.

---

# 24. CURRENT SCIENTIFIC HEADLINE

The project is **not finished**.

The current strongest truth is:

> **Individual opportunity allocation uncertainty is a real, replicated,
> player-specific football signal. For RB carries and WR targets, a
> team-conserving joint Dirichlet model materially improves historical
> probability quality across 2023, 2024, and 2025 while preserving exact player
> means and exact team opportunity totals.**

At the same time:

> **Raw player mean accuracy is still unresolved, Week-6 prospective locking is
> source-gated by exact target-week roster identity, and individual
> efficiency-tail uncertainty is now the next non-time-gated player-specific
> root cause to attack.**

That is where the next chat must resume.
