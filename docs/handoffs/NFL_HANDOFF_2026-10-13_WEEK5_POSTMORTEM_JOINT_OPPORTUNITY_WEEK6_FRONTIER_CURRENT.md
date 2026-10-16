# NFL HANDOFF — 2026-10-13 — WEEK-5 POSTMORTEM / JOINT OPPORTUNITY UNCERTAINTY / WEEK-6 FRONTIER CURRENT

GitHub is canonical. This handoff is the authoritative deep continuity document
for the next chat as of 2026-10-13 morning.

Repo:
- `dkaps6/imtiredofthis`

Active research container:
- branch: `research-individual-opportunity-roadmap-2026-10-08`
- pre-handoff branch head: `3e90aec415385a4a06d32319233107e057ee8d32`
- draft PR: **#672**
- continuity issue: **#535**

Production main at handoff preparation:
- `812ac2cde79fe3846174613a2f2777e57b34b739`
- `Docs: preserve model-backed alt-line parlay construction method`

The research branch contains many research-only descendants of main. Do not
confuse branch-head research commits with production promotion.

---

## 1. START RULE — DO THIS, NOT A RESTART

Read in this order:

1. `AGENTS.md`
2. ONLY the newest TOP checkpoint in `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. this file
4. `docs/research/WEEKS1_5_CONTINUITY_WEEK5_POSTMORTEM_SYNTHESIS_V1.md`
5. `docs/research/PLAYER_OPPORTUNITY_JOINT_ALLOCATION_DIRICHLET_V1_HISTORICAL_RESULT.md`
6. `docs/research/PLAYER_OPPORTUNITY_JOINT_ALLOCATION_DIRICHLET_V1_PROSPECTIVE_SHADOW_MECHANICS_RESULT.md`
7. `docs/research/WEEK6_INDIVIDUAL_OPPORTUNITY_LOCK_READINESS_RESULT.md`
8. newest comments on Issue #535 / PR #672
9. live main / active research branch / exact-head Actions

Do **not** recursively read old handoffs.

Do **not** rerun W1-W4 all-player replay, Player Landscape, Player Output
Component Decomposition, completed Week-5 postmortem, or closed matchup
experiments merely to reacquire context.

If live GitHub moved after this handoff, reconcile the newer branch state first.
Newer frozen plans/results on the same research branch beat this narrative.

---

## 2. USER INTENT / NON-NEGOTIABLE MODEL VISION

The user is not asking for a generic position model.

The model end state is:

> project every player as an individual football player in the exact game he is
> about to play, using his current role, usage, team environment, offense,
> opponent, individual history, efficiency, matchup interaction, injuries /
> personnel, and uncertainty.

Conceptually:

`player projection = active-role state x team opportunity x individual share x individual efficiency x player-specific matchup/context + individual uncertainty`

Important interpretation from completed decomposition:
- individual-player opportunity/share error remains the biggest systematic
  projection problem;
- that does **not** mean generic team/position averaging is the solution;
- it means the model must become better at assigning the offense's finite
  opportunities to the correct individual player and quantifying uncertainty
  around that assignment.

User explicitly wants:
- all positions;
- true individualization;
- football-reasonable projections, not raw averages;
- fewer / higher-confidence wagers;
- own-line thinking eventually;
- model-backed alt lines;
- no fabricated certainty;
- no repeated failed research under new names.

The user is frustrated because it is now Week 6 and feels progress has not
translated consistently into winning tickets. Do not respond by abandoning the
project or by claiming the model is solved. Keep execution moving.

---

## 3. PRODUCTION / WEEK-5 LIVE AUTHORITY — DO NOT LOSE THIS

Canonical Week-5 preserved-odds Full Slate authority:
- run: `37860592615` SUCCESS
- artifact: `11585973062`
- digest:
  `sha256:bf52aa4bc30bf6e8cc6029a0a86c0f33cdfaa09f587c38a138f23cef6b7436f4`
- production behavior authority head:
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
- ZERO second OddsAPI acquisition was used for canonical replay;
- do not spend another paid OddsAPI credit without explicit user authorization;
- preserved replay prices are stale and must never be labeled current /
  "Bettable Now".

Production main remains separate from research PR #672.

---

## 4. WEEK 1-5 POSTMORTEM — CURRENT FACTUAL TRUTH

Canonical synthesis:
`docs/research/WEEKS1_5_CONTINUITY_WEEK5_POSTMORTEM_SYNTHESIS_V1.md`

### Weeks 1-4 selected board

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
- mean stated prob 80.52%
- realized win rate 53.60%
- 57 eligible clustered slices
- 0 BH-FDR survivors

Do not resurrect raw-edge worship as confidence.

### Week 5 selected board

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
- 2,017 decided
- 51.96%
- -17.93u
- approximate simple cumulative ROI -0.89%

### Week-5 raw model vs market

Full football board:
- model MAE: 18.9288
- preserved median-line MAE: 17.8431
- model closer than line: 50.26%
- model direction hit rate vs median line: 55.32%

Interpretation:
**Week-5 profitability came primarily from downstream selection / price handling,
not from a solved raw mean-projection layer.**

Do not say the model is fixed because the selected board made money.

### Week-5 top failure decomposition

Top 40 absolute final-model player-market misses:
- efficiency-dominant: **19**
- opportunity-dominant: **14**
- mixed opportunity/efficiency: **7**

This creates two separate root-cause lanes:
1. opportunity / role allocation uncertainty;
2. individual efficiency-tail uncertainty.

Both matter.

---

## 5. USER-FACING TICKET POSTMORTEM — SEPARATE DOWNSTREAM DEFECT

Canonical:
`docs/research/WEEK5_USER_FACING_TICKET_CONCENTRATION_POSTMORTEM_V1.md`

Do not use these outcomes to retune football science.

### Watson

User reported Deshaun Watson finished with 155 passing yards.

Canonical:
- model 238.23
- actual 155
- predicted attempts 24.77
- actual attempts 33
- predicted YPA 7.28
- actual YPA 4.70
- label: `MIXED_OPPORTUNITY_EFFICIENCY`

Watson appeared across multiple user-facing tickets.

Interpretation:
- genuine player projection failure;
- NOT primarily alt-line construction failure;
- cross-ticket repeated exposure turned one wrong player thesis into multiple
  ticket losses.

Process defect:
`CROSS_TICKET_PLAYER_CONCENTRATION`

### BUF-LAR Monday core alt ticket

Suggested:
- Josh Allen 200+ pass -> actual 210 HIT
- Puka Nacua 50+ rec -> actual 138 HIT
- Kyren Williams 50+ rush -> actual 49 MISS by 1

This was a 2/3 ticket and a one-yard threshold miss.

Interpretation:
`ALT_LINE_NEAR_MISS__NOT_EQUIVALENT_TO_WATSON_PROJECTION_FAILURE`

Higher-payout expansion also included:
- Keon Coleman 40+ -> actual 7 MISS
- Dalton Kincaid 40+ -> actual 7 MISS

Stafford standalone:
- model ~242.3
- recommended U278.5
- actual 383
- large opportunity-dominant miss / bad player-game thesis.

### Required downstream controls

Prospectively:
1. cap repeated non-lock player exposure across paid tickets;
2. model mean > alt line does NOT equal calibrated survival probability;
3. tag failure type before reusing a thesis;
4. track ticket-level / portfolio-level survival;
5. avoid one player, one game, one position seam, or one correlated thesis
   dominating multiple tickets;
6. keep lottery entertainment tickets separate from the core process.

The selected board can be good while the user-facing ticket layer is bad.

---

## 6. CORE SCIENTIFIC FOUNDATION — COMPLETED / DO NOT REDO

Completed:
- W1-W4 all-player replay
- Player Landscape Transmission Audit
- Player Output Component Decomposition
- availability parity / opportunity audits
- WR/TE trajectory historical integration work
- Week-5 full model postmortem / player failure casebook
- Weeks 1-5 continuity synthesis

Key decomposition:
opportunity oracle beats efficiency oracle in all 8 primary cells.

ACT-only corrected approximate opportunity-oracle mass:
- QB team pass opportunity ~60.3% attempt-MAE oracle removal
- RB carry share ~64.5%
- RB target share ~76.5%
- WR target share ~68.6%
- TE target share ~77.1%

The active roadmap therefore remains **individual opportunity allocation first**,
with efficiency-tail work now becoming the next independent root-cause lane
because opportunity uncertainty has finally produced a major historical PASS.

---

## 7. MAJOR NEW PASS — PLAYER-SPECIFIC OPPORTUNITY SHARE VOLATILITY

### V1 corrected 2024-2025

Canonical:
- run `38292658912` SUCCESS
- exact head `1d224c8cc5804eec110f9c9750d60d0f506bc7e4`
- artifact `11767472231`
- digest
  `sha256:021a9fc79f28b4568b8663be6e57d194686b785ccbfd4c6d0a088d7e44a4f5a3`
- result:
  `docs/research/PLAYER_OPPORTUNITY_SHARE_STABILITY_V1_RESULT.md`

Important repair:
an earlier all-zero result was INVALID because raw roster GSIS IDs (`00-...`)
did not match PlayerForm identity keys (`gsis:00-...`).

Final corrected run added a non-degenerate-share integrity gate.

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

All relevant game/player clustered CI lower bounds >0.

### Untouched 2023 holdout

Canonical:
- run `38294106319` SUCCESS
- exact head `cad154e0f2fd0bdd2d1d5f502eda77c13ae2f636`
- artifact `11767104207`
- digest
  `sha256:e5a042d84ba575fedd31148f9c5e9a5bebc9bd8f1ae36e2decd9df173280e8ce`
- result:
  `docs/research/PLAYER_OPPORTUNITY_SHARE_STABILITY_2023_HOLDOUT_RESULT.md`

2023 also confirms all four:
- RB carry rho +.6409
- RB target +.6116
- WR target +.4686
- TE target +.5699

Disposition:
`PLAYER_OPPORTUNITY_SHARE_STABILITY_THREE_SEASON_CONFIRMED`

This is one of the strongest player-specific signals found in the project.

It changes uncertainty logic, NOT mean opportunity.

---

## 8. V2 EXCESS ROLE VOLATILITY — SURVIVES FINITE-COUNT NOISE

Result:
`docs/research/PLAYER_OPPORTUNITY_SHARE_VOLATILITY_MECHANISM_V2_RESULT.md`

Canonical authority from PR/Issue continuity:
- run `38295515693` SUCCESS
- exact head `81cd2beeb6f8da9fd5e485d70daee546285e6905`
- artifact `11768432238`
- digest
  `sha256:7b31dd29e4b94674d6948cf84af24fae7181d1a4f502a07beaa43919d4d7f3a6`

Disposition:
`PLAYER_OPPORTUNITY_EXCESS_VOLATILITY_EXISTS`

V2 removes expected finite-count sampling variance already represented by
fixed-share multinomial allocation and partials out prior mean share + expected
sampling SD.

Three-season confirmed:
- RB carry share
- RB target share
- WR target share

TE target share:
- NOT confirmed under frozen all-three-season gate;
- 2023 player-cluster CI low -0.0134;
- do not rescue TE by pooling or changing thresholds.

Representative excess-SD rho:
- RB carry: .4989 / .4609 / .5024
- RB target: .3293 / .3053 / .3049
- WR target: .2502 / .2927 / .2730

Meaning:
there is genuine player-specific role/share instability beyond ordinary
multinomial count noise.

---

## 9. MARGINAL BETA-BINOMIAL UNCERTAINTY — HISTORICAL PASS / FAIL SPLIT

Canonical:
- run `38296726460` SUCCESS
- exact head `b94e2188b6fd0c20b41263b6dc9c8161b151d5da`
- artifact `11768483266`
- digest
  `sha256:403360d3245e78507cfe7646022291a676535fe6bf90d607ba85f89f900eae4c`
- result:
  `docs/research/PLAYER_OPPORTUNITY_ALLOCATION_UNCERTAINTY_BETABINOMIAL_V1_RESULT.md`

Candidate preserves expected count exactly:
`E[count] = N * prior_share_mean`

PASS all 2023/2024/2025:
- **RB carry share**
  - CRPS improvement +.1814 / +.1437 / +.1033
  - all game/player cluster CI lower bounds >0
  - paired NLL improves all three
- **WR target share**
  - CRPS improvement +.0184 / +.0253 / +.0277
  - all game/player cluster CI lower bounds >0
  - paired NLL improves all three

FAIL CLOSED:
- RB target share:
  - 2025 point CRPS +.0020
  - game CI low -0.00132
  - player CI low -0.00159
- TE target was never authorized into this family because V2 failed its
  three-season mechanism gate.

Do not reopen RB target or TE under this exact candidate.

---

## 10. MAJOR CURRENT PASS — JOINT / TEAM-CONSERVING DIRICHLET ALLOCATION

This is the most important current scientific frontier.

Canonical historical result:
- run `38401069806` SUCCESS
- exact scoring head `5f35f96f0a43b9963691a7768a330b32b4854a5b`
- artifact `11815490905`
- digest
  `sha256:2b336114dc5ef0145c0a40706266da850a0ff1d25c8935958481b996333fce29`
- result:
  `docs/research/PLAYER_OPPORTUNITY_JOINT_ALLOCATION_DIRICHLET_V1_HISTORICAL_RESULT.md`

Passing cells:
- **RB_CARRY_SHARE**
- **WR_TARGET_SHARE**

Both:
`JOINT_ALLOCATION_DIRICHLET_HISTORICAL_PASS_FREEZE_FORWARD_SHADOW`

No pass/authorization for:
- RB target
- TE target

Candidate properties:
- exact canonical player means preserved;
- exact finite team opportunity totals conserved;
- player-specific latent role uncertainty comes from strictly-prior excess
  share variance;
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

### RB carry joint log-score improvement

2023:
- +3.42013
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

### WR target joint log-score improvement

2023:
- +.78218
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

All historical frozen gates pass.

Interpretation:
the current fixed-share multinomial allocation is too narrow for individual RB
carry allocation and individual WR target allocation.

This is a **distribution / uncertainty improvement**, not a new mean allocator.

---

## 11. PROSPECTIVE JOINT SHADOW — MECHANICS READY

Result:
`docs/research/PLAYER_OPPORTUNITY_JOINT_ALLOCATION_DIRICHLET_V1_PROSPECTIVE_SHADOW_MECHANICS_RESULT.md`

Mechanics authority:
- run `38402018307` SUCCESS
- exact head `fb4136b639ac4b2281a57a6d3a383c97b5c79fe6`

Future manual runner:
`.github/workflows/research-future-week-joint-opportunity-shadow-v1.yml`

Certified:
- exact target-week player-state authority required;
- stale prior-week roster rejected;
- only RB carry / WR target accepted;
- canonical mean q preserved exactly;
- alpha/alpha0 equals canonical q within 1e-12;
- room probability mass conserved;
- strictly-prior source week required;
- zero-excess / unauthorized players stay ineligible;
- no production simulator import/reference;
- no target-week outcomes.

This shadow is research-only and does not mutate production pricing.

---

## 12. WEEK-6 OPPORTUNITY LOCK — CURRENT HARD BLOCK

Canonical readiness result:
`docs/research/WEEK6_INDIVIDUAL_OPPORTUNITY_LOCK_READINESS_RESULT.md`

Authority:
- run `38399980031`
- exact head `70090477c416541ce2475b218ea502f3d9833091`
- diagnostic artifact `11812829801`
- disposition:
  `PLAYER_STATE_FUTURE_WEEK_NOT_READY`

Ready:
- Week-5 stats complete
- Week-5 snaps complete
- schedule context available
- usage coverage among played ~99.458%
- RB/WR/TE snap coverage ~99.379%
- stable ID coverage ~99.848%
- chronology violations 0
- 28 scheduled Week-6 teams

Hard blocker:
- target week = 6
- roster source week = 5
- `target_week_roster_source_ready = false`
- target-week injury rows available = 0

No Week-6 immutable lock was legally created:
- no share-volatility lock
- no RB carry/snap lock
- no RB receiving lock
- no WR/TE trajectory lock
- no immutable opportunity lock pack
- no prospective joint Dirichlet shadow lock

**Do not substitute Week-5 roster identity.**

Immediate operational rule:
as soon as exact Week-6 roster authority appears, rerun the existing frozen
future-week lock workflow and then run/freeze the joint opportunity shadow.

Do not change any rules merely to make Week 6 pass.

---

## 13. EXISTING FUTURE-WEEK OPPORTUNITY MECHANICS — READY

Mechanics are already built/certified for:
- exact target-week player state
- RB carry/snap allocation
- RB receiving-room allocation
- WR/TE target-share trajectory
- raw share-volatility lock
- joint RB-carry / WR-target opportunity shadow

There is also a fail-closed weekly opportunity lock-pack orchestration layer.

The current delay is source readiness, not missing architecture.

Do not spend research time reinventing these components while waiting on roster
publication.

---

## 14. QB TEAM PASS-INTENT SIDE LANE — USEFUL BUT PASSIVE

This is a narrow upstream QB team-pass-opportunity source experiment, not the
main player-specific roadmap.

### Corrected archive transport

V1I-R1:
- run `38186579290` SUCCESS
- artifact `11719552673`
- digest
  `sha256:c371cde47ac800fb9665dcdd511d33c6626e59a4f47ba3f45e10fe17f7f063bd`
- corrected combined transport 141/158 = **89.24%**
- transport gate 80% PASS
- original geography-only false-positive V1I count is superseded

### Semantic automation closed

V1J independent NLI:
- run `38186677190`
- failed safety gate / no promotion

V1K blinded precision:
- run `38187415229`
- 4/32 genuinely eligible explicit target-game intent
- 25/32 generic/descriptive
- 3/32 insufficient context
- blanket V1G promotion rejected

Do not:
- manually adjudicate all 378;
- retune V1F;
- cascade another NLI model;
- promote historical V1G wholesale.

### V1L prospective collector

Initial:
- run `38187787580`
- artifact `11719983641`
- 28/28 teams attempted
- 0 retained sources

Monday attempt 2:
- same run, attempt 2
- artifact `11752326387`
- digest
  `sha256:415a04ebfe254f2b18665d641943ed9dbb6996dff6cca613856e92bd39736d0b`
- 0 retained sources again

Interpretation:
source/time-gated, not mechanically broken.

Let V1L continue prospectively when genuine first-party coach/OC media appears.
Do not loosen source families.

QB protected production science:
- M89/M90
- QB C2

Closed:
- generic QB mean/YPA
- same-data attempts repackaging
- schedule/rest D1
- PBP D2

QB intent remains one possible upstream team pass-volume input only.

---

## 15. PLAYER-SPECIFIC MATCHUP WORK — WHAT CLOSED

Do not restart generic matchup work.

### WR/TE depth interaction

Source was strongly reconstructable:
- 7,184 source-ready player-weeks
- WR 4,666
- TE 2,518

Exact direct depth-YPT transmission failed:
- WR worsened in both 2024 and 2025
- TE improved/supportive in 2025 but not 2024
- frozen replication gate failed
- scorer also violated untreated-output isolation

Result:
`docs/research/PLAYER_ROLE_MATCHUP_DEPTH_INTERACTION_PREDICTIVE_V1_RESULT.md`

Do not rescue by:
- changing depth bands;
- shrinkage;
- slot/deep/WR1 subgroup mining;
- switching post-hoc to completion interaction.

### RB individual run-location interaction

Source qualified:
- 2,654 source-ready RB player-weeks

Predictive result:
- run `38258940285` SUCCESS
- exact scoring head `20230872a13729f8c5269b653c46ffec030c1c78`
- artifact `11752168196`
- digest
  `sha256:476000f0f6814af60f956f71bf96708442057f6a4b55a37e975bdc0bfac77ed2`
- disposition:
  `RB_RUN_LOCATION_MATCHUP_HISTORICAL_FAIL_CLOSED`

2024:
- MAE 23.7181 -> 23.6520
- +.0661 point improvement
- clustered CIs cross 0

2025:
- MAE 23.2285 -> 23.2506
- -.0221
- RMSE slightly worse

Do not rescue by tuning left/middle/right, bounds, success rate, inside/outside
subgroups, or M95 recombination.

### Earlier closed matchup families

Also remain closed:
- generic defense-v-position multipliers
- Football Matchup Transmission simple formulas
- M95A/M95B
- Coverage-v2 team man/zone
- direct historical WR-CB assignment unless a genuinely validated source exists
- universal target-depth distribution transform

The correct lesson is NOT "matchups do not matter."
It is: these exact transmission formulas did not earn promotion.

---

## 16. RB / WR / TE MEAN-ALLOCATION STATUS

### RB receiving

Historical W1-W4 player-specific receiving-room allocation improved:
- targets +7.60% MAE
- receptions +3.83%
- rec yards +2.89%
- rush+rec +1.66%

Preserve that science.

### RB carry/snap

Frozen future allocator exists:
- 50% recent carry share
- 50% snap fraction

Week-5 first prospective lock did not beat recent-carry control.

Do not promote from one lock and do not retune to Week-5 outcome.

### WR/TE target trajectory

Historically supported / prospective mechanics exist.

Week-5:
- WR essentially flat/slightly worse
- TE small positive first lock

Do not weaken exact eligibility:
- >=4 prior same-season same-team target-team games.

No broad new mean allocator was promoted from Week 5.

---

## 17. NEXT NON-TIME-GATED FOOTBALL SCIENCE — EFFICIENCY-TAIL UNCERTAINTY

Now that opportunity uncertainty has a major three-season PASS and prospective
shadow machinery is built, the other major Week-5 failure family must be
attacked separately.

Week-5 top-40 misses:
- efficiency-dominant 19
- opportunity-dominant 14
- mixed 7

Next open scientific question:

> Does strictly-prior individual efficiency volatility / regime state identify
> next-game efficiency-tail risk for the individual player, while preserving the
> existing mean projection?

Potential cells must be reconciled against all prior research before freezing:
- QB passing efficiency / YPA tail uncertainty
- receiver YPT / catch-efficiency tail uncertainty
- RB YPC tail uncertainty

Do NOT begin by fitting Week-5 misses.

Required protocol:
1. audit already-completed efficiency/distribution work to prevent retest;
2. define strictly-prior player-specific feature(s);
3. freeze exact historical diagnostic before scoring outcomes;
4. separate mean bias from tail/uncertainty;
5. require multi-season replication and clustered support;
6. no sportsbook;
7. no Week-5 player-specific tuning;
8. preserve promoted mean science;
9. if mechanism passes, freeze separate distribution candidate before scoring;
10. no production promotion without prospective evidence.

Anti-retest reminders:
- target-depth dispersion was a player-specific difficulty signal, but the
  universal symmetric width transform failed;
- generic QB mean/YPA is closed;
- player matchup direct multipliers above are closed;
- Opportunity State Conflict Uncertainty was null;
- generic `rules_volatility_mult` represents yardage/efficiency noise, not
  role allocation uncertainty.

Before opening a new efficiency lane, inspect exact existing research docs / Issue
#535 for overlap.

---

## 18. WEEK-6 IMMEDIATE EXECUTION ORDER

At next-chat start:

### First
Reconcile:
- live research branch
- main
- PR #672
- Issue #535 latest comments
- exact-head Actions

### Then check Week-6 source readiness

If exact Week-6 roster authority is NOW available:
1. rerun the frozen future-week opportunity lock pack;
2. require exact Week-6 identity;
3. require complete Week-5 stats/snaps;
4. freeze raw share-volatility state;
5. freeze RB carry/snap, RB receiving, WR/TE trajectory;
6. build the joint Dirichlet shadow for RB carry + WR target only;
7. preserve artifact/run/digest BEFORE Week-6 outcomes;
8. do not mutate production;
9. do not grade until games settle.

If exact Week-6 roster authority is STILL unavailable:
- leave lock fail-closed;
- do NOT substitute Week 5 roster;
- continue the efficiency-tail historical lane;
- optionally refresh QB V1L first-party media collector without loosening rules.

### Parallel downstream work
Ticket construction can proceed as a separate risk layer:
- concentration cap
- survival calibration
- portfolio accounting

But do not allow ticket outcomes to choose football-model features.

---

## 19. WEEK-6 / PROSPECTIVE LOCK DO-NOTS

Do not:
- use Week-6 game outcomes before lock;
- use target-game participation to construct the pregame universe;
- carry Week-5 roster forward as Week-6 authority;
- use sportsbook prices upstream;
- spend OddsAPI without explicit approval;
- silently production-promote the Dirichlet shadow;
- apply passing cells to RB targets or TE targets;
- weaken eligibility because Week 6 is urgent.

Fail closed is correct behavior.

---

## 20. CURRENT ACTIONS / RED-X INTERPRETATION

On research head `3e90aec415385a4a06d32319233107e057ee8d32` at handoff preparation:

Green:
- Repo CI run `38402198481` SUCCESS
- Backtest Historical Input Validation `38402198342` SUCCESS
- Historical Benchmark Clean Rebuild V1 `38402198378` SUCCESS
- Prospective Joint Opportunity Shadow mechanics `38402018307` SUCCESS
- Joint historical scoring `38401069806` SUCCESS

Red but currently infrastructure/noise, NOT scientific invalidation:
- Backtest Weeks 1-2 Full Board `38402198463`:
  - fails in `grade_market_track_record_gsis_v1.py`
  - Polars `.to_pandas()` -> pyarrow import
  - pyarrow requires NumPy >=2.0, repo env has NumPy 1.26.4
- Replay Paid Full Slate Artifact Once `38402198332`:
  - fails in TE-R5P snap loader for the same Polars -> pandas -> pyarrow /
    NumPy incompatibility
  - `FETCH_LIVE_ODDS=false`
  - this is NOT a new OddsAPI pull
  - canonical Week-5 production authority remains intact

These should eventually be repaired mechanically (e.g. Arrow-free Polars bridge
or compatible pyarrow pin) without changing football science.

Do not mistake these PR-side red checks for a failure of the joint allocation
research result.

---

## 21. CLOSED / PROTECTED SCIENCE — DO NOT REOPEN

Protect:
- QB M89/M90
- QB C2
- M38 WR1 anchor
- WR-R15
- TE-R5P
- RB receiving-room historical improvement
- exact WR/TE trajectory eligibility
- production Week-5 preserved board authority

Closed:
- generic QB mean/YPA
- QB same-data attempts repackaging
- schedule/rest D1
- PBP D2
- historical QB-intent blanket automation
- WR-R3 combined calibration
- coverage-v2
- generic FMT direct matchup transforms
- M95A/B
- M96 RB retrospective program
- universal target-depth symmetric distribution transform
- player-depth direct YPT V1
- RB run-location direct YPC V1
- RB target Beta-Binomial V1
- TE target Beta-Binomial / joint family under current mechanism
- Opportunity State Conflict uncertainty

Do not reopen closed science because a Week-5 player missed.

A new attempt must use genuinely new information or a genuinely different
football mechanism.

---

## 22. GITHUB / CONCURRENCY RULES

GitHub is canonical.

If Issue/PR comments conflict:
1. live production main is canonical for production;
2. live research branch is canonical for that lane;
3. newest frozen plan/result on branch beats stale comment prose;
4. exact-head Actions verify mechanics;
5. result artifact beats a green workflow label.

Before every write:
- re-fetch research branch head;
- do not overwrite concurrent commits.

Do not claim a run is active without checking Actions.

Do not fabricate a result from chat memory.

---

## 23. USER COMMUNICATION / WORK STYLE

The user wants execution, not repeated roadmap speeches.

Preferred behavior:
- take the wheel;
- keep moving autonomously;
- give concise status after meaningful milestones;
- when a run fails mechanically, fix it and rerun until resolved;
- do not ask the user to restate project history;
- do not restart completed research;
- distinguish real science failures from CI mechanics;
- do not call a one-yard miss the same thing as a catastrophic projection miss;
- maintain football-language interpretation of what each mechanism means.

User is frustrated by:
- repeated timeouts;
- apparent restarts;
- research that does not translate to individual players;
- parlays dying by one leg;
- Week 6 arriving before the model feels finished.

Respond with concrete movement, not reassurance alone.

---

## 24. EXACT NEXT ACTIONS — CURRENT PRIORITY

Unless GitHub has moved materially, the next chat should:

1. Reconcile live branch/main/PR #672/Issue #535/Actions.
2. Check whether exact Week-6 weekly roster authority now exists.
3. If yes:
   - dispatch the frozen Week-6 opportunity lock pack;
   - freeze the joint Dirichlet RB-carry / WR-target prospective shadow;
   - preserve run/artifact/digest before any Week-6 result;
   - remain research-only.
4. If no:
   - keep Week-6 lock fail-closed;
   - immediately open the historical **individual efficiency-tail uncertainty**
     lane after an anti-retest audit.
5. In parallel or immediately after the lock:
   - mechanically repair the two Arrow/NumPy PR-side red workflows without
     changing science.
6. Refresh V1L prospective QB first-party source collection when appropriate;
   never loosen source families.
7. Do not spend OddsAPI unless user explicitly authorizes a new paid pull.
8. Keep ticket-construction research downstream and separate.

The project is not finished. The correct current scientific headline is:

> **Individual opportunity allocation uncertainty is now a real three-season
> validated signal for RB carries and WR targets under a team-conserving joint
> model; Week-6 prospective lock is source-gated; raw player mean accuracy and
> individual efficiency tails remain unresolved.**
