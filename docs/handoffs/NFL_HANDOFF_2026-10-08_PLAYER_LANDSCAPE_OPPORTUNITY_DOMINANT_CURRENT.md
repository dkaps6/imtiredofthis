# NFL HANDOFF — 2026-10-08 — PLAYER LANDSCAPE / OPPORTUNITY-DOMINANT CURRENT

GitHub is canonical. Chat memory is secondary.

This handoff supersedes the 2026-10-07 Player-Centric All-Positions checkpoint for
immediate continuation.

Repository: `dkaps6/imtiredofthis`

## 0. Immediate user intent

The user wants the NFL model to remain an **individual-player model**, not merely
a position/team model with player names attached.

The target mental model is:

> For this specific QB/RB/WR/TE, given his current identity, availability, role,
> recent workload, room hierarchy, team opportunity environment, individual
> efficiency, opponent defense, matchup interaction, injuries/vacancies, and
> uncertainty, what should his own usage and output be this week?

The user explicitly asked whether we are really doing this at the individual
player level. The answer after the completed audits is:

- **YES:** the current stack has a genuine individual-player core.
- **NO:** the entire football landscape does not yet transmit into every final
  player projection.
- **MOST IMPORTANT:** the completed W1-W4 decomposition shows the dominant
  remaining error source across every tested player market is **opportunity /
  workload allocation**, not generic per-opportunity efficiency.

Do not pivot away from this conclusion.

The user also explicitly does **not** want unfinished work abandoned. Before the
current checkpoint, the prior player-centric lanes were reconciled and either
completed, frozen prospectively, source-blocked, or closed.

---

## 1. Canonical repo state at handoff

Main before this continuity update:

`7336994c8c5c5b232648ff8d899bca806628cff8`

Active research branch:

`research-player-output-component-decomposition-v1`

Current branch head after docs-only result-authority update:

`35fb319cad55c4588124cdd5b8e56522d6838054`

Latest **scientific exact-head** successful run:

- workflow: `Research Player Output Component Decomposition V1`
- run: `37777319154`
- source head: `cb1e511adcb58e40fc41d0949306083a304e83dd`
- conclusion: **SUCCESS**
- tests: SUCCESS
- decomposition: SUCCESS
- certification: SUCCESS
- strict repo audit: SUCCESS
- artifact upload: SUCCESS

Artifact:

- id: `11550446909`
- name: `player-output-component-decomposition-v1-37777319154`
- digest:
  `sha256:7acfc453258a49f3aadf0d0ea386b13827eb715f1dfb055b96505f733320d204`
- expires: 2027-01-06

Canonical result doc on active branch:

`docs/research/PLAYER_OUTPUT_COMPONENT_DECOMPOSITION_V1_RESULT.md`

No relevant research Action is intentionally left running at this handoff.
Always verify live Actions before claiming otherwise.

---

## 2. Current scientific headline

Frozen disposition from the final W1-W4 component decomposition:

`OPPORTUNITY_DOMINANT_ACROSS_ALL_PRIMARY_MARKETS__INDIVIDUAL_ROLE_SHARE_ALLOCATION_NEXT`

This is the strongest current organizing result.

The decomposition used completed 2026 Weeks 1-4 individual player-game
projections and asked two counterfactual questions:

1. **Opportunity oracle:** what if the model had the correct workload
   (attempts/carries/targets) but kept its own effective per-opportunity output?
2. **Efficiency oracle:** what if the model had the correct realized
   per-opportunity output but kept its own projected workload?

In all eight predeclared cells, correcting opportunity removes more MAE than
correcting efficiency.

| Position | Market | Baseline MAE | Opp oracle MAE | Opp MAE removed | Eff MAE removed* | Dominant |
|---|---|---:|---:|---:|---:|---|
| QB | pass_yards | 73.995 | 56.328 | **23.9%** | 12.7% | OPPORTUNITY |
| RB | rush_yards | 19.846 | 10.893 | **45.1%** | 7.2% | OPPORTUNITY |
| RB | rec_yards | 10.466 | 5.747 | **45.1%** | 27.5% | OPPORTUNITY |
| RB | receptions | 1.180 | 0.505 | **57.2%** | **-2.4%** | OPPORTUNITY |
| WR | rec_yards | 22.275 | 12.739 | **42.8%** | 28.2% | OPPORTUNITY |
| WR | receptions | 1.511 | 0.689 | **54.4%** | 11.6% | OPPORTUNITY |
| TE | rec_yards | 15.615 | 7.026 | **55.0%** | 23.5% | OPPORTUNITY |
| TE | receptions | 1.427 | 0.453 | **68.3%** | 8.1% | OPPORTUNITY |

*Efficiency percentages are on the frozen positive-actual-opportunity subset and
are not population-identical to the opportunity percentages.

Opportunity error is also more strongly correlated with final output error in
every market:

- QB pass yards: **0.814** vs efficiency 0.491
- RB rush yards: **0.828** vs 0.401
- RB rec yards: **0.789** vs 0.619
- RB receptions: **0.923** vs 0.330
- WR rec yards: **0.749** vs 0.553
- WR receptions: **0.871** vs 0.383
- TE rec yards: **0.788** vs 0.515
- TE receptions: **0.907** vs 0.336

Interpretation:

> Do not respond to the player's remaining misses by opening another generic
> YPT/YPC/YPA/catch-rate, coverage, defense-vs-position, or game-script
> multiplier. The main bottleneck is allocating the right amount of opportunity
> to the named player.

This does **not** mean matchup or efficiency is irrelevant. It means they are
secondary to the current opportunity problem.

---

## 3. Final decomposition integrity / grading-source details

Parents:

- all-player W1-W4 point scoreboard:
  run `37683439543`
- replay-matched baseline opportunity rows:
  run `37687979574`

Do not pair the ACT-only opportunity variant to the baseline point scoreboard.
That mixes model variants and was correctly rejected during development.

Population:

- paired rows: **4,017**
- oracle-scoreable rows: **3,991**
- model-effective-efficiency unavailable rows: **21**
- max baseline reconstruction gap:
  `2.842170943040401e-14`
- max full-actual identity gap:
  `5.684341886080802e-14`
- parameters fit: **0**
- threshold searches: **0**
- sportsbook inputs upstream: **false**
- paid OddsAPI: **false**
- automatic promotion: **false**

Actual target grading was hardened during this chat.

Why:
the frozen weekly-stat artifact contained impossible target semantics in a small
number of rows (receiving production with zero reported targets).

Final grading authority:

- QB pass attempts: frozen replay-matched opportunity artifact
- RB carries: frozen replay-matched opportunity artifact
- RB/WR/TE targets: completed-game nflverse PBP target counts, **grading only**
- identity route:
  PBP receiver GSIS ID -> validated weekly-roster aliases -> canonical player
  key
- zero-target / zero-output rows may use explicit frozen zero-use fallback

The PBP grading source is downstream only and never enters the prediction.

Final run facts:

- target rows graded by PBP: **2,056**
- verified zero-use fallback rows: **1,360**
- resolved target-count discrepancy rows: **0**
- grading identity mismatch rows: **0**
- source-conflict exclusions: **5 rows / 2 player-weeks**

Explicit excluded player-weeks:

- Deebo Samuel Sr. — Week 3
- David Montgomery — Week 4

Reason:

`GRADING_SOURCE_CONFLICT_UNRESOLVED_PBP_TARGET`

Those rows are retained as evidence but excluded from all oracle comparisons.
Do not fabricate a target count to force them in.

---

## 4. Player Landscape Transmission Audit V1 — COMPLETE

Branch:

`research-player-landscape-transmission-audit-v1`

Canonical result doc:

`docs/research/PLAYER_LANDSCAPE_TRANSMISSION_AUDIT_V1_RESULT.md`

Completed run:

`37716318460` SUCCESS

Artifact:

`11523478868`

Disposition:

`INDIVIDUAL_PLAYER_CORE_CONFIRMED__LANDSCAPE_TRANSMISSION_INCOMPLETE`

What it proved:

The current model is **not** a position model with player names attached.

All nine required player markets have:

- individual usage state;
- individual efficiency state;
- material opponent context.

The dynamic Week-5 pregame trace followed named players through:

`identity -> availability -> player state -> room state -> team environment ->
opponent environment -> opportunity -> efficiency -> final mean -> distribution`

The audit selected deterministic LOW / MEDIAN / HIGH projected-workload players
for QB/RB/WR/TE without outcome selection.

Static architecture inventory:

- concrete feature/mechanism families: **51**
- expanded position-market transmission rows: **227**
- actively consumed rows: **143 / 227 = 63.0%**
- gap/blocked/prospective/closed rows: **84 / 227 = 37.0%**

Do **not** interpret 37% as “37% of model quality missing.” It is an architecture
inventory count with duplicated markets and intentionally closed/source-blocked
families.

Important architecture result:

- QB is more complete because of the M89/M90 specialist path.
- The larger incomplete-transmission problem is generic RB/WR/TE.
- But the later output decomposition says opportunity allocation is the primary
  problem to solve before chasing more matchup/efficiency transmission.

Representative Week-5 pregame trace examples from the successful audit were
named-player projections such as Tyler Shough, Jahmyr Gibbs, Chris Olave and
Brock Bowers. They are architecture evidence, not recommendations or bets.

---

## 5. All-player / all-position W1-W4 replay — COMPLETE

Branch:

`research-all-player-all-position-replay-v1`

Successful run:

`37683439543`

Artifact:

`11509468659`

Digest:

`sha256:abb5bb7ba85d13306c0f64f83dd7c1e9840847c3ad308b18d859724121b8a711`

Canonical result:

`docs/research/ALL_PLAYER_ALL_POSITION_REPLAY_V1_RESULT.md`

Coverage:

- 4,488 point rows
- 1,837 unique player-weeks
- QB 128
- RB 471
- WR 748
- TE 490

Current-stack point MAE diagnostic:

- QB pass yards: 73.99
- RB rec yards: 10.46
- RB receptions: 1.182
- RB rush yards: 19.11
- RB rush+rec yards: 24.37
- WR rec yards: 22.30
- WR receptions: 1.514
- TE rec yards: 15.62
- TE receptions: 1.427

The strongest player-level finding was workload compression:

- low-workload players are systematically overprojected;
- high-workload focal players are systematically underprojected.

Among active weekly-stat rows, final point error strongly orders with realized
opportunity.

Examples frozen in the replay result:

- WR rec yards, 1-2 actual targets: bias +10.88 yards
- WR rec yards, 9+ targets: bias -45.16
- TE rec yards, 1-2 targets: +8.06
- TE rec yards, 9+ targets: -40.34
- RB rush yards, 1-3 carries: +13.19
- RB rush yards, 15+ carries: -35.81
- QB pass yards, <=20 attempts: +110.00
- QB pass yards, 41+ attempts: -83.30

The full-roster replay also exposed a participation/zero-opportunity problem:
many pregame roster rows project material usage despite no weekly stat row.

Do not “fix” this by post-hoc dropping zero-stat players after outcomes.
Participation/role must be modeled pregame.

---

## 6. WR/TE target-depth distribution — COMPLETE / universal transform NOT supported

Player Target Depth Distribution Week-5 shadow was frozen:

- branch:
  `research-player-target-depth-distribution-shadow-v1`
- run:
  `37677366697` SUCCESS
- artifact:
  `11507970080`
- artifact digest:
  `sha256:ecd7c8683d65bf9701353f0dbccb3ace59c992157bdf75b2b87191b08a8f314f`
- row digest:
  `sha256:1a5b6a914517310cdfb0fe5c92fcba7a77a28b6a988664c0f7dcc49cd2b8eb2a`
- 277 WR/TE players
- feature available 210/277
- no Week-5 outcomes
- no sportsbook
- no fitted parameters
- receiving means invariant

Frozen scale formula:

`depth_scale = sqrt(prior8_target_depth_sd / position_anchor)`

Anchors:

- WR: `10.02786868561449`
- TE: `6.735519692444827`

The W1-W4 all-player replay then showed the universal symmetric transform did
**not** improve pooled WR/TE receiving-yard distribution accuracy:

- baseline CRPS: **13.43749**
- shadow CRPS: **13.50947**
- shadow is slightly worse.

Disposition:

`TARGET_DEPTH_FULL_SYMMETRIC_DISTRIBUTION_TRANSFORM_NOT_CONFIRMED_W1_W4`

Descriptively, highest-dispersion players benefited while low-dispersion players
were harmed, but **do not** post-hoc choose a threshold from the same outcomes.

The underlying player-specific target-depth dispersion signal remains real.
The universal transform is not promoted.

---

## 7. WR/TE target-share trajectory — CONFIRMED / prospective Week-5 shadow frozen

Historical signal authority:

- run `37638269235` SUCCESS
- disposition:
  `PLAYER_TARGET_SHARE_TRAJECTORY_SIGNAL_CONFIRMED`

Frozen feature:

- latest 2 completed same-team target-share games vs earlier completed
  same-season same-team games
- `trajectory_delta = recent2_share - earlier_share`

Expected sign:
rising role means current promoted entitlement is more likely to underpredict.

Result:

- WR pooled rho: -0.07646
- TE pooled rho: -0.09876
- combined rho: -0.08496
- all confirmation gates passed.

Week-5 prospective shadow:

- run `37654382316` SUCCESS
- artifact `11497153776`
- row digest:
  `sha256:afbfd7f360c50fcd4850c0967be40f9a333da1bd5f835cdc676c2e88d777c1f3`
- 277 WR/TE rows
- 247 changed
- M38 WR1 anchor frozen
- WR2+ pool conserved
- TE room conserved
- team target mass conserved
- no Week-5 outcomes
- no sportsbook
- no fitted parameters
- production unchanged.

Critical rule:

**2026 W1-W4 cannot exercise exact Trajectory V1.**

It requires four prior same-season team games.
Do not weaken this eligibility rule after seeing outcomes.

Historical Week-5+ seasons may be used for an exact frozen integration test.

At this checkpoint (morning of 2026-10-08), Week 5 is not a settled prospective
sample. Do not grade or modify the frozen Week-5 trajectory lock before outcomes
are legally available.

---

## 8. RB player-level work — current state

### RB carry/snap player-state allocation Week-5 shadow

Authority:

- run `37560824479` SUCCESS
- 98 RB/HB/FB identities
- 30 teams
- prospective Week-5
- no Week-5 outcomes
- no sportsbook
- no production change

Frozen rule:
50/50 recent carry-share + snap-fraction allocation.

Preserve the M96 retrospective stop.
Do **not** apply this Week-5 room allocation retrospectively to 2026 W1-W4.

### RB receiving-room share — retrospective impact CONFIRMED

Branch:

`research-rb-receiving-room-share-impact-v1`

Run:

`37703522415` SUCCESS

Artifact:

`11518826839`

Frozen no-fit mechanism:

- state: strict-prior `prior_rb_room_share`
- preserve total RB receiving-room target entitlement
- redistribute only inside the RB/FB room
- no coefficient fit
- no threshold search
- no sportsbook
- rushing held exactly unchanged.

W1-W4 individual-player impact across 436 RB/FB player-games:

- target MAE: **7.60% better**
- receptions MAE: **3.83% better**
- receiving-yards MAE: **2.89% better**
- rush+receiving-yards MAE: **1.66% better**
- rush-yards projections: **exactly unchanged**

Target and receiving-yard MAE improved in **all four completed weeks**.

This is direct proof that better individual opportunity allocation can improve
final player projections.

### RB receiving-room Week-5 prospective lock

Branch:

`research-rb-receiving-room-share-week5-lock-v1`

Successful run:

`37705464974`

Artifact:

`11518604971`

Final lock:

- exact frozen RB identities: **98**
- teams/rooms: **30**
- strict-prior receiving-history coverage: **98/98**
- identity bridge: exact GSIS
- parameters fit: 0
- Week-5 outcomes read: 0
- sportsbook inputs: 0
- production changed: false

Do not alter the rule after the W1-W4 retrospective result.

---

## 9. Gap crosswalk / anti-retest findings completed in this chat

Canonical crosswalk on active decomposition branch:

`docs/research/PLAYER_LANDSCAPE_GAP_DISPOSITION_V1.md`

### Route participation / YPRR

Status:

`SOURCE_PARITY_BLOCKED_FOR_HISTORICAL_PREDICTIVE_TEST`

Canonical PlayerForm only fills route rate/YPRR when a real routes/routes_run
field exists.

Do **not** substitute nflverse participation `route` labels for routes run.

RB prior source disposition already said:

`LIVE_RB_ROUTE_VOLUME_CONFIRMED_HISTORICAL_WEEKLY_PARITY_NOT_CLEARED`

This source limitation applies to historical WR/TE route-participation testing
as well unless a real semantically equivalent historical source is acquired.

### WR-R3 combined player calibration

The older handoff saying it was “never built” was stale.

It was built and executed later.

Authority:

- branch `research-wr-r3-combined-calibration`
- run `34726509088`
- artifact `10309091155`
- final disposition:
  `NO_ACTIONABLE_WR_R3_COMBINED_CALIBRATION`

Do not rebuild, rescue, or retune it.

### Coverage-v2 team man/zone

Already ablation-tested and effectively near-null.

Do not reopen as the same feature family.

### Player-level WR-CB assignment

Current/live evidence can exist, but a free reproducible historical assignment
ground truth is not available.

Status:

`SOURCE_PARITY_BLOCKED`

Do not infer historical assignments from nflverse participation.

### Generic game-script / Vegas confirmation

Existing research established:

- game-market variables add essentially no incremental improvement to the core
  team-volume baseline;
- Vegas line is a noisy descriptor of realized game script;
- pregame confirmation-likelihood work ended:
  `NO_ACTIONABLE_PREGAME_CONFIRMATION_STATE`
- no combined promotion.

Do not route player-prop/sportsbook data upstream into football projection.

### Simple Football Matchup Transmission V1 formulas

All three frozen integration candidates failed closed:

- RB opponent pass-rate-faced
- WR true PROE
- TE opponent pass-success allowed

Architecture gaps remain real, but these exact formulas cannot be rescued.

### M95A/M95B

Generic RB role x run-defense ideas are closed.

Do not create a renamed “bad run defense => boost RB” rule.

---

## 10. What the current model already does at the individual level

This should be preserved when explaining the project to the user.

The model already has real player-specific transmission:

### Identity / availability
- named player
- team
- opponent
- pregame roster membership
- role where consumed

### Individual usage
- target share
- rush share
- QB pass-attempt share/opportunity
- lagged same-player history through ML
- lagged same-player state through State

### Room hierarchy
- M38 WR hierarchy
- TE-R5P
- WR-R15
- alpha-receiver vacancy redistribution
- RB room shadows under prospective evaluation

### Individual efficiency
- YPT
- catch rate
- YPC
- QB YPA/efficiency state

### Team environment
- expected plays / pace
- generic pass/rush split
- richer QB attempt environment under M89/M90

### Opponent context
- pressure
- man/zone/middle coverage state
- RB box state
- role x coverage target multipliers

So the correct description is:

> **Individual-player core + material positional shrinkage + promoted player
> specialists + partial team/opponent transmission.**

It is not a position model with names attached.

But the complete player landscape still does not fully transmit, especially in
generic RB/WR/TE.

The decomposition says the next priority is opportunity allocation, not trying
to solve every missing matchup field at once.

---

## 11. Position-by-position status NOW

### QB

Status:
**player core ready; residual opportunity/attempt volume is the next legal
player-level question.**

Protect:

- M89/M90
- C2
- broad QB mean research freeze
- prior QB receiver pair YPT closure

Do not reopen a generic YPA/mean feature hunt.

The W1-W4 component decomposition says:

- opportunity oracle removes 23.9% MAE
- efficiency oracle removes 12.7%
- opportunity error correlation with output error 0.814.

Any next QB player-level lane should localize:

- starter probability/identity
- pass-attempt workload
- team pass-volume assignment

before another efficiency mechanism.

### RB

Status:
**opportunity remains dominant; one receiving-share mechanism is retrospectively
confirmed and two Week-5 room mechanisms are prospectively frozen.**

Protect:

- M96 retrospective stop
- no Week-5 carry/snap shadow back-application to W1-W4
- closed generic RB width/mean and M95A/M95B families.

Current result:

- rush yards workload oracle removes 45.1% MAE
- rec yards 45.1%
- receptions 57.2%

RB receiving-room W1-W4 result is already a real final-projection improvement.

### WR

Status:
**target/opportunity allocation is the main residual problem.**

Protect:

- M38
- WR-R15
- target-share trajectory exact eligibility rule
- WR-R3 closure
- WR-CB source block
- target-depth universal-transform failure.

Current result:

- rec yards workload oracle removes 42.8% MAE
- receptions removes 54.4%
- efficiency is secondary.

Do not open another generic receiving-efficiency correction first.

### TE

Status:
**target/opportunity allocation is the main residual problem.**

Protect:

- TE-R5P
- trajectory eligibility
- target-depth universal-transform failure
- closed TE width work.

Current result:

- rec yards workload oracle removes 55.0% MAE
- receptions removes 68.3%.

This is the strongest opportunity-dominant position.

---

## 12. EXACT NEXT ACTION

Do not restart any completed audit.

Do not go back to generic matchup hunting.

Do not wait idly for Week-5 outcomes.

The next chat should:

1. Verify live `main`, the active decomposition branch, relevant Actions, and
   Issue #535.
2. Read the newest handoff only; do not recursively read old handoffs.
3. Treat `PLAYER_OUTPUT_COMPONENT_DECOMPOSITION_V1_RESULT.md` as the current
   scientific decision gate.
4. Build a **bounded individual-opportunity next-step roadmap** from the already
   completed evidence before opening another candidate:
   - identify, by QB/RB/WR/TE, which remaining pregame role/share/allocation seam
     is legal, source-valid, not already closed, and testable;
   - explicitly separate:
     a) participation / active-role probability,
     b) team opportunity volume,
     c) room share / player allocation.
5. Reconcile this roadmap against already-completed opportunity studies so no
   lane is duplicated:
   - RB receiving-room impact already confirmed;
   - RB carry/snap Week-5 shadow already frozen;
   - WR/TE target-share trajectory already confirmed and Week-5 shadow frozen;
   - exact trajectory cannot operate on 2026 W1-W4;
   - route participation is historical-source-blocked.
6. For WR/TE, the most obvious legal validation path is an **exact frozen
   historical Week-5+ integration test of Target Share Trajectory V1** using
   historical seasons where the four-prior-same-season-game rule is eligible.
   Preserve M38 / WR-R15 / TE-R5P and team target mass.
7. For QB, any next candidate must address attempt/starter/team-volume state,
   not generic passing efficiency.
8. For RB, do not violate the M96 retrospective stop. Keep current Week-5 room
   locks sealed prospectively and use only legal historical/current-season
   evidence.
9. If a new cross-position participation/zero-state mechanism is considered,
   first reconcile the completed historical-availability parity and opportunity
   audits. Do not invent an outcome-selected cutoff.
10. Freeze the exact mechanism before scoring it.
11. Keep efficiency/matchup research secondary unless opportunity allocation for
    that market is already resolved.
12. No production promotion from diagnostics alone.
13. No paid OddsAPI without explicit user authorization.

Current date context at handoff:
2026-10-08 morning. Week-5 prospective locks are still intended to remain
outcome-blind. Do not grade them prematurely.

---

## 13. Mandatory anti-retest / governance rules

- GitHub is canonical.
- Read `AGENTS.md` first.
- Read only the newest top checkpoint in `CURRENT_NFL_RESEARCH_HANDOFF.md`.
- Do not recursively load historical handoffs.
- Do not ask the user to re-explain the project.
- Do not restart completed research.
- Do not weaken a frozen eligibility rule after seeing outcomes.
- Do not fit a threshold from the same sample used to discover the pattern.
- Do not claim an Action is active unless live GitHub Actions says queued or
  in-progress.
- Do not silently promote a diagnostic.
- Do not use sportsbook/player-line data upstream in independent football
  projections.
- Do not spend OddsAPI credits without explicit user approval.
- Preserve raw private GSIS boundaries.
- Keep important milestones checkpointed to GitHub because the UI repeatedly
  times out for this user.

---

## 14. User interaction / working style

The user wants the assistant to take the wheel and execute.

They do **not** want:
- repeated “what do you want to do?” questions when the next action is clear;
- vague updates;
- work restarted because a chat rolled over;
- old failed lanes reopened under new names;
- long waits for prospective results when legal retrospective/current-season
  diagnostics can be run now.

They do want:
- direct status updates with exact runs/heads/artifacts;
- reasoning tied to final individual-player projections;
- all positions handled, not just one;
- explicit protection against leaving unfinished work behind;
- robust checkpointing in GitHub before a chat ends.

If a run fails:
- inspect the exact failure;
- fix the data/source/identity plumbing without weakening the frozen science;
- rerun the exact contract;
- state clearly whether the failure was mechanical or scientific.

---

## 15. Minimal read list for the next chat

Read in this order:

1. `AGENTS.md`
2. newest top checkpoint in `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. this file
4. `docs/research/PLAYER_OUTPUT_COMPONENT_DECOMPOSITION_V1_RESULT.md`
   from branch `research-player-output-component-decomposition-v1`
5. `docs/research/PLAYER_LANDSCAPE_GAP_DISPOSITION_V1.md`
   from the same branch
6. `docs/research/PLAYER_LANDSCAPE_TRANSMISSION_AUDIT_V1_RESULT.md`
   from branch `research-player-landscape-transmission-audit-v1`
7. `docs/research/ALL_PLAYER_ALL_POSITION_REPLAY_V1_RESULT.md`
   from branch `research-all-player-all-position-replay-v1`
8. RB receiving-room W1-W4 result and Week-5 lock docs
9. Target Share Trajectory result + Week-5 lock
10. Target Depth Distribution W1-W4 result / Week-5 lock
11. Issue #535 newest continuity comment
12. live main / relevant branches / Actions

Then continue immediately from the exact next action above.

Do not recursively read older handoffs unless a targeted anti-retest question
requires one.
