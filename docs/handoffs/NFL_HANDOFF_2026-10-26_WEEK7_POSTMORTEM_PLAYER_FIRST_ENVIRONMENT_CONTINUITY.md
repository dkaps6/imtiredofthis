# NFL HANDOFF — 2026-10-26 — WEEK-7 SUNDAY/SNF FORENSIC + PLAYER-FIRST / ENVIRONMENT-TRANSMISSION CONTINUITY

Repository: `dkaps6/imtiredofthis`

This is the authoritative deep handoff for the next ChatGPT session after the 2026-10-25 Week-7 Sunday + Sunday Night Football betting/postmortem conversation.

The user explicitly wants the next session to feel like the exact same assistant continuing. **Do not ask the user to reteach PRISM, the player-first philosophy, the Sunday tickets, the Mahomes/Walker discussion, the Claude/GPT role split, or why opponent/environment transmission is still open.**

GitHub is canonical. Live GitHub evidence newer than this handoff supersedes it.

---

## 1. Mandatory read/reconcile order

Read in this exact order:

1. `AGENTS.md`
2. ONLY the newest TOP checkpoint in `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. this deep handoff
4. newest Issue #535 comments, especially the comment IDs listed below
5. live `main`, open PRs, exact-head Actions/artifacts
6. targeted research/evidence files explicitly named here only as needed

Do **not** recursively read old handoffs.

Important control-room comments for this state:

- GPT Week-7 user escalation / environment-transmission assignment: **#535 comment 6417917738**
- GPT Week-7 ticket evidence preservation: **6418739459**
- Claude Week-7 forensic / construction diagnosis: **6419138959**
- Claude player-level gate proposal: **6419872553**
- Claude correction that identity history is recoverable: **6420003475**
- GPT response / accepted construction diagnosis + transmission remains open: **6420198257**

At the final pre-handoff reconciliation, **6420198257 was the newest visible #535 comment**. Always re-fetch before acting.

---

## 2. Permanent assignment — do not regress

PRISM is an **individual-player projection model**.

The scientific unit is the named player in his actual current football state for the exact market.

Canonical user language:

> **THE PLAYER SHOULD HAVE THE RULES. THE POSITION SHOULD NOT HAVE THE PLAYER.**

> **ANOMALIES ARE QUESTIONS, NOT TRENDS.**

Position MAE/RMSE is downstream QA only. Never make aggregate positional MAE the optimization objective.

For each named player, the intended reasoning chain is:

1. player identity / what he has shown he is;
2. current health / active / starter / role state;
3. current opportunity and opportunity trajectory;
4. teammate competition / finite team opportunity;
5. offense / QB / coaching / team environment;
6. exact opponent / matchup / coverage / front / pressure / game environment;
7. market-specific translation;
8. named-player reliability / uncertainty;
9. sportsbook comparison **only after** the football projection is frozen.

Do **not** scrap/rebuild PRISM. Current identity/state/environment work is additive high-resolution arbitration over substantial existing qualified science.

Always distinguish:
- `SOURCE READY`
- `SOURCE BLOCKED`
- `PREDICTIVE CLOSED`
- `QUALIFIED`

A source block is not a predictive failure.
A failed additive candidate does not erase older qualified science.

---

## 3. Control-room roles

Per user direction:

- **Claude = active executor / research lead**
- **GPT-5.6 Sol = independent auditor / project control**
- shared asynchronous lane = GitHub Issue #535
- GPT uses `[GPT AUDIT]` / `[GPT OBJECTION]`
- Claude uses `[CLAUDE]` / `[CLAUDE STATUS]` / `[CLAUDE QUESTION FOR GPT]`

GPT should audit evidence, scope, anti-retest compliance, and production boundaries. Do not silently take over Claude's research branches.

If Claude posts new evidence or asks a question, answer in #535 and preserve exact comment/run/head/artifact references.

---

## 4. Live repository / production state at this handoff

Pre-handoff live `main`:

`97ecae88f391a692cf694971c2bf2f34c04cf593`

This is PR #695's evidence-only merge:
- **PR #695 merged**
- preserves Week-7 Sunday/SNF ticket construction evidence
- no model/science/gate change
- no paid OddsAPI acquisition

The continuity merge created from this handoff will advance `main` by docs only. Next session must re-fetch the actual final main SHA.

Canonical paid Week-7 Full Slate remains:

- run `39849371821` — SUCCESS
- job `120071136220`
- source head `4dc178b59da8708dbea978b8e54e753b1e2f75c5`
- artifact `12456298985`
- digest `sha256:d1a1fb48e5bb920c5291f35b7457b5870801b689751deb2d742f98c555603a85`
- pricing status CURRENT
- sportsbook downstream-only
- 14/14 games eligible, 0 withheld
- 328 priced players
- 732 compact player-market rows
- 1,447 exact bookmaker offer rows
- 2,894 priced side rows
- unresolved core player identity rows = 0

Durable Week-7 paid boundary:

- `data/market_track_record/boards/2026_wk07.csv`
- `data/market_track_record/game_odds/2026_wk07.csv`
- `data/market_track_record/raw_props/2026_wk07.csv`

All are tied to paid source run `39849371821`.

**NO additional paid OddsAPI call is authorized.**
Reuse the durable Week-7 boundary.

### CI / operations at pre-handoff main

PR #695 merge triggered:

- Archive Market Track Record run `40341879091` — **SUCCESS**
- Repo CI run `40341289972` — **FAILURE**, but still exactly the known stale workflow-contract assertion:
  - 660 passed
  - 1 skipped
  - 1 failed
  - failing test: `test_full_slate_main_push_defaults_to_no_credit_mode`
  - workflow is already stricter than the stale exact-string assertion

**Never weaken the live-odds authorization guard to make this stale test pass.**

Other open PRs at final reconciliation include:
- #691 — stale/superseded continuity PR; do not blindly merge
- #690 — tiny order-insensitive RB Stage-1 source-cert assertion repair, previously audit-cleared; scientific disposition remains SOURCE_BLOCKED
- #672 — large/draft research lineage; do not wholesale merge
- older research/ops PRs remain; re-fetch before asserting exact status

---

## 5. Week-7 Sunday/SNF ticket evidence is now durable

The user explicitly asked that **everything we constructed in chat** be preserved, including Sunday, RB/non-RB versions, ATD, SNF, and pricing screenshots/transcriptions.

Canonical evidence root:

`docs/research/evidence/2026-10-25_week7_ticket_ledger/`

Merged by PR #695.

Preserved there:

- Sunday non-RB initial draft
- Sunday non-RB final Safe/Moderate/Medium/High/Lottery
- Sunday RB-integrated final Safe/Moderate/Medium/High/Lottery
- Sunday ATD fun/uncertified cards
- KC-SEA SNF ATD detour cards
- KC-SEA SNF regular alt-line Safe/Moderate/Medium/High/Lottery
- `ticket_legs.csv` — one row per created leg
- 68 readable FanDuel prices transcribed from 21 user screenshots
- screenshot filename -> player/market mapping
- SHA-256 per original screenshot
- explicit provenance note

The installed GitHub connector could not ingest the raw JPEG bytes directly from ChatGPT conversation attachments. The repo truthfully does **not** claim those JPEG binaries are on main. The exact image hashes and price transcription are durable.

Do not reconstruct the tickets from memory. Use this evidence package.

Important historical-integrity note:
- the SNF ATD `SEA_SCRIPT_LABEL_AS_CREATED` card was preserved exactly as created, including the contemporaneous Kenneth Walker team-label mistake. Do not silently rewrite historical ticket evidence.

---

## 6. User-reported Week-7 betting result that triggered the forensic

The user reported:

- across all Sunday tickets — non-RB, RB-integrated, ATD, and Sunday Night Football — **only one ticket hit**
- **none of the SNF regular alt-line tickets hit**
- user specifically highlighted Patrick Mahomes and Kenneth Walker III as repeated SNF anchors that killed boards

This felt catastrophically bad and led to the model/environment discussion.

Do not dismiss the user's concern with “parlays are volatile.”
But do not use the final ticket count as direct proof of predictive-science failure either.

---

## 7. Claude's Week-7 forensic — accepted construction diagnosis

Claude independently audited the durable #695 ledger in #535 comment **6419138959**.

Core accepted findings:

### 7.1 Ticket-level expectation

Claude calculated the 15 FINAL all-yardage parlays had approximately:

- **2.36 model-implied expected winning tickets**
- observed user-reported winners = 1

Claude's conclusion: one winner is roughly ~1 SD below that naive expectation and **not itself a statistical anomaly**.

Approximate naive model-implied full-ticket probabilities:

- Safe: ~41–47%
- Moderate: ~18–20%
- Medium: ~9–16%
- High: ~2–4%
- Lottery: ~0.3–0.9%

Important:
- same-game correlation makes some tickets, especially SNF, worse/more complicated than simple independent-product estimates
- a ~41% four-leg ticket should **not** have been called “Safe”

### 7.2 We bet against the model's main-line direction too often

Claude found **25 of 57 unique yardage legs (~44%)** were OVER alt bets while PRISM's positive main-line edge was on the UNDER.

Examples called out:

- Christian McCaffrey receiving: model main-line direction UNDER, yet we used bought-down OVER 25+
- George Kittle receiving: model main-line direction UNDER, yet we used OVER 40+
- Puka Nacua receiving: model main-line direction UNDER, yet initial draft used OVER 50+
- Kenneth Walker III rushing: model had ~66.05 vs 84.5 and strong UNDER direction, yet we used OVER 50+/60+/70+

This is a major **betting-layer construction defect**.

### 7.3 Buy-down cushion was treated as probability

We repeatedly reasoned:

> projection is X, alt is much lower, therefore it is “safe.”

But `model_sd` already existed on the board and showed many thresholds were nowhere near “safe.”

Claude's examples:

Patrick Mahomes pass yards:
- proj ~224.84
- sd ~54.75
- 175+ ~81.9% model-clear probability
- 200+ ~67.5%
- 225+ ~49.9%

Kenneth Walker III rush yards:
- proj ~66.05
- sd ~28.10
- 50+ ~71.6%
- 60+ ~58.5%
- 70+ ~44.4%

Important caveat:
Week-6 `cushion_sd` was empirically flat as a ranking metric in at least one pass-yard comparison. So sigma-normalization is **necessary context but not sufficient as a selector**.

Do not turn this into “just divide by sigma and we're fixed.”

### 7.4 Portfolio concentration was bad

Claude found high repeated-player exposure across 30 tickets, including:

- JSN on ~60%
- AJ Barner ~37%
- Brock Purdy ~30%
- Rodgers / Tre Tucker / Nico Collins / Kenneth Walker III ~27% each

A single named-player miss could kill a huge fraction of the portfolio.

This is a downstream construction/exposure failure, not something to repair by changing PRISM means.

### 7.5 Market allocation was questionable

Claude compared ticket leg mix with the recent Week-6 graded ledger and noted:

- ~41% of 163 legs were rec_yards
- ~29% pass_yards
- ~23% ATD
- ~7% rush_yards

Week-6 had:
- rec_yards weak/negative
- pass_yards weak/negative
- rush_yards strongest
- ATD uncertified

Treat this as a **decision-layer warning**, not a permanent market ban. One week's market results do not establish immutable truth.

---

## 8. Critical GPT ruling — construction failure does NOT close transmission

GPT answered Claude in #535 comment **6420198257**.

The exact boundary to preserve:

> **The ticket evidence does not prove transmission failure.**
>
> **It also does not prove transmission is adequate.**

Disposition:

- Week-7 parlay failure = primarily a **construction / threshold / exposure problem**
- opponent/environment transmission = still an **independent open architecture/research problem**
- neither cancels the other

This distinction is mandatory.

Do not let a future session say:
- “Sunday proved the model's defense handling is broken” — too strong
- “Claude proved defense transmission is fine” — also too strong

The betting layer definitely failed.
The model transmission lane remains open because prior code-path audits independently established real information-loss gaps.

---

## 9. Opponent/environment transmission — independently established architecture gap

Before SNF, the user asked a crucial question:

> We have tons of defensive data. Is each named-player projection actually using all the right pieces of that matchup/environment information?

The answer from the code audit was:

**PRISM collects a lot of defensive/environment data, but generic RB/WR/TE paths do not consume all of it at the resolution the user expects.**

Previously established Phase-A / live-code facts include:

- `def_rush_epa` reaches TeamContext but is not consumed by generic RB `matchup_multipliers()` / simulation
- `explosive_play_rate_allowed` exists but is not consumed by generic skill-position matchup rules
- WR / TE / RB YPT allowed exist upstream but are dropped before generic player projection
- outside / slot YPT allowed exist upstream but are dropped
- defensive YBC / stuff fields exist upstream but are dropped
- generic `project_game_script()` hardcodes `pass_share = 0.57`
- generic lead/trail probabilities do not actually move the pass/rush split in simulation
- generic RB rushing matchup changes efficiency mainly through coarse light/heavy box thresholds
- QB M89/M90 is materially richer and must not be conflated with generic skill-position logic

Separate Lane-B audit in #535 comment `6353823672` established:

> already-qualified Stage-2 conditional allocation research does **not** currently reach the canonical production board.

That is a production integration gap, not a predictive failure.

### Correct architecture question

Not:

> “Is Seattle a good defense?”

But:

> “Which legal pregame opponent/environment features should affect THIS named player's opportunity, efficiency, distribution and uncertainty — and which of those actually reach production?”

Examples:
- JSN and Kupp should not necessarily receive the same KC WR adjustment if alignment/route family differs
- Barner should care more about TE/middle-field defensive behavior
- Walker should care about box/YBC/stuff/rush EPA/explosive-run suppression
- Mahomes should care about pressure/pass EPA/success/YPA/explosive suppression through the richer QB path

Do not solve with a dumb defense-vs-position scalar.

---

## 10. Walker is the preferred first transmission case study

Claude objected that Mahomes is a poor first test of missing transmission because M89/M90 already consumes richer opponent state.

GPT agreed in **6420198257**.

**Start generic skill-position decomposition with Kenneth Walker III.**

Read-only decomposition should answer:

1. player/current-role baseline absent exact opponent adjustment;
2. every legal pregame opponent run-defense input available;
3. which inputs actually reached production;
4. which existed but were dropped;
5. exact effect on opportunity/volume;
6. exact effect on efficiency;
7. exact effect on final mean;
8. exact effect on sigma/uncertainty, if any;
9. whether rushing-vs-receiving defensive asymmetry is represented;
10. whether existing qualified Stage-2 allocation would alter opportunity if/when properly integrated.

Then use representative receiving cases:
- JSN vs Kupp — same offense, different receiving identities/alignment
- Barner — TE-specific environment
- Nico Collins / Juwan Johnson as useful Week-7 misses only if they answer a pregame mechanism question

Do not build candidates from the Week-7 outcome.
Week 7 is a forensic/case-finding set.

Any predictive candidate must be frozen before scoring and tested leakage-safe / prospectively.

---

## 11. New betting-layer rule direction — preserve as open framework, not overfit rule

The old informal behavior:

> main-line model wants UNDER, but a bought-down OVER is below the projection, so we can still use it

is no longer acceptable as an unexamined default.

GPT ruling in **6420198257**:

When proposed alt direction conflicts with canonical main-line model edge direction, default should be **PASS** unless a preregistered, evidence-backed alt-threshold framework explicitly clears the conflict.

Do **not** invent a conflict threshold from Week 7.

For every future candidate alt leg, the decision record should ideally contain:

- canonical football mean
- distribution authority / model SD
- sportsbook main line
- canonical main-line edge direction
- exact alt threshold
- estimated P(clear threshold)
- exact alt price / implied probability
- threshold EV if a valid price/probability authority exists
- player-state confidence
- opponent/environment conflict flag
- current ticket/player exposure count

Sportsbook remains downstream-only.

### “Safe” naming correction

Do not label a full parlay “Safe” merely because legs are bought down.

Full-ticket tiering should become probability-aware and dependency-aware.

At minimum:
- show estimated ticket hit probability when available;
- show same-game concentration;
- show largest named-player exposure;
- do not let one player become a universal anchor across all tiers.

No arbitrary numeric exposure cap has been scientifically frozen yet. Do not invent one post hoc.

---

## 12. ATD remains a separate uncertified lane

Dedicated anytime-TD science is **not certified**.

ATD tickets from Sunday were explicitly a fun side lane.

Do not:
- use ATD outcomes to validate or invalidate yardage science;
- silently incorporate ATD into PRISM's certified portfolio selector;
- treat ATD losses as evidence of a yardage projection failure.

If user wants ATD again, clearly label it uncertified/fun unless the science changes.

---

## 13. Player-identity proposal from Claude — key correction and current direction

Claude comment **6419872553** argued that player-conditional truth may be blocked by evaluation grain because many promotion gates are pooled.

Core useful proposal:

- preregister player partitions/archetypes before scoring
- partitions come from football semantics, not observed errors
- require replication across independent seasons / windows
- pooled population must not materially harm
- declared partition must improve consistently
- minimum support required
- report per-player heterogeneity even when candidate closes
- no top-N rescue
- no post-hoc named-player carveout

GPT directionally approved this in **6420198257**.

### Claude correction — identity state is recoverable

Claude initially overstated that prior-season player identity state was lost.

He corrected himself in **6420003475**.

Correct state:

The strictly-prior Player Identity Card is buildable now from published stable football sources, including repository-used nflverse loaders:

- `load_pbp`
- `load_player_stats`
- `load_snap_counts`
- `load_schedules`
- historical participation where provider coverage exists

Therefore the identity layer is **not blocked**.

For a target week `w`, build identity strictly from weeks `< w`.
That is leakage-safe and does not require expired historical run artifacts.

What is actually lost for some old weeks:
- our exact past pregame projection boards for Weeks 3–5
- exact sportsbook lines for those weeks
- exact Ourlads/T-75 as-run snapshots from those weeks

That limits grading of our historical as-run predictions.
It does **not** prevent building historical football identity.

### Durability still matters going forward

Current durable board contains final outputs and only limited explanatory football state.

Going forward, preserve enough as-run player/team/opponent state to explain:
- what role the model believed
- what opportunity state it believed
- what opponent environment it consumed
- what availability state applied
- what projection components/lineage produced the final mean/distribution

This is a durability/auditability problem, not a reason to delay descriptive identity work.

---

## 14. Existing science boundaries that still stand

Do not regress these:

- `PRISM_RETURN_STATE_NEXT_ROLE_PERSISTENCE_V1_QUALIFIED_STATE_ONLY` = QUALIFIED
- `PRISM_PLAYER_TEAM_CONTEXT_TRANSITION_V1_SOURCE_READY` = SOURCE READY
- RB carry-count identity bridge = SOURCE BLOCKED, not predictive failure
- RB receiving team-context reset = PREDICTIVE CLOSED
- TE receiving team-context reset = PREDICTIVE CLOSED
- RB rushing Stage-1 participation-authority reset = SOURCE BLOCKED
- qualified RB Stage-2 conditional allocation exists in research
- direct high-value-share V1 closure is narrow; do not generalize to “scoring-area usage useless”
- generic QB narrowing remains closed/retracted
- WR-R15 active
- TE-R5P active
- QB M89/M90 + C2 active
- RB P3 / R22 Week-1-only authorities are not active in Week 7
- direct WR/CB matchup unavailable/gated off
- dedicated ATD model not certified

Do not reopen closed science because Week 7 was ugly.

---

## 15. Roadmap / exact next research actions

The earlier user-approved roadmap was B -> C -> D -> E:

- B: Stage-2 -> production wiring audit
- C: team scoring-territory drive creation
- D: applicability as strictly-prior player property
- E: opponent/environment transmission

Lane B read-only audit is now factually complete enough to know:
**qualified Stage-2 allocation does not reach production.**

The exact earlier next action was a **fresh Stage-2 source-readiness re-probe**, because the last Week-7 readiness probe failed closed when legal Week-6 prior-state data were incomplete; Week 6 is now complete.

Do not lose that work.

Current additive priorities after live reconciliation:

1. **Continue Claude's descriptive Player Identity Card construction** from strictly-prior football history. This fits nothing and promotes nothing.
2. **Fresh Stage-2 source-readiness re-probe** under the already-frozen readiness contract if not already completed by a newer #535 comment.
3. **Walker-first opponent/environment projection decomposition**, read-only.
4. Extend decomposition to representative WR/TE identities.
5. Freeze any new transmission candidate before scoring; no outcome-driven candidate creation.
6. Develop the player-conditional promotion-gate design under anti-retest constraints.
7. Preserve richer football-state/explanation evidence going forward.
8. Separately design/evaluate a proper alt-threshold / portfolio-construction framework; do not change PRISM means to repair a betting-layer defect.

If Claude posts newer work after this handoff, audit that first rather than duplicating it.

---

## 16. Week-7 SNF numbers / context worth preserving

Frozen canonical Week-7 examples from the paid board:

- Patrick Mahomes pass yards ~224.84
- Kenneth Walker III rush yards ~66.05
- JSN rec yards ~84.63
- Travis Kelce rec yards ~41.68
- AJ Barner rec yards ~30.62
- Cooper Kupp rec yards ~30.47
- Sam Darnold pass yards ~217.70

Pregame manual football audit identified Seattle as a very difficult KC environment and KC as a strong Seattle environment.

That manual review is why the user asked whether all this opponent information was actually entering projections.

Do not use those manual observations post hoc as fitted coefficients.
Use them as examples of the information-transmission question.

---

## 17. Bet-slip construction continuity before Week 7 vs now

Old document:
`docs/research/MODEL_BACKED_ALT_LINE_PARLAY_CONSTRUCTION_V1.md`

The Week-6 SF-WAS anecdote had encouraged:
football projection -> current role/environment -> actual line -> bought-down alt -> coherent script.

Week 7 showed the implementation of that idea had serious defects:
- too much “cushion means safe”
- too much alt-OVER use against canonical main-line UNDER direction
- too much repeated-player concentration
- insufficient use of model distribution / clearance probability
- tier labels not tied to realistic full-ticket survival probability
- too much uncertified ATD allocation

Do not discard the whole concept.
Treat Week 7 as evidence that **V1 construction needs an evidence-backed V2**, separate from PRISM predictive science.

Until a V2 is frozen:
- model direction conflict => default PASS
- exact price matters
- model SD / clearance probability must be visible
- ticket probability must be presented honestly
- repeated anchor concentration must be surfaced
- ATD remains separate

---

## 18. User behavior / response preferences for next chat

The user is highly sensitive to the assistant forgetting the assignment or retreating to generic aggregate metrics.

When discussing model problems:
- talk named players;
- talk actual current role/environment;
- explain what data reached the projection and what did not;
- do not use positional MAE as the goal;
- do not blindly agree with user claims without checking evidence;
- distinguish model failure from betting construction failure;
- distinguish source block from predictive failure;
- preserve prior qualified work.

When discussing tickets:
- give exact player/market/threshold;
- explain why the number is backed by both PRISM and football;
- show model direction conflict if present;
- use exact user sportsbook screenshots/prices when provided;
- never invent SGP combined payout;
- avoid universal anchors;
- clearly separate fun/uncertified ATD from certified markets.

If user says “Claude sent something,” re-fetch #535.

If user says “keep going,” continue the exact active canonical research lane after live reconciliation; do not restart.

---

## 19. Final continuity state / anti-reteach summary

At the final pre-handoff reconciliation:

- pre-handoff main = `97ecae88f391a692cf694971c2bf2f34c04cf593`
- PR #695 ticket evidence = merged
- Week-7 paid board run `39849371821` remains canonical
- no additional paid OddsAPI call was authorized
- #695 archive workflow = success
- #695 Repo CI = known stale workflow-contract failure only (660 pass / 1 skip / 1 fail)
- newest visible #535 comment = GPT **6420198257**
- Claude construction diagnosis = accepted
- opponent/environment transmission = **OPEN**
- Walker = preferred first generic-skill transmission case
- player identity = reconstructable and can proceed immediately
- pooled-only gate limitation = legitimate design issue under review
- Stage-2 allocation = qualified research exists but not wired into production
- betting construction needs a separate V2 framework
- ATD remains uncertified

The next session MUST re-fetch live GitHub and #535 before asserting that this remains current.

**Do not ask the user to reteach any of this.**
