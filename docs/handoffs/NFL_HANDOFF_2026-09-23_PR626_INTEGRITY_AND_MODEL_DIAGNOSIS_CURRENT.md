# NFL HANDOFF — 2026-09-23 — PR #626 INTEGRITY REPAIR + LIVE MODEL DIAGNOSIS

GitHub is canonical over chat memory. This handoff supersedes the prior
2026-09-22 active handoff for current priority/state.

## 0. START HERE / MEMORY-EFFICIENT READ ORDER

Read only this sequence before acting:

1. `AGENTS.md`
2. `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. this file
4. Issue #535 comments after the handoff checkpoint that points here
5. live PR #626 head/checks/reviews/unresolved threads

Do **not** reread the entire historical Issue #535 thread or restart old
experiments unless this handoff explicitly points to them.

Production main immediately before these docs-only handoff commits:

`e400fa6fc9c8cacc022bba938618d8bba6142bc2`

The docs-only handoff commits will advance `main`; the final Issue #535
checkpoint records the new exact main SHA.

---

# 1. EXECUTIVE STATE

The current priority is still:

> **Finish PR #626 correctly, then immediately return to model-performance /
> new-information research.**

Do not get stuck in infrastructure after #626 is genuinely closed.

### Closed / do not reopen

- PR #625 — merged/finished.
- PR #627 — merged/finished.
- M96 exposed retrospective RB router family — CLOSED by standing prohibition.
- WR/TE 2026 snap source-continuation work — completed in PR #627; do not duplicate.

### Current open work

PR #626:

`Make the market track record durable, reproducible and readable per position`

Branch:

`claude/nice-cori-4qkwqs`

**Physical GitHub head at handoff:**

`a8973fc2e398a7b842736dd0481857391d387829`

PR state at handoff:

- open
- mergeable = true
- mergeable_state = unstable
- six unresolved review threads
- Repo CI on this head: SUCCESS
- W1/W2 full-board backtest/provenance workflow on this head: SUCCESS
- latest preserved no-paid Full Slate replay on this head: FAILURE
- an earlier preserved replay on the exact same SHA succeeded

The replay difference is now diagnosed as a deterministic moving-source
reproducibility bug, not random football drift.

**Do not merge #626 yet.**

---

# 2. CRITICAL DISTINCTION: CLAUDE'S BACKTEST WAS NOT DUPLICATED

The user explicitly asked about this.

Claude already completed the original Week-1/Week-2 full-board grading lane and
preserved the raw paid boards. GPT-5.6 did **not** independently rerun the
football model from scratch or start a second competing model backtest.

What happened after Claude finished:

1. Claude preserved the real W1/W2 priced boards and initially graded one
   player-market bet under a consensus-line convention.
2. Codex review exposed correctness defects in how that preserved board was
   being turned into a betting scorecard.
3. GPT-5.6 then repeatedly repaired/regraded the **same preserved Claude boards**
   so the scorecard would match actual production decision semantics and
   sportsbook settlement semantics.
4. That repair work uncovered several real evaluation/plumbing defects:
   - arbitrary duplicate-row selection;
   - consensus-side / retained-real-line contradictions;
   - grading mean-vs-line direction instead of the deployed Best Snapshot EV side;
   - counting nonpositive-EV markets as bets instead of PASS;
   - grading inactive/DNP props as zero instead of VOID;
   - treating roster presence as participation;
   - fractional probability edges binned as if they were 0-100 percentages;
   - heterogeneous prices tested with an iid single-p binomial null;
   - within-game prop dependence ignored;
   - void rows contaminating accuracy denominators;
   - board provenance restamped with later postgame metadata;
   - replay dependence on a moving current NFL.com injury page.

So the time spent after Claude's backtest was **scoreboard integrity repair**,
not duplicate model research.

The raw paid boards Claude preserved remain the source evidence.

---

# 3. CURRENT PRODUCTION-ALIGNED SCORECARD — STILL PROVISIONAL

The current branch's production-decision findings are in:

`docs/production/BACKTEST_2026_WEEKS_1_2_FINDINGS.md`

Current selected-settlement ledger:

`data/market_track_record/graded/2026_wk01_wk02_graded.csv`

Current branch output before the remaining six review blockers are closed:

- selected settlement rows: **810**
- decided bets: **802**
- DNP voids: **8**
- stats-table outcomes: **791**
- snap-confirmed verified-zero outcomes: **11**
- unresolved positive-EV rows excluded: **2**
- current decided record: **397-405 (49.5%)**
- current units: **-44.09u**
- Week 1: **204-205**
- Week 2: **193-200**

By major area:

- QB pass yards: **36-17, +14.99u**
- TE overall: **58-75, -21.45u**
- TE receiving yards: **28-41, -16.09u**
- TE receptions: **30-34, -5.37u**
- rush+receiving yards: **33-33**, with model bias about **-20.67 yd**

**These exact numbers are not yet final canonical numbers.**

Why:
- final-board quarantine handling is still under review and can alter the exact
  production record;
- provenance/replay integrity blockers are still open.

Do not resurrect or quote the old 438-428 / 439-427 / 440-426 records as
production scorecards. Those were intermediate conventions and are superseded.

---

# 4. WHAT THE CURRENT SCOREBOARD REPAIR ACTUALLY CHANGED

The canonical grading intent now mirrors the deployed downstream betting layer:

For every real book+line offer:

1. use archived side-specific `fair_prob` and captured American odds;
2. compute expected ROI for OVER and UNDER;
3. take the higher-EV side for that concrete offer;
4. across real offers for that player-market, take the highest-EV offer;
5. if best EV <= 0, production says PASS and no bet is graded;
6. material exact-EV ties that imply different wagers fail closed;
7. identical-wager ties use deterministic normalized provider identity;
8. sportsbook data remains downstream only and never feeds the football projection.

Postgame settlement now intends:

- real weekly-stat row -> settled actual;
- missing weekly-stat row + positive postgame snap participation -> verified zero;
- explicit inactive/DNP + no participation for supported captured book -> VOID;
- ambiguous/missing participation evidence -> unresolved/fail closed.

The live scorecard work therefore taught us that **yes, parts of the
evaluation/decision plumbing were incorrectly wired**. That is different from
proving the upstream football models are mathematically worthless.

---

# 5. CURRENT REMOTE PR #626 REVIEW BLOCKERS — EXACTLY SIX THREADS

At handoff, the physical GitHub head is still
`a8973fc2e398a7b842736dd0481857391d387829`.

## 5A. All-PASS / zero-selected-bet crash — one defect, three duplicate P2 threads

Threads:

- `PRRT_kwDOQAMuU86k_S5e`
- `PRRT_kwDOQAMuU86lApQc`
- `PRRT_kwDOQAMuU86lApRK`

Problem:

If every player-market has nonpositive EV (or every best offer fails closed),
`select_model_bet(board)` returns an empty frame.

Current remote code then continues into identity/postgame resolution and can:

- index a missing `gsis_id` column in `grade_market_track_record_gsis_v1.grade()`;
- hit `pd.concat([])` in `backtest_full_report_v1.build_graded()`.

Required repair:

- immediately short-circuit after empty selection;
- return a structured zero-selected/zero-decided summary;
- if detail output is requested, write a header/schema-valid empty CSV;
- independent full-report path must return a downstream-safe empty graded frame;
- add focused all-nonpositive-EV regression coverage;
- preserve all non-empty behavior exactly.

Important: Codex produced several summaries claiming local commits
(`a285736`, `69511a...`, `95840d...`, `36ea176...`) implementing this,
but those commits did **not** advance the physical GitHub PR head. They are
NON-CANONICAL patch hints only.

Do not assume this is fixed until the remote PR head actually changes and tests
run on that exact remote SHA.

## 5B. Historical preserved replay depends on moving current NFL.com page — P1

Thread:

`PRRT_kwDOQAMuU86lEIJt`

Exact-head evidence:

- failing replay: run `35824360266`
- earlier same-SHA successful replay: run `35816381724`
- Repo CI `35824360282` = SUCCESS
- W1/W2 full-board/provenance run `35824360270` = SUCCESS

Diagnosed cause:

The replay is pinned to historical Week 2, but the injury-scope certification
fetches the current live NFL.com injuries page. NFL.com rolled from Week 2 to
Week 3, causing:

`expected=2 detected=3`

This makes the supposedly preserved/offline replay time-dependent.

Required repair:

- historical replay must consume immutable/preserved Week-2 injury-scope
  evidence;
- validate provenance/digest, season/week, exact scheduled-team set, resolved
  scope states and official-source identity;
- retain the byte-for-byte preserved injury-row mutation guard;
- do not weaken/remove the scope gate;
- do not substitute current Week-3 data;
- no paid odds pull.

A Codex local summary proposed downloading immutable Week-2 scope evidence from
successful same-head replay `35816381724`; useful design hint, but the claimed
local commit `36ea176...` is not on GitHub.

## 5C. Provenance verifier is still self-referential — P1

Thread:

`PRRT_kwDOQAMuU86lEIJy`

Current manifest:

`data/market_track_record/ORIGIN_PROVENANCE_V1.json`

Current verifier:

`scripts/operations/verify_market_board_origin_v1.py`

Problem:

The verifier currently proves that a board matches a digest stored in the same
editable repository/PR. A simultaneous edit to board + manifest can therefore
pass. That proves internal consistency, not immutable origin provenance.

Frozen origin evidence already recovered:

### Week 1
- source run: `34650067599`
- source SHA: `be061eaf23372f080db3911d3b4919120c744c53`
- original artifact: `10283817522`
- original artifact digest:
  `a70c90023632059476cc688070fb21ad63b9ebd14aa4db102844b9e342188359`
- original artifact expired
- no-paid replay `34910010508` logs
  `FULL_SLATE_SOURCE_RUN_ID=34650067599`
- surviving recovery artifact: `10607998417`
- canonical non-provenance content digest:
  `b830655c127c089cf1980f1af230444bfde11f8ad27c77b0051bcfd7c123ef82`

### Week 2
- source run: `35282021679`
- source SHA: `c6ec55be70d6e05bbd1dbae83d7d5c86ac8aa00a`
- origin artifact: `10523345092`
- origin artifact digest:
  `6024ed21d7032e6e6440145277d9d4b74e98f319ef286962bac2a786a8d28e3c`
- raw `props_priced_clean.csv` SHA256:
  `1f38c995011a54fddaa73457c4db0fdb746ee2be027c735ee25d842f8160b40c`
- uploaded workbook SHA256:
  `962ef8dc048783bd90a7441652d910180e05b1b96a9608f0203607c6668b189c`
- canonical non-provenance content digest:
  `d461ef415e86d6b5a986bd39893f2a6e40d3d07852cf419bee11ba0afc1793d2`

Required next step:

Anchor the board digest check to **independently immutable origin/recovery
evidence**, not merely to an editable manifest value. Preserve the source
run/SHA checks too.

## 5D. Final-board quarantine is not applied to the current production record — P2

Thread:

`PRRT_kwDOQAMuU86lHtql`

File:

`data/manual_final_board_quarantine.csv`

Current review cites Carson Wentz Week 2.

The quarantine file explicitly says all Wentz Week-2 props should be pulled
from the published board until the MIN starter state resolves.

However, recovered origin evidence establishes an important nuance:

- Week-2 origin Full Slate run `35282021679` failed in pricing step 31 **after**
  `props_priced_clean.csv` existed;
- final-board quarantine step 32 was therefore skipped;
- always-run workbook build/artifact upload still succeeded;
- the actual uploaded workbook contains Wentz in Best Snapshot Edges,
  Bet Evidence V0 and Master Betting Board;
- the raw origin artifact also contains 12 Wentz rows.

So there is a real semantic conflict to resolve:

A. grade exactly what the failed-but-uploaded origin artifact emitted, or  
B. grade the intended/published production board after applying the repository's
final-board quarantine policy.

Do not silently choose based on which record looks better.

The project's concept of a **production betting track record** should be made
explicit and then the quarantine applied consistently. Codex currently treats
the quarantine as authoritative and says at least Wentz pass-yards and
rush-yards selected bets should be suppressed.

This blocker can change exact W/L/units.

---

# 6. RESOLVED #626 DEFECTS — DO NOT REDISCOVER

These issues were found and repaired earlier on the PR branch. Do not start
over on them unless a new review proves regression.

### Durable board/archive issues
- raw W1/W2 boards are committed and no longer live only on 7-day artifacts;
- expired Week-1 workflow dependency was replaced by committed durable data;
- archive-retention backstop was repaired to run even after earlier failure.

### Grading / selection issues
- arbitrary `drop_duplicates(keep=last)` behavior was removed;
- consensus-vs-real-line straddle inconsistency was investigated;
- then the more important semantic defect was found: production does not grade
  mean-vs-line direction at all;
- scorecard now intends to reproduce downstream Best Snapshot EV selection;
- nonpositive-EV markets PASS.

### Settlement issues
- inactive/DNP is no longer supposed to become an automatic zero;
- postgame snaps distinguish participated-zero from DNP;
- supported captured-book DNP rows VOID;
- row-level `book_title` fallback for legacy blank provider keys was added;
- void rows were removed from accuracy denominators.

### Statistical issues
- heterogeneous-price iid-binomial significance was replaced;
- game clustering is accounted for before BH/FDR;
- fractional `edge_pct` units were separated from native stat-unit
  projection-line gap bins;
- shared `edge_bucket()` was restored to stat-unit semantics;
- probability-edge slice analysis has its own fractional binning.

### Provenance string repair
Original Full Slate source run/SHA were restored:
- W1 `34650067599` / `be061eaf...`
- W2 `35282021679` / `c6ec55be...`

What remains open is independent anchoring of those claims, not the source
strings themselves.

---

# 7. WHAT WE LEARNED ABOUT "IS THE MODEL/MATH WIRED WRONG?"

The user's frustration is justified. Do not answer this as if nothing was
wrong.

### Confirmed wiring/evaluation problems

We have proven that multiple scoreboard/decision/inference layers were wrong
or insufficiently reproducible:

- the original duplicate-row wager selector was nondeterministic;
- later grading did not initially reproduce production EV-side selection;
- PASS markets were initially counted as bets;
- DNPs could be counted as zeros;
- edge units were misinterpreted in one analysis path;
- heterogeneous odds and within-game dependence were initially mishandled;
- void rows contaminated an accuracy denominator;
- provenance/replay behavior was not truly immutable.

Those defects are real.

### But this does NOT mean all football projection science is fake

Independent historical football-only work still supports real signal:

Current-Season State Persistence V1:
- 2025 replication improved 9/10 tested production-aligned metrics;
- RB rush-share MAE improved **21.53%**;
- TE target-share MAE improved **11.66%**;
- WR target-share MAE improved **9.30%**;
- QB YPA full-season blend improved **4.82%**;
- RB YPC was the non-replicating exception.

PR #627 prospectively restored available 2026 strict-prior snap participation
to WR-R15/TE-R5P beginning Week 3 after a schedule/bye-aware freshness gate.

RB PD2 qualified historically as a mean-neutral distribution-calibration
improvement; it was never supposed to fix RB mean/carries/YPC.

### Genuine live model weaknesses also remain

Even after production-decision alignment, the W1/W2 board still indicates:

- probability layer materially overconfident;
- model distributions too narrow relative to realized error;
- `edge_pct` not monotonic with performance;
- TE is the clearest live positional weakness;
- rush+receiving yards has a very large low projection bias / construction seam;
- production-selected board is heavily UNDER-skewed with systemic low bias.

So the honest synthesis is:

> **Some evaluation/decision plumbing was wrong, and some football/model
> authorities are genuinely weak live. Both are true.**

Do not tell the user that years of backtesting were meaningless. Also do not
pretend the current model is elite merely because historical gates passed.

---

# 8. LIVE SCIENTIFIC SIGNALS TO PRESERVE AFTER #626 CLOSES

Treat exact W/L/units as provisional until the remaining PR blockers close, but
these mechanisms are strong enough to preserve as follow-up leads.

## QB passing yards

Current production-aligned branch result:
- **36-17, +14.99u**

This remains the strongest early live market.

Do not retune QB off two weeks. Keep frozen and score prospectively.

## TE

Current production-aligned branch result:
- overall **58-75, -21.45u**
- receiving yards **28-41, -16.09u**
- receptions **30-34, -5.37u**

Historical/current-state evidence:
- TE target-share state is highly persistent;
- TE-R1 attribution said roughly 45% of receiving-yard error mass is
  entitlement/targets and ~55% catch-rate + YPR efficiency;
- 2026 snap participation was missing from TE-R5P through W2;
- PR #627 restores strict-prior 2026 snaps prospectively beginning W3.

This intersection makes TE receiving yards the strongest likely next model
diagnostic.

The question is **entitlement failure vs efficiency/distribution failure**,
not "TE is inverted."

## Rush + receiving yards

Current branch:
- 66 decided selected bets
- model projection mean about 53.8
- selected line mean about 73.3
- actual mean about 74.5
- model bias about **-20.7 yd**

This is a high-value construction/seam audit target.

## Probability / distribution layer

Current branch findings:
- >70% stated-probability band realizes near 50%;
- realized error SD is materially wider than stated `model_sd` across major
  markets;
- no global two-week SD multiplier is authorized.

Evaluate authority-by-authority.

## Cluster-aware significance

Current branch's repaired inference reported no slice surviving BH/FDR.
QB pass yards had an encouraging raw game-cluster p-value but did not survive
the multiple-comparisons gate.

Treat it as a prospective lead, not a certified betting edge.

---

# 9. CURRENT-SEASON STATE / PR #627 — DO NOT LOSE THIS

Current-Season State Persistence V1 canonical run:

- run `35741758765`
- artifact `10699781744`
- 41,745 player-metric observations
- 2026 outcomes used = 0
- sportsbook inputs = 0
- 9/10 tested metrics improved on 2025 replication

Core interpretation:

> historical prior + faster current opportunity/role updating + heavier
> efficiency shrinkage

Strongest state:
- RB rush share
- WR target share
- TE target share

Weak/noisy early state:
- RB YPC
- individual QB YPA at exactly two games
- raw WR/TE efficiency/catch-rate

PR #627 is already merged and prospectively enables 2026 strict-prior snaps for
WR-R15 / TE-R5P beginning Week 3. It uses a schedule-aware/bye-aware
immediately-prior-week freshness gate. No coefficient refit.

Do not duplicate or reopen #627.

---

# 10. NEXT MODEL-RESEARCH ORDER AFTER #626

As soon as #626 is genuinely closed:

## First: TE receiving-yards entitlement-vs-efficiency

Use:

**live weakness + historical target-share persistence + newly restored 2026
strict-prior snap participation + existing TE-R1 attribution + deployable data**

Questions:

1. How much of W1/W2 TE miss came from target/route/snap entitlement?
2. How much came from catch rate / YPR / distribution width?
3. Does W3 current-snap activation move entitlement in the historically
   expected direction?
4. Are losses concentrated in role changes, room competition, or translation
   from targets to yards?
5. Can we improve with genuinely new pregame state rather than refitting to
   exposed W1/W2 outcomes?

No sportsbook line may be used upstream.

## Strong parallel sanctioned lane: RB teammate vacancy propagation

Existing production correctly removes unavailable RBs but does not explicitly
transfer vacated opportunity to teammates.

This is strongly motivated by:
- RB rush share = strongest state-persistence signal;
- M96A proved opportunity dominates low/high workload errors;
- live/free injury/depth/snap information exists.

Do not reopen exposed M96 router/threshold variants.

## Separate audit: rush+receiving construction

Investigate the ~-20 yd low-bias seam separately from ordinary RB/WR/TE tuning.

---

# 11. EXACT NEXT ACTION QUEUE FOR THE NEXT CHAT

### Step 1 — verify live state
Read the memory-efficient sequence at the top. Verify:
- current `main`;
- PR #626 physical GitHub `head_sha`;
- current exact unresolved review threads;
- current workflow checks.

Do not trust Codex summaries that mention commits not reachable from the PR
head.

### Step 2 — land the all-PASS repair physically on the PR branch
One implementation should close the three duplicate P2 threads.

After writing, explicitly re-fetch PR #626 and verify `head_sha` changed.

### Step 3 — repair historical injury-scope replay deterministically
Use immutable Week-2 scope evidence, not the moving current NFL.com page.
Retain all fail-closed scope/provenance/mutation checks.

### Step 4 — strengthen board-origin verification
Make the verifier depend on independently immutable origin/recovery evidence,
not only board + editable manifest consistency.

### Step 5 — resolve the final-board quarantine definition
Decide/document what the canonical production betting record represents:
actual emitted failed-run artifact vs policy-correct published final board.
Apply quarantine consistently. Do not optimize the choice for W/L.

### Step 6 — rerun exact-head gates
Required:
- focused tests;
- Repo CI;
- W1/W2 full-board/provenance workflow;
- preserved no-paid Full Slate replay;
- final exact-head Codex review;
- zero unresolved correctness/integrity threads.

No paid OddsAPI pull.

### Step 7 — merge #626 only when genuinely clean
Then update the canonical handoff / Issue #535 closure note.

### Step 8 — immediately pivot to real model research
Start TE entitlement-vs-efficiency, with RB vacancy propagation queued in
parallel. Do not spend another research cycle inventing scoreboard plumbing
once #626 is closed.

---

# 12. DO NOT DO THESE THINGS

Do not:

- restart Claude's original W1/W2 board acquisition/backtest;
- recompute old 438-428 as if it were canonical;
- trust local/container Codex commits that never advanced GitHub;
- merge #626 with unresolved correctness/provenance/settlement defects;
- reopen PR #625;
- duplicate PR #627;
- reopen exposed M96 retrospective router variants;
- retune QB because of two live weeks;
- chase two-game RB YPC;
- globally widen every distribution from two weeks of live data;
- use sportsbook lines as upstream football features;
- make a paid OddsAPI request without explicit user authorization;
- abandon an unfinished integrity loop because a more interesting model idea appears.

---

# 13. USER EXPECTATIONS / WORKING STYLE

The user wants:

- exact continuity;
- no re-explaining old work;
- GitHub as canonical;
- closed loops;
- honest disagreement/pushback when warranted;
- actual model improvement, not endless plumbing;
- current-season learning without overreacting to tiny samples;
- Claude and GPT-5.6 to share progress through GitHub.

The user is currently frustrated because the live record looks poor despite a
large historical research program.

The next chat should not dismiss that frustration or say "everything is fine."

The correct framing is:

1. yes, multiple scorekeeping/decision/inference layers were genuinely wrong;
2. fixing them was necessary and explains why the headline changed repeatedly;
3. that does not erase the historical football-only evidence;
4. the repaired live board still exposes genuine model weaknesses, especially
   TE, calibration/distribution width and rush+receiving construction;
5. after #626 closes, the project must return to those scientific problems
   immediately.

---

# 14. ONE-PARAGRAPH EXECUTIVE HANDOFF

Claude's original W1/W2 full-board work is complete and the raw paid boards are
preserved; GPT-5.6 did not duplicate that model backtest, but repaired the way
those same boards were converted into a production betting scorecard after
Codex exposed numerous real integrity defects. The scorecard now intends to
reproduce deployed Best Snapshot EV selection, PASS nonpositive-EV markets,
void confirmed DNPs, verify zeroes with snaps, use fractional edge units
correctly and perform game-cluster-aware inference. The current branch reports
810 selected settlement rows / 802 decided bets / 397-405 / -44.09u, with QB
pass yards 36-17 and TE 58-75, but exact numbers remain provisional because PR
#626 is still open at physical remote head
`a8973fc2e398a7b842736dd0481857391d387829` with six unresolved review threads:
three duplicate all-PASS crash findings, one P1 historical replay dependence on
the moving NFL.com injury page, one P1 self-referential provenance verifier and
one P2 final-board quarantine issue that can change W/L. Several Codex repair
summaries refer to local commits that never reached GitHub and must not be
treated as done. Finish those four unique defects, run exact-head CI/backtest/
no-paid replay/final review, merge #626 only when clean, then pivot immediately
to TE receiving-yards entitlement-vs-efficiency using the already-proven
current-season target-share signal and PR #627's newly restored W3+ snap state,
with RB teammate-vacancy opportunity propagation as the strongest sanctioned
parallel mean-information lane.
