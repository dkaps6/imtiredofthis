# NFL HANDOFF — 2026-09-22 — CURRENT-SEASON STATE + LIVE W1/W2 SCOREBOARD + NEXT MODEL IMPROVEMENT

**Status:** CURRENT  
**Repository:** `dkaps6/imtiredofthis`  
**GitHub is canonical over chat memory.**  
**Purpose:** allow the next GPT-5.6 chat to resume immediately without replaying the last conversation.

---

## 0. READ THIS FIRST — CURRENT PRIORITY

The user wants the project back on **actual model improvement and genuinely new football information**.

Do not restart old research, do not repeat PR #625, and do not confuse distribution plumbing with mean-model improvement.

The active idea is now:

> learn from completed 2026 football every week, but use historical evidence to determine **which live signals persist** and **how aggressively to trust them**.

The latest evidence strongly supports:

> **historical prior + faster current opportunity/role updating + heavier efficiency shrinkage**

The two immediately relevant work streams are:

1. finish/repair Claude's Week-1/Week-2 full-board grading PR #626 so the exact live scorecard is canonical;
2. use that scorecard together with the completed current-season state research to choose the next position/market improvement lane.

The user explicitly believes every position can still improve. Do not treat QB as "finished" simply because pass yards is currently the strongest live market.

---

# 1. EXACT CURRENT REPOSITORY STATE

Substantive production/research state after the latest merge:

`main = 5ecd4da5a7ba7ab314f11ae9e8cff9ef7f6308fc`

That is the merge commit for PR #627.

A later documentation-only handoff commit may move `main`; verify live `main` before mutation.

### Closed / merged

- **PR #625** — RB PD2 forward-shadow implementation validation — MERGED.
- **PR #627** — prospectively continue WR/TE 2026 snap state into Week 3 — MERGED.

### Open

- **PR #626** — `Make the market track record durable, reproducible and readable per position`
  - branch: `claude/nice-cori-4qkwqs`
  - current head at handoff: `c19399903c5c70635b47bdfd748510df23dfffab`
  - mergeable/clean at handoff, BUT **NOT merge-ready** because four Codex review threads remain unresolved, including one P1 that can affect exact grading.

---

# 2. PR #625 — COMPLETELY CLOSED; DO NOT REOPEN

PR #625 built the RB PD2 prospective/shadow infrastructure. It did **not** improve RB rushing-yard point means.

Final merge:

`2a8b4e5622df3917c672212b16a8189c2bab7921`

Week-2 completed-history certification:

- run `35732450688` SUCCESS
- artifact `10695992072`
- digest `sha256:a84aa005dfdd35d16f891c40c1cafbad5c2779aca3dad16214624d0216e052bb`
- 108 / 108 Week-2 RB/FB rows verified
- 93 weekly-stats outcomes + 15 roster-confirmed verified zeroes
- 0 exclusions
- cumulative certified 2026 history = 215 rows, Weeks `[1,2]`
- sportsbook inputs used for football projection = 0

Final Week1+2 forward history:

- run `35732782075` SUCCESS
- artifact `10696312599`
- digest `sha256:1205ea152b4702e19d12379c79e29ff7fc20fd2a632a61f7c519980f64b648fb`
- 1,608 state rows
- 169 players
- 889 scoreable rows
- `completed_2026_through_week=2`
- zero prospective outcomes used

The prospective lock now writes artifacts first and requires a 15-minute pre-kickoff persistence buffer.

PD2 is **mean-neutral distribution calibration**:
- pooled CRPS historical gain +1.218%
- high-difficulty CRPS +2.624%
- point MAE unchanged by design

Do not describe #625 as an RB mean-model accuracy fix.

---

# 3. USER'S LIVE-SEASON QUESTION — NOW ANSWERED WITH DATA

The user asked whether a serious model should learn each week from what is actually happening in the current season: usage, role, defenses, teammates, etc., while still shrinking small samples toward history.

We froze and ran **Current-Season State Persistence V1** specifically to answer that.

Branch:

`research-current-season-state-persistence-v1`

Frozen plan:

`docs/research/CURRENT_SEASON_STATE_PERSISTENCE_V1_PLAN.md`

Result:

`docs/research/CURRENT_SEASON_STATE_PERSISTENCE_V1_RESULT.md`

Canonical run:

- `35741758765` SUCCESS
- run head `28d79153f723765b01a447c2023b47108b14138c`
- artifact `10699781744` / `current-season-state-persistence-v1`
- digest `sha256:2d81bcb5222136ae812575b15a0d79952d27a03e68034cd312aa0f927503868a`
- row-level panel: 41,745
- frozen tests 4/4 PASS
- strict repo audit PASS
- sportsbook inputs = 0
- 2026 realized outcomes used = 0
- production changed = 0

## 3A. Existing production was already partially live

Production was not blind to 2026:

### PlayerForm V2

Uses prior season + completed current-season games with fixed pseudo-prior:

`w_current = current_games / (current_games + 4)`

So:
- after 1 game = 20% current
- after 2 games = 33.3%
- after 3 games = 42.9%
- after 4 games = 50%

Metrics include:
- target share
- rush share
- route rate where supplied
- YPRR
- YPT
- YPC
- YPA
- receptions per target

### QB / team context

Promoted M89/M90 semantics use the last eight completed games strictly before target week. Current-year games naturally replace prior-year games.

Includes:
- PROE
- neutral pace
- pressure rate allowed/generated
- pass attempts/dropback
- offensive pass rate
- defensive pass rate faced
- defensive pass EPA/success/YPA allowed
- offensive YPA/pass EPA
- estimated plays

So the frontier is not "start using current-season data." The frontier is **explicit state changes and better live source continuation**.

## 3B. 2025 untouched replication — 9/10 metrics improved

| Position | Metric | Prior MAE | Current-only MAE | Blend-4 MAE | Blend improvement |
|---|---|---:|---:|---:|---:|
| QB | YPA | 1.7386 | 1.7993 | **1.6548** | **4.82%** |
| RB | rush share | 0.1486 | **0.1096** | 0.1166 | **21.53%** |
| RB | target share | 0.04611 | 0.04418 | **0.04268** | **7.45%** |
| RB | YPC | **1.9135** | 2.1243 | 1.9160 | **FAIL / -0.13%** |
| TE | catch rate | 0.2598 | 0.2683 | **0.2547** | 1.97% |
| TE | target share | 0.05153 | 0.04643 | **0.04553** | **11.66%** |
| TE | YPT | 3.9903 | 4.1334 | **3.8234** | **4.18%** |
| WR | catch rate | 0.2340 | 0.2501 | **0.2332** | 0.35% |
| WR | target share | 0.06773 | 0.06352 | **0.06144** | **9.30%** |
| WR | YPT | 5.0035 | 5.0457 | **4.7787** | **4.49%** |

State-delta Spearman highlights:
- RB rush share: **+0.6256**
- TE target share: **+0.4485**
- WR target share: **+0.4083**
- RB target share: +0.3610
- QB YPA: +0.1913

The strongest live signal is **role/opportunity state**, not raw RB efficiency.

## 3C. Exact two-completed-game / Week-3 analogue

2025 rows with exactly two completed current-season games before target:

- RB rush share: prior `0.1288` -> blend **`0.1113`**
- RB target share: prior `0.04462`; current-only **`0.04106`**; blend `0.04170`
- WR target share: prior `0.05933` -> blend **`0.05451`**
- TE target share: prior `0.03941` -> blend **`0.03428`**
- RB YPC: prior **`2.0697`** beats blend `2.1239` and current-only `2.6333`
- QB YPA: early two-game current signal is too noisy; prior-only beats blend
- WR/TE early catch-rate and raw efficiency are also much noisier than role state

Operational interpretation:

> **Update opportunity/role aggressively enough to matter; keep efficiency much more anchored to history early.**

This should become a standing architectural principle, not a one-off Week-3 trick.

---

# 4. WR/TE 2026 SNAP GAP — FOUND, TESTED, AND NOW MERGED VIA PR #627

The state audit found:

nflverse 2026 snap counts were already available:
- W1: 1,492 rows, all 32 teams, complete offense snaps/pct
- W2: 1,502 rows, all 32 teams, complete offense snaps/pct

But WR-R15 and TE-R5P shared a loader hardcoded to source seasons 2020-2025.

Thus the promoted entitlement layers could not see current 2026 snap participation.

## 4A. Research source-continuation authority

Canonical research branch:

`research-wr-te-2026-snap-source-continuation-v1`

Canonical run:

- `35742765095` SUCCESS
- head `741ae291aa43f7b7f42aa8047d36e18a1ffa064d`
- artifact `10699873027`
- digest `sha256:dbd15f7f9d2eb34d2c28aec07fb71dc0dccb80e8f6d40f0165b92c91eee62ddd`
- historical 2020-2025 source parity: 150,909 rows exact
- Week-1 invariance exact
- strict-prior gates PASS
- 2026 source: 2,994 rows
- Week-2 feature comparison showed many rows gain current-season snap state:
  - 279 target rows with any feature change in the Week-2 analogue
  - 91 gain prior1 same-team
  - 43 gain prior1 any-team
- Week-3 analogue also showed broad current-state feature change

Disposition:

`WR_TE_2026_SNAP_SOURCE_CONTINUATION_READY`

### Important duplicate-lane warning

A later GPT branch `research-current-season-state-persistence-v1` briefly started a second implementation/workflow after the UI timeout. It was recognized as duplication and abandoned once the canonical existing continuation branch/PR was discovered.

Do **not** treat the later duplicate workflow/script as authority. The canonical science is:
- `research-wr-te-2026-snap-source-continuation-v1`
- run `35742765095`
- artifact `10699873027`

## 4B. PR #627 production continuation — MERGED

PR title:

`Prospectively continue WR/TE snap state into 2026 Week 3`

Final PR head:

`33f973678b6dbb3def9c23a90c6c9edf898c328d`

Merge commit:

`5ecd4da5a7ba7ab314f11ae9e8cff9ef7f6308fc`

Exact-head validation:
- Repo CI `35745912040` SUCCESS
- preserved paid Week-2 replay `35745912090` SUCCESS
- final Codex review: no major issues
- unresolved review threads: 0
- no paid/live OddsAPI pull

### Activation boundary

This intentionally does **not** rewrite Weeks 1-2 production:

- <=2025: source 2020-2025
- 2026 W1: legacy source
- 2026 W2: legacy source
- **2026 W3+: source may include 2026**

No model coefficients/assets/features/pool logic changed.

### Late P1 that was fixed before merge

Codex correctly noted that "2026 exists" was not enough. A delayed feed could contain Week 1 but be missing completed Week 2 and still activate stale history.

Final merged behavior now fails closed unless:
- the immediately prior regular-season week exists;
- every team actually scheduled that week is present;
- usable offense snaps and offense pct exist;
- the gate is schedule-sized / bye-aware, not hardcoded to 32 teams.

Both TE-R5P and WR-R15 audits expose the freshness proof.

---

# 5. CLAUDE'S WEEK-1/WEEK-2 FULL-BOARD BACKTEST — COMPLETE DATA, BUT PR #626 STILL NEEDS REPAIR

Claude completed the live-slate grading work and posted the full evidence in Issue #535.

PR:

`#626 — Make the market track record durable, reproducible and readable per position`

Branch:

`claude/nice-cori-4qkwqs`

Current head at handoff:

`c19399903c5c70635b47bdfd748510df23dfffab`

Canonical full-board run:

- `35740684289` SUCCESS
- run head `ca5a001bb817e495ca2d74b1e03ce43886493f43`
- artifact `10699408992` / `backtest_weeks_1_2_graded`
- digest `sha256:afed85e578adfd8ad9a9dff1564f41dc590315574a74b67ecdda99d5195829d7`

Committed source files on Claude branch:
- `data/market_track_record/boards/2026_wk01.csv`
- `data/market_track_record/boards/2026_wk02.csv`
- `data/market_track_record/graded/2026_wk01_wk02_graded.csv`
- `docs/production/BACKTEST_2026_WEEKS_1_2_FINDINGS.md`

Reproduction command:

`PYTHONPATH=. python scripts/operations/backtest_full_report_v1.py --season 2026 --weeks 1,2`

### Board sizes

- W1 raw ledger: 3,048 rows
- W2 raw ledger: 3,178 rows
- gradable markets after anytime-TD exclusion: W1 1,674 / W2 1,644
- current one-bet-per-player-market graded log: 866 rows
- anytime TD remains ungraded by standing policy

### Claude's currently reported scorecard — PROVISIONAL UNTIL #626 P1 IS REPAIRED

Overall:
- 438-428 (50.6%)
- -48.64 units
- model MAE 18.08 vs Vegas 17.11

By position:
- QB 64-46 (58.2%), +11.21 units
- RB 152-145 (51.2%), -14.86
- WR 162-155 (51.1%), -14.97
- TE 60-82 (42.3%), -30.01

By market:
- pass_yards 38-20 (65.5%), +13.60
- rush_yards 78-73 (51.7%), -3.58
- receptions 146-147 (49.8%), -25.78
- rush_rec_yards 33-34 (49.3%), -4.87
- rec_yards 143-154 (48.1%), -28.02

Week over week:
- W1 total 224-214
- W2 total 214-214
- W2 TE 24-46

These exact W/L/unit/line-error numbers must be treated as **provisional** until PR #626 is repaired/regraded because of the P1 below.

## 5A. Claude's major diagnostic findings

Even with the grading caveat, several mechanisms are important enough that the next chat should retain them.

### A. Probability layer appears dramatically overconfident

Claude reported:
- >70% stated-probability band: 355 bets
- mean stated probability ~81.4%
- realized hit ~50.1%

The broad observation is consistent with older clean-cohort overconfidence work.

### B. Simulated distributions appear too narrow

Reported model SD versus realized projection-error SD:

- pass yards: 54.44 vs 78.67 = ~1.45x
- rush yards: 15.23 vs 28.13 = ~1.85x
- receiving yards: 17.40 vs 31.18 = ~1.79x
- receptions: 1.62 vs 2.17 = ~1.33x

This plausibly explains the extreme fair probabilities.

Do **not** immediately rescale all distributions from two live weeks. This needs frozen validation and awareness of existing position-specific distribution authorities (including QB C2 and RB PD2).

### C. `edge_pct` does not rank performance monotonically

Claude's quintiles were non-monotonic; largest declared edges also carried very large negative projection bias.

Interpretation: large edge may currently be a marker of **projection disagreement/error**, not useful certainty.

### D. RB rush+receiving yards has a very large low-bias / construction concern

Claude reported:
- mean model projection ~53.7
- mean line ~72.7
- mean actual ~73.9
- model bias around -20.2 yards

The component-market biases do not fully explain the combined-market shortfall. Treat this as a likely **construction/seam audit target**, not proof of one particular cause.

### E. TE is a serious weakness

The live board strongly indicates TE is a problem. Claude's interpretation evolved from "hole" to "no-edge market, not necessarily inverted":

- model often sits close to the posted TE number;
- W2 TE is especially poor;
- receiving yards appears worse than receptions;
- the mechanism may be inability to separate winners/losers rather than a simple sign/bias inversion.

This matters enormously because the state-persistence work independently says **TE target-share state is highly persistent**, and PR #627 just restored current 2026 snap participation to the TE entitlement path beginning Week 3.

That intersection makes TE a very high-value next diagnostic lane.

### F. QB passing yards is the standout early positive market

Reported 38-20 across W1-W2 under Claude's current grading method and robust to several quoted-line selection variants Claude manually compared.

However:
- n is only 58;
- bets cluster within games;
- Week 1 did not cover all 32 teams;
- exact canonical grading still awaits #626 P1 repair.

Do not retune QB based on this. Preserve parameters and continue prospective Weeks 3-6 evaluation.

### G. Systemic low projection bias / UNDER-heavy board

Claude reported the board was roughly 568 UNDER vs 298 OVER and model bias more negative than line bias.

Treat this as a cross-position diagnostic, not evidence that "unders are good."

---

# 6. PR #626 — EXACT OPEN REVIEW DEFECTS

**Do not merge PR #626 yet.**

At handoff there are four unresolved Codex threads.

## P1 — grading line/side inconsistency

Thread:

`PRRT_kwDOQAMuU86kxoOU`

Review comment DB ID:

`4072883652`

File:

`scripts/operations/grade_market_track_record_v1.py`

Current implementation:
- computes median consensus line;
- chooses model side against consensus;
- filters book rows to that side;
- then keeps nearest quoted line.

Problem:
a retained UNDER/OVER book row can have a quoted line on the opposite side of the projection relative to the consensus decision. Codex cited Terry McLaurin:
- projection 50.33
- quotes 49.5 and 52.5
- consensus side UNDER
- current tie break can retain an UNDER at 49.5 even though the model projection is OVER 49.5.

That can corrupt:
- W/L
- Vegas error
- closer-rate
- edge slices
- units / odds interpretation

**Next chat must fix this first and regenerate/reconfirm the scorecard.**

Do not "fix" by choosing whichever line makes the result look better. Freeze a deterministic outcome-independent rule.

Reasonable approaches include:
- grade all prediction/error/W-L geometry against an explicitly stored synthetic consensus line and use a separately documented deterministic odds convention; or
- retain only a real quote whose own line is compatible with the model side, with a predeclared abstention rule if none exists.

Pick one rule prospectively/deterministically, test row-order invariance and straddle cases, then regenerate both weeks.

## P2 — expired Week-1 hit-rate workflow source

Thread:

`PRRT_kwDOQAMuU86kxoOi`

Comment:

`4072883667`

File:

`.github/workflows/week1-qb-passyards-hitrate.yml`

It still downloads expired run/artifact `34910010508`.

Fix by using the committed Week-1 board or surviving durable ledger artifact. Do not rely on an expired seven-day replay artifact.

## P2 — archive retention backstop skips after commit failure

Thread:

`PRRT_kwDOQAMuU86kxoOq`

Comment:

`4072883678`

File:

`.github/workflows/archive-market-track-record-v1.yml`

The long-retention upload step has a `hashFiles` condition but lacks `always()`.

If the prior commit/push step fails, GitHub's implicit success condition skips the backstop—the exact failure mode it is supposed to protect.

Fix condition to preserve the ledger even after prior-step failure while retaining the file-existence guard.

## P2 — heterogeneous-odds significance null is not exact

Thread:

`PRRT_kwDOQAMuU86kxoOx`

Comment:

`4072883685`

File:

`scripts/research/slice_graded_track_record_v1.py`

Current code averages payout then runs a single-probability binomial test.

Because bets have heterogeneous prices, wins are not identically distributed Bernoulli trials under that null. Exact count distribution is Poisson-binomial, or use an appropriate price-aware/unit-return test.

This can make p-values too optimistic and affects BH/FDR declarations.

Fix before treating any "significant profitable slice" as canonical.

---

# 7. IMPORTANT PROCESS DEFECT CLAUDE FOUND — MARKET ARCHIVE HAD NEVER WORKED

Before Claude's repair work, all historical `Archive Market Track Record` runs had died because a no-live Full Slate has no ledger directory and:

`git add data/market_track_record/boards/`

failed under `set -e`.

This is why the live boards were almost lost to 7-day artifact expiry.

Week-1 original source artifact already expired; Claude recovered it only because a surviving related artifact contained the board.

The raw W1/W2 boards are now committed on PR #626's branch. Preserve them.

The archive workflow fix is directionally correct but its P2 backstop bug must still be repaired before merge.

---

# 8. CURRENT BEST SYNTHESIS — WHAT THE MODEL IS TELLING US

The strongest new combined insight is not "current season good" in the abstract.

It is:

## Opportunity / entitlement updates are where two weeks already matter

Historical evidence:
- RB rush share: very strong persistence
- WR target share: strong persistence
- TE target share: strong persistence

Live evidence:
- RB is still mediocre overall and known to have opportunity problems in high/low workloads
- WR is roughly neutral early
- TE is badly underperforming
- current 2026 WR/TE snap participation was missing from R15/R5P before Week 3

Thus the highest-value near-term work should emphasize:
- role changes
- snap share
- carry/target entitlement
- room competition
- injury-created vacancy
- target-pool / position-room changes

## Early efficiency needs more shrinkage

Historical evidence:
- RB YPC does not improve from two-game/current-season state
- QB individual YPA is noisy after two games
- WR/TE YPT and catch rate are much less stable than target share

Do not chase two-game hot/cold efficiency.

## Distribution calibration is also a real cross-position issue

Claude's live result suggests the probability layer is too confident and empirical distributions too narrow.

But this must be handled authority-by-authority:
- QB C2 already exists
- RB PD2 is in prospective shadow
- WR/TE distribution work has historical lineage
- do not apply a single global live SD multiplier from two weeks without frozen historical/held-out support

---

# 9. POSITION-SPECIFIC NEXT LEADS

## QB

Current production:
- mean M89/M90
- distribution C2

Current-season team context already updates through rolling-eight completed games.

Live W1-W2 passing-yard performance is very encouraging, but sample is small.

Next action:
- preserve parameters
- continue prospective scorecard
- audit distribution calibration separately if the broad SD-narrowness finding survives repaired #626 grading / later weeks
- do not reopen ordinary same-information QB model-zoo work

## RB

Standing retrospective prohibition remains in force.

M96A already proved:
- overall opportunity recovery ~7.68 yd
- efficiency recovery ~6.73 yd
- low/high workload = opportunity-dominant
- middle workload = efficiency-dominant

M96E achieved +0.141791 yd MAE improvement vs required +0.150000; shortfall only 0.008209. Do not reopen exposed 2025 router tuning.

Strongest sanctioned new live lead remains:

**backfield teammate availability / injury-created vacancy propagation**

Why:
- production removes unavailable RBs but does not explicitly transfer missing carries to successors;
- current-season RB rush share is the strongest state-persistence signal found;
- prior-week snap counts are deployable;
- M95G/M95H already showed current role/depth has workload signal.

Second live lead:
- prior-week snap share / backfield concentration.

Do not update YPC aggressively from two games.

## WR

Authority:
- M38 WR1
- WR-R15 WR2+

Current-season target share is strongly persistent.
PR #627 now lets R15 see 2026 strict-prior snaps beginning W3 after freshness validation.

Watch whether W3+ entitlement improves before inventing a new WR formula.

Longer-term new-data frontier:
- strict-prior receiver release spacing / route-conditioned spacing from BDB.

## TE

Authority:
- TE-R5P

This is likely the most urgent live weakness after #626 repair.

Historical attribution:
- TE error approximately 45% opportunity / 55% efficiency;
- catch rate ~34% and YPR ~21% of error mechanism.

New combined clue:
- TE target-share state is strongly persistent;
- 2026 snap participation was absent from R5P through W2;
- PR #627 fixes that prospectively for W3+;
- live W2 TE was very poor.

Next TE work should distinguish:
1. did stale entitlement/participation contribute materially?
2. after W3 current snaps are active, does entitlement move in the expected direction?
3. is the remaining TE miss mostly efficiency / distribution / matchup?

Do not call TE simply "inverted."

---

# 10. ADVANCED / NEW-DATA FRONTIER — STILL ACTIVE

Ordinary aggregate feature hunting has diminishing returns.

Highest-value genuinely new relational information remains:

1. receiver/defender route geometry and true coverage responsibility
2. blocker-rusher assignment + protection geometry
3. RB first-contact/tackle/run geometry
4. run concept / point of attack / blocking responsibility
5. ball / throw-window geometry

Recovered BDB results still matter:
- receiver spacing persistence is real;
- route-conditioned spacing has usable strict-prior persistence;
- protection interaction geometry has persistence;
- these have not yet proven downstream outcome lift.

Do not replace current-state work with a giant feature dump. Use frozen hypotheses tied to observed model error.

---

# 11. SPORTSBOOK / FOOTBALL BOUNDARY

Permanent philosophy:

`GAME STATE -> TEAM OPPORTUNITY -> POSITION/ROOM POOL -> PLAYER ENTITLEMENT -> PLAYER EFFICIENCY -> JOINT MC -> PROJECTIONS/DISTRIBUTIONS -> FAIR PROBABILITIES -> SPORTSBOOK COMPARISON`

Sportsbook data is downstream benchmarking/pricing.

Do not train upstream football projections to market lines.

Claude's market-track-record lane is a **scoreboard**, not a source of football truth.

No paid OddsAPI pull is authorized without explicit user approval.

---

# 12. EXACT NEXT ACTION QUEUE

When the next chat starts:

### Step 1 — verify live GitHub state
Read:
1. `AGENTS.md`
2. `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. this file
4. latest Issue #535 comments after `5779220450`
5. PR #626 current review threads/head/checks

Do not assume the SHAs above are still current.

### Step 2 — close PR #626 correctly
Address all four review findings.

**P1 first.** Freeze a deterministic grading-line convention that cannot depend on outcome or row order.

Regenerate/reconfirm the full W1/W2 scorecard after the P1.

Then:
- fix expired W1 workflow source;
- fix archive backstop with `always()`;
- replace the homogeneous binomial null with a price-aware significance method.

Run:
- targeted tests
- Repo CI
- relevant backtest workflow
- preserved replay if required by PR contract
- final Codex exact-head review

Do not merge until zero unresolved correctness/integrity findings.

### Step 3 — compare repaired #626 scoreboard to Current-State Persistence V1
Use the intersection of:
- actual live weakness
- historical persistence
- missing or newly restored live state
- deployable pregame data
- no closed-family retest

Likely first high-value diagnostic:
**TE receiving-yards / entitlement-vs-efficiency after PR #627 current-snap activation.**

RB vacancy propagation remains an excellent parallel sanctioned mean-information lane.

### Step 4 — build weekly learning process
The project should eventually produce each Tuesday:
- authority-exact model performance by position/market
- deduplicated betting scoreboard
- opportunity vs efficiency diagnosis where supported
- current-season state changes since prior week
- which changes historical walk-forward evidence says are persistent
- only then candidate experiments

The user wants the model to continuously learn from what the NFL is currently telling us without overreacting to tiny samples.

---

# 13. THINGS NOT TO DO

Do not:
- reopen #625;
- rerun Week-2 RB PD2 certification;
- duplicate PR #627;
- merge #626 before fixing the review findings;
- canonize the current 438-428 exact record before #626 regrade;
- retune QB off 58 pass-yard bets;
- chase two-game RB YPC;
- reopen M96 exposed retrospective router variants;
- use sportsbook lines as upstream football features;
- apply one global SD multiplier from two weeks of live error;
- abandon unfinished work because a new idea appears.

---

# 14. USER EXPECTATIONS / WORKING STYLE

The user strongly values:
- exact continuity;
- honest pushback;
- not redoing already-tested ideas;
- closing loops;
- real model improvement rather than plumbing for its own sake;
- historical evidence plus live-season adaptation;
- position-by-position football reasoning;
- autonomous progress unless a consequential decision actually requires input.

The user was especially pleased with the current-season state audit because it connected:
- repository architecture,
- historical walk-forward evidence,
- current live data availability,
- a concrete missing production seam,
- and an actionable prospective fix.

Maintain that standard.

---

# 15. ONE-PARAGRAPH EXECUTIVE HANDOFF

PR #625 is closed and gave RB PD2 a clean prospective distribution-calibration path, not a mean fix. Current-Season State Persistence V1 then proved on 2022-2025 walk-forward data that completed current-season role/opportunity state matters materially: RB rush share, WR target share and TE target share update quickly, while RB YPC and early efficiency remain noisy. That audit discovered that WR-R15/TE-R5P were not consuming available 2026 snap data; canonical source-continuation research passed and PR #627 has now merged prospectively for Week 3+, with a schedule-aware fail-closed prior-week freshness gate. In parallel Claude completed the full W1/W2 live-board backtest and preserved the raw boards/866-row log in PR #626. Its diagnostics show major probability overconfidence/narrow distributions, TE weakness, a rush+receiving construction concern and strong early QB pass-yards performance, but PR #626 still has four unresolved Codex review defects—one P1 that can change exact grading—so repair/regrade it before treating exact W/L/units as canonical. After that, use the repaired live scoreboard together with the state-persistence evidence to attack the highest-value position/market, likely TE entitlement/efficiency first and RB vacancy propagation in parallel, while keeping efficiency heavily shrunk early and continuing genuinely new-data research.

