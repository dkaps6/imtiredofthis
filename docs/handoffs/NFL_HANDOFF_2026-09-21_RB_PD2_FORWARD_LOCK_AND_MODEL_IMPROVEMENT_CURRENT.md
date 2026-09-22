# NFL HANDOFF — 2026-09-21 — RB PD2 FORWARD LOCK + RETURN TO MODEL IMPROVEMENT

**Repository:** `dkaps6/imtiredofthis`  
**GitHub is canonical over chat memory.**  
**Claude is temporarily unavailable because the user is out of Claude credits. GPT-5.6 owns this lane solo until Claude returns.**

---

## 0. USER DIRECTION — IMPORTANT

The user explicitly said:

- finish the current RB-PD2 forward-lock work; do **not** leave it half-finished;
- but recognize that this work is **not itself fixing RB mean accuracy**;
- once this implementation is finished, return immediately to actual model-improvement work;
- RB is still not reliably solved;
- the wider objective remains improving the production model across RB and the other positions, not spending another session exchanging status comments;
- do not let the project become administration-only.

This handoff therefore has two sequential priorities:

1. **finish the forward-lock implementation correctly;**
2. **pivot immediately into metric-improvement work.**

Do not start another documentation-only detour after Priority 1 unless a real blocker requires it.

---

## 1. READ ORDER

Read these in order before changing anything:

1. `AGENTS.md`
2. `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. this file
4. `docs/research/RB_PD2_FORWARD_SHADOW_CONFIRMATION_V1_PLAN.md`
5. `docs/research/RB_PD2_FORWARD_SHADOW_ACTIVATION_V1.md`
6. `docs/research/RB_PD2_YARD_DIFFICULTY_MC_WIDTH_V1_PLAN.md`
7. `docs/research/RB_PD2_YARD_DIFFICULTY_MC_WIDTH_V1_AMENDMENT3.md`
8. `docs/research/RB_NEW_DATA_READINESS_V1.md` on branch `research/rb-new-data-readiness-v1@355ce5c7`
9. the latest Issue #535 comments and PR #625 review threads

Also preserve the standing M96E retrospective stop in
`CURRENT_NFL_RESEARCH_HANDOFF.md` and `NFL_MASTER_CONTINUITY_RECORD.md`.

---

## 2. CURRENT CANONICAL / BRANCH STATE

Production `main`:

`76e5e0452a88018b95ebc212fb0d1b21a2ca90a3`

Active branch:

`research-rb-pd2-forward-shadow-confirmation-v1`

Current code-bearing head before this handoff commit:

`ab7e268429d4e33000d84388246aaaf1aad3f99c`

Open PR:

`#625 — WIP: RB PD2 forward-shadow implementation validation`

PR #625 is currently:

- open;
- mergeable;
- merge state = clean;
- exact-head Repo CI run `35633599108` / run number 986 = **SUCCESS**;
- exact-head preserved paid Full Slate replay run `35633599273` / run number 327 = **SUCCESS**.

Do **not** merge PR #625 yet.

There is exactly one intentionally unresolved review blocker remaining:

PR review thread:

`PRRT_kwDOQAMuU86ka3Dq`

Automated review comment:

`4063865061`

Issue:

> Advance history before locking each target week.

GPT-5.6 explicitly acknowledged the blocker in reply `4064969778` and left the
thread unresolved on purpose.

---

## 3. WHAT THE CURRENT RB-PD2 WORK ACTUALLY DOES

This is crucial.

The current forward/shadow work is **distribution-calibration validation**.

It does **not** improve RB point projection / mean MAE by itself.

Historical qualified result:

`RB_YARD_DIFFICULTY_MC_WIDTH_QUALIFIED`

Canonical historical run:

`35039152022`

Historical result:

- pooled CRPS improvement: **+1.218%**;
- high-difficulty-quartile CRPS improvement: **+2.624%**;
- player-cluster bootstrap P(candidate better): **1.0**;
- crossed player × game bootstrap P(candidate better): **1.0**;
- interval coverage improved;
- Brier >=100 improved;
- 4/4 season robustness;
- **point MAE unchanged by design** because the transform is exactly mean-neutral.

Therefore the forward study can eventually improve:

- distributional CRPS;
- interval calibration / coverage;
- high-yardage tail probabilities;
- threshold Brier scores;
- betting probability quality downstream;

but it does **not** solve the core RB mean problem: carries/opportunity, YPC/efficiency,
or final rushing-yard mean accuracy.

The user specifically challenged this distinction. Keep it explicit in all future work.

---

## 4. FROZEN RB-PD2 SCIENCE — DO NOT RETUNE

Historical difficulty state:

- last 8 strictly-prior same-player games;
- minimum prior games = 4;
- raw statistic = prior rushing-yard MAE;
- strict-prior rolling empirical percentile;
- reference minimum = 100;
- same-week rows score before insertion;
- width onset = 0.50;
- width cap = 0.30.

Frozen multiplier:

`1 + .30 * clip((difficulty_score - .50) / .50, 0, 1)`

Candidate distribution:

1. baseline = exact production empirical `adjusted_outcomes`;
2. widen around the baseline mean;
3. truncate at zero;
4. rescale to exact original mean;
5. no Normal approximation;
6. exact candidate/baseline mean parity <=1e-8.

Do not retune any of this.

Forward support:

- >=8 distinct prospectively locked NFL weeks;
- >=400 unique eligible player-games.

Primary scientific gate:

- paired pooled CRPS gain >0;
- 10,000-rep game-cluster bootstrap;
- 95% percentile CI lower bound >0;
- seed 42027.

Crossed player × game robustness:

- P(candidate CRPS lower) >=0.95.

Guardrails:

- pooled 80/90 coverage gaps non-worse and at least one improves;
- same in high-difficulty group;
- Brier >=100 strictly better;
- Brier >=50 and >=75 non-worse;
- exact point-mean parity.

PASS:

`RB_PD2_YARD_DIFFICULTY_WIDTH_FORWARD_CONFIRMED`

Support-sufficient fail:

`NO_ACTIONABLE_RB_PD2_YARD_DIFFICULTY_WIDTH_FORWARD_CONFIRMATION`

Insufficient season support:

`NO_FORWARD_CONFIRMATION_INSUFFICIENT_SUPPORT`

Even PASS only enables a separate promotion review.

---

## 5. WHAT IS ALREADY IMPLEMENTED

The following is real code, not planning:

### §7 exact empirical capture

Integrated from the accepted Claude contract and hardened further.

Current behavior:

- exact production empirical `adjusted_outcomes` captured in-process;
- sportsbook-expanded rows collapse to one scientific football key;
- repeated football key must have identical draw digest / count / target mean / mc_proj;
- mismatch invalidates research;
- exact float64 arrays persisted losslessly;
- digest verified;
- session provenance;
- run/session isolation;
- no stale-buffer carryover;
- no week filter at the capture seam;
- research failures become sentinels and do not abort canonical pricing.

Real-slate dry-run evidence from Claude before credits ran out:

- run `35545877615`;
- 52 captured RB player-games;
- 31 teams;
- zero football-key collisions;
- mean 1.92 books per player-game;
- zero digest mismatches;
- zero dropped-before-seam;
- valid session;
- 52/52 NPZ round trips;
- 52/52 frozen §8 transforms succeeded on 25,000-draw arrays;
- production board untouched.

A second run `35548422673` confirmed sportsbook `commence_time` is absent at
the pricing source; therefore kickoff authority correctly comes from the NFL
schedule provider rather than odds metadata.

### §3 historical difficulty state

Implemented.

2025 predictor history uses the frozen 2024-OOS ensemble weights:

- MC = `0.5569542426070742`
- ML = `0.4430457573929258`
- State = 0
- fit scope = `all_2024_oos_frozen_for_2025`

Player identity uses the canonical `player_clean_key` path.

Stable hard identity fingerprints:

- `manual_name_overrides_sha256`
- `canonical_names_py_sha256`

The mutable weekly `roles_ourlads.csv` whole-file SHA is retained as diagnostic
provenance only, not a hard cross-week equality gate.

### Certified 2026 Week-1 history

Workflow:

`RB PD2 Completed 2026 Week1 History V1`

Green run:

`35619889214`

Artifact:

`rb-pd2-completed-2026-week1-history-v1`

Artifact ID:

`10649235022`

Result:

- 107 verified completed Week-1 rows;
- exact P3/STACK1 parent values retained;
- parity recomputed mechanically, not trusted from a boolean;
- zero prospective observations created.

### Combined forward-history artifact

Green build:

`35620321000`

Artifact:

`rb-pd2-forward-history-v1`

Artifact ID:

`10648116139`

Digest:

`sha256:52e56ce090576480d9b8cf4ec1999bfaf9e6884e6a0d1266efc075f1afac3af8`

Verified content at that checkpoint:

- 1,500 rows;
- 168 players;
- 808 scoreable rows;
- maximum strict-prior reference N = 882;
- includes all 107 certified Week-1 rows.

### §8 transform

Implemented against the persisted empirical arrays.

### §9 immutable lock

Implemented.

Hard lock checks include:

- valid capture receipt;
- stable identity fingerprint match;
- exact baseline digest;
- canonical team/opponent schedule match;
- authoritative `kickoff_utc`;
- capture timestamp < kickoff;
- lock timestamp < kickoff;
- lock/capture on or after frozen prospective start;
- >=4 prior games;
- reference N >=100;
- exact mean neutrality;
- no sportsbook inputs in candidate;
- no production mutation;
- no outcome present at lock.

### Live Full Slate wiring

`.github/workflows/full-slate.yml` now has manual input:

`rb_pd2_shadow_capture`

Default is truly OFF/empty.

When explicitly enabled with a paid live Full Slate:

1. production pricing runs;
2. exact §7 capture occurs;
3. forward-history artifact is restored;
4. §8/§9 lock assembly runs;
5. dedicated research artifact uploads immediately.

Research history restore, lock assembly and lock upload are non-blocking for
canonical pricing.

No paid OddsAPI pull was triggered during this implementation work.

---

## 6. LATE REVIEW DEFECTS — STATUS

PR #625 automated review raised four later issues.

Three are fixed and their threads are resolved:

### Fixed — artifact identity verification

The workflow now verifies the exact pinned artifact:

- artifact ID;
- artifact name;
- artifact digest;
- expired=false;

before download.

### Fixed — contain every shadow exception

Production now enters research through no-raise wrappers.

Unexpected failures in expectation/capture/completeness/finalization cannot abort
canonical pricing.

### Fixed — strict lineage certification parsing

No `astype(bool)` truthiness trap.

Missing, malformed and literal `"False"` certification fail closed.

### STILL OPEN — history freshness before every target week

This is the only remaining blocker.

The assembler now correctly enforces:

`history completed through target_week - 1`

and requires current-season history weeks to be contiguous.

That is good.

But the live workflow still pins the history artifact from run `35620321000`,
which only contains completed 2026 Week 1.

Therefore:

- a Week-3 capture will correctly fail as stale unless completed Week 2 is
  leakage-safely certified and appended first;
- later target weeks would fail similarly if the history artifact is not advanced.

This is not a modeling/science issue. It is the final operational requirement to
make observation #1 legitimate.

Do not merge PR #625 until this path is implemented and tested.

---

## 7. EXACT NEXT ACTION — FINISH THE CURRENT WORK

The next chat should **not** restart §7/§8/§9.

It should attack only the remaining history-advance blocker.

Required sequence:

1. Verify current branch / PR / exact head first.
2. Re-read unresolved PR #625 review thread `PRRT_kwDOQAMuU86ka3Dq`.
3. Design the smallest leakage-safe way to advance predictor history through every
   completed week before the next target lock.
4. For the first Week-3 lock, Week 2 must be certified after Week 2 is complete.
5. Week-2 history must use a legitimate pregame football projection lineage;
   do not reconstruct it from outcomes or sportsbook information.
6. Append completed Week-2 outcomes only after the pregame projection frame is
   frozen/certified.
7. Rebuild forward history through Week 2.
8. Update/pin the exact green artifact ID + digest used by the live lock.
9. Add regression proof that:
   - target Week 3 accepts history through Week 2;
   - history through only Week 1 fails closed;
   - future-week history fails closed;
   - missing week / noncontiguous current-season history fails closed.
10. Re-run Repo CI + preserved replay + automated review.
11. Resolve the final review thread only after the real path is green.
12. Merge PR #625.
13. Update `main` handoff immediately after merge.

The 2026 Week-2 slate is not yet fully complete at this handoff time; MNF is still
pending. Do not fabricate Week-2 final outcomes before completion.

A paid Full Slate is **not** required to finish this plumbing.

The first future paid Full Slate with shadow capture still requires explicit user
authorization because it consumes OddsAPI credits.

---

## 8. AFTER PR #625 IS FINISHED — RETURN TO ACTUAL MODEL IMPROVEMENT

Do not spend another session extending shadow infrastructure unless a real defect
appears.

The user wants actual metric improvement.

### First principle

Separate:

- **mean / point projection quality** — MAE, RMSE, bias, correlation;
- **distribution quality** — CRPS, coverage, Brier/tail probabilities.

PD2 addresses the second category only.

### RB: what the evidence says

M96A is canonical and must not be rerun.

2025 RB/FB n=1,393:

- pregame rush-yard MAE = 21.0312;
- perfect carries with frozen efficiency = 13.3535;
- opportunity recovery = 7.6777 yards;
- perfect efficiency with frozen carries = 14.3055;
- efficiency recovery = 6.7256 yards;
- opportunity-dominant games = 59.73%;
- efficiency-dominant games = 40.27%.

Regime finding:

- 0–5 carries: opportunity problem;
- 6–10: mixed / efficiency slightly larger;
- 11–14: efficiency problem;
- 15–19: efficiency problem;
- 20+ and 25+: large opportunity problem.

Do not re-derive those numbers.

### M96E prohibition

Further retrospective RB router/threshold/feature variants against exposed 2025
are forbidden.

Only two RB continuations are sanctioned:

1. genuinely prospective / untouched 2026 evidence;
2. a separately justified new-data source that does not retune against exposed
   historical outcomes.

### Strongest new-data lead already found

Claude's readiness branch:

`research/rb-new-data-readiness-v1@355ce5c7`

Document:

`docs/research/RB_NEW_DATA_READINESS_V1.md`

Key finding:

**the current production model has no mechanism for an unavailable teammate RB
to increase another back's carries.**

Existing live/free data already include:

- teammate `report_status`;
- `practice_status`;
- injury rows keyed by `gsis_id`;
- derivable backfield vacancy state.

Current production injury logic:

- WR1 absence can redistribute target share;
- a player's own injury can haircut his own workload;
- there is no RB-to-RB rushing opportunity propagation.

This is a materially more promising **mean-information** direction than more
historical threshold tuning because it is genuinely new pregame information and
directly targets the high-volume opportunity failure identified by M96A.

### Pending production-correctness audit

Claude was assigned an audit of the current
`OUT/DOUBTFUL/IR/PUP` self-haircut rule, but Claude is now out of credits and
has not delivered the audit.

GPT-5.6 must take this over solo.

Question:

Does a confirmed inactive RB retain nonzero rushing opportunity anywhere in the
actual preserved/live production path, or does a later layer remove/zero him?

Do not assume the source-line behavior equals final production behavior.

Audit first.

If a confirmed inactive player really reaches pricing with nonzero rushing
opportunity, treat it as a production-correctness issue separately from new
research.

### Snap-share lead

Free 2026 `load_snap_counts` exists.

Current readiness evidence:

- 2026 weeks 1–2 available;
- 211 RB/FB rows;
- 117 players;
- all 32 teams;
- fields include `offense_snaps` and `offense_pct`.

Limitations:

- postgame data, so pregame use must be lagged / strictly prior;
- PFR identity needs canonical resolution;
- snap share is a coarse opportunity proxy, not routes run.

This is another legitimate new-data lane after the first forward lock /
prospective protocol is ready.

### Routes run

nflverse participation ends at 2025.

No live 2026 route-run source currently exists in the repo.

If exact routes are desired, the user may need a commercial current-season
source. Do not build a historical routes feature that cannot be computed live.

---

## 9. WIDER POSITION MODEL-IMPROVEMENT PROGRAM

After the RB forward implementation is landed, do not assume RB is the only
problem.

Use the completed Week-2 slate, once final, as a fresh diagnostic checkpoint.

Build a current production performance table by position/market with at minimum:

- sample size;
- MAE;
- RMSE;
- bias;
- correlation;
- CRPS where empirical distributions exist;
- 80/90 coverage where applicable;
- relevant threshold Brier scores;
- miss decomposition: opportunity vs efficiency vs tail where already supported.

Compare to the correct frozen/authority-exact baseline for each position.

Then choose the next lane from evidence, not preference.

Known boundaries:

### QB

M89/M90 mean synthesis and QB-C2 distribution are existing authorities.

Broad generic QB mean hunting is closed unless:

- a specific architecture diagnostic identifies a failed layer; or
- genuinely new independent football information becomes available.

Do not reopen QB Conditional Analog V1.

### WR

WR receiving-yard translation/efficiency remains a known weakness, especially
high-efficiency/right-tail games.

M38 + WR_R15 remain production authorities.

WR receptions/opportunity are comparatively healthy.

The QB-C2 -> WR1 shared-tail selector is closed.

Before any new WR candidate:

- anti-retest audit;
- feature-availability audit;
- freeze one genuinely new leakage-safe mechanism.

### TE

Do not mutate TE-R5P without evidence from the current full-slate grade that TE
is a priority failure.

### RB receiving

R23-R27D conventional receiving-efficiency family is closed.

Do not reopen without genuinely new strict-prior football information.

### General rule

Do not chase sportsbook lines as football features.

Sportsbook remains downstream comparison / calibration evidence only.

---

## 10. WEEK-2 CHECKPOINT AFTER MNF

Once all Week-2 games are final:

1. run/inspect the authoritative Week-2 production grade;
2. certify Week-2 RB predictor history for Week-3 forward locks;
3. build a position/market performance table;
4. identify the largest remaining production error surfaces;
5. distinguish mechanical/data correctness problems from scientific model
   weaknesses;
6. start the highest-value sanctioned model-improvement lane.

This is where the project returns to direct metric improvement.

---

## 11. CLAUDE STATUS

Claude is temporarily unavailable because the user is out of credits.

Do not wait for Claude.

Do not assign work that blocks progress.

Continue autonomously with GPT-5.6.

When Claude credits return:

- give Claude the updated GitHub handoff;
- use Issue #535 for independent review/challenge;
- do not make Claude repeat completed work.

Latest useful Claude artifact before credits ended:

`research/rb-new-data-readiness-v1@355ce5c7`

Claude's injury self-haircut audit assignment was not completed.

---

## 12. CLOSED / DO-NOT-REOPEN FAMILIES

Do not rescue or retune:

- QB Conditional Analog V1;
- Historical Analog State V1;
- Role/Room Concentration V1;
- Event-Regime Reliability V1;
- RB PD3/PD4/PD5 mean-correction family;
- RB receiving efficiency R23-R27D;
- defensive-front cohesion;
- OL cohesion;
- BDB2023 blocker-rusher pairing;
- M77;
- M80-M81;
- M95T detached tail overlays;
- M96 retrospective router/threshold/feature work against exposed 2025;
- QB-C2 -> WR1 shared-tail selector.

Do not trigger a paid OddsAPI pull without explicit user authorization.

---

## 13. SUCCESSFUL STOP POINT FOR THIS CHAT

At handoff creation:

- production `main` = `76e5e0452a88018b95ebc212fb0d1b21a2ca90a3`;
- active branch code head = `ab7e268429d4e33000d84388246aaaf1aad3f99c`;
- PR #625 = clean/mergeable;
- exact-head Repo CI = green;
- exact-head preserved replay = green;
- three late review defects = fixed and resolved;
- one review blocker remains open: history must advance through target_week-1;
- no paid odds pull has been run;
- no prospective observation exists yet;
- no target outcome has been graded under the forward study;
- no production science has been changed by PD2;
- Claude is unavailable; GPT-5.6 must continue solo.

**Immediate next move: finish the history-advance path, land PR #625, then pivot to actual model-improvement work.**
