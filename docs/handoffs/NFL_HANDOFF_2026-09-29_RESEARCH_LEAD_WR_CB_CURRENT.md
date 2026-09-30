# NFL HANDOFF — 2026-09-29 — RESEARCH LEAD / WR-CB SOURCE FRONTIER CURRENT

GitHub is canonical. This handoff is designed so the next chat can continue as the same research lead without re-reading the entire project history.

## 0. READ ORDER — DO NOT RECURSE

Read only, in this order:

1. `AGENTS.md`
2. `CURRENT_NFL_RESEARCH_HANDOFF.md` — newest top checkpoint only
3. this file
4. Issue #535 from comment `5899500522` onward, with special attention to:
   - `5899645764` research-lead split
   - `5899710345` RB route-volume source result
   - `5899966307` legacy coverage mechanical-materiality audit
   - `5900122307` CLV PR #663 checkpoint
   - `5901422664` / `5901433362` legacy coverage retirement / merge
   - `5901608387` free FantasyAlarm WR-CB archive discovery
   - `5901927972` later contradictory WR-CB source note — treat with caution; reconcile against live repo
   - GSIS checkpoint `5881240970`
5. query GitHub live for:
   - `main`
   - `research-week3-postmortem-execution-v1`
   - `research-wr-cb-free-archive-v1` / PR #665
   - `research-market-snapshot-history-v1` / PR #663
   - `research-rb-opponent-defender-injury-readiness-v1`
   - `repair-specialist-rng-isolation-v1`
   - PR #662
   - current Actions / CI

Do not recursively read old handoffs unless a concrete integrity question requires one.

---

## 1. CANONICAL LIVE PRODUCTION STATE

At handoff write, canonical `main` is:

`deea0f68202c9ab6e85fa616f7444d0008ee10af`

This is newer than the earlier Week-3 continuity main `a99be87e...`.

### Critical production fact: legacy coverage heuristic is REMOVED

PR #664:
- title: `Production: retire unsupported legacy WR coverage heuristic`
- merged
- merged main SHA: `deea0f68202c9ab6e85fa616f7444d0008ee10af`
- Repo CI `36648664504` passed before merge
- preserved replay red state was unrelated expired artifact `run_35282021679`

Live-code verification on current main:
- `rules_v2.py` no longer defines `coverage_penalty()`
- `simulation_rules.py` contains only a retirement comment; it does not call the function
- static WR coverage multipliers `0.92 / 0.94 / 1.06 / 1.04` are gone

Do **not** re-add or resurrect this heuristic merely because later Issue comments describe it as active. Those comments are stale/concurrent-chat drift relative to live main.

No replacement coefficient was added.
No paid WR-CB source was added.
No OddsAPI spend occurred.

---

## 2. WEEK 3 / WEEKS 1-3 SCIENCE IS CLOSED

Canonical Weeks 1-3:
- 1,240 decided bets
- 629-611
- 50.7%
- -40.72u
- model MAE 17.55 vs selected-line MAE 16.36
- model signed bias -5.36
- no simple position/market/side slice survives the frozen game-cluster-aware multiple-comparisons gate

Do not rerun:
- Weeks 1-3 settlement/postmortem
- RB Vacancy W3
- Public Intent W3
- Receiving Rule Semantics W3
- RB-PD2 W3 Observation #1
- Availability -> Opportunity W3 grading
unless a concrete integrity defect is found.

### Frozen Week-3 dispositions

RB Vacancy:
`WEEK3_OBSERVATIONAL_MIXED`

Public Intent:
`WEEK3_PUBLIC_INTENT_DIRECTIONALLY_INFORMATIVE`
observational only

Receiving Rule Semantics:
no cell promoted; continue exact prospective cells unchanged

RB-PD2:
`NO_FORWARD_CONFIRMATION_INSUFFICIENT_SUPPORT`
- 1 / 8 required weeks
- 46 / 400 rows
- HOLD; not PASS, not scientific FAIL

Availability -> Opportunity:
`NO_VALID_POSTGAME_GRADING_CONTRACT_DO_NOT_GRADE_POST_HOC`
- structural rule-order gap remains confirmed
- no Week-3 predictive PASS/FAIL exists

Consolidated matrix:
`docs/research/WEEK3_PROSPECTIVE_DISPOSITION_MATRIX_V1.md`

---

## 3. PROJECTION-AUTHORITY NEW SCIENCE

Weeks 1-3 line-conflict diagnostic was frozen before result inspection.

Primary result:
`NO_CLEAR_CURRENT_SEASON_AUTHORITY_LINE_CONFLICT_SIGNAL`

Crossing the sportsbook line itself was not a useful failure state:
- crossed n=119 / 43 games
- crossed win rate 50.42%
- non-crossed 50.76%
- clustered intervals cross zero

Do not create a crossed-line exclusion rule.

### Secondary predeclared discovery-only state

Same-side late authority movement:

`SAME_SIDE_STRENGTHENED`
- n=250
- 44.0%
- -38.15u
- -15.26% ROI
- final MAE worse than MC

`SAME_SIDE_WEAKENED`
- n=471
- 55.20%
- +24.44u
- +5.19% ROI
- final MAE better than MC

This is discovery-only because Weeks 1-3 outcomes were already exposed.

Forward plan:
`docs/research/WEEK4_PLUS_PROJECTION_AUTHORITY_MOVE_DIRECTION_FORWARD_V1_PLAN.md`

Support floor:
- >=8 future eligible weeks beginning Week 4
- >=400 strengthened
- >=400 weakened

Before then:
`FORWARD_OBSERVATION_ONLY_INSUFFICIENT_SUPPORT`

No sportsbook input may move upstream into football generation.

---

## 4. RESEARCH-LEAD SPLIT / CLAUDE COORDINATION

Issue #535 comment `5899645764` established the collaboration model.

GPT-5.6 owns:
- RB route-volume source readiness / prospective capture
- cross-audit of Claude's opponent-injury conclusion
- architecture research such as CLV evidence retention

Claude was assigned:
- opponent-defender injury propagation into matchup context
- anti-retest first
- source-readiness before predictive candidate
- no Week-3 outcome fitting
- no sportsbook input upstream

### Important current collaboration state

A branch now exists:

`research-rb-opponent-defender-injury-readiness-v1`

At the last Issue #535 sweep there was **no Claude result comment yet** after the assignment.

Next chat must query that branch and Issue #535 before duplicating Claude's work.
If Claude has posted, cross-audit it.
If he has only pushed branch work, inspect the actual branch/files before starting a parallel lane.

Use Issue #535 as the shared lab notebook.

---

## 5. RB ROUTE-VOLUME FRONTIER

Result doc:
`docs/research/RB_ROUTE_VOLUME_SOURCE_READINESS_V1.md`

Disposition:
`LIVE_RB_ROUTE_VOLUME_CONFIRMED_HISTORICAL_WEEKLY_PARITY_NOT_CLEARED`

What changed:
- public live 2026 player route-volume data exists
- HeatRadar exposes current weekly Routes / Route % / TPRR / YPRR / target share
- StatRankings exposes current 2026 plus historical season selectors

Critical semantic guard:
- nflverse participation `route` is a route label for the primary receiver on a play
- it is **not** equivalent to total player routes run
- do not create a fake historical bridge by counting nflverse route labels

Prospective plan:
`docs/research/RB_ROUTE_VOLUME_PROSPECTIVE_CAPTURE_V1_PLAN.md`

Implementation on `research-week3-postmortem-execution-v1`:
- `scripts/research/materialize_rb_route_volume_prospective_capture_v1.py`
- fail-close tests in `tests/test_materialize_rb_route_volume_prospective_capture_v1.py`

The prospective idea:
- archive free weekly/current route data
- difference immutable cumulative snapshots where valid
- cross-check independent current sources
- never coerce missing player to zero
- trades/provider corrections fail closed

Do not freeze a predictive route-volume model until the parity/source gate clears.

No paid route source without user approval.

---

## 6. LEGACY COVERAGE HEURISTIC — WHY IT WAS REMOVED

Before removal, an exact outcome-free Week-3 mechanical replay used:
- run `36293274478`
- artifact `10923570170`
- digest `sha256:5a3d4f64592c70553e66dd51bb3bff45263d2900f4d270e370353fa60ea1c480`

Reproduction gates were exact to floating-point tolerance.

Week-3 current-stack state before PR #664:
- 161 WR football rows
- 131 affected = 81.4%
- 27 / 30 teams affected
- zero frozen nonblank `primary_cb` assignments
- every active adjustment came from generic `heavy_zone`

Remove-only deterministic mechanical effect:
- affected WR final target entitlement: about -2.63%
- affected WR YPT: -3.846%
- affected entitlement×YPT mean kernel: about -6.37%
- target-conservation spillover: TE and RB/FB pools about +2.9% on affected teams

Disposition before removal:
`LEGACY_COVERAGE_HEURISTIC_MECHANICALLY_MATERIAL_LOW_SELECTIVITY_UNVALIDATED`

The old 2025 no-coverage ablation was near-neutral/slightly favorable to removal and never positively validated the fixed coefficients.

This mechanical audit did **not** use Week-3 outcomes.

PR #664 then removed the unsupported heuristic from production.

The previously frozen Week-4+ remove-only shadow plan is now operationally superseded by the merged production removal. Do not automatically run it as if the old heuristic were still production control. If a future qualification study is desired, freeze a new contract that treats current no-coverage production as the control.

---

## 7. WR-CB PLAYER-LEVEL SOURCE FRONTIER — CURRENT NUANCED STATE

The user explicitly does **not** want to pay for WR-CB matchup data.

### Production state

Direct player-level WR↔CB assignments are already fail-closed when unavailable:
- Week-3 frozen artifact had zero `primary_cb` assignments
- production validators gate direct matchup unavailable state
- do not infer fake assignments from on-field defenders or nflverse

Production decision:
`PLAYER_LEVEL_WR_CB_ASSIGNMENT_PRODUCTION_UNAVAILABLE_KEEP_GATED_OFF`

Even before PR #664, player-level direct assignments were not the thing driving the old coverage effect; the generic team-zone fallback was.

### Free-source discovery

A later source audit found public FantasyAlarm WR/CB archive pages apparently spanning 2021-2026 with explicit pregame WR↔CB pairings/alignment buckets.

Research branch:
`research-wr-cb-free-archive-v1`

Draft PR:
**#665 — Research: recover free historical WR-CB assignment archive**

Known implementation checkpoints from Issue #535:
- source-readiness doc:
  `docs/research/WR_CB_FREE_HISTORICAL_ARCHIVE_SOURCE_READINESS_V1.md`
- manifest-driven acquirer
- 16 seed pages spanning 2021-2026
- parser fail-close tests
- live-source audit workflow
- editorial Safe/Moderate/Risky grade marked `NOT_MODEL_ELIGIBLE`

### Critical contradiction to resolve FIRST

Issue #535 contains two concurrent-chat conclusions:
- `5901608387`: free historical FantasyAlarm archive may unblock true assignment science
- `5901927972`: later note says no stable machine-reproducible historical archive is yet cleared

Do **not** pick one from chronology alone.

Exact next action for WR-CB:
1. query PR #665 and branch `research-wr-cb-free-archive-v1` live;
2. inspect its current source-audit result / CI / acquisition output;
3. decide whether the archive actually clears:
   - free
   - stable/retrievable
   - explicit pregame WR↔CB factual pairing
   - week/game grain
   - sufficient historical coverage
   - parse/identity quality
   - timestamp/temporal integrity
4. if source gate fails: close research source lane, keep direct matchup gated off, spend $0;
5. if source gate clears: proceed research-only to the already source-blocked `TOP_WEAPON_ESCAPE_HATCH` style test using factual pairing/alignment only.

Even if PR #665 clears, **do not restore the deleted legacy 0.92/0.94/1.06/1.04 heuristic**.
A true assignment model would be entirely new science with a new frozen contract.

Do not use FantasyAlarm editorial matchup grades as model features unless separately justified; initial eligible information is factual pairing/alignment only.

---

## 8. DEFENSIVE-SCHEME HISTORY — DO NOT CONFUSE WITH WR-CB ASSIGNMENT

The repo has already tested team-level defensive scheme/context:
- strict-prior man/zone rates
- coverage family labels
- box rates
- pressure / defensive context

Prior authoritative results include:
- M56 richer static/lagged defensive matchup -> `SIGNAL_SCREEN_FAILED`
- M83 conditional adaptive defensive gameplan -> `NO_DEFENSIVE_ADAPTATION_MECHANISM`
- 2025 team-level coverage man/zone ablation -> near-neutral/slightly favorable to removal

So do not reopen generic man/zone or defensive-context research under a new name just because true WR-CB source work is active.

The potential novelty in PR #665 is **player-level responsibility/alignment**, not generic zone/man tendency.

---

## 9. CLV / MARKET SNAPSHOT ARCHITECTURE — PR #663

Branch:
`research-market-snapshot-history-v1`

Draft PR:
**#663 — Research: preserve immutable market snapshots for CLV**

Purpose:
- preserve append-only snapshots from already-acquired Full Slate market data
- zero marginal odds spend
- never fetch odds itself
- market evidence stays downstream only

Frozen timing:
- T0-30 before kickoff: `VALID_T30_CLOSE` and may be called CLV
- T30-60: late-market movement
- >T60: pregame movement only
- no later same-book capture: no valid close
- no research-only paid pull merely to obtain a close

Architecture:
- exact source `props_raw.csv.fetched_at`
- exact event kickoff `commence_time`
- same-book identity
- append-only immutable snapshots
- existing latest-line ledger remains separate

Two existing provenance problems were found:
1. old archive workflow stamped archive-workflow SHA rather than triggering Full Slate SHA
2. old archive logic could resolve current runtime week rather than assert source-board week

### Latest known mechanical CI issue

Run `36648161557` failed one new unit test after 624 passed:
`test_snapshot_archive_is_append_only_per_run`

Failure was a KeyError around transient `operation_status` after the immutable manifest fix.

The companion replay run `36648161564` failed before model execution because old preserved source artifact `run_35282021679` is expired/not found.

Important: concurrent work may have pushed newer PR #663 commits after those run IDs. Therefore the next chat must query PR #663 live before editing or rerunning anything.

Do not interpret the expired replay failure as a CLV scientific failure.

---

## 10. SPECIALIST RNG REPAIR

Branch:
`repair-specialist-rng-isolation-v1`

Original failed run:
`36504441918`

Diagnosis:
- isolation scope passed
- TE/WR protected arrays had zero drift
- intended specialist arrays changed
- QB-C2 invariants passed
- paid artifact/source authority passed
- red gate was exact research-to-production finite-sample fingerprint parity

Concrete defect identified:
- frozen research semantic seed labels:
  - `TE_CHANGED_ROOM`
  - `WR_CHANGED_ROOM`
- production candidate had renamed them:
  - `TE_R5P_ROOM`
  - `WR_R15_ROOM`

Because room label is part of the deterministic RNG key, the rename changes the finite substream.

Narrow patch:
`30333ec2b853d5ba7bd420b74d63c53f097b5fff`

No football coefficients/inputs/selection logic were changed.

Do not call PASS until the exact frozen post-patch fingerprint validation is observed live.
Do not weaken the gate.

This RNG lane remains important because future full-array A/B research needs deterministic unrelated-array isolation.

---

## 11. GSIS

PR #662 remains research-only/draft.

Private Week-3 snapshot:
- baseline-only for prospective temporal work
- do not publish raw GSIS

Lineup Detail:
`INCREMENTAL_CURRENT_STATE_CONFIRMED`
but temporal state:
`NEEDS_PROSPECTIVE_HISTORY`

Formation Usage:
`INCREMENTAL_CURRENT_STATE_CONFIRMED`

Lineup Combinations:
`MOSTLY_REDUNDANT_ONE_NARROW_FIELD`
- only `Unique Starting Lineups` worth prospective capture

Do not broaden GSIS acquisition into closed/duplicative reports.

---

## 12. RESEARCH PRIORITY FROM HERE

The project is not at "rerun Week 3." It is now a forward-information / architecture phase.

Priority order:

### A. Resolve PR #665 WR-CB source gate
This is the immediate continuation of the user's last question.
Determine whether free FantasyAlarm archive is genuinely production-grade enough for historical research.
Spend $0.
Keep production direct-matchup gated off regardless.

### B. Cross-audit Claude opponent-injury lane
Inspect `research-rb-opponent-defender-injury-readiness-v1` and any new Issue #535 comment before doing duplicate work.

### C. Finish PR #663 mechanics
Only after querying the current live branch/head; do not blindly patch the already-seen failure if another chat fixed it.

### D. Validate specialist RNG repair
Exact frozen fingerprint only.

### E. Continue prospective captures
- RB-PD2 unchanged
- authority move-direction Week-4+
- RB route-volume
- GSIS lineup/formation temporal snapshots

No outcome-driven Week-3 production rescue.

---

## 13. DO-NOT-DO LIST

- no paid OddsAPI without explicit user approval
- no paid WR-CB source
- no reintroduction of deleted coverage heuristic
- no fake WR-CB reconstruction from nflverse
- no editorial matchup grade as a model feature by default
- no Week-3 post-hoc coefficient fit
- no regrading closed Week-3 lanes
- no RB-PD2 PASS/FAIL before support floors
- no generic man/zone retest
- no scalar probability-calibration retread
- no recursive old-handoff crawl
- no raw GSIS publication
- no weakening RNG equivalence gates

---

## 14. CONTINUITY / CONCURRENT-CHAT RULE

Multiple chats are actively touching this repo.

When Issue comments disagree:
1. live `main` code is canonical for production state;
2. live branch/PR code is canonical for that research lane;
3. newest frozen plan/result on that branch beats an older narrative comment;
4. Issue #535 is coordination, not a substitute for source-code verification.

Before editing any open branch, query its current head and latest Actions.

The most important reconciled example:
- stale Issue wording says legacy coverage rule is active;
- live `main@deea0f68...` proves it has been removed.

Do not undo validated concurrent work.

---

## 15. USER INTENT / WORKING STYLE

The user is exhausted and explicitly asked GPT-5.6 to take the research lead.
Do not make them re-explain the project or choose every next micro-step.

The user can loop Claude in. Coordinate through Issue #535 with clear ownership and cross-audits.

The standard remains:
- rigorous
- frozen before outcomes
- source-parity first
- anti-retest
- no fake precision
- no paid-source dependency unless explicitly approved
- push useful work forward without asking unnecessary clarifying questions

